"""
Result analysis functions for ECLIPSE instrument response simulations.

This module provides functions for loading, analyzing, and visualizing
instrument response simulation results.
"""

import warnings

import numpy as np
import astropy.units as u
import astropy.constants as const
import sunpy.map
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any
from ndcube import NDCube
from tqdm import tqdm

from .results_file import load_results
from .utils import has_wrong_velocity_sign



def _to_canonical_scalar(val):
    """
    Convert a value to a canonical scalar for parameter comparison.

    Quantities are reduced to their SI value so comparisons are unit-agnostic
    (e.g. ``40 * u.s`` and ``40000 * u.ms`` both give ``40.0``).
    Offset-unit quantities (e.g. Celsius) are converted to Kelvin first.
    """
    if not hasattr(val, "unit"):
        return val
    try:
        return float(val.si.value)
    except Exception:
        try:
            return float(val.to(u.K, equivalencies=u.temperature()).value)
        except Exception:
            return float(val.value)


def load_instrument_response_results(filepath: str | Path,
                                     allow_wrong_velocity_sign: bool = False,
                                     ) -> Dict[str, Any]:
    """
    Load instrument response results and reconstruct signals for compatibility.
    Fit statistics are kept with units separated.

    Parameters
    ----------
    filepath : str or Path
        Path to the results file, ``run/result/<config name>.h5``. A pickle,
        as ECLIPSE 0.11.0 and earlier wrote them, is read with a warning
        unless it is named as an HDF5 file, and a ``.pkl`` name that no
        longer exists reads the ``.h5`` beside it.
    allow_wrong_velocity_sign : bool, optional
        Load a results file made from a synthesis file that an older ECLIPSE
        wrote for a view along x or z, with a warning instead of an error.
        Every velocity in such a file has the wrong sign, and its spectra are
        mirrored in wavelength about the rest wavelength of each line, so
        anything that interacts with a blend or another feature on one side
        of a line can differ too.  Older views along y were already right and
        load without it.

    Returns
    -------
    dict
        Dictionary containing all results and metadata with reconstructed signals.
    """
    data = load_results(filepath, _stacklevel=3)
    # An old pickle can hold anything.
    if not isinstance(data, dict):
        raise ValueError(f"{filepath} does not hold the results of an instrument simulation.")

    # Refuse results made from synthesis files written before the Doppler
    # sign was fixed, for the views whose sign it changed.  Uniform intensity
    # runs have no synthesis file and no velocities, so they are unaffected.
    cube_sim = data.get("cube_sim")
    if cube_sim is not None and has_wrong_velocity_sign(cube_sim.meta):
        axis = (cube_sim.meta or {}).get("integration_axis", "z")
        message = (
            f"{filepath} was made from a synthesis file written by an older "
            "ECLIPSE, which used the simulation velocity along the line of "
            "sight without turning it into a velocity away from the observer. "
            f"For this view along {axis}, every velocity in it has the wrong "
            "sign: flows towards the observer are redshifted."
        )
        if not allow_wrong_velocity_sign:
            raise ValueError(
                message + " Re-run the synthesis and then the simulation with "
                "this version, or pass allow_wrong_velocity_sign=True to load "
                "it anyway."
            )
        warnings.warn(message, stacklevel=2)

    # A file that is not what the simulation writes is refused naming the
    # file. A signal's unit has to be a unit, as two arrays could multiply to
    # far more than either, and a signal the file holds once is multiplied
    # once, however many combinations share it.
    signals = {}
    try:
        combinations = data["results"]["all_combinations"]
        for param_key, combination_results in tqdm(combinations.items(), desc="Reconstructing results", leave=False):
            # Refuse files written before the cube axis order was fixed (issue
            # #12).  Those store signals as (x, y, wavelength) with an HPLT-first
            # WCS; the maps made from one here would come out transposed.
            wcs_ctype = combination_results["first_signal_wcs"].wcs.ctype
            if str(wcs_ctype[1]).startswith("HPLT"):
                raise ValueError(
                    f"{filepath} was written by an older ECLIPSE that stored "
                    "cubes as (x, y, wavelength). Cubes are now "
                    "(y, x, wavelength). Re-run the simulation with this "
                    "version to regenerate the file."
                )
            # Reconstruct signal NDCubes
            for kind in ("dn", "photon"):
                signal = combination_results[f"first_{kind}_signal_data"]
                unit = combination_results[f"first_{kind}_signal_unit"]
                if not isinstance(unit, u.UnitBase):
                    raise ValueError(f"{filepath}: the {kind} signal's unit is of type "
                                     f"{type(unit).__name__}, not a unit.")
                if (id(signal), unit) not in signals:
                    signals[id(signal), unit] = signal * unit
                combination_results[f"first_{kind}_signal"] = NDCube(
                    signals[id(signal), unit], wcs=combination_results["first_signal_wcs"])
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"{filepath} does not hold the results of an instrument simulation: "
                         f"{type(error).__name__}: {error}") from error

    return data


def get_parameter_combinations(results: Dict[str, Any]) -> List[Dict]:
    """
    Get all parameter combinations that were simulated.

    Returns a list of the ``parameters`` dicts (one per combination), each
    using ``section.attribute`` key names.  This is more useful than the raw
    hash keys stored internally.

    Parameters
    ----------
    results : dict
        Results dictionary from load_instrument_response_results.

    Returns
    -------
    list of dict
        One parameters dict per simulated combination.
    """
    return [combo["parameters"] for combo in results["results"]["all_combinations"].values()]


# Results files written before fitted components were stored by name hold
# only the per-parameter arrays.
_UNNAMED_FITS = (
    "This results file was written before fitted components were stored by "
    "name, so it only holds per-parameter statistics. Re-run the simulation "
    "with this version to get them."
)


def _get_fit_stats(combination_results: Dict[str, Any], data_type: str) -> Dict[str, Any]:
    """The fit statistics of one signal, or a clear error if there are none."""
    fit_stats_key = f"{data_type}_fit_stats"
    if fit_stats_key not in combination_results:
        raise ValueError(f"No {fit_stats_key} found in combination results")
    fit_stats = combination_results[fit_stats_key]
    if fit_stats is None:
        raise ValueError(
            f"'{data_type}' signal was not fitted for this combination. "
            f"Check the 'fit_signals' setting in your YAML config."
        )
    return fit_stats


def list_fit_components(combination_results: Dict[str, Any],
                        data_type: str = "dn") -> List[str]:
    """
    Names of the fitted components, in fit order.

    Parameters
    ----------
    combination_results : dict
        Results for a specific parameter combination.
    data_type : str, optional
        Either "dn" or "photon".

    Returns
    -------
    list of str
        The names to pass as ``component`` to :func:`analyse_fit_statistics`
        and :func:`create_sunpy_maps_from_combo`.  The primary component's
        name is in the fit statistics under ``primary_component``.
    """
    fit_stats = _get_fit_stats(combination_results, data_type)
    if "components" not in fit_stats:
        raise ValueError(_UNNAMED_FITS)
    return list(fit_stats["components"])


def analyse_fit_statistics(
    combination_results: Dict[str, Any],
    rest_wavelength: u.Quantity | None = None,
    data_type: str = "dn",
    fit_config=None,
    component: str | None = None,
) -> Dict[str, Any]:
    """
    Velocity, line width and intensity statistics of one fitted component.
    
    Parameters
    ----------
    combination_results : dict
        Results for a specific parameter combination.
    rest_wavelength : u.Quantity, optional
        The rest wavelength velocities are measured from.  Results files
        record each component's own, and a value given here has to match
        it.  Only files written before components were stored by name need
        it.
    data_type : str, optional
        Either "dn" or "photon" to specify which fit statistics to analyze.
    fit_config : FitConfig, optional
        Only used for files written before components were stored by name,
        to find the primary component's parameters.
    component : str, optional
        Name of the component to analyse, from :func:`list_fit_components`.
        Defaults to the primary component.
        
    Returns
    -------
    dict
        ``v_first``, ``v_mean``, ``v_std``, ``v_true`` and ``v_err`` (truth
        minus mean) for the velocity, and ``w_first``, ``w_mean`` and
        ``w_std`` for the Gaussian width, where ``first`` is the first Monte
        Carlo iteration.  Files with components stored by name also give
        ``component``, ``rest_wavelength``, ``tied``, ``w_true``,
        ``i_first``, ``i_mean`` and ``i_std`` for the intensity (the fitted
        line's counts), ``failed_fits`` and ``n_iterations``.
    """
    fit_stats = _get_fit_stats(combination_results, data_type)
    ground_truth = combination_results["ground_truth"]
    fit_truth_data = ground_truth["fit_truth_data"]
    fit_truth_units = ground_truth["fit_truth_units"]

    if "components" in fit_stats:
        name = fit_stats["primary_component"] if component is None else component
        if name not in fit_stats["components"]:
            raise ValueError(
                f"No fitted component is named {name!r}. The components are "
                f"{list(fit_stats['components'])}."
            )
        comp = fit_stats["components"][name]
        rest = comp["rest_wavelength"]
        if rest_wavelength is not None and not u.isclose(rest_wavelength, rest,
                                                         rtol=1e-6):
            raise ValueError(
                f"rest_wavelength is {rest_wavelength}, but component "
                f"{name!r} was fitted at {rest}, and its velocities are "
                f"measured from that. Leave rest_wavelength out, or choose a "
                f"different component with component=."
            )
        truth = ground_truth["components"][name]
        velocity, width, intensity = comp["velocity"], comp["width"], comp["intensity"]
        return {
            "component": name,
            "rest_wavelength": rest,
            "tied": comp["tied"],
            "v_first": velocity["first"],
            "v_mean": velocity["mean"],
            "v_std": velocity["std"],
            "v_err": truth["velocity"] - velocity["mean"],
            "v_true": truth["velocity"],
            "w_first": width["first"],
            "w_mean": width["mean"],
            "w_std": width["std"],
            "w_true": truth["width"],
            "i_first": intensity["first"],
            "i_mean": intensity["mean"],
            "i_std": intensity["std"],
            "failed_fits": fit_stats["failed_fits"],
            "n_iterations": fit_stats["n_iterations"],
            "fit_stats": fit_stats,
            "fit_truth_data": fit_truth_data,
            "fit_truth_units": fit_truth_units,
        }

    if component is not None:
        raise ValueError(f"component={component!r} cannot be chosen. {_UNNAMED_FITS}")
    if rest_wavelength is None:
        raise ValueError(f"rest_wavelength is needed for this file. {_UNNAMED_FITS}")

    # Determine parameter indices for the primary component
    if fit_config is not None and not fit_config.is_single:
        idx_center = fit_config.idx_center
        idx_sigma = fit_config.idx_sigma
    else:
        idx_center = 1
        idx_sigma = 2

    # Extract data and units
    first_data = fit_stats["first_fit_data"]  # Shape: (ny, nx, n_params)
    mean_data = fit_stats["mean_data"]        # Shape: (ny, nx, n_params)
    std_data = fit_stats["std_data"]          # Shape: (ny, nx, n_params)
    units = fit_stats["units"]                # List of n_params astropy units
    
    # Get center statistics for the primary component
    center_unit = units[idx_center]
    center_first_q = first_data[..., idx_center] * center_unit
    center_mean_q = mean_data[..., idx_center] * center_unit
    center_std_q = std_data[..., idx_center] * center_unit
    
    # Get width statistics for the primary component
    width_unit = units[idx_sigma]
    width_first_q = first_data[..., idx_sigma] * width_unit
    width_mean_q = mean_data[..., idx_sigma] * width_unit
    width_std_q = std_data[..., idx_sigma] * width_unit
    
    # Convert centers to velocities using simple formula
    # v = (lambda - lambda0) / lambda0 * c
    def centers_to_velocity(centers_q, lambda0):
        """Convert wavelength centers to velocities"""
        velocity = ((centers_q - lambda0) / lambda0 * const.c).to(u.km / u.s)
        return velocity
    
    # Convert to velocities
    v_first = centers_to_velocity(center_first_q, rest_wavelength)
    v_mean = centers_to_velocity(center_mean_q, rest_wavelength)
    v_true = centers_to_velocity(fit_truth_data[..., idx_center] * fit_truth_units[idx_center], rest_wavelength)
    v_err = v_true - v_mean
    
    # Convert center std to velocity std using differential: dv/dlambda = c/lambda
    c = const.c.to(u.km / u.s)
    v_std = (c * center_std_q / rest_wavelength).to(u.km / u.s)
    
    return {
        "v_first": v_first,
        "v_mean": v_mean,
        "v_std": v_std,
        "v_err": v_err,
        "v_true": v_true,
        "w_first": width_first_q,
        "w_mean": width_mean_q,
        "w_std": width_std_q,
        "fit_stats": fit_stats,
        "fit_truth_data": fit_truth_data,
        "fit_truth_units": fit_truth_units,
    }


def _same_parameter(name: str, asked, stored) -> bool:
    """
    Whether *asked* is the value of parameter *name* that a combination was run with, *stored*.

    Compared in the stored value's own unit, to a part in 1e9 of its size:
    compared in SI to 1e-8 of at least one, every length under 10 nm, a
    microroughness or a thickness, was the same as every other. A quantity
    of another kind is refused rather than matched, as 5 m matched 5 s. A
    number with no unit is compared with the stored value in SI, as before.
    """
    if isinstance(asked, u.Quantity) and isinstance(stored, u.Quantity):
        try:
            asked = asked.to_value(stored.unit, equivalencies=u.temperature())
        except u.UnitConversionError:
            raise ValueError(f"{name} was run in {stored.unit}, a unit of "
                             f"{stored.unit.physical_type}; {asked} is not.") from None
        stored = stored.value
    else:
        asked, stored = _to_canonical_scalar(asked), _to_canonical_scalar(stored)
    numbers = (int, float, np.integer, np.floating)
    if (isinstance(asked, numbers) and isinstance(stored, numbers)
            and not isinstance(asked, bool) and not isinstance(stored, bool)):
        return bool(np.isclose(asked, stored, rtol=1e-9, atol=0.0))
    return asked == stored


def get_results_for_combination(results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Get results for a specific parameter combination.

    Parameters are specified as keyword arguments using the full
    ``section.attribute`` names stored in the results, e.g.::

        get_results_for_combination(results, **{"simulation.expos": 40*u.s, "simulation.slit_width": 0.2*u.arcsec})
        get_results_for_combination(results, **{"detector.qe_euv": 0.76})

    Use ``summary_table(results)`` to see all available parameter names and
    their values across combinations.

    Parameters
    ----------
    results : dict
        Results dictionary from :func:`load_instrument_response_results`.
    **kwargs
        Parameter name-value pairs to match, using ``section.attribute`` names.
        Values should be astropy Quantities where the stored value has units.

    Returns
    -------
    dict
        Results for the matched parameter combination.

    Raises
    ------
    ValueError
        If zero or more than one combination matches.
    """
    all_combinations = results["results"]["all_combinations"]

    if not kwargs:
        if len(all_combinations) == 1:
            return next(iter(all_combinations.values()))
        raise ValueError(
            f"No parameters specified but {len(all_combinations)} combinations exist. "
            "Use summary_table(results) to list all combinations, then specify "
            "enough parameters to select a unique one."
        )

    query = dict(kwargs)

    matches = [combo_results for combo_results in all_combinations.values()
               if all(key in combo_results["parameters"]
                      and _same_parameter(key, value, combo_results["parameters"][key])
                      for key, value in query.items())]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        raise ValueError(
            f"No combination matches: {kwargs}\n"
            "Use summary_table(results) to see all available combinations."
        )
    else:
        raise ValueError(
            f"{len(matches)} combinations match the specified parameters {kwargs}. "
            "Add more parameters to narrow the selection. "
            "Use summary_table(results) to see all available combinations."
        )


def get_dem_data_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract DEM data from loaded instrument response results.
    
    Parameters
    ----------
    results : dict
        Results dictionary from load_instrument_response_results.
        
    Returns
    -------
    dict
        Dictionary containing DEM data with keys:
        - 'dem_map': DEM(T) map (numpy array, shape ny, nx, nT)
        - 'em_tv': EM(T,v) map (numpy array, shape ny, nx, nT, nv)
        - 'logT_centres': Temperature bin centers (numpy array)
        - 'v_edges': Velocity bin edges (numpy array)
        - 'goft': Contribution function data (dict)
        - 'logT_grid': Temperature grid used for interpolation (numpy array)
        - 'logN_grid': Density grid used for interpolation (numpy array)
        
    Raises
    ------
    KeyError
        If DEM data is not found in the results, which it never is: no
        released version wrote it there.

    .. deprecated:: 0.12.0
        The DEM and the emission measure are in the synthesis file; read
        them with :func:`euvst_response.read_synthesis_products`.
    """
    warnings.warn("get_dem_data_from_results is deprecated and will be removed in a future "
                  "release: no released version wrote DEM data into the results. The DEM is "
                  "in the synthesis file; read it with "
                  "euvst_response.read_synthesis_products(path, keys=['dem_map', 'em_tv']).",
                  FutureWarning, stacklevel=2)
    if "dem_data" not in results:
        raise KeyError(
            "DEM data is not in the results, as no released version wrote it there. It is "
            "in the synthesis file: read it with "
            "euvst_response.read_synthesis_products(path, keys=['dem_map', 'em_tv'])."
        )
    
    return results["dem_data"]


def summary_table(results: Dict[str, Any]) -> None:
    """
    Print a summary table of all parameter combinations.

    Column headers are discovered dynamically from the stored parameters, so
    the table automatically reflects whatever was swept or overridden - no
    code changes needed when new parameters are added.

    Parameters
    ----------
    results : dict
        Results dictionary from :func:`load_instrument_response_results`.
    """
    all_combinations = results["results"]["all_combinations"]

    if not all_combinations:
        print("No parameter combinations found.")
        return

    # Print run metadata if available
    if "software_version" in results:
        print(f"Software version : {results['software_version']}")
    if "git_commit_id" in results:
        print(f"Git commit       : {results['git_commit_id']}")
    if "software_version" in results or "git_commit_id" in results:
        print()

    # Discover all parameter names (excluding non-display fields)
    _skip = {"simulation.pinhole_sizes", "simulation.pinhole_positions"}
    param_names: List[str] = []
    for combo in all_combinations.values():
        for k in combo["parameters"]:
            if k not in param_names and k not in _skip:
                param_names.append(k)
    param_names.sort()

    def _fmt(val) -> str:
        if hasattr(val, "unit"):
            return f"{val.value:.4g} {val.unit}"
        return str(val)

    col_w = 24
    header = " | ".join(f"{n:<{col_w}}" for n in param_names)
    sep = "-" * len(header)
    print(header)
    print(sep)
    for combo in all_combinations.values():
        params = combo["parameters"]
        row = [f"{_fmt(params.get(n, 'N/A')):<{col_w}}" for n in param_names]
        print(" | ".join(row))
    print(sep)
    print(f"Total combinations: {len(all_combinations)}")

    sweep_dims = results.get("results", {}).get("sweep_dimensions", {})
    if sweep_dims:
        print("\nSwept dimensions:")
        for dim, vals in sweep_dims.items():
            print(f"  {dim}: {vals}")

    # The components are the same in every combination and for both signals.
    first_combo = next(iter(all_combinations.values()))
    fit_stats = first_combo.get("dn_fit_stats") or first_combo.get("photon_fit_stats")
    if fit_stats and "components" in fit_stats:
        print("\nFitted components:")
        for name, comp in fit_stats["components"].items():
            notes = ["primary"] if name == fit_stats["primary_component"] else []
            notes += [f"{key} tied to {source}"
                      for key, source in comp["tied"].items() if source is not None]
            print(f"  {name}" + (f" ({', '.join(notes)})" if notes else ""))


def _utc_date(value, what: str) -> str:
    """
    *value* as an ISO date and time in UTC, which sunpy reads.

    A date with an offset, as YAML gives an unquoted time ending in Z, was
    written as it came; sunpy could not read it and stamped the maps with
    the time they were made instead, different on every call.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            raise ValueError(f"{what} {value!r} is not a date and time sunpy can read; give "
                             f"it in ISO form, such as 2024-03-20T00:00:00.") from None
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment.isoformat()


def _resolve_date_obs(combination_results: Dict[str, Any], date_obs) -> str:
    """
    Settle on the observation date to write into the maps.

    A synthetic scene has no intrinsic date, so there are only two honest
    sources: one the caller gives, or the date an EIS run was calibrated
    against, which is a real observing date the user already chose.  Without
    either, sunpy would fall back to the current time, which makes the maps
    different on every run and silently wrong to anything that uses the date.

    Parameters
    ----------
    combination_results : dict
        Results for one parameter combination.
    date_obs : str, datetime.datetime, datetime.date or None
        The caller's date, if any.

    Returns
    -------
    str
        An ISO-8601 date string for the ``DATE-OBS`` keyword.
    """
    if date_obs is not None:
        return _utc_date(date_obs, "date_obs")

    telescope = combination_results.get("config_objects", {}).get("telescope")
    calibration_date = getattr(telescope, "date", None)
    if calibration_date:
        return _utc_date(calibration_date, "The EIS calibration date")

    raise ValueError(
        "These maps have no observation date. A synthesised scene does not "
        "carry one, and sunpy would otherwise stamp the maps with the time "
        "the code happened to run, so they would differ between runs and "
        "mislead anything that uses the date. Pass date_obs, for example "
        "date_obs='2024-03-20T00:00:00'. An EIS run configured with a "
        "time-dependent calibration uses that calibration date instead."
    )


def _map_header(wcs_2d, date_obs: str, bunit: str):
    """
    FITS header for one output map: the WCS, the unit, and the vantage point.

    The observer keywords are not a guess.  The whole radiometric chain works
    at exactly one astronomical unit (``photons_to_pixel_counts`` divides the
    pixel area by ``const.au ** 2``), and the cubes are laid out about disc
    centre, so that is the vantage point the data already describes.  Writing
    it down stops sunpy assuming an Earth-based observer, which is close for a
    low-Earth-orbit mission but is an assumption sunpy makes silently and
    which no longer matches the data if the chain's distance ever changes.

    Parameters
    ----------
    wcs_2d : astropy.wcs.WCS
        The 2D celestial WCS for the map.
    date_obs : str
        Observation date, from :func:`_resolve_date_obs`.
    bunit : str
        Unit string for the map data.

    Returns
    -------
    astropy.io.fits.Header
    """
    header = wcs_2d.to_header()
    header["DATE-OBS"] = date_obs
    header["BUNIT"] = bunit
    # Heliographic Stonyhurst position of the observer: on the Sun-disc-centre
    # line, one au from the emitting surface, as the radiometric chain has
    # it, so one au and a solar radius from the Sun's centre, which is what
    # DSUN_OBS measures. Latitude and longitude are zero for the same reason
    # the WCS reference is disc centre - ECLIPSE models no B0 angle.
    header["HGLN_OBS"] = 0.0
    header["HGLT_OBS"] = 0.0
    header["DSUN_OBS"] = (const.au + const.R_sun).to_value(u.m)
    header["RSUN_REF"] = const.R_sun.to_value(u.m)
    return header


def create_sunpy_maps_from_combo(
    combination_results: Dict[str, Any],
    cube_reb=None,
    rest_wavelength: u.Quantity | None = None,
    data_type: str = "dn",
    precision_requirement: u.Quantity = 2.0 * u.km / u.s,
    exposure_time_results: List[Dict[str, Any]] | None = None,
    fit_config=None,
    date_obs=None,
    component: str | None = None,
) -> Dict[str, Any]:
    """
    Create SunPy maps from combination results using the new fit statistics structure.
    
    Parameters
    ----------
    combination_results : dict
        Results for a specific parameter combination from get_results_for_combination().
    cube_reb : NDCube, optional
        NDCube with helioprojective WCS to use for all maps.
        If not provided, the WCS stored in the combination results is used.
    rest_wavelength : u.Quantity, optional
        The rest wavelength velocities are measured from.  Results files
        record each component's own, and a value given here has to match
        it.  Files written before components were stored by name use it, and
        default to 195.119 A (Fe XII).
    data_type : str, optional
        Either "dn" or "photon" to specify which fit statistics to use for velocity/width maps.
    precision_requirement : u.Quantity, optional
        Velocity precision requirement for exposure time map (default: 2.0 km/s).
    exposure_time_results : list of dict, optional
        List of results from get_results_for_combination() for different exposure times.
        If provided, will create an exposure time map showing minimum exposure needed.
    fit_config : FitConfig, optional
        Only used for files written before components were stored by name,
        to find the primary component's centre and width.
    date_obs : str or datetime, optional
        Observation date written to every map. A synthesised scene has no
        date of its own, so this has to come from the caller. An EIS run
        configured with a time-dependent calibration uses that calibration
        date when this is not given; anything else raises rather than let
        sunpy stamp the maps with the time the code ran.
    component : str, optional
        Name of the fitted component to map, from
        :func:`list_fit_components`.  Defaults to the primary component.

    Returns
    -------
    dict
        Dictionary of SunPy maps with keys:
        - 'total_photons': Total photons (summed along wavelength) from first MC iteration
        - 'total_dn': Total DN (summed along wavelength) from first MC iteration
        - 'velocity_from_fit': Velocity from first fit of first MC iteration
        - 'velocity_mean': Mean velocity across all MC iterations
        - 'velocity_std': Velocity uncertainty (standard deviation)
        - 'velocity_err': Velocity error (truth - mean)
        - 'line_width_from_fit': Line width from first fit of first MC iteration  
        - 'line_width_mean': Mean line width across all MC iterations
        - 'line_width_std': Line width uncertainty (standard deviation)
        - 'intensity_from_fit', 'intensity_mean', 'intensity_std': The same
          for the fitted line's counts, and 'failed_fits': the number of
          failed fits per pixel (files with components stored by name only)
        - 'exposure_time': Minimum exposure time required to reach precision (if exposure_time_results provided)
    """
    
    date_obs = _resolve_date_obs(combination_results, date_obs)

    if rest_wavelength is None and "components" not in _get_fit_stats(
            combination_results, data_type):
        # Older files do not record the rest wavelength, and this was the
        # default when they were written.
        rest_wavelength = 195.119 * u.AA

    # Handle optional exposure time analysis
    if exposure_time_results is not None:
        # Create analysis_per_exp from the list
        analysis_per_exp = {}
        for result in exposure_time_results:
            # Extract exposure time from parameters
            exposure_time = result["parameters"]["simulation.expos"].to_value(u.s)
            # Create analysis for this exposure
            analysis = analyse_fit_statistics(result, rest_wavelength, data_type,
                                              fit_config=fit_config, component=component)
            analysis_per_exp[exposure_time] = analysis
    else:
        analysis_per_exp = None
    
    # Extract 2D helioprojective WCS from the cube or stored signal WCS.
    # The cubes are (ny, nx, nwave) against a (WAVE, HPLN, HPLT) WCS, so the
    # celestial part is already in SunPy's (HPLN, HPLT) order and the data
    # can go into the maps as it is.
    if cube_reb is not None:
        wcs_2d = cube_reb.wcs.celestial
    else:
        wcs_2d = combination_results["first_signal_wcs"].celestial

    # Get the data arrays - now only first iteration is saved
    first_photon_signal = combination_results["first_photon_signal"]  # Shape: (ny, nx, nwave)
    first_dn_signal = combination_results["first_dn_signal"]         # Shape: (ny, nx, nwave)
    maps = {}

    # --- Total photons map (before detector effects) ---
    total_photons_data = first_photon_signal.data.sum(axis=2)  # Sum along wavelength
    total_photons_unit = first_photon_signal.unit * u.pix

    maps['total_photons'] = sunpy.map.Map(
        total_photons_data, _map_header(wcs_2d, date_obs, str(total_photons_unit)))

    # --- Total DN map (after detector effects) ---
    total_dn_data = first_dn_signal.data.sum(axis=2)  # Sum along wavelength
    total_dn_unit = first_dn_signal.unit * u.pix

    maps['total_dn'] = sunpy.map.Map(
        total_dn_data, _map_header(wcs_2d, date_obs, str(total_dn_unit)))
    
    # --- Get velocity, width and intensity analysis for this combination ---
    analysis = analyse_fit_statistics(combination_results, rest_wavelength, data_type,
                                      fit_config=fit_config, component=component)

    # --- Velocity maps ---
    for key, name in [("v_first", "velocity_from_fit"), ("v_mean", "velocity_mean"),
                      ("v_std", "velocity_std"), ("v_true", "velocity_true"),
                      ("v_err", "velocity_err")]:
        velocity = analysis[key].to(u.km / u.s)
        maps[name] = sunpy.map.Map(
            velocity.value, _map_header(wcs_2d, date_obs, str(velocity.unit)))

    # --- Line width maps, in Angstrom ---
    for key, name in [("w_first", "line_width_from_fit"), ("w_mean", "line_width_mean"),
                      ("w_std", "line_width_std")]:
        maps[name] = sunpy.map.Map(
            analysis[key].to_value(u.AA), _map_header(wcs_2d, date_obs, str(u.AA)))

    # --- Intensity and failed-fit maps (files with components stored by name) ---
    if "i_mean" in analysis:
        for key, name in [("i_first", "intensity_from_fit"), ("i_mean", "intensity_mean"),
                          ("i_std", "intensity_std")]:
            maps[name] = sunpy.map.Map(
                analysis[key].value, _map_header(wcs_2d, date_obs, str(analysis[key].unit)))
        maps['failed_fits'] = sunpy.map.Map(
            analysis["failed_fits"].astype(float), _map_header(wcs_2d, date_obs, ""))
    
    # --- Exposure time map (minimum required for precision) ---
    if analysis_per_exp is not None:
        exp_times = sorted(analysis_per_exp.keys())
        nlevels = len(exp_times)
        shape = next(iter(analysis_per_exp.values()))["v_std"].shape
        best_exp = np.full(shape, np.nan)

        # The shortest exposure time, in seconds, that meets the precision
        # requirement in each pixel. It held the exposure's index instead,
        # under a unit of seconds, so that a saved map, or one read without
        # the colorbar helper, gave 0, 1, 2 for the exposures.
        for s in exp_times:
            vstd = analysis_per_exp[s]["v_std"].to_value(u.km / u.s)
            msk = (vstd <= precision_requirement.to_value(u.km / u.s)) & np.isnan(best_exp)
            best_exp[msk] = s

        # Pixels that don't meet the precision requirement even at the
        # longest exposure need longer than any run: above the scale.
        best_exp[np.isnan(best_exp)] = np.inf

        # One colour per exposure time, each spanning halfway to the next.
        times = np.asarray(exp_times, dtype=float)
        if nlevels > 1:
            middles = 0.5 * (times[1:] + times[:-1])
            bounds = np.concatenate([[times[0] - (middles[0] - times[0])], middles,
                                     [times[-1] + (times[-1] - middles[-1])]])
        else:
            bounds = np.array([0.5 * times[0], 1.5 * times[0]])
        cmap = ListedColormap(plt.get_cmap("viridis")(np.linspace(0, 1, nlevels)))
        cmap.set_over("white")
        cmap.set_bad("gray")  # Change bad color so we can distinguish from over
        norm = BoundaryNorm(bounds, nlevels)

        maps['exposure_time'] = sunpy.map.Map(
            best_exp, _map_header(wcs_2d, date_obs, 's'))
        maps['exposure_time'].plot_settings.update(dict(cmap=cmap, norm=norm))

        # Store exposure time information for custom colorbar formatting
        maps['exposure_time']._exposure_times = exp_times
        maps['exposure_time']._exposure_indices = list(range(nlevels))
    
    # Set appropriate visualization settings for common map types
    # Also ensure correct aspect ratio for all maps
    map_names = list(maps.keys())

    # Set aspect ratio metadata for all maps to ensure correct plotting
    cdelt_x = wcs_2d.wcs.cdelt[0]
    cdelt_y = wcs_2d.wcs.cdelt[1]
    aspect_ratio = cdelt_y / cdelt_x
    for map_name in map_names:
        maps[map_name].plot_settings.update({
            'aspect': aspect_ratio,
        })
    
    # Set specific color maps and ranges
    maps['total_photons'].plot_settings.update(dict(cmap="afmhot", norm="log"))
    maps['total_dn'].plot_settings.update(dict(cmap="afmhot", norm="log"))
    maps['velocity_from_fit'].plot_settings.update(dict(cmap="RdBu_r", vmin=-15, vmax=15))
    maps['velocity_mean'].plot_settings.update(dict(cmap="RdBu_r", vmin=-15, vmax=15))
    maps['velocity_std'].plot_settings.update(dict(cmap="magma", vmin=0))
    maps['line_width_from_fit'].plot_settings.update(dict(cmap="Purples"))
    maps['line_width_mean'].plot_settings.update(dict(cmap="Purples"))
    maps['line_width_std'].plot_settings.update(dict(cmap="Purples"))
    if 'intensity_mean' in maps:
        maps['intensity_from_fit'].plot_settings.update(dict(cmap="afmhot"))
        maps['intensity_mean'].plot_settings.update(dict(cmap="afmhot"))
        maps['intensity_std'].plot_settings.update(dict(cmap="magma", vmin=0))
        maps['failed_fits'].plot_settings.update(dict(cmap="Greys", vmin=0))
    if 'exposure_time' in maps:
        maps['exposure_time'].plot_settings.update(dict(origin="lower"))

    return maps


def format_exposure_time_colorbar(map_obj, colorbar, precision_requirement: u.Quantity = 2.0 * u.km / u.s):
    """
    Format the colorbar for an exposure time map with proper tick labels.
    
    Parameters
    ----------
    map_obj : sunpy.map.Map
        The exposure time map object (should have _exposure_times attribute).
    colorbar : matplotlib.colorbar.Colorbar
        The colorbar object to format.
    precision_requirement : u.Quantity, optional
        Velocity precision requirement for the title (default: 2.0 km/s).
    """
    # A tick at each exposure time, in the middle of its colour, written as
    # given: to one decimal, 0.25 s read as 0.2.
    tick_positions = list(map_obj._exposure_times)
    tick_labels = [f"{exp_time:g}" for exp_time in map_obj._exposure_times]
    
    colorbar.set_ticks(tick_positions)
    colorbar.set_ticklabels(tick_labels)
    
    # Create title with precision requirement
    precision_val = precision_requirement.to_value(u.km / u.s)
    title = f"Minimum exposure time to reach $\\sigma_v \\leq {precision_val:.1f}$ km/s [s]"
    colorbar.set_label(title)