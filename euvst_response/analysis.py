"""
Result analysis functions for ECLIPSE instrument response simulations.
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

from .config import Detector_SWC, _old_names_used
from .deprecated import get_dem_data_from_results  # noqa: F401  (deprecated)
from .results_file import load_results
from .utils import has_wrong_velocity_sign



def _to_canonical_scalar(val):
    """
    Value as a plain scalar, for comparing parameters.

    Quantities become their SI value, so `40 * u.s` and `40000 * u.ms` both
    give 40.0. Offset units such as Celsius are converted to Kelvin first.
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
    Read a results file written by `eclipse`.

    Parameters
    ----------
    filepath : str or Path
        Results file, `run/result/<config name>.h5`. Results pickles from
        ECLIPSE 0.11.0 and earlier are also read, with a warning, and a
        `.pkl` name that no longer exists reads the `.h5` file beside it.
        Reading a pickle runs any code it holds, so only read pickles you
        trust.
    allow_wrong_velocity_sign : bool, optional
        Warn instead of raising for results made from a synthesis that an
        older ECLIPSE wrote, under the older velocity convention, for a view
        along x or z. Their velocities have the opposite sign to this
        version's. Default False.

    Returns
    -------
    dict
        Results. `results["results"]["all_combinations"]` holds those of
        each combination, and `get_results_for_combination` picks one out.
    """
    data = load_results(filepath, _stacklevel=3)
    # An old pickle can hold anything.
    if not isinstance(data, dict):
        raise ValueError(f"{filepath} does not hold the results of an instrument simulation.")

    # Refuse results made from synthesis files written under the older
    # velocity convention, for the views whose sign it changed. Uniform
    # intensity runs have no synthesis file and no velocities, so are
    # unaffected.
    cube_sim = data.get("cube_sim")
    if cube_sim is not None and has_wrong_velocity_sign(cube_sim.meta):
        axis = (cube_sim.meta or {}).get("integration_axis", "z")
        message = (
            f"{filepath} was made from a synthesis file written by an older "
            "ECLIPSE, which took the line-of-sight velocity straight from the "
            "simulation's velocity component. ECLIPSE now uses the velocity "
            f"away from the observer. For this view along {axis}, the two "
            "conventions have opposite signs, so the velocities in it are "
            "opposite in sign to this version's."
        )
        if not allow_wrong_velocity_sign:
            raise ValueError(
                message + " Re-run the synthesis and then the simulation with "
                "this version, or pass allow_wrong_velocity_sign=True to load "
                "it anyway."
            )
        warnings.warn(message, stacklevel=2)

    # A malformed file is refused with an error naming it. A signal's unit
    # must be a unit, since two arrays could multiply to far more than
    # either. A signal shared by several combinations is multiplied once.
    signals = {}
    try:
        combinations = data["results"]["all_combinations"]
        for param_key, combination_results in tqdm(combinations.items(), desc="Reconstructing results", leave=False):
            # Refuse files written before the cube axis order changed (issue
            # #12). They store signals as (x, y, wavelength) with an HPLT-first
            # WCS, so their maps would come out transposed.
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
    Settings of every simulated combination.

    Parameters
    ----------
    results : dict
        Results from `load_instrument_response_results`.

    Returns
    -------
    list of dict
        One dict per combination, of its settings keyed by section and key,
        such as `"simulation.expos"`.
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
    """Fit statistics of one signal; raises ValueError if there are none."""
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
    Names of the fitted components, in the order they were fitted.

    Parameters
    ----------
    combination_results : dict
        Results of one combination, from `get_results_for_combination`.
    data_type : str, optional
        Fits to use: `"dn"` for the signal in DN (default), or `"photon"`
        for the photons.

    Returns
    -------
    list of str
        Names to pass as `component` to `analyse_fit_statistics` and
        `create_sunpy_maps_from_combo`.
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
    Per-pixel precision of one fitted line's velocity, width and intensity.

    Parameters
    ----------
    combination_results : dict
        Results of one combination, from `get_results_for_combination`.
    rest_wavelength : u.Quantity, optional
        Rest wavelength the velocities are measured from. Results files store
        each component's own, so this is only needed for files written
        before components were stored by name. If given, it must match.
    data_type : str, optional
        Fits to use: `"dn"` for the signal in DN (default), or `"photon"`
        for the photons.
    fit_config : FitConfig, optional
        Only needed for files written before components were stored by name.
    component : str, optional
        Component name, from `list_fit_components`. Default the primary
        component.

    Returns
    -------
    dict
        Mostly maps, with one value per pixel. `v_first`, `w_first` and
        `i_first`: velocity, width and intensity from the first Monte Carlo
        iteration. `v_mean`, `w_mean` and `i_mean`: their means over the
        iterations. `v_std`, `w_std` and `i_std`: their standard deviations.
        `v_true` and `w_true`: true velocity and width. `v_err`: true
        velocity minus the mean. `failed_fits` and `n_iterations`: number of
        failed fits, and of iterations. `component`, `rest_wavelength` and
        `tied`: component name, its rest wavelength and what it is tied to.
        `fit_stats`, `fit_truth_data` and `fit_truth_units`: fit statistics
        and fit to the truth, as stored in the results. The width is the
        Gaussian sigma and the intensity the fitted line's counts. Files
        written before components were stored by name give only `v_first`,
        `v_mean`, `v_std`, `v_true`, `v_err`, `w_first`, `w_mean`, `w_std`,
        `fit_stats`, `fit_truth_data` and `fit_truth_units`.
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
    Whether the value `asked` for parameter `name` matches the `stored` one.

    Quantities are compared in the stored value's unit, to a part in 1e9.
    Compared in SI to 1e-8 of at least one, every length under 10 nm, such
    as a microroughness or a thickness, would equal every other. A quantity
    of a different kind raises ValueError rather than matching (5 m would
    otherwise match 5 s), as does a quantity with a unit for a setting run
    as a plain number. A number with no unit is compared with the stored
    value in SI. A list, tuple or array matches a stored list or tuple item
    by item, so a list from YAML matches a tuple the run stored.
    """
    if isinstance(stored, (list, tuple)):
        if isinstance(asked, np.ndarray) and asked.ndim > 0:
            asked = list(asked)
        return (isinstance(asked, (list, tuple)) and len(asked) == len(stored)
                and all(_same_parameter(name, one, other) for one, other in zip(asked, stored)))
    numbers = (int, float, np.integer, np.floating)
    if isinstance(asked, u.Quantity) and isinstance(stored, (bool, np.bool_)):
        raise ValueError(f"{name} is a switch, true or false; {asked} is not.")
    if isinstance(asked, u.Quantity) and isinstance(stored, numbers):
        if not asked.unit.is_equivalent(u.dimensionless_unscaled):
            raise ValueError(f"{name} was run as a number with no unit; {asked} is a "
                             f"{asked.unit.physical_type}.")
        asked = asked.to_value(u.dimensionless_unscaled)
    if isinstance(asked, u.Quantity) and isinstance(stored, u.Quantity):
        try:
            asked = asked.to_value(stored.unit, equivalencies=u.temperature())
        except u.UnitConversionError:
            raise ValueError(f"{name} was run in {stored.unit}, a unit of "
                             f"{stored.unit.physical_type}; {asked} is not.") from None
        stored = stored.value
    else:
        asked, stored = _to_canonical_scalar(asked), _to_canonical_scalar(stored)
    if (isinstance(asked, numbers) and isinstance(stored, numbers)
            and not isinstance(asked, bool) and not isinstance(stored, bool)):
        return bool(np.isclose(asked, stored, rtol=1e-9, atol=0.0))
    return asked == stored


def get_results_for_combination(results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Results of one combination, selected by its settings.

    Settings are named by section and key, as `summary_table` prints them.
    Keyword arguments cannot contain a dot, so pass them as a dictionary:

        get_results_for_combination(results, **{"simulation.expos": 40 * u.s,
                                                 "simulation.slit_width": 0.2 * u.arcsec})

    Parameters
    ----------
    results : dict
        Results from `load_instrument_response_results`.
    **kwargs
        Settings to match, with units where they have them. Not needed if
        there is only one combination.

    Returns
    -------
    dict
        Results of the matching combination.

    Raises
    ------
    ValueError
        If no combination matches, or more than one does.
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

    renamed = _old_names_used(Detector_SWC, {key[len("detector."):] for key in kwargs
                                             if key.startswith("detector.")},
                              "detector.", error=ValueError)
    query = {}
    for key, value in kwargs.items():
        name = key[len("detector."):]
        if key.startswith("detector.") and name in renamed:
            key = f"detector.{renamed[name]}"
        query[key] = value

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


def summary_table(results: Dict[str, Any]) -> None:
    """
    Print a summary of the results.

    Prints the ECLIPSE version and git commit that made the results, the
    settings of every combination, the swept settings and the names of the
    fitted components.

    Parameters
    ----------
    results : dict
        Results from `load_instrument_response_results`.
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
    `value` as an ISO date and time in UTC, a form sunpy can read.

    sunpy cannot read a date with an offset, such as YAML gives for an
    unquoted time ending in Z, and would stamp the maps with the time they
    were made instead, different on every call.
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
    Observation date to write into the maps.

    A synthetic scene has no date of its own. The date is either given by
    the caller or taken from the calibration date of an EIS run, a real
    observing date the user chose. Without either, sunpy would use the
    current time, making the maps differ on every run and silently wrong
    for anything that uses the date.

    Parameters
    ----------
    combination_results : dict
        Results for one parameter combination.
    date_obs : str, datetime.datetime, datetime.date or None
        Date given by the caller, if any.

    Returns
    -------
    str
        ISO-8601 date string for the `DATE-OBS` keyword.
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
    FITS header for one map: WCS, unit and observer position.

    The radiometric chain works at exactly one astronomical unit
    (`photons_to_pixel_counts` divides the pixel area by `const.au ** 2`),
    and the cubes are centred on disc centre, so that is the observer
    position the data describe. Writing it in the header stops sunpy from
    silently assuming an Earth-based observer. That is close for a
    low-Earth-orbit mission, but would not match the data if the chain's
    distance changed.

    Parameters
    ----------
    wcs_2d : astropy.wcs.WCS
        2D celestial WCS of the map.
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
    # Observer in Heliographic Stonyhurst coordinates, on the line through
    # disc centre. The radiometric chain puts it one au from the emitting
    # surface, so DSUN_OBS, measured from the Sun's centre, is one au plus a
    # solar radius. Latitude and longitude are zero, and the WCS reference
    # is disc centre, because ECLIPSE models no B0 angle.
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
    Make SunPy maps of one combination's results.

    Parameters
    ----------
    combination_results : dict
        Results of one combination, from `get_results_for_combination`.
    cube_reb : NDCube, optional
        Cube whose coordinates the maps take. Default the coordinates stored
        in the results.
    rest_wavelength : u.Quantity, optional
        Rest wavelength the velocities are measured from. Only needed for
        files written before components were stored by name. If given, it
        must match.
    data_type : str, optional
        Fits to use: `"dn"` for the signal in DN (default), or `"photon"`
        for the photons.
    precision_requirement : u.Quantity, optional
        Velocity precision for the `exposure_time` map. Default 2 km/s.
    exposure_time_results : list of dict, optional
        Combinations at each exposure time. If given, the maps include
        `exposure_time`: the shortest of these exposures at which each
        pixel's velocity is measured to within `precision_requirement`.
    fit_config : FitConfig, optional
        Only needed for files written before components were stored by name.
    date_obs : str or datetime, optional
        Observation date of the maps. A synthesised scene has no date of its
        own, so this is needed, except for an EIS run with a dated
        calibration, whose date is used.
    component : str, optional
        Component to map, by name from `list_fit_components`. Default the
        primary component.

    Returns
    -------
    dict
        Maps by name. `total_dn` and `total_photons`: first iteration's
        signal, summed over wavelength. `velocity_from_fit`,
        `line_width_from_fit` and `intensity_from_fit`: first iteration's
        fit. `velocity_mean`, `line_width_mean` and `intensity_mean`: means
        over the iterations. `velocity_std`, `line_width_std` and
        `intensity_std`: standard deviations. `velocity_true` and
        `velocity_err`: true velocity, and true velocity minus the mean.
        `failed_fits`: number of failed fits in each pixel. `exposure_time`:
        shortest exposure that reaches `precision_requirement`, if
        `exposure_time_results` is given. Files written before components
        were stored by name have no intensity maps and no `failed_fits`.
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
    
    # 2D helioprojective WCS from the cube or the stored signal WCS. The cubes
    # are (ny, nx, nwave) with a (WAVE, HPLN, HPLT) WCS, so the celestial part
    # is already in SunPy's (HPLN, HPLT) order and the data need no transpose.
    if cube_reb is not None:
        wcs_2d = cube_reb.wcs.celestial
    else:
        wcs_2d = combination_results["first_signal_wcs"].celestial

    # Get the data arrays; only the first iteration is saved
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

        # Shortest exposure time, in seconds, that meets the precision
        # requirement in each pixel.
        for s in exp_times:
            vstd = analysis_per_exp[s]["v_std"].to_value(u.km / u.s)
            msk = (vstd <= precision_requirement.to_value(u.km / u.s)) & np.isnan(best_exp)
            best_exp[msk] = s

        # Pixels that miss the requirement even at the longest exposure need
        # longer than any run, so they go above the scale.
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
    Label the colorbar of an exposure time map with its exposure times.

    Parameters
    ----------
    map_obj : sunpy.map.Map
        Exposure time map, with an `_exposure_times` attribute.
    colorbar : matplotlib.colorbar.Colorbar
        Colorbar to format.
    precision_requirement : u.Quantity, optional
        Velocity precision requirement for the colorbar label. Default 2.0 km/s.
    """
    # One tick per exposure time, in the middle of its colour, labelled as
    # given so that 0.25 s reads as 0.25.
    tick_positions = list(map_obj._exposure_times)
    tick_labels = [f"{exp_time:g}" for exp_time in map_obj._exposure_times]
    
    colorbar.set_ticks(tick_positions)
    colorbar.set_ticklabels(tick_labels)
    
    # Create title with precision requirement
    precision_val = precision_requirement.to_value(u.km / u.s)
    title = f"Minimum exposure time to reach $\\sigma_v \\leq {precision_val:.1f}$ km/s [s]"
    colorbar.set_label(title)