import os
import sys
import re
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
import astropy.constants as const
from tqdm import tqdm
import psutil
import dill
import h5py
from ndcube import NDCube
from astropy.wcs import WCS
from .utils import (element_data, proton_electron_ratio_once, require_uniform_grid, usable_cpus,
                    velocity_centers_to_edges, velocity_grid, view_axis_and_side, view_name,
                    OBSERVER_SIDE, VELOCITY_CONVENTION)
from .synthesis_file import (SpectralLine, Synthesis, _pixel_edges, _to_length, write_line_cubes,
                             write_synthesis)
from .emission_measure import (EmissionMeasure, is_emission_measure_file,
                               read_emission_measure, read_emission_measure_layout)
from .band import (BAND_STEP, BandLines, IonLines, band_lines_for, band_spectra,
                   band_wavelengths, ion_contribution_function, same_temperatures)
from .projection import PROJECTIONS, CurvedSkyView, SkyView, disk_geometry
from .continuum import (CONTINUUM_PREFIX, compute_continuum_fiasco, continuum_spectra,
                        continuum_windows)
from .atmosphere import (AXES, NUMPY_AXIS, Atmosphere, _offer_database_build,
                         electron_density_from_protons, mass_per_electron, read_atmosphere,
                         require_mass_per_electron)
# Deprecated MURaM file readers and dynamic mode, importable from here.
from .deprecated import (_dynamic_mode_cubes, apply_cube_cropping,  # noqa: F401
                         build_composite_cubes_mhd, compute_slice_timestep_mapping_mhd,
                         discover_timesteps, get_file_for_timestep, load_cube,
                         load_muram_files, read_timestep_time)

##############################################################################
# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------
##############################################################################

def create_atmosphere_ndcube(
    data: np.ndarray | u.Quantity,
    voxel_dx: u.Quantity,
    voxel_dy: u.Quantity, 
    voxel_dz: u.Quantity,
) -> NDCube:
    """
    NDCube of a simulation cube, with solar coordinates from its cell sizes.

    x and y are centred on zero, and z starts at zero.

    Parameters
    ----------
    data : np.ndarray or u.Quantity
        Cube of shape (nz, ny, nx); `data[k]` is a horizontal slice indexed
        `[y, x]`.
    voxel_dx, voxel_dy, voxel_dz : u.Quantity
        Cell size along each axis.

    Returns
    -------
    NDCube
        Cube with coordinates in Mm.
    """
    nz, ny, nx = data.shape

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['SOLX', 'SOLY', 'SOLZ']
    wcs.wcs.cunit = ['Mm', 'Mm', 'Mm']

    # Reference pixels (1-indexed for WCS)
    wcs.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2, 1]  # Z starts at first pixel

    wcs.wcs.crval = [0, 0, 0]  # X,Y centered at origin, Z starts at 0

    wcs.wcs.cdelt = [
        voxel_dx.to(u.Mm).value,
        voxel_dy.to(u.Mm).value,
        voxel_dz.to(u.Mm).value
    ]

    # A plain array has no unit.
    return NDCube(np.asarray(getattr(data, "value", data)),
                  wcs=wcs,
                  unit=getattr(data, "unit", None))


def _match_candidates(line_name: str, wavelengths_aa: np.ndarray, observed: np.ndarray) -> tuple:
    """
    Indices of the transitions matching `line_name`, and how they were matched.

    The match is, in order of preference:

    - Transitions whose wavelength, rounded to the decimals in the name,
      equals the name's: observed ones if any, else theoretical ones.
    - The nearest observed line. CHIANTI has many weak theoretical
      transitions within a few mA of strong lines, up to 1e12 times fainter,
      which a name slightly off an observed wavelength would otherwise pick.
    - The nearest theoretical line, for an ion with no observed lines.

    All transitions at the chosen wavelength are returned, so the brightest
    can be taken.
    """
    number = line_name.split("_", 1)[1]
    decimals = len(number.partition(".")[2])
    target = float(number)
    named = np.round(wavelengths_aa, decimals) == np.round(target, decimals)
    if (named & observed).any():
        pool, how = named & observed, "named"
    elif named.any():
        pool, how = named, "named"
    elif observed.any():
        pool, how = observed, "nearest observed"
    else:
        pool, how = np.ones_like(named), "nearest theoretical"
    distance = np.where(pool, np.abs(wavelengths_aa - target), np.inf)
    return np.flatnonzero(distance == distance.min()), how


def _ion_part(ion_name: str, temperature_K: np.ndarray, rows: np.ndarray, densities_cm3,
              abundance: str, ion_kwargs: dict, step: int, g_tables: dict):
    """
    Fill in G(T, N) of one ion's candidate transitions at some of its temperatures.

    fiasco is called on `step` of the temperatures in `rows` at a time, and
    the candidate transitions are written to those rows of `g_tables`.
    Returns the number of temperatures at which levels were left out, and
    log10 of the temperature from which the single-ion model is used, or None.
    """
    dropped, single_ion = 0, None
    for start in range(0, rows.size, step):
        chunk = rows[start:start + step]
        g, chunk_dropped, chunk_single_ion = ion_contribution_function(
            ion_name, temperature_K[chunk], densities_cm3, abundance, ion_kwargs)
        dropped += chunk_dropped
        if single_ion is None:
            single_ion = chunk_single_ion
        for (line_name, idx), table in g_tables.items():
            table[chunk] = g[:, :, idx].to(u.erg * u.cm**3 / u.s).value
        del g
    return dropped, single_ion


def _compute_single_ion(args):
    """G(T, N) of the requested lines of one ion, for a worker process.

    fiasco is imported here so each spawned process has its own HDF5
    handles. G is worked out only where CHIANTI gives the ion an ionisation
    fraction above zero, and is zero elsewhere, as the ion is absent. With
    `temperature_chunk`, fiasco is called on that many temperatures at a
    time and only the requested lines are kept, so memory use is that of one
    chunk. Otherwise up to `n_threads` threads can each take part of the
    temperatures."""

    import fiasco
    import logging

    (elem, stage, temperature_K, densities_cm3, abundance, lines,
     hdf5_dbase_root, temperature_chunk, n_threads) = args
    ion_name = f'{elem} {stage}'
    n_temperatures = len(temperature_K)
    step = n_temperatures if temperature_chunk is None else temperature_chunk

    # A spawned process re-reads ~/.fiasco/fiascorc and misses a database the
    # parent selected in memory, so the root is passed in.
    ion_kwargs = {} if hdf5_dbase_root is None else {
        'hdf5_dbase_root': hdf5_dbase_root}

    # Silence fiasco's warnings about missing proton (.psplups) and
    # autoionisation (.auto, .rrlvl) data. These are gaps in CHIANTI, which
    # fiasco and IDL both handle by leaving out proton rates or using the
    # single-ion model, and the warnings fire once per density.
    fiasco_logger = logging.getLogger('fiasco')
    prev_level = fiasco_logger.level
    fiasco_logger.setLevel(logging.ERROR)

    try:
        with proton_electron_ratio_once(temperature_K) as shared:
            # The transitions and the ionisation fractions are read once, on
            # the whole grid; no level populations are worked out for this.
            ion = fiasco.Ion(ion_name, temperature_K * u.K, abundance=abundance, **ion_kwargs)
            bound = ion.transitions.is_bound_bound
            bb_wl = ion.transitions.wavelength[bound]
            # CHIANTI stores theoretical wavelengths as negative numbers; fiasco
            # gives them positive and marks observed ones with this flag.
            observed = np.asarray(ion.transitions.is_observed[bound])
            matches = {line_name: _match_candidates(line_name, bb_wl.to_value(u.AA), observed)
                       for line_name, _ in lines}
            g_tables = {(line_name, int(idx)): np.zeros((n_temperatures, densities_cm3.size))
                        for line_name, (indices, _) in matches.items() for idx in indices}
            fraction = np.nan_to_num(np.asarray(ion.ionization_fraction, dtype=float))
            present = np.flatnonzero(fraction > 0)
            threads = min(n_threads, present.size // _LEAST_TEMPERATURES_PER_THREAD)
            # Without a shared ratio each thread would work it out again.
            if threads > 1 and shared:
                # The ratio of protons to electrons is worked out once, here,
                # and the threads share it. Each thread fills in its own rows.
                ion.proton_electron_ratio
                spans = np.array_split(present, threads)
                with ThreadPoolExecutor(threads) as pool:
                    parts = list(pool.map(
                        lambda span: _ion_part(ion_name, temperature_K, span, densities_cm3,
                                               abundance, ion_kwargs, span.size, g_tables),
                        spans))
            else:
                parts = [_ion_part(ion_name, temperature_K, present, densities_cm3, abundance,
                                   ion_kwargs, step, g_tables)]
    finally:
        fiasco_logger.setLevel(prev_level)

    dropped = sum(n for n, _ in parts)
    # The coolest part beyond the level-resolved rates says where they stop.
    starts = [log_t for _, log_t in parts if log_t is not None]
    single_ion = min(starts) if starts else None

    results = {}
    for line_name, target_wl_aa in lines:
        target_wl = target_wl_aa * u.AA
        indices, how = matches[line_name]
        tables = {}
        for idx in indices:
            g_tn = g_tables[(line_name, int(idx))].T
            tables[int(idx)] = np.nan_to_num(g_tn, nan=0.0, posinf=0.0, neginf=0.0)
        # Of the transitions at one wavelength, take the brightest.
        idx = max(tables, key=lambda i: tables[i].max())
        matched_wl = bb_wl[idx]

        results[line_name] = {
            "g_tn": tables[idx],
            "atom": int(ion.atomic_number),
            "ion": stage,
            "target_wl_cm": float(target_wl.to(u.cm).value),
            "matched_wl_aa": float(matched_wl.to(u.AA).value),
            "match": how,
            "observed": bool(observed[idx]),
            "transition": idx,
            "delta_aa": float(abs(matched_wl - target_wl).to(u.AA).value),
            # Database the contribution functions came from; fiasco fills in
            # the fiascorc default if none was chosen.
            "hdf5_dbase_root": str(ion.hdf5_dbase_root),
            "ion_name": ion.ion_name,
            "dropped": dropped,
            "single_ion": single_ion,
        }
    return results


# Line name: element, ionisation stage and wavelength in Angstrom, such as
# Fe12_195.1190.
_LINE_NAME = re.compile(r'^([A-Z][a-z]?)(\d+)_(\d+\.?\d*)$')


def _parse_line_name(name: str) -> Tuple[str, int, float]:
    """
    Element, ionisation stage and wavelength in Angstrom of a line name.

    Names look like `Fe12_195.1190`. The element must exist, the stage must
    be from 1 (neutral) to its atomic number plus one, and the wavelength
    must be above zero. Checking here catches a bad name before the
    atmosphere is read; a wavelength of zero would match the shortest
    transition.
    """
    match = _LINE_NAME.match(name)
    if not match:
        raise ValueError(f"Cannot parse line name '{name}'. Expected format like "
                         f"'Fe12_195.1190'.")
    elem, stage, wavelength = match.group(1), int(match.group(2)), float(match.group(3))
    try:
        atomic_number, _ = element_data(elem)
    except ValueError:
        raise ValueError(f"Line name '{name}': {elem} is not an element.") from None
    if not 1 <= stage <= atomic_number + 1:
        raise ValueError(f"Line name '{name}': {elem} has ionisation stages 1 to "
                         f"{atomic_number + 1}, got {stage}.")
    if not (np.isfinite(wavelength) and wavelength > 0):
        raise ValueError(f"Line name '{name}': the wavelength must be above zero, in "
                         f"Angstrom.")
    return elem, stage, wavelength

# Threads sharing one ion's temperatures. fiasco holds Python's lock for much
# of its work, so beyond about 8 threads an ion takes longer, not less.
_MOST_THREADS_PER_ION = 8
# Fewest temperatures a thread takes, so that each has a fair share of work.
_LEAST_TEMPERATURES_PER_THREAD = 8

# Default grids for G(T, n_e): log10 T from 4 to 9 in steps of 0.05, and
# log10 n_e from 7 to 13 in steps of 0.3. density_grid extends the density
# grid on the same points as far as an atmosphere needs.
_LOGT_MIN, _LOGT_MAX, _N_T = 4.0, 9.0, 101
_LOGN_MIN, _LOGN_MAX, _N_N = 7.0, 13.0, 21


def compute_goft_fiasco(
    line_names: List[str],
    abundance: str = "sun_coronal_2021_chianti",
    logT_min: float = _LOGT_MIN,
    logT_max: float = _LOGT_MAX,
    nT: int = _N_T,
    logN_min: float = _LOGN_MIN,
    logN_max: float = _LOGN_MAX,
    nN: int = _N_N,
    precision: type = np.float64,
    n_workers: int = 0,
    hdf5_dbase_root=None,
    temperature_chunk: int | None = None,
) -> Tuple[Dict[str, dict], np.ndarray, np.ndarray]:
    """
    Contribution function G(T, n_e) of each line, from CHIANTI via fiasco.

    Each line is matched to a CHIANTI transition, as described in the docs
    page on naming spectral lines, and placed at CHIANTI's wavelength, `wl0`,
    not the name's. G includes the proton-to-electron ratio, so it is per
    emission measure n_e^2 dh. With two or more ions and workers, the ions
    run in spawned processes, so a calling script needs an
    `if __name__ == "__main__":` guard. Workers beyond one per ion become
    threads, up to 8 for each ion, each taking part of its temperatures.

    Parameters
    ----------
    line_names : list of str
        Line names, such as `["Fe12_195.1190", "Fe09_171.073"]`.
    abundance : str, optional
        CHIANTI abundance set. Default `"sun_coronal_2021_chianti"`.
    logT_min, logT_max : float, optional
        Range of log10(T / K).
    nT : int, optional
        Number of temperatures, evenly spaced in log10(T) across that range.
    logN_min, logN_max : float, optional
        Range of log10(n_e / cm-3).
    nN : int, optional
        Number of densities, evenly spaced in log10(n_e) across that range.
    precision : type, optional
        `np.float32` or `np.float64` (default).
    n_workers : int, optional
        Number of workers. Two or more ions are shared between processes,
        and the workers beyond one per ion become threads sharing an ion's
        temperatures. Default 0 uses every CPU this process may use (as
        allocated by SLURM).
    hdf5_dbase_root : str or Path, optional
        CHIANTI database to read. Default is fiasco's, set in
        `~/.fiasco/fiascorc`.
    temperature_chunk : int, optional
        Temperatures passed to fiasco at a time, to use less memory, with one
        thread for each ion. The result is the same. Default None (the whole
        grid).

    Returns
    -------
    goft_dict : dict
        Entry per line name, holding `wl0` (CHIANTI wavelength), `g_tn` (G on
        the density and temperature grid, shape (nN, nT), in erg cm3 / s),
        `atom` and `ion` (atomic number and ionisation stage) and
        `hdf5_dbase_root` (database used).
    logT_grid : np.ndarray
        Temperatures, as log10(T / K).
    logN_grid : np.ndarray
        Densities, as log10(n_e / cm-3).
    """
    # fiasco's contribution_function, G = Ab(X) f_{X,k} (N_j / N) A_ij dE_ij
    # / n_e, leaves out n_H / n_e, which is multiplied in here.
    # Workers are spawned, for HDF5 safety, so they do not see fiasco.defaults
    # set in the parent. The database is passed down, and each worker reports
    # the root it used, which is checked against the request.
    # fiasco's memory grows with the number of temperatures, as it solves
    # them together. Chunks are computed in turn, at the cost of reading the
    # atomic data again.
    if temperature_chunk is not None and temperature_chunk < 1:
        raise ValueError(
            f"temperature_chunk must be a positive number of temperatures, "
            f"not {temperature_chunk}."
        )

    logT_grid = np.linspace(logT_min, logT_max, nT)
    logN_grid = np.linspace(logN_min, logN_max, nN)

    temperature_K = 10.0 ** logT_grid
    densities_cm3 = 10.0 ** logN_grid

    # ---- group the lines by ion ----
    ion_lines: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}

    for name in line_names:
        elem, stage, wl = _parse_line_name(name)
        ion_lines.setdefault((elem, stage), []).append((name, wl))

    # Workers beyond one per ion become threads, each taking part of an
    # ion's temperatures.
    n_ions = len(ion_lines)
    if n_workers <= 0:
        n_workers = usable_cpus()
    n_threads = max(1, min(n_workers // max(n_ions, 1), _MOST_THREADS_PER_ION,
                           nT // _LEAST_TEMPERATURES_PER_THREAD))
    # A chunk smaller than the grid limits the memory, so it keeps one thread.
    if temperature_chunk is not None and temperature_chunk < nT:
        n_threads = 1

    # Build worker arguments (all picklable plain types / numpy arrays)
    dbase_root = None if hdf5_dbase_root is None else str(hdf5_dbase_root)
    worker_args = [
        (elem, stage, temperature_K, densities_cm3, abundance, lines, dbase_root,
         temperature_chunk, n_threads)
        for (elem, stage), lines in ion_lines.items()
    ]
    if temperature_chunk is not None and temperature_chunk < nT:
        print(f"  {nT} temperatures in chunks of {temperature_chunk}")

    # Offer to build the database here, since spawned workers cannot ask.
    _offer_database_build(dbase_root)

    # ---- dispatch: parallel for 2+ ions, serial otherwise ----
    use_parallel = n_workers > 1 and n_ions > 1

    if use_parallel:
        import multiprocessing as mp
        pool_size = min(n_workers, n_ions)
        ctx = mp.get_context("spawn")
        print(f"  Parallel: {pool_size} workers for {n_ions} ions"
              + (f", each with up to {n_threads} threads" if n_threads > 1 else ""))
        with ctx.Pool(pool_size) as pool:
            all_results = pool.map(_compute_single_ion, worker_args)
    else:
        if n_threads > 1:
            print(f"  Up to {n_threads} threads share the ion's temperatures")
        all_results = [_compute_single_ion(a) for a in tqdm(
            worker_args, desc="Computing G(T,N)", unit="ion"
        )]

    # ---- collect results ----
    goft_dict: Dict[str, dict] = {}
    for result in all_results:
        # What was done to work out the ion's populations, once per ion.
        info = next(iter(result.values()), None)
        if info is not None and info["dropped"]:
            print(f"  {info['ion_name']}: levels nothing populates were left out at "
                  f"{info['dropped']} temperatures, and the rest solved exactly")
        if info is not None and info["single_ion"] is not None:
            print(f"  {info['ion_name']}: CHIANTI's recombination and ionisation rates of each "
                  f"level stop short of the hottest temperatures, so from log T "
                  f"{info['single_ion']:.2f} its level populations are worked out with "
                  f"fiasco's single-ion model, as for ions that have no such rates")
        for line_name, info in result.items():
            used = info["hdf5_dbase_root"]
            if dbase_root is not None and used != dbase_root:
                raise RuntimeError(
                    f"A G(T,N) worker built its Ion against the CHIANTI "
                    f"database at {used} instead of the requested "
                    f"{dbase_root}, so the request did not reach it. Its "
                    f"contribution functions would come from the wrong "
                    f"atomic data."
                )
            print(
                f"  {line_name}: requested {info['target_wl_cm']*1e8:.4f} Angstrom, "
                f"matched {info['matched_wl_aa']:.4f} Angstrom, "
                f"{'observed' if info['observed'] else 'theoretical'} "
                f"(delta={info['delta_aa']:.4f} Angstrom)"
            )
            # No line of the ion matches the name's wavelength to its digits.
            if info["match"] == "nearest observed":
                warnings.warn(
                    f"{line_name}: no line of this ion is at that wavelength to the digits "
                    f"given, so the nearest line CHIANTI has observed is used, at "
                    f"{info['matched_wl_aa']:.4f} Angstrom, {info['delta_aa']:.4f} Angstrom "
                    f"from the name, and synthesised there. To synthesise another line, "
                    f"theoretical ones included, give CHIANTI's wavelength for it.",
                    UserWarning, stacklevel=2)
            elif info["match"] == "nearest theoretical":
                warnings.warn(
                    f"{line_name}: no line of this ion is at that wavelength to the digits "
                    f"given, and CHIANTI has observed none of its lines, so the nearest of its "
                    f"theoretical wavelengths is used, at {info['matched_wl_aa']:.4f} Angstrom, "
                    f"{info['delta_aa']:.4f} Angstrom from the name, and synthesised there.",
                    UserWarning, stacklevel=2)
            same = [other for other, seen in goft_dict.items()
                    if (seen["atom"], seen["ion"], seen["transition"])
                    == (info["atom"], info["ion"], info["transition"])]
            if same:
                raise ValueError(
                    f"{same[0]} and {line_name} are the same line, CHIANTI's at "
                    f"{info['matched_wl_aa']:.4f} Angstrom, which would be synthesised twice "
                    f"and summed. Give it once.")
            goft_dict[line_name] = {
                # Place the line at its CHIANTI wavelength, not at the
                # name's digits.
                "wl0": info["matched_wl_aa"] * u.AA.to(u.cm) * u.cm,
                "transition": info["transition"],
                "g_tn": info["g_tn"].astype(precision),
                "atom": info["atom"],
                "ion": info["ion"],
                "hdf5_dbase_root": used,
            }

    return goft_dict, logT_grid.astype(precision), logN_grid.astype(precision)


##############################################################################
# ---------------------------------------------------------------------------
#  DEM and G(T) helpers
# ---------------------------------------------------------------------------
##############################################################################

def compute_dem(
    logT_cube: np.ndarray,
    logN_cube: np.ndarray,
    voxel_dh_cm: float | np.ndarray,
    logT_grid: np.ndarray,
    integration_axis: str = "z",
    *,
    sky: Optional[SkyView] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DEM(T) and emission-measure-weighted mean electron density <n_e>(T).

    Parameters
    ----------
    logT_cube : np.ndarray
        3D array of log10(T/K) values.
    logN_cube : np.ndarray
        3D array of log10(n_e / cm^-3) values.
    voxel_dh_cm : float or np.ndarray
        Cell depth along the integration axis in cm: one value for all cells,
        or one per cell along that axis.
    logT_grid : np.ndarray
        Temperature bin centres, as log10(T / K).
    integration_axis : str
        View: the axis to integrate along, `"x"`, `"y"` or `"z"`, optionally
        signed for the side of the box to look from, such as `"-x"`; see
        `line_of_sight_velocity`. `"z"` with `sky`.
    sky : SkyView, optional
        Box seen from a position on the disk, along a slanted line of sight
        instead of an axis. `voxel_dh_cm` is then the cells' height. Keyword
        only. Default None.

    Returns
    -------
    dem_map : np.ndarray
        DEM in cm^-5 per dex, with the two remaining spatial axes in image
        order (row, column). Shape by axis:
        - "x": (nz, ny, nT)
        - "y": (nz, nx, nT)
        - "z": (ny, nx, nT)
        - with `sky`: (*sky.image_shape, nT)
    avg_ne : np.ndarray
        Mean electron density in each temperature bin, in cm^-3, same shape
        as `dem_map`.
    """
    dem, avg_ne, _ = _compute_dem(logT_cube, logN_cube, voxel_dh_cm, logT_grid,
                                  integration_axis, sky)
    return dem, avg_ne


def _compute_dem(logT_cube, logN_cube, voxel_dh_cm, logT_grid, integration_axis, sky):
    """`compute_dem`, with each cell's n_e^2 dh, which `build_em_tv` takes."""
    nT = len(logT_grid)
    axis, _ = view_axis_and_side(integration_axis)

    dlogT, logT_edges = _temperature_bins(logT_grid)

    ne = 10.0 ** logN_cube.astype(np.float64)
    dh = along_line_of_sight(voxel_dh_cm, axis)
    w2 = np.broadcast_to(ne**2 * dh, logT_cube.shape)  # weights for EM
    w3 = np.broadcast_to(ne**3 * dh, logT_cube.shape)  # weights for EM*n_e

    # Split each cell between the temperature bins either side of it, as in
    # build_em_tv, so the DEM matches its emission measure and each bin's
    # density is that of its plasma.
    lower, share, inside = _neighbouring_bins(logT_cube, logT_grid, logT_edges)
    parts, (n_rows, n_cols) = _pixel_parts(logT_cube.shape, integration_axis, sky)
    em = np.zeros(n_rows * n_cols * nT)   # cm^-5
    em_n = np.zeros_like(em)              # cm^-5 * n_e
    for cells, pixel, weight, _ in parts:
        lower_c, share_c, inside_c = lower[cells], share[cells], inside[cells]
        w2_c, w3_c = w2[cells], w3[cells]
        for step, of_T in ((0, 1.0 - share_c), (1, share_c)):
            in_bin = (pixel * nT + np.minimum(lower_c + step, nT - 1)).ravel()
            of_T = np.where(inside_c, of_T, 0.0)
            if weight is not None:
                of_T = of_T * weight
            np.add.at(em, in_bin, (w2_c * of_T).ravel())
            np.add.at(em_n, in_bin, (w3_c * of_T).ravel())
    if sky is not None:
        em = sky.to_image(em.reshape(n_rows * n_cols, nT))
        em_n = sky.to_image(em_n.reshape(n_rows * n_cols, nT))
        n_rows, n_cols = sky.image_shape
    em, em_n = em.reshape(n_rows, n_cols, nT), em_n.reshape(n_rows, n_cols, nT)

    dem = em / dlogT
    # Zero where there is no plasma; the division alone would leave
    # uninitialised memory there.
    avg_ne = np.divide(em_n, em, out=np.zeros_like(em), where=em > 0.0)
    return dem, avg_ne, w2


def _temperature_bins(logT_grid: np.ndarray) -> Tuple[float, np.ndarray]:
    """Width and edges of the DEM temperature bins centred on `logT_grid`."""
    dlogT = logT_grid[1] - logT_grid[0] if len(logT_grid) > 1 else 0.1
    logT_edges = np.concatenate([
        [logT_grid[0] - dlogT/2],
        logT_grid[:-1] + dlogT/2,
        [logT_grid[-1] + dlogT/2]
    ])
    return dlogT, logT_edges


def _neighbouring_bins(values: np.ndarray, centres: np.ndarray,
                       edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split each value between the two bins either side of it.

    Returns the index of the bin whose centre is at or below each value, the
    share going to the next bin up, and whether the value is inside the
    outer bin edges.

    The share rises linearly from 0 at one centre to 1 at the next, so the
    two centres weighted by their shares average to the value: 2.4 km/s on a
    5 km/s grid is 0.52 in the 0 km/s bin and 0.48 in the 5 km/s bin. A
    value between an end centre and its outer edge goes wholly to that bin.
    A value beyond the edges is in no bin, and its share is zero.
    """
    values = np.asarray(values, dtype=np.float64)
    inside = (values >= edges[0]) & (values < edges[-1])
    n = centres.size
    lower = np.clip(np.searchsorted(centres, values, side="right") - 1, 0, max(n - 2, 0))
    if n > 1:
        share = np.clip((values - centres[lower]) / (centres[lower + 1] - centres[lower]),
                        0.0, 1.0)
    else:
        share = np.zeros(values.shape)
    return lower, np.where(inside, share, 0.0), inside


def _image_pixels(shape: Tuple[int, int, int], integration_axis: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Image pixel that each cell of a (z, y, x) cube is seen in.

    Returns each cell's flat pixel index (row times the number of columns
    plus column) and the image shape (rows, columns), as `compute_dem` and
    `build_em_tv` use them.

    Seen from the side opposite :data:`OBSERVER_SIDE`, the image is mirrored
    left to right, as an observer on that side would see it.
    """
    axis, side = view_axis_and_side(integration_axis)
    z, y, x = np.indices(shape, sparse=True)
    rows, columns = {"x": (z, y), "y": (z, x), "z": (y, x)}[axis]
    n_rows, n_columns = rows.size, columns.size
    if side != OBSERVER_SIDE[axis]:
        columns = n_columns - 1 - columns
    return np.broadcast_to(rows * n_columns + columns, shape), (n_rows, n_columns)


def _log_cubes(temperature: np.ndarray, electron_density: np.ndarray,
               precision: type) -> Tuple[np.ndarray, np.ndarray]:
    """log10 of the temperature and electron density, in `precision`.

    A cell with no electrons gets a log density of minus infinity, so it
    adds nothing to the emission measure. A cell with no temperature gets
    log T = 0, below every temperature grid.
    """
    logN_cube = np.log10(electron_density, where=electron_density > 0.0,
                         out=np.full_like(electron_density, -np.inf)).astype(precision)
    logT_cube = np.log10(temperature, where=temperature > 0.0,
                         out=np.zeros_like(temperature)).astype(precision)
    return logT_cube, logN_cube


def _log_cubes_by_slab(temperature: np.ndarray, density, to_electron_density, precision: type,
                       slab: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    `_log_cubes`, one slab of the first axis at a time.

    The electron density, which can be float64, is then never held for the
    whole box. `to_electron_density` turns a slab of `temperature` and of
    `density` into electron density in cm^-3.
    """
    logT_cube = np.empty(temperature.shape, dtype=precision)
    logN_cube = np.empty(temperature.shape, dtype=precision)
    if slab is None:
        # Planes whose few float64 arrays fit in EM_BLOCK_BYTES.
        slab = max(1, int(EM_BLOCK_BYTES // (4 * 8 * temperature[0].size)))
    for first in range(0, temperature.shape[0], slab):
        cut = slice(first, first + slab)
        logT_cube[cut], logN_cube[cut] = _log_cubes(
            temperature[cut], to_electron_density(temperature[cut], density[cut]), precision)
    return logT_cube, logN_cube


def density_grid(
    temperature: np.ndarray,
    electron_density: np.ndarray,
    logT_grid: np.ndarray,
    precision: type = np.float64,
) -> Tuple[float, float, int]:
    """Density grid on which to compute G(T, n_e) for an atmosphere.

    The contribution functions are taken at the mean density at each
    temperature along each line of sight, and are zero off their grid. This
    grid has the default grid's points (every 0.3 in log10 n_e, through
    10**7 cm^-3) and reaches just past the densities of all cells with a
    temperature on `logT_grid`. No mean density falls off it, and a narrow
    density range needs fewer points.

    Parameters
    ----------
    temperature, electron_density : np.ndarray
        Temperature of every cell in K and electron density in cm^-3.
    logT_grid : np.ndarray
        DEM temperature grid, as returned by :func:`compute_goft_fiasco`.
    precision : type
        Precision of the synthesis; the densities are rounded to it as in
        :func:`synthesise_cubes`.

    Returns
    -------
    logN_min, logN_max : float
    nN : int
        Grid for :func:`compute_goft_fiasco`. The default grid if no cell
        has electrons at a temperature on `logT_grid`.
    """
    return _density_grid_from_points(
        _density_points(temperature, electron_density, logT_grid, precision))


def _density_grid_from_points(points: Optional[Tuple[int, int]]) -> Tuple[float, float, int]:
    """`density_grid`'s result from `_density_points`'s."""
    if points is None:
        return _LOGN_MIN, _LOGN_MAX, _N_N
    first, last = points
    return _density_point(first), _density_point(last), last - first + 1


def band_density_grid(logN_min: float, logN_max: float) -> np.ndarray:
    """
    Densities, as log10 n_e, on which to compute a band's lines for a synthesis.

    The default grid, extended to `logN_min` and `logN_max` if they lie
    beyond it. A band's lines take hours to compute, and on this grid they
    serve every later synthesis within the default grid.
    """
    step = (_LOGN_MAX - _LOGN_MIN) / (_N_N - 1)
    first = min(0, int(round((logN_min - _LOGN_MIN) / step)))
    last = max(_N_N - 1, int(round((logN_max - _LOGN_MIN) / step)))
    return np.array([_density_point(index) for index in range(first, last + 1)])


def _density_point(index: int) -> float:
    """log10 n_e at 0-based point `index` of the default density grid."""
    return _LOGN_MIN + index * (_LOGN_MAX - _LOGN_MIN) / (_N_N - 1)


def _density_points(temperature: np.ndarray, electron_density: np.ndarray,
                    logT_grid: np.ndarray, precision: type) -> Optional[Tuple[int, int]]:
    """First and last :func:`_density_point` indices of :func:`density_grid`.

    None if no cell has electrons at a temperature on `logT_grid`.
    """
    return _density_points_from_logs(*_log_cubes(temperature, electron_density, precision),
                                     logT_grid)


def _density_points_from_logs(logT_cube: np.ndarray, logN_cube: np.ndarray,
                              logT_grid: np.ndarray) -> Optional[Tuple[int, int]]:
    """`_density_points` from the cells' logs, as `_log_cubes` gives them."""
    _, logT_edges = _temperature_bins(logT_grid)
    counted = ((logT_cube >= logT_edges[0]) & (logT_cube < logT_edges[-1])
               & np.isfinite(logN_cube))
    if not counted.any():
        return None

    return _points_just_past(np.min(logN_cube, where=counted, initial=np.inf),
                             np.max(logN_cube, where=counted, initial=-np.inf))


def _points_just_past(lowest: float, highest: float) -> Tuple[int, int]:
    """
    :func:`_density_point` indices just past `lowest` and `highest`, as log10 n_e.
    """
    step = (_LOGN_MAX - _LOGN_MIN) / (_N_N - 1)
    # Positions in steps from the default grid's first point, rounded so a
    # density on a point to within floating-point error counts as on it.
    lowest, highest = (round((float(extreme) - _LOGN_MIN) / step, 9)
                       for extreme in (lowest, highest))
    # Take the points just past them, so a mean density a rounding error
    # beyond the lowest or highest is still on the grid.
    return int(np.ceil(lowest)) - 1, int(np.floor(highest)) + 1


def interpolate_g_on_dem(
    goft: Dict[str, dict],
    avg_ne: np.ndarray,
    logT_grid: np.ndarray,
    logN_grid: np.ndarray,
    logT_goft: np.ndarray,
    precision: type = np.float32,
) -> None:
    """
    Each line's contribution function at each pixel's density and temperature.

    G is interpolated linearly from the `compute_goft_fiasco` grid and is
    zero outside it. The result is stored in each line's entry of `goft` as
    `g`, shape (rows, columns, temperatures), for `synthesise_spectra`.

    Parameters
    ----------
    goft : dict
        Lines from `compute_goft_fiasco`, modified in place.
    avg_ne : np.ndarray
        Electron density in each pixel at each temperature, in cm-3, shape
        (rows, columns, temperatures). If the cells along the line of sight
        differ in density, give their mean weighted by emission measure, as
        the synthesis does. For a DEM, give the density you choose.
    logT_grid : np.ndarray
        Temperatures at which to take G, as log10(T / K).
    logN_grid : np.ndarray
        Densities G was computed on, from `compute_goft_fiasco`.
    logT_goft : np.ndarray
        Temperatures G was computed on, from `compute_goft_fiasco`.
    precision : type, optional
        `np.float32` (default) or `np.float64`.
    """
    lines = _g_on_pixels(goft, avg_ne, logT_grid, logN_grid, logT_goft, precision)
    for name, g in tqdm(lines, total=len(goft), desc="interpolating G", unit="line",
                        leave=False):
        goft[name]["g"] = g


def _g_on_pixels(goft: Dict[str, dict], avg_ne: np.ndarray, logT_grid: np.ndarray,
                 logN_grid: np.ndarray, logT_goft: np.ndarray, precision: type):
    """Each line's name and G, one line at a time, as `interpolate_g_on_dem` stores it."""
    nT, n_rows, n_cols = len(logT_grid), *avg_ne.shape[:2]

    # Build query points for interpolation
    logNe_flat = np.log10(avg_ne, where=avg_ne > 0.0,
                         out=np.zeros_like(avg_ne)).transpose(2, 0, 1).ravel()
    logT_flat = np.broadcast_to(logT_grid[:, None, None],
                               (nT, n_rows, n_cols)).ravel()
    query_pts = np.column_stack((logNe_flat, logT_flat))

    for name, info in goft.items():
        rgi = RegularGridInterpolator(
            (logN_grid, logT_goft), info["g_tn"],
            method="linear", bounds_error=False, fill_value=0.0
        )
        g_flat = rgi(query_pts)
        yield name, g_flat.reshape(nT, n_rows, n_cols).transpose(1, 2, 0).astype(precision)


##############################################################################
# ---------------------------------------------------------------------------
#  Build EM(T,v) and synthesise spectra
# ---------------------------------------------------------------------------
##############################################################################

# A view looks along one axis of the box from one side. An axis alone, such
# as "x", looks from OBSERVER_SIDE (in utils): +1 for the side of increasing
# coordinate, -1 for the other. From there the line cube from
# create_line_cube is the right way round: the column axis crossed with the
# row axis points at the observer, at +z from above and at +x and -y from
# the sides. A signed view, such as "-x", looks from that side; from the
# side opposite OBSERVER_SIDE the image is mirrored left to right
# (_image_pixels).
SIGNED_VIEWS = ("+x", "-x", "+y", "-y", "+z", "-z")


def line_of_sight_velocity(velocity, integration_axis: str):
    """
    Velocity away from the observer, from velocity along the integration axis.

    Simulation velocities are positive towards increasing coordinate. Seen
    from above, an upflow (positive z velocity) moves towards the observer,
    so its line-of-sight velocity is negative, a blueshift. Each line is
    placed at `lambda_0 (1 + v / c)`.

    Parameters
    ----------
    velocity : np.ndarray or u.Quantity
        Velocity along the integration axis, positive towards increasing
        coordinate.
    integration_axis : str
        View: `"x"`, `"y"` or `"z"`, seen from the side given by
        :data:`OBSERVER_SIDE`, or a signed axis for the side, such as `"-x"`
        for the view from -x.

    Returns
    -------
    np.ndarray or u.Quantity
        Velocity away from the observer, in the same units.
    """
    _, side = view_axis_and_side(integration_axis)
    return -side * velocity


def _pixel_parts(shape: Tuple[int, int, int], integration_axis: str, sky: Optional[SkyView]):
    """
    Cells, the pixels they go to and their shares, and the grid they are summed on.

    Returns `(parts, grid_shape)`. Each part is `(cells, pixel, share, towards)`:
    `cells` indexes the box's arrays, `pixel` is the flat grid pixel each
    cell goes to, and `share` its share there, or None for all of it.
    `towards` is as for `SkyView.parts`, None along an axis. Along an axis
    each cell is in one pixel. With `sky` the grid may be the box's base,
    which `sky.to_image` maps to the image.
    """
    if sky is None:
        pixel, image_shape = _image_pixels(shape, integration_axis)
        return [(Ellipsis, pixel, None, None)], image_shape
    if view_axis_and_side(integration_axis) != ("z", +1):
        raise ValueError(f"A box seen from a position on the disk is seen from above, so its "
                         f"integration_axis is 'z', not {integration_axis!r}.")
    return sky.parts(), sky.accumulation_shape


def build_em_tv(
    logT_cube: np.ndarray,
    vel_cube: np.ndarray,
    logT_grid: np.ndarray,
    vel_grid: np.ndarray,
    ne_sq_dh: np.ndarray,
    integration_axis: str = "z",
    *,
    sky: Optional[SkyView] = None,
) -> np.ndarray:
    """
    Emission measure EM(row, column, T, v) in cm^-5.

    Each cell's emission measure is split between the temperature and
    velocity bins either side of it, by how close it is to each
    (:func:`_neighbouring_bins`). Its mean velocity is then the cell's own,
    and its contribution function is interpolated in temperature. A value
    past the outermost bin centre but inside that bin goes wholly to it.

    Parameters
    ----------
    logT_cube : np.ndarray
        3D temperature cube, shape (nz, ny, nx).
    vel_cube : np.ndarray
        3D velocity cube along the integration axis, or with a curved `sky`,
        three cubes of velocity along the box's x, y and z.
    logT_grid : np.ndarray
        Temperature bin centres.
    vel_grid : np.ndarray
        Velocity bin centres.
    ne_sq_dh : np.ndarray
        n_e^2 dh of each cell.
    integration_axis : str
        View: the axis to integrate along, `"x"`, `"y"` or `"z"`, optionally
        signed for the side of the box to look from, such as `"-x"`; see
        `line_of_sight_velocity`. `"z"` with `sky`.
    sky : SkyView, optional
        Box seen from a position on the disk, along a slanted line of sight
        instead of an axis. Keyword only. Default None.

    Returns
    -------
    em_tv : np.ndarray
        Emission measure, with the two remaining spatial axes in image order
        (row, column). Shape by axis:
        - "x": (nz, ny, nT, nv)
        - "y": (nz, nx, nT, nv)
        - "z": (ny, nx, nT, nv)
        - with `sky`: (*sky.image_shape, nT, nv)
    """
    axis, _ = view_axis_and_side(integration_axis)
    print("  Building 4-D emission-measure cube along "
          + (f"{axis}-axis..." if sky is None else "the slanted line of sight..."))
    em_tv, beyond, seen = _em_tv(logT_cube, vel_cube, logT_grid, vel_grid, ne_sq_dh,
                                 integration_axis, sky)
    _warn_beyond_velocity_grid(beyond, seen, vel_grid)
    return em_tv


def _velocity_edges(vel_grid) -> Tuple[np.ndarray, np.ndarray]:
    """Velocity bin centres and edges in cm/s; a plain array is taken to be in cm/s."""
    v_centres = (vel_grid.to_value(u.cm / u.s) if isinstance(vel_grid, u.Quantity)
                 else np.asarray(vel_grid, dtype=float))
    return v_centres, velocity_centers_to_edges(v_centres)


def _warn_beyond_velocity_grid(beyond: float, seen: float, vel_grid) -> None:
    """Warn that plasma faster than the velocity grid is missing from the spectra."""
    # Such plasma is kept in the DEM but not the spectra. The default grid
    # misses the fastest flows of a flare.
    if beyond > 0:
        _, v_edges = _velocity_edges(vel_grid)
        warnings.warn(
            f"{100 * beyond / seen:.3g} per cent of the emission measure is "
            f"from plasma faster than the velocity grid reaches, "
            f"{v_edges[0] / 1e5:.1f} to {v_edges[-1] / 1e5:.1f} km/s, and emits beyond the "
            f"synthesised wavelengths, so it is not in the spectra. Widen --vel-lim to keep "
            f"it.", UserWarning, stacklevel=3)


def _em_tv(logT_cube, vel_cube, logT_grid, vel_grid, ne_sq_dh, integration_axis,
           sky) -> Tuple[np.ndarray, float, float]:
    """
    `build_em_tv`'s cube, plus the emission measure beyond the velocity grid and the total.
    """
    _, logT_edges = _temperature_bins(logT_grid)
    v_centres, v_edges = _velocity_edges(vel_grid)

    # Split each cell between the temperature and velocity bins either side
    # of it. Taking the nearest bin would put a 2.4 km/s flow on the default
    # 5 km/s grid at rest. With the split, the mean velocity is the cell's,
    # the line's variance grows by at most a quarter of a bin squared, and G
    # is interpolated in temperature.
    lower_T, share_T, in_T = _neighbouring_bins(logT_cube, logT_grid, logT_edges)
    # On a curved surface each part of a cell has its own line of sight, and
    # so its own velocity along it, computed below.
    curved = sky is not None and sky.curved
    in_temperature = ne_sq_dh * in_T
    beyond, seen = 0.0, 0.0
    if not curved:
        lower_v, share_v, in_v = _neighbouring_bins(vel_cube, v_centres, v_edges)
        beyond, seen = in_temperature[~in_v].sum(), in_temperature.sum()

    # Sum along the line of sight into each image pixel's bins.
    nT, nv = len(logT_grid), len(v_centres)
    parts, (n_rows, n_cols) = _pixel_parts(logT_cube.shape, integration_axis, sky)
    em_tv = np.zeros(n_rows * n_cols * nT * nv)
    for cells, pixel, weight, towards in parts:
        of_cell = in_temperature[cells] if weight is None else in_temperature[cells] * weight
        if curved:
            # The velocity away from the observer.
            away = -(towards[:, 0] * vel_cube[0][cells] + towards[:, 1] * vel_cube[1][cells]
                     + towards[:, 2] * vel_cube[2][cells])
            lower_v_c, share_v_c, in_v_c = _neighbouring_bins(away, v_centres, v_edges)
            beyond += of_cell[~in_v_c].sum()
            seen += of_cell.sum()
        else:
            lower_v_c, share_v_c, in_v_c = lower_v[cells], share_v[cells], in_v[cells]
        in_pixel = np.where(in_v_c, of_cell, 0.0)
        lower_T_c, share_T_c = lower_T[cells], share_T[cells]
        for step_T, of_T in ((0, 1.0 - share_T_c), (1, share_T_c)):
            in_bin_T = pixel * nT + np.minimum(lower_T_c + step_T, nT - 1)
            for step_v, of_v in ((0, 1.0 - share_v_c), (1, share_v_c)):
                in_bin = in_bin_T * nv + np.minimum(lower_v_c + step_v, nv - 1)
                np.add.at(em_tv, in_bin.ravel(), (in_pixel * of_T * of_v).ravel())

    if sky is not None:
        em_tv = sky.to_image(em_tv.reshape(n_rows * n_cols, nT * nv))
        n_rows, n_cols = sky.image_shape
    return em_tv.reshape(n_rows, n_cols, nT, nv), beyond, seen


def synthesise_spectra(
    goft: Dict[str, dict],
    em_tv: np.ndarray,
    vel_grid: u.Quantity | np.ndarray,
    logT_grid: np.ndarray,
) -> None:
    """
    Spectrum of each line in every pixel, from the emission measure EM(T, v).

    Each temperature and velocity bin adds a Gaussian of that temperature's
    thermal width, shifted by that velocity and scaled by the emission
    measure times G. The spectra are stored in each line's entry of `goft`
    as `si`, in erg / (s cm2 sr cm), on the wavelengths `wl_grid`, which are
    `lambda_0 (1 + v / c)` on the velocity grid.

    Parameters
    ----------
    goft : dict
        Lines with `g` from `interpolate_g_on_dem`, modified in place.
    em_tv : np.ndarray
        Emission measure in each bin, in cm-5, shape (rows, columns,
        temperatures, velocities).
    vel_grid : u.Quantity or np.ndarray
        Velocity bin centres, evenly spaced and increasing. A plain array is
        taken to be in cm/s.
    logT_grid : np.ndarray
        Temperature bin centres, as log10(T / K).
    """
    vel_grid = _spectral_velocity_grid(vel_grid)
    for line, data in tqdm(goft.items(), desc="spectra", unit="line", leave=False):
        data["wl_grid"] = (vel_grid * data["wl0"] / const.c + data["wl0"]).cgs
        data["si"] = _line_spectra(em_tv, data["g"], _line_profiles(data, vel_grid, logT_grid))


def _spectral_velocity_grid(vel_grid) -> u.Quantity:
    """Velocity grid in cm/s, checked to be even; a plain array is taken to be in cm/s."""
    # The Doppler shifts are in cm/s. A grid in km/s taken as cm/s
    # would shift every line by a factor of 1e5 too little.
    vel_grid = (vel_grid.to(u.cm / u.s) if isinstance(vel_grid, u.Quantity)
                else np.asarray(vel_grid, dtype=float) * (u.cm / u.s))
    # create_line_cube sets CDELT from the first step alone, so an uneven grid
    # would give a wrong wavelength axis.
    require_uniform_grid(vel_grid, "vel_grid")
    return vel_grid


def _line_profiles(data: dict, vel_grid: u.Quantity, logT_grid: np.ndarray) -> np.ndarray:
    """
    Normalised Gaussian of each temperature and velocity bin, in 1/cm.

    On the line's wavelengths `data["wl_grid"]`, shape (temperatures,
    velocities, wavelengths).
    """
    kb = const.k_B.cgs.value
    c_cm_s = const.c.cgs.value
    wl0 = data["wl0"].cgs.value  # cm
    wl_grid = data["wl_grid"].cgs.value  # (n_lambda,)

    _, atomic_weight = element_data(int(data["atom"]))
    atom_weight_g = (atomic_weight * u.u).cgs.value

    # Thermal width per T-bin: sigma_T (nT,)
    sigma_T = wl0 * np.sqrt(kb * (10 ** logT_grid) / atom_weight_g) / c_cm_s

    # Doppler-shifted center for each v-bin: (nv,)
    lam_cent = wl0 * (1 + vel_grid.value / c_cm_s)

    # Build phi(T,v,lambda) as (nT,nv,n_lambda)
    delta = wl_grid[None, None, :] - lam_cent[None, :, None]
    phi = np.exp(-0.5 * (delta / sigma_T[:, None, None]) ** 2)
    phi /= sigma_T[:, None, None] * np.sqrt(2 * np.pi)
    return phi


def _line_spectra(em_tv: np.ndarray, g: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    One line's spectra in each pixel of `em_tv`, in erg / (s cm2 sr cm).

    On the wavelengths of `phi` from `_line_profiles`. `g` is the line's
    contribution function on the same pixels and temperatures, or one that
    broadcasts to them.
    """
    # One row at a time, as EM(row,col,T,v) * G(T) can be as large as em_tv.
    g = np.broadcast_to(g, em_tv.shape[:3])
    spec_map = np.zeros((*em_tv.shape[:2], phi.shape[-1]))
    for row in range(em_tv.shape[0]):
        weighted = em_tv[row] * g[row, ..., None]
        # Collapse T and v: dot ((nT,nv) , (nT,nv)) -> (n_cols,n_lambda)
        spec_map[row] = np.tensordot(weighted, phi, axes=([1, 2], [0, 1]))
    return spec_map / (4 * np.pi)


# Largest block of the emission measure cube held at once, in bytes, when
# synthesising a block of image rows at a time.
EM_BLOCK_BYTES = 2**30


def _synthesise_by_rows(lines: Dict[str, dict], logT_cube: np.ndarray, logN_cube: np.ndarray,
                        los_velocity: np.ndarray, dh_cm, logT_grid: np.ndarray,
                        logN_grid: np.ndarray, vel_grid: u.Quantity, integration_axis: str,
                        precision: type) -> Tuple[np.ndarray, np.ndarray]:
    """
    DEM, density, G and spectra, a block of image rows at a time.

    Neither the whole emission measure cube nor any float64 array over the
    whole box is held. Each image row is a slab of the box, so the result
    is the same as for the whole box. Each line in `lines` gets `g`,
    `wl_grid` and `si`, as from `interpolate_g_on_dem` and
    `synthesise_spectra`. Returns the DEM and the density, as from
    `compute_dem`.
    """
    axis, _ = view_axis_and_side(integration_axis)
    print(f"  Building the DEM and the emission measure in (T,v) along {axis}-axis, a "
          f"block of rows at a time...")
    vel_grid = _spectral_velocity_grid(vel_grid)
    shape = logT_cube.shape
    row_axis = 1 if axis == "z" else 0
    n_rows = shape[row_axis]
    n_cols = shape[{"x": 1, "y": 2, "z": 2}[axis]]
    nT = len(logT_grid)
    # Bytes per row: its emission measure, or about 16 numbers per cell used
    # by compute_dem and _em_tv, whichever is larger.
    per_row = max(n_cols * nT * vel_grid.size,
                  16 * logT_cube[(slice(None),) * row_axis + (0,)].size) * 8
    step = max(1, int(EM_BLOCK_BYTES // per_row))

    dem_map = np.empty((n_rows, n_cols, nT))
    avg_ne = np.empty((n_rows, n_cols, nT))
    profiles = {}
    for name, data in lines.items():
        data["wl_grid"] = (vel_grid * data["wl0"] / const.c + data["wl0"]).cgs
        data["g"] = np.empty((n_rows, n_cols, nT), dtype=precision)
        data["si"] = np.zeros((n_rows, n_cols, vel_grid.size), dtype=precision)
        profiles[name] = _line_profiles(data, vel_grid, logT_grid)
    los_velocity = np.broadcast_to(los_velocity, shape)
    beyond, seen = 0.0, 0.0
    for first in tqdm(range(0, n_rows, step), desc="rows", unit="block", leave=False):
        rows = slice(first, min(first + step, n_rows))
        cut = (slice(None),) * row_axis + (rows,)
        logT_rows, logN_rows = logT_cube[cut], logN_cube[cut]
        dem_map[rows], avg_ne[rows], ne_sq_dh = _compute_dem(
            logT_rows, logN_rows, dh_cm, logT_grid, integration_axis, None)
        em_tv, more, total = _em_tv(logT_rows, los_velocity[cut], logT_grid, vel_grid,
                                    ne_sq_dh, integration_axis, None)
        del ne_sq_dh
        beyond, seen = beyond + more, seen + total
        for name, g in _g_on_pixels(lines, avg_ne[rows], logT_grid, logN_grid, logT_grid,
                                    precision):
            lines[name]["g"][rows] = g
            lines[name]["si"][rows] = _line_spectra(em_tv, g, profiles[name])
    _warn_beyond_velocity_grid(beyond, seen, vel_grid)
    return dem_map, avg_ne


def synthesise_cubes(
    temperature: np.ndarray,
    electron_density: np.ndarray,
    los_velocity: np.ndarray,
    dh_cm,
    goft: Dict[str, dict],
    logT_grid: np.ndarray,
    logN_grid: np.ndarray,
    vel_grid: u.Quantity,
    integration_axis: str,
    precision: type,
    *,
    continuum: Optional[Dict[str, dict]] = None,
    sky: Optional[SkyView] = None,
    return_density: bool = False,
    keep_em_tv: bool = True,
) -> Union[Tuple[Dict[str, dict], np.ndarray, np.ndarray],
           Tuple[Dict[str, dict], np.ndarray, np.ndarray, np.ndarray]]:
    """
    Spectra of every line from one set of cubes (a whole box or a strip of it).

    Parameters
    ----------
    temperature : np.ndarray
        Temperature of every cell in K, shape `(nz, ny, nx)`.
    electron_density : np.ndarray
        Electron density of every cell in cm^-3, same shape.
    los_velocity : np.ndarray
        Velocity away from the observer in cm/s, same shape, or with a
        curved `sky`, three cubes of velocity along the box's x, y and z.
    dh_cm : float or np.ndarray
        Cell depth along the line of sight in cm, one value for all cells or
        one per cell along that axis.
    goft : dict
        Contribution functions from :func:`compute_goft_fiasco`, on
        `logT_grid` and `logN_grid`. Not modified; the entries are copied.
    logT_grid, logN_grid : np.ndarray
        Grids the contribution functions are tabulated on. The DEM uses the
        same temperature grid.
    vel_grid : u.Quantity
        Velocity bin centres, evenly spaced.
    integration_axis : str
        View: `"x"`, `"y"` or `"z"`, optionally signed for the side of the
        box to look from, such as `"-x"`; see `line_of_sight_velocity`.
    precision : type
        `np.float32` or `np.float64`.
    continuum : dict, optional
        Continuum to add, by entry name, each with `"wl_grid"` (its
        wavelengths) and `"free"` and `"two_photon"` from
        `continuum.compute_continuum_fiasco` on these temperatures and
        densities. Keyword only. Default None (no continuum).
    sky : SkyView, optional
        Box seen from a position on the disk, along a slanted line of sight;
        see :func:`compute_dem`. `integration_axis` is then `"z"`, `dh_cm`
        the cells' height and `los_velocity` the velocity away from the
        observer. Keyword only. Default None.
    return_density : bool, optional
        Also return the electron density for the contribution functions: at
        each temperature of each pixel, the mean over its cells weighted by
        emission measure. Keyword only. Default False.
    keep_em_tv : bool, optional
        Return the emission measure cube. False builds it a block of image
        rows at a time, using far less memory, and returns None instead.
        With `sky` it is built whole either way. Keyword only. Default True.

    Returns
    -------
    lines : dict
        Copy of `goft` whose entries also hold `"g"` (contribution function
        on the DEM), `"wl_grid"` and `"si"` (specific intensity, shape
        `(rows, columns, wavelength)`). With `continuum`, it also has an
        entry per continuum name, with `"wl_grid"`, `"si"` and `"wl0"` (the
        middle of its wavelengths), and no `"atom"` or `"ion"`.
    dem_map : np.ndarray
        As returned by :func:`compute_dem`.
    em_tv : np.ndarray or None
        As returned by :func:`build_em_tv`, or None without `keep_em_tv`.
    electron_density : np.ndarray
        Only with `return_density`: density in cm^-3, same shape as
        `dem_map`, zero where there is no plasma at a temperature.
    """
    logT_cube, logN_cube = _log_cubes(temperature, electron_density, precision)
    return _synthesise_from_logs(
        logT_cube, logN_cube, los_velocity, dh_cm, goft, logT_grid, logN_grid, vel_grid,
        integration_axis, precision, continuum=continuum, sky=sky,
        return_density=return_density, keep_em_tv=keep_em_tv)


def _warn_off_density_grid(dem_map: np.ndarray, avg_ne_map: np.ndarray,
                           logN_grid: np.ndarray) -> None:
    """Warn of emission measure at densities the contribution functions do not cover."""
    # The contribution functions are zero off their density grid.
    occupied = dem_map > 0
    log_ne = np.log10(avg_ne_map, where=avg_ne_map > 0,
                      out=np.full(avg_ne_map.shape, -np.inf))
    off_grid = occupied & ((log_ne < logN_grid[0]) | (log_ne > logN_grid[-1]))
    if off_grid.any():
        warnings.warn(
            f"{100 * dem_map[off_grid].sum() / dem_map[occupied].sum():.3g} per cent of the "
            f"emission measure is at electron densities outside the {10 ** logN_grid[0]:.0e} "
            f"to {10 ** logN_grid[-1]:.0e} cm^-3 the contribution functions are worked out "
            f"for, where they are taken as zero.", UserWarning, stacklevel=3)


def _synthesise_from_logs(logT_cube, logN_cube, los_velocity, dh_cm, goft, logT_grid,
                          logN_grid, vel_grid, integration_axis, precision, *, continuum=None,
                          sky=None, return_density=False, keep_em_tv=True):
    """
    `synthesise_cubes` from the cells' logs, as `_log_cubes` gives them.
    """
    lines = {name: dict(info) for name, info in goft.items()}
    if keep_em_tv or sky is not None:
        dem_map, avg_ne_map, ne_sq_dh = _compute_dem(logT_cube, logN_cube, dh_cm, logT_grid,
                                                     integration_axis, sky)
        _warn_off_density_grid(dem_map, avg_ne_map, logN_grid)
        interpolate_g_on_dem(lines, avg_ne_map, logT_grid, logN_grid, logT_grid, precision)
        em_tv = build_em_tv(logT_cube, los_velocity, logT_grid, vel_grid, ne_sq_dh,
                            integration_axis, sky=sky)
        del ne_sq_dh
        synthesise_spectra(lines, em_tv, vel_grid, logT_grid)
        for name in goft:
            lines[name]["si"] = lines[name]["si"].astype(precision, copy=False)
        if not keep_em_tv:
            em_tv = None
    else:
        em_tv = None
        dem_map, avg_ne_map = _synthesise_by_rows(
            lines, logT_cube, logN_cube, los_velocity, dh_cm, logT_grid, logN_grid, vel_grid,
            integration_axis, precision)
        _warn_off_density_grid(dem_map, avg_ne_map, logN_grid)

    # The continuum is not Doppler shifted, so it uses the DEM, which unlike
    # em_tv keeps plasma faster than the velocity grid.
    if continuum:
        emission_measure = dem_map * _temperature_bins(logT_grid)[0]
        for name, table in continuum.items():
            grid = u.Quantity(table["wl_grid"]).to(u.cm)
            spectra = continuum_spectra(emission_measure, avg_ne_map, logN_grid,
                                        table["free"], table["two_photon"])
            lines[name] = {"si": spectra.astype(precision), "wl_grid": grid,
                           "wl0": grid[grid.size // 2], "atom": None, "ion": None}
    if return_density:
        return lines, dem_map, em_tv, avg_ne_map
    return lines, dem_map, em_tv


def _cell_size_mm(cube: NDCube, pixel_axis: int) -> float:
    """World size in Mm of one pixel of `cube` along `pixel_axis`.

    Measured between the edges of the first pixel, so it needs neither a
    second pixel nor a CDELT, which a cropped cube's WCS does not have.
    """
    # A cropped cube wraps its WCS in a high-level object; use the low-level
    # one underneath.
    wcs = getattr(cube.wcs, "low_level_wcs", cube.wcs)
    low = [0.0] * wcs.pixel_n_dim
    high = [0.0] * wcs.pixel_n_dim
    low[pixel_axis], high[pixel_axis] = -0.5, 0.5
    world_low = wcs.pixel_to_world_values(*low)
    world_high = wcs.pixel_to_world_values(*high)
    unit = u.Unit(wcs.world_axis_units[pixel_axis])
    return ((world_high[pixel_axis] - world_low[pixel_axis]) * unit).to_value(u.Mm)


def _world_at(coords: u.Quantity, crpix: float) -> float:
    """
    Value of the uniform grid `coords` at 1-based pixel `crpix`, as a float.

    A reference pixel in the middle of an axis with an even number of pixels
    falls between two of them, so its value is interpolated rather than
    taken from the pixel below.
    """
    if coords.size == 1:
        return coords[0].value
    step = (coords[-1] - coords[0]).value / (coords.size - 1)
    return coords[0].value + (crpix - 1) * step


def create_line_cube(
    line_name: str,
    line_data: dict,
    spatial_cube: NDCube,
    intensity_unit: u.Unit,
    integration_axis: str = "z",
    image_columns: Optional[Tuple[int, int]] = None,
) -> NDCube:
    """
    NDCube of one line's synthesised spectra, with the image's coordinates.

    Parameters
    ----------
    line_name : str
        Line name.
    line_data : dict
        The line's entry from `compute_goft_fiasco`, with its spectra `si`
        and their wavelengths `wl_grid` added by `synthesise_spectra`.
    spatial_cube : NDCube
        Atmosphere cube, such as from `create_atmosphere_ndcube`, whose
        coordinates the image takes.
    intensity_unit : u.Unit
        Unit of the spectra.
    integration_axis : str, optional
        View used in the synthesis: `"x"`, `"y"` or `"z"` (default),
        optionally signed for the side of the box, such as `"-x"`; see
        `line_of_sight_velocity`. Seen from the side opposite
        :data:`OBSERVER_SIDE`, the image is mirrored left to right, and its
        column coordinate is the box coordinate with its sign flipped,
        increasing to that observer's right.
    image_columns : tuple of int, optional
        For spectra of a strip of the image only: the strip's first column
        and the number of columns of the whole image. Default None, for the
        whole image.

    Returns
    -------
    NDCube
        Spectra indexed `[row, column, wavelength]` like an image, so summing
        over wavelength gives an array that plots the right way up.
    """
    view = view_name(integration_axis)
    los_axis, side = view_axis_and_side(view)
    # An axis with unequal cells has no single CDELT. Only the line of sight
    # may be like that, since it is integrated out here.
    nonuniform = (spatial_cube.meta or {}).get("nonuniform_axes", [])
    stretched = [axis for axis in AXES
                 if axis != los_axis and axis in nonuniform]
    if stretched:
        raise ValueError(
            f"The {', '.join(stretched)} axis of the atmosphere is not evenly "
            f"spaced, so it cannot be an image axis of a view along "
            f"{los_axis}. Only the line of sight may be stretched.")

    # The simulation cubes are (z, y, x), so integrating one axis out leaves
    # 'si' already in (row, column, wavelength) order for every view.
    cube_data = line_data["si"]
    first_column, n_columns = ((0, cube_data.shape[1]) if image_columns is None
                               else image_columns)

    # Cell sizes as the world size of one pixel of the reference cube, which
    # works for an axis one cell wide and for any WCS, cropped ones included.
    cell_size = [_cell_size_mm(spatial_cube, pixel_axis) for pixel_axis in range(3)]

    # The WCS has a single CDELT, so the wavelength grid must be uniform.
    wl_step = require_uniform_grid(line_data["wl_grid"].to(u.cm), "wl_grid")

    # The box axes the image's columns and rows run along. The rows of a
    # side view run up z from the box's first cell; those of the view from
    # above run along y and are centred, like the columns.
    column_axis, row_axis = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[los_axis]
    n_rows, _, nl = cube_data.shape
    # Each box axis's place among the WCS pixel axes (x, y, z); the reference
    # cube's array axes are (z, y, x).
    wcs_index = {"x": 0, "y": 1, "z": 2}

    def world(axis):
        return spatial_cube.axis_world_coords(2 - wcs_index[axis])[0].to(u.Mm)

    spatial_axes = ["WAVE", f"SOL{column_axis.upper()}", f"SOL{row_axis.upper()}"]
    spatial_units = ["cm", "Mm", "Mm"]
    spatial_cdelt = [wl_step, cell_size[wcs_index[column_axis]], cell_size[wcs_index[row_axis]]]
    spatial_crpix = [(nl + 1) / 2, (n_columns + 1) / 2,
                     1 if row_axis == "z" else (n_rows + 1) / 2]
    spatial_crval = [
        _world_at(line_data["wl_grid"].to(u.cm), spatial_crpix[0]),
        _world_at(world(column_axis), spatial_crpix[1]),
        world("z")[0].value if row_axis == "z" else _world_at(world(row_axis), spatial_crpix[2]),
    ]

    # Seen from the other side, the image is mirrored (_image_pixels), so
    # flip the sign of the column coordinate. The columns are centred, so the
    # middle one stays put.
    if side != OBSERVER_SIDE[los_axis]:
        spatial_crval[1] = -spatial_crval[1]
    # A strip keeps the whole image's coordinates.
    spatial_crpix[1] -= first_column

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = spatial_axes
    wcs.wcs.cunit = spatial_units
    wcs.wcs.crpix = spatial_crpix
    wcs.wcs.crval = spatial_crval
    wcs.wcs.cdelt = spatial_cdelt

    return NDCube(
        cube_data,
        wcs=wcs,
        unit=intensity_unit,
        meta={
            "line_name": line_name,
            "rest_wav": line_data["wl0"],
            "atom": line_data["atom"],
            "ion": line_data["ion"],
            "integration_axis": view,
            "velocity_convention": VELOCITY_CONVENTION,
            "spatial_reference": spatial_cube.meta if hasattr(spatial_cube, 'meta') else None
        }
    )

def continuum_tables(windows: Dict[str, u.Quantity], logT_grid: np.ndarray,
                     logN_grid: np.ndarray, abundance: str, n_workers: int = 0,
                     hdf5_dbase_root=None) -> Dict[str, dict]:
    """
    Continuum per unit emission measure over each of a set of windows.

    Parameters
    ----------
    windows : dict
        Wavelengths of each continuum entry by name, as from `continuum_windows`.
    logT_grid, logN_grid, abundance, n_workers, hdf5_dbase_root
        As for `compute_continuum_fiasco`, which is called once for all the
        windows.

    Returns
    -------
    dict
        For each entry, its `"wl_grid"`, and `"free"` and `"two_photon"` on it.
    """
    wavelength = np.concatenate([grid.to_value(u.cm) for grid in windows.values()]) * u.cm
    free, two_photon = compute_continuum_fiasco(
        wavelength, logT_grid, logN_grid, abundance=abundance, n_workers=n_workers,
        hdf5_dbase_root=hdf5_dbase_root)
    tables, start = {}, 0
    for name, grid in windows.items():
        stop = start + grid.size
        tables[name] = {"wl_grid": grid, "free": free[:, start:stop],
                        "two_photon": two_photon[:, :, start:stop]}
        start = stop
    return tables


# Options that describe an atmosphere, which an emission measure does not have.
ATMOSPHERE_ONLY_OPTIONS = ("integration_axis", "crop_x", "crop_y", "crop_z", "downsample",
                           "mass_per_electron")


def check_source_options(args) -> None:
    """
    Reject options that do not suit the input, an atmosphere or an emission measure.

    An emission measure file takes none of the atmosphere options, and needs
    an electron density or a pressure if it holds no density. An atmosphere
    takes neither. Raises ValueError.
    """
    if not getattr(args, "emission_measure", None):
        for name in ("electron_density", "pressure"):
            if getattr(args, name, None) is not None:
                raise ValueError(f"--{name.replace('_', '-')} is for an emission measure file, "
                                 f"given with --emission-measure. An atmosphere has its own "
                                 f"densities.")
        return
    # The parser notes each of these options that appeared on the command
    # line, so one typed at its default value is caught too.
    noted = getattr(args, "given_options", ())
    given = [name for name in ATMOSPHERE_ONLY_OPTIONS + MURAM_LAYOUT_OPTIONS + DYNAMIC_OPTIONS
             if name in noted]
    given += [name for name in ("position", "rotation", "projection")
              if getattr(args, name, None) is not None]
    given += [name for name in ("curvature", "no_wrap") if getattr(args, name, False)]
    if args.slit_rest_time is not None:
        given.insert(0, "slit_rest_time")
    if args.atmosphere:
        given.insert(0, "atmosphere")
    if given:
        flags = ", ".join("--" + name.replace("_", "-") for name in given)
        raise ValueError(f"--emission-measure is the emission measure of an image already, "
                         f"so {flags} would not be used. Those options describe an "
                         f"atmosphere.")
    if Path(args.output_name).suffix.lower() in (".pkl", ".pickle", ".dill"):
        raise ValueError(f"An emission measure is synthesised into a synthesis file, which "
                         f"is HDF5: give --output-name a name ending in .h5, not "
                         f"{args.output_name}.")
    if args.electron_density is not None and args.pressure is not None:
        raise ValueError("Give --electron-density or --pressure, not both.")
    for name, unit, example in (("electron_density", u.cm**-3, "1e9 cm-3"),
                                ("pressure", u.K / u.cm**3, "1e15 K cm-3")):
        value = getattr(args, name)
        if value is None:
            continue
        try:
            value = u.Quantity(value).to_value(unit)
        except (TypeError, ValueError, u.UnitConversionError):
            value = None
        if value is None or np.ndim(value) != 0 or not (np.isfinite(value) and value > 0):
            raise ValueError(f"--{name.replace('_', '-')} takes one value above zero, such as "
                             f"'{example}', got {getattr(args, name)!r}.")
    # A file of another kind is reported when it is read.
    if (args.electron_density is None and args.pressure is None
            and is_emission_measure_file(args.emission_measure)):
        with h5py.File(args.emission_measure, "r") as f:
            if "electron_density" not in f:
                raise ValueError(
                    "An emission measure has no densities, so the contribution functions "
                    "need an electron density to be taken at: give --electron-density, such "
                    "as '1e9 cm-3', or --pressure, such as '1e15 K cm-3', or put an "
                    "electron_density in the file.")


def emission_measure_density(measure: EmissionMeasure, electron_density=None,
                             pressure=None) -> np.ndarray:
    """
    Electron density at each temperature of each pixel of an emission measure, in cm^-3.

    Parameters
    ----------
    measure : EmissionMeasure
        DEM or VDEM.
    electron_density : u.Quantity, optional
        One density for all the plasma, in place of any in `measure`.
    pressure : u.Quantity, optional
        Electron pressure n_e T, such as in K cm^-3, giving each temperature
        the density at that pressure, in place of any in `measure`.

    Returns
    -------
    np.ndarray
        Shape (ny, nx, n_temperatures).

    Raises
    ------
    ValueError
        If neither a density nor a pressure is given and `measure` has no
        density.
    """
    shape = (*measure.shape, measure.temperature.size)
    if pressure is not None:
        density = (u.Quantity(pressure) / measure.temperature).to_value(u.cm**-3)
    elif electron_density is not None:
        density = u.Quantity(electron_density).to_value(u.cm**-3)
    elif measure.electron_density is not None:
        density = measure.electron_density.to_value(u.cm**-3)
        if density.ndim == 2:
            density = density[:, :, np.newaxis]
    else:
        raise ValueError("The emission measure has no electron density: give one, or a "
                         "pressure.")
    return np.broadcast_to(np.asarray(density, dtype=float), shape)


def _line_as_band(info: dict, logT_grid: np.ndarray, logN_grid: np.ndarray,
                  abundance: str) -> BandLines:
    """One line from `compute_goft_fiasco`, as the lines of a band, for `band_spectra`."""
    rest = info["wl0"].to_value(u.Angstrom)
    ion = IonLines(name=f"{info['atom']} {info['ion']}", atom=int(info["atom"]),
                   stage=int(info["ion"]), mass=element_data(int(info["atom"]))[1],
                   wavelength=np.array([rest]), observed=np.array([True]),
                   temperatures=np.arange(logT_grid.size),
                   g=np.asarray(info["g_tn"], dtype=float)[np.newaxis])
    return BandLines(band=(rest, rest), logT_grid=logT_grid, logN_grid=logN_grid,
                     abundance=abundance, hdf5_dbase_root=str(info["hdf5_dbase_root"]),
                     ions=[ion])


def _synthesise_emission_measure(args, precision: type, band, band_grid, band_step) -> None:
    """
    Synthesise spectra from an emission measure file: a DEM or a VDEM.

    A DEM's emission is at rest, and a VDEM's is Doppler shifted by its own
    velocity bins. Each line is a Gaussian of its thermal width, integrated
    over the wavelength bins, as for a band (:func:`eclipse.band.band_spectra`).
    A line's wavelengths are set by `--vel-res` and `--vel-lim`.
    """
    check_source_options(args)
    intensity_unit = u.erg / u.s / u.cm**2 / u.sr / u.cm
    print("Synthesis from an emission measure file")
    print(f"  Emission measure: {args.emission_measure}")
    # Only the columns asked for are read, as a VDEM of a whole image is large.
    layout = read_emission_measure_layout(args.emission_measure)
    columns = None
    if getattr(args, "image_x", None):
        columns = image_columns_between(_to_length(layout["x_edges"]), args.image_x)
        print(f"  Image columns {columns.start} to {columns.stop - 1} of "
              f"{layout['x_edges'].size - 1} kept")
    measure = read_emission_measure(args.emission_measure, columns=columns)
    logT_grid = np.log10(measure.temperature.to_value(u.K))
    em_tv = measure.emission_measure.to_value(u.cm**-5).astype(float)
    if measure.velocity is None:
        # A DEM: all of the emission is in one bin, at rest.
        em_tv, velocity = em_tv[..., np.newaxis], np.zeros(1) * u.cm / u.s
    else:
        velocity = measure.velocity.to(u.cm / u.s)
    density = emission_measure_density(measure, args.electron_density, args.pressure)
    em_t = em_tv.sum(axis=3)
    print(f"  {em_t.shape[0]} by {em_t.shape[1]} pixels, {logT_grid.size} temperatures from "
          f"log T {logT_grid[0]:.2f} to {logT_grid[-1]:.2f}, "
          f"{'a DEM' if measure.velocity is None else f'a VDEM with {velocity.size} velocities'}")

    # Default density grid points, to just past the plasma's densities.
    logN_min, logN_max, nN = _LOGN_MIN, _LOGN_MAX, _N_N
    occupied = em_t > 0
    if occupied.any():
        first, last = _points_just_past(np.log10(density[occupied].min()),
                                        np.log10(density[occupied].max()))
        logN_min, logN_max, nN = _density_point(first), _density_point(last), last - first + 1
    logN_grid = np.linspace(logN_min, logN_max, nN)

    dbase_root = getattr(args, "hdf5_dbase_root", None)
    lines, goft = {}, {}
    if band is None:
        print(f"Computing contribution functions via fiasco at 10^{logN_min:.1f} to "
              f"10^{logN_max:.1f} cm^-3")
        goft, logT_goft, _ = compute_goft_fiasco(
            args.lines, abundance=args.abundance, precision=np.float64,
            logT_min=float(logT_grid[0]), logT_max=float(logT_grid[-1]), nT=logT_grid.size,
            logN_min=logN_min, logN_max=logN_max, nN=nN, n_workers=args.n_workers,
            hdf5_dbase_root=dbase_root,
            temperature_chunk=getattr(args, "goft_temperature_chunk", None))
        if not same_temperatures(logT_goft, logT_grid):
            raise ValueError("The contribution functions came back on other temperatures "
                             "than the emission measure's.")
        dbase_root = next(iter(goft.values()))["hdf5_dbase_root"] if goft else None
        window = velocity_grid(u.Quantity(args.vel_res), u.Quantity(args.vel_lim),
                               ("--vel-res", "--vel-lim"))
        fastest = np.abs(velocity.value[em_tv.any(axis=(0, 1, 2))]).max(initial=0.0)
        if fastest > window.value[-1]:
            warnings.warn(
                f"The emission measure has plasma at up to {fastest / 1e5:.4g} km/s, beyond "
                f"the {window.value[-1] / 1e5:.4g} km/s each line's wavelengths reach, so "
                f"part of its light is left out. Set --vel-lim wide enough for it.",
                UserWarning, stacklevel=2)
        for name, info in goft.items():
            grid = (window * info["wl0"] / const.c + info["wl0"]).cgs
            # One line needs no workers.
            spectra = band_spectra(_line_as_band(info, logT_grid, logN_grid, args.abundance),
                                   em_tv, density, velocity, grid, n_workers=1)
            lines[name] = {"si": spectra.to_value(intensity_unit), "wl_grid": grid,
                           "wl0": info["wl0"], "atom": info["atom"], "ion": info["ion"]}
    else:
        # Compute a band's lines on the whole image's densities, so one file
        # serves every strip. The file is named for the emission measure's
        # temperatures.
        whole = density
        if (args.electron_density is None and args.pressure is None
                and layout["electron_density"] is not None):
            whole = layout["electron_density"].to_value(u.cm**-3)
        first, last = _points_just_past(np.log10(whole.min()), np.log10(whole.max()))
        name = f"band_lines_logT_{logT_grid[0]:.2f}-{logT_grid[-1]:.2f}_{logT_grid.size}.h5"
        band_lines = band_lines_for(
            args.band_lines or Path(args.output_dir) / name, band, logT_grid, logN_grid,
            abundance=args.abundance, n_workers=args.n_workers, hdf5_dbase_root=dbase_root,
            wider_grid=band_density_grid(_density_point(first), _density_point(last)))
        dbase_root = band_lines.hdf5_dbase_root
        print(f"Calculating the spectrum of the band's {band_lines.n_lines} lines at "
              f"{band_grid.size} wavelengths")
        grid = (band_grid * u.Angstrom).to(u.cm)
        spectra = band_spectra(band_lines, em_tv, density, velocity, grid,
                               n_workers=args.n_workers)
        lines[f"band_{band_grid[0]:.4f}-{band_grid[-1]:.4f}"] = {
            "si": spectra.to_value(intensity_unit), "wl_grid": grid,
            "wl0": grid[grid.size // 2], "atom": None, "ion": None}
    print(f"  CHIANTI database: {dbase_root}")

    if getattr(args, "continuum", False):
        if band is not None:
            # On the band's own wavelengths.
            windows = {f"{CONTINUUM_PREFIX}_{band_grid[0]:.4f}-{band_grid[-1]:.4f}":
                       (band_grid * u.Angstrom).to(u.cm)}
        else:
            windows = continuum_windows([info["wl_grid"] for info in lines.values()])
        print(f"Computing the continuum over {len(windows)} window"
              f"{'' if len(windows) == 1 else 's'} via fiasco")
        tables = continuum_tables(windows, logT_grid, logN_grid, args.abundance,
                                  args.n_workers, dbase_root)
        for name, table in tables.items():
            grid = table["wl_grid"]
            lines[name] = {"si": continuum_spectra(em_t, density, logN_grid, table["free"],
                                                   table["two_photon"]),
                           "wl_grid": grid, "wl0": grid[grid.size // 2],
                           "atom": None, "ion": None}

    held = {name: SpectralLine(
        intensity=info["si"].astype(precision) * intensity_unit, wavelength=info["wl_grid"],
        rest_wavelength=info["wl0"],
        **{key: int(info[key]) for key in ("atom", "ion") if info.get(key) is not None})
        for name, info in lines.items()}
    synthesis = Synthesis(lines=held, x_edges=measure.x_edges, y_edges=measure.y_edges,
                          source=measure.source, time=measure.time)
    products = {
        # Per unit log T, as the synthesis of an atmosphere keeps it.
        "dem_map": em_t / _temperature_bins(logT_grid)[0],
        **({"em_tv": em_tv} if getattr(args, "keep_vdem", False) else {}),
        "electron_density": density,
        "logT_grid": logT_grid,
        "vel_grid": velocity,
        "logN_grid": logN_grid,
        "goft": goft,
        "dynamic_mode": {"enabled": False},
        "config": {
            "precision": precision.__name__,
            "keep_vdem": bool(getattr(args, "keep_vdem", False)),
            "emission_measure": args.emission_measure,
            "electron_density": args.electron_density,
            "pressure": args.pressure,
            "vel_res": u.Quantity(args.vel_res),
            "vel_lim": u.Quantity(args.vel_lim),
            "intensity_unit": str(intensity_unit),
            "lines": args.lines or [],
            "band": None if band is None else list(band),
            "band_step": None if band is None else band_step,
            "image_x": getattr(args, "image_x", None),
            "abundance": args.abundance,
            "continuum": bool(getattr(args, "continuum", False)),
            "hdf5_dbase_root": dbase_root,
            "velocity_convention": VELOCITY_CONVENTION,
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_name
    write_synthesis(synthesis, output_file, products=products)
    print(f"Saved results to {output_file} ({os.path.getsize(output_file) / 1e6:.2f} MB)")
    print("Synthesis complete!")


def image_x_edges(spatial_cube: NDCube, integration_axis: str) -> u.Quantity:
    """
    Pixel boundaries across the slit, in Mm, of the image of a view of `spatial_cube`.

    These are the `x_edges` a synthesis of the whole image is written with.
    """
    n_rows, n_columns = _image_pixels(spatial_cube.data.shape, integration_axis)[1]
    grid = np.array([1.0, 2.0]) * u.cm
    blank = {"si": np.broadcast_to(0.0, (n_rows, n_columns, 2)), "wl_grid": grid,
             "wl0": grid[0], "atom": None, "ion": None}
    cube = create_line_cube("image", blank, spatial_cube, u.dimensionless_unscaled,
                            integration_axis)
    return _pixel_edges(cube, 1)


def image_columns_between(x_edges: u.Quantity, limits) -> slice:
    """
    Image columns with any part between two positions across the slit.

    Parameters
    ----------
    x_edges : u.Quantity
        Pixel boundaries across the slit, increasing.
    limits : sequence of two
        Positions, as lengths on the Sun or angles seen from 1 AU, in the
        coordinates of `x_edges`.

    Raises
    ------
    ValueError
        If the positions are not two lengths or angles in increasing order,
        or no column lies between them.
    """
    try:
        low, high = (_to_length(u.Quantity(value)).to_value(u.Mm) for value in limits)
    except (TypeError, ValueError, u.UnitConversionError):
        raise ValueError(f"--image-x takes two positions across the slit, such as '-1 Mm' "
                         f"'1 Mm', got {list(limits)}.") from None
    if not low < high:
        raise ValueError(f"--image-x runs from one position to a larger one, got "
                         f"{list(limits)}.")
    edges = x_edges.to_value(u.Mm)
    kept = np.flatnonzero((edges[1:] > low) & (edges[:-1] < high))
    if kept.size == 0:
        raise ValueError(f"No part of the image, which runs from {edges[0]:g} to "
                         f"{edges[-1]:g} Mm across the slit, is between --image-x "
                         f"{list(limits)}.")
    return slice(int(kept[0]), int(kept[-1]) + 1)


def image_strip(cube: np.ndarray, integration_axis: str, columns: slice) -> np.ndarray:
    """
    Cells of a (z, y, x) cube seen in the image columns `columns`.

    For a view along `integration_axis`. The strip's image is those columns
    of the whole image, in order.
    """
    axis, side = view_axis_and_side(integration_axis)
    column_axis = {"x": 1, "y": 2, "z": 2}[axis]
    n_columns = cube.shape[column_axis]
    first, last = columns.start, columns.stop
    # Seen from the other side the image is mirrored left to right
    # (_image_pixels), so the strip is counted from the other end.
    if side != OBSERVER_SIDE[axis]:
        first, last = n_columns - last, n_columns - first
    kept = [slice(None)] * 3
    kept[column_axis] = slice(first, last)
    return cube[tuple(kept)]



##############################################################################
# ---------------------------------------------------------------------------
#                 M A I N   W O R K F L O W
# ---------------------------------------------------------------------------
##############################################################################

# Location and layout of MURaM's own files, for dynamic mode and the
# deprecated static route. An atmosphere file holds all of this, so giving
# both is an error.
MURAM_LAYOUT_OPTIONS = ("data_dir", "temp_file", "rho_file", "vx_file", "vy_file",
                        "vz_file", "cube_shape", "voxel_dx", "voxel_dy", "voxel_dz")

# Options read only in dynamic mode, which a static synthesis from an
# atmosphere file would ignore.
DYNAMIC_OPTIONS = ("slit_width", "temp_dir", "temp_filename", "rho_dir", "rho_filename",
                   "vx_dir", "vx_filename", "vy_dir", "vy_filename", "vz_dir",
                   "vz_filename", "time_dir", "time_filename")

# Documentation of the atmosphere file and how to write one.
ATMOSPHERE_DOCS = "https://solarc-eclipse.readthedocs.io/en/stable/files/atmosphere/"
# Documentation of observing a time series of atmosphere files, which
# replaces the deprecated dynamic mode.
TIME_SERIES_DOCS = "https://solarc-eclipse.readthedocs.io/en/stable/observe/rasters/"


class _NotedOption(argparse.Action):
    """Store the value and record that the option was given, even at its default."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        given = getattr(namespace, "given_options", None)
        if given is None:
            given = set()
            namespace.given_options = given
        given.add(self.dest)


def build_parser() -> argparse.ArgumentParser:
    """Command line parser for `synthesise-spectra`."""
    parser = argparse.ArgumentParser(
        description="Synthesise solar spectra from 3D MHD simulation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    files = parser.add_argument_group("Input and output")
    files.add_argument("--config", type=str, default=None,
                       help="A run file, whose 'atmosphere', 'synthesis_file' and "
                            "'synthesis:' section give the settings. Options given here "
                            "override it.")
    files.add_argument("--atmosphere", type=str, default=None,
                       help="The ECLIPSE atmosphere file (HDF5) to synthesise "
                            f"from; see {ATMOSPHERE_DOCS} for how to write one. "
                            "Required, except in dynamic mode.")
    files.add_argument("--emission-measure", type=str, default=None,
                       help="In place of --atmosphere, an emission measure file (HDF5) to "
                            "synthesise from: a DEM, or with velocities a VDEM.")
    # MURaM file options of the deprecated static route, hidden from the
    # help. They still work, with a warning.
    for flag, default in (("--temp-file", "temp/eosT.0270000"),
                          ("--rho-file", "rho/result_prim_0.0270000"),
                          ("--vx-file", "vx/result_prim_1.0270000"),
                          ("--vy-file", "vy/result_prim_3.0270000"),
                          ("--vz-file", "vz/result_prim_2.0270000")):
        files.add_argument(flag, type=str, default=default, action=_NotedOption,
                           help=argparse.SUPPRESS)
    files.add_argument("--output-dir", type=str, default="./run/input",
                       help="Output directory for results")
    files.add_argument("--output-name", type=str, default="synthesised_spectra.h5",
                       help="Output filename, an HDF5 synthesis file; a name ending in .pkl "
                            "writes the deprecated pickle instead")
    files.add_argument("--keep-vdem", action="store_true",
                       help="Also keep each pixel's emission measure by temperature and "
                            "velocity (its VDEM) in the synthesis file. It is often larger "
                            "than the spectra.")

    lines = parser.add_argument_group("Lines and atomic data")
    lines.add_argument("--lines", nargs="+", default=None,
                       help="Line specifications (e.g. Fe12_195.1190 Fe09_171.073)")
    lines.add_argument("--band", nargs=2, type=str, default=None, metavar=("FROM", "TO"),
                       help="In place of --lines, synthesise every CHIANTI line between "
                            "two wavelengths as one spectrum (e.g. '169 Angstrom' "
                            "'213.5 Angstrom'), as a full-CCD frame needs")
    lines.add_argument("--band-step", type=str, default=f"{BAND_STEP} Angstrom",
                       help="Wavelength step of the band's spectrum")
    lines.add_argument("--band-lines", type=str, default=None,
                       help="File the contribution functions of the band's lines are kept "
                            "in, and read back from if it is there already. By default "
                            "band_lines.h5 in the output directory.")
    lines.add_argument("--abundance", type=str, default="sun_coronal_2021_chianti",
                       help="CHIANTI abundance dataset name for fiasco")
    lines.add_argument("--continuum", action="store_true",
                       help="Also synthesise the free-free, free-bound and two-photon "
                            "continuum under the lines' windows, as entries of its own "
                            "beside the lines")
    lines.add_argument("--electron-density", type=str, default=None,
                       help="With --emission-measure, the electron density to take the "
                            "contribution functions at, such as '1e9 cm-3', in place of "
                            "any in the file")
    lines.add_argument("--pressure", type=str, default=None,
                       help="With --emission-measure, the electron pressure, such as "
                            "'1e15 K cm-3', which gives each temperature its own electron "
                            "density, in place of --electron-density")
    lines.add_argument("--hdf5-dbase-root", type=str, default=None,
                       help="CHIANTI HDF5 database to use, overriding the one "
                            "in ~/.fiasco/fiascorc. Use this to run against a "
                            "second CHIANTI version without changing the "
                            "default for other work.")
    lines.add_argument("--mass-per-electron", "--mean-mol-wt",
                       dest="mass_per_electron", type=float, default=None,
                       action=_NotedOption,
                       help="Mass of the plasma per free electron, in atomic mass "
                            "units, which turns a mass density into an electron "
                            "density. By default it is worked out from --abundance "
                            "for a fully ionised plasma, about 1.16 for coronal "
                            "abundances. --mean-mol-wt is the old name; ECLIPSE "
                            "0.11.0 and earlier used 1.29, the value for a neutral gas. "
                            "Not used when the atmosphere gives an electron density.")

    view = parser.add_argument_group("The view and the part of the box")
    view.add_argument("--integration-axis", choices=["x", "y", "z", *SIGNED_VIEWS],
                      default="z", action=_NotedOption,
                      help="The axis to look along, x, y or z, from +x, -y or above; or the "
                           "axis with a sign for the side to look from, such as -x")
    view.add_argument("--position", nargs=2, type=str, default=None,
                      help="See the box from this position on the solar disk: the "
                           "helioprojective x and y of its centre, west and north of disk "
                           "centre (e.g. '600 arcsec' '-200 arcsec'). The box's x is taken "
                           "to point west, y north and z up, unless turned by --rotation. "
                           "The atmosphere file needs the velocity along each axis the line "
                           "of sight leans along")
    view.add_argument("--rotation", type=str, default=None,
                      help="With --position, turn the box about the vertical by this angle, "
                           "from its x pointing west towards north (e.g. '30 deg'). "
                           "Default 0 deg")
    view.add_argument("--projection", choices=PROJECTIONS, default=None,
                      help="With --position, how cells are shared between sky pixels: "
                           "'two-step' (default) along the line of sight onto the box's "
                           "base, then onto the sky by area; or 'exact', each cell straight "
                           "onto the sky, which is slower")
    view.add_argument("--curvature", action="store_true",
                      help="With --position, lay the box on the curved solar surface, each "
                           "column along its own vertical, and share its cells between sky "
                           "pixels exactly. The atmosphere file needs the velocity along x, "
                           "y and z")
    view.add_argument("--no-wrap", action="store_true",
                      help="With --position, end lines of sight at the box's sides, rather "
                           "than taking the box to repeat sideways")
    view.add_argument("--crop-x", nargs=2, type=str, default=None, action=_NotedOption,
                      help="Crop in x direction: x_min x_max (e.g. '-50 Mm' '50 Mm')")
    view.add_argument("--crop-y", nargs=2, type=str, default=None, action=_NotedOption,
                      help="Crop in y direction: y_min y_max (e.g. '-50 Mm' '50 Mm')")
    view.add_argument("--crop-z", nargs=2, type=str, default=None, action=_NotedOption,
                      help="Crop in z direction: z_min z_max (e.g. '0 Mm' '20 Mm')")
    view.add_argument("--downsample", type=int, default=1, action=_NotedOption,
                      help="Downsampling factor (1 = no downsampling)")
    view.add_argument("--image-x", nargs=2, type=str, default=None, metavar=("FROM", "TO"),
                      help="Synthesise only the part of the image between two positions "
                           "across the slit, in the image's own x coordinates (e.g. "
                           "'-1 Mm' '1 Mm'). The emission is still summed along the "
                           "whole line of sight.")

    grids = parser.add_argument_group("Grids and precision")
    grids.add_argument("--vel-res", type=str, default="5.0 km/s",
                       help="Velocity resolution (e.g. '5.0 km/s')")
    grids.add_argument("--vel-lim", type=str, default="300.0 km/s",
                       help="Velocity limit +/- (e.g. '300.0 km/s')")
    grids.add_argument("--precision", choices=["float32", "float64"], default="float64",
                       help="Numerical precision")

    computing = parser.add_argument_group("Computing")
    computing.add_argument("--n-workers", type=int, default=0,
                           help="Number of workers for fiasco G(T,N): processes for "
                                "the ions, and threads within each "
                                "(0 = all CPUs, default: 0)")
    computing.add_argument("--goft-temperature-chunk", type=int, default=None,
                           help="Compute G(T,N) this many temperatures at a time "
                                "rather than the whole grid at once. This lowers "
                                "fiasco's peak memory, which for an ion with many "
                                "levels can be several GB, at the cost of reading "
                                "the atomic data again for each chunk. The result "
                                "is the same.")

    # Deprecated dynamic mode (time-varying synthesis), which reads MURaM's
    # own files. Its options are hidden from the help and still work, with a
    # warning.
    dynamic_group = parser.add_argument_group()
    dynamic_group.add_argument("--slit-rest-time", type=str, default=None,
                       help="Slit rest time per position (e.g. '40 s'). "
                            "Enables dynamic mode when specified.")
    dynamic_group.add_argument("--slit-width", type=str, default=None,
                       action=_NotedOption,
                       help="Slit width (e.g. '0.2 arcsec', required for dynamic mode)")
    dynamic_group.add_argument("--data-dir", type=str, default="data/atmosphere",
                       action=_NotedOption,
                       help="Directory containing the MURaM files")
    dynamic_group.add_argument("--cube-shape", nargs=3, type=int, default=[512, 768, 256],
                       action=_NotedOption,
                       help="Cube dimensions in the file's storage order (nx nz ny)")
    dynamic_group.add_argument("--voxel-dx", type=str, default="0.192 Mm",
                       action=_NotedOption,
                       help="Voxel size in x (e.g. '0.192 Mm')")
    dynamic_group.add_argument("--voxel-dy", type=str, default="0.192 Mm",
                       action=_NotedOption,
                       help="Voxel size in y (e.g. '0.192 Mm')")
    dynamic_group.add_argument("--voxel-dz", type=str, default="0.064 Mm",
                       action=_NotedOption,
                       help="Voxel size in z (e.g. '0.064 Mm')")

    # Directory arguments for dynamic mode
    dynamic_group.add_argument("--temp-dir", type=str, default=None,
                       action=_NotedOption,
                       help="Directory containing temperature files (for dynamic mode)")
    dynamic_group.add_argument("--temp-filename", type=str, default="eosT",
                       action=_NotedOption,
                       help="Temperature filename prefix before timestep suffix")
    dynamic_group.add_argument("--rho-dir", type=str, default=None,
                       action=_NotedOption,
                       help="Directory containing density files (for dynamic mode)")
    dynamic_group.add_argument("--rho-filename", type=str, default="result_prim_0",
                       action=_NotedOption,
                       help="Density filename prefix before timestep suffix")
    dynamic_group.add_argument("--vx-dir", type=str, default=None,
                       action=_NotedOption,
                       help="Directory containing vx files (for dynamic mode)")
    dynamic_group.add_argument("--vx-filename", type=str, default="result_prim_1",
                       action=_NotedOption,
                       help="Vx filename prefix before timestep suffix")
    dynamic_group.add_argument("--vy-dir", type=str, default=None,
                       action=_NotedOption,
                       help="Directory containing vy files (for dynamic mode)")
    dynamic_group.add_argument("--vy-filename", type=str, default="result_prim_3",
                       action=_NotedOption,
                       help="Vy filename prefix before timestep suffix")
    dynamic_group.add_argument("--vz-dir", type=str, default=None,
                       action=_NotedOption,
                       help="Directory containing vz files (for dynamic mode)")
    dynamic_group.add_argument("--vz-filename", type=str, default="result_prim_2",
                       action=_NotedOption,
                       help="Vz filename prefix before timestep suffix")
    dynamic_group.add_argument("--time-dir", type=str, default="header",
                       action=_NotedOption,
                       help="Directory containing header files (for dynamic mode)")
    dynamic_group.add_argument("--time-filename", type=str, default="Header",
                       action=_NotedOption,
                       help="Header filename prefix before timestep suffix")

    for action in dynamic_group._group_actions:
        action.help = argparse.SUPPRESS

    return parser


def _config_option(words: list) -> Optional[str]:
    """The run file given with --config among command-line words, or None."""
    for index, word in enumerate(words):
        if word == "--config" and index + 1 < len(words):
            return words[index + 1]
        if word.startswith("--config="):
            return word.split("=", 1)[1]
    return None


def parse_arguments(argv=None):
    """Parse command line arguments for spectrum synthesis."""
    words = list(sys.argv[1:] if argv is None else argv)
    # A run file's settings go first, so options given here override them.
    config = _config_option(words)
    if config is not None:
        from .run_file import load, runs, synthesis_arguments

        found = runs(*load(config)[:2])
        if len(found) != 1:
            raise ValueError(f"{config} has several runs; synthesise them with "
                             f"'eclipse synthesise {config}', which takes --only to pick one.")
        words = synthesis_arguments(found[0][1], found[0][0]) + words
    # argparse reads the -x in "--integration-axis -x" as an option, so join
    # it as "--integration-axis=-x".
    joined, i = [], 0
    while i < len(words):
        if (words[i] == "--integration-axis" and i + 1 < len(words)
                and words[i + 1] in SIGNED_VIEWS):
            joined.append(f"--integration-axis={words[i + 1]}")
            i += 2
        else:
            joined.append(words[i])
            i += 1
    return build_parser().parse_args(joined)


def check_atmosphere_options(args) -> None:
    """
    Reject options that --atmosphere makes meaningless; warn if it is absent.

    Without an atmosphere file, static mode reads MURaM's own files and
    dynamic mode a time series of them; both are deprecated. The MURaM
    layout and dynamic mode options would be ignored with --atmosphere, so
    they raise an error.
    """
    if not args.atmosphere:
        if args.slit_rest_time is not None:
            warnings.warn(
                f"Dynamic mode (--slit-rest-time) is deprecated and will be "
                f"removed in a future release: write the snapshots as atmosphere "
                f"files and observe them as a time series in the instrument run, "
                f"as described at {TIME_SERIES_DOCS}, with 'direction: decreasing' "
                f"under 'raster:' to scan the way dynamic mode does.",
                FutureWarning, stacklevel=2)
        else:
            warnings.warn(
                f"No --atmosphere was given, so the synthesis is reading "
                f"MURaM's own files from {args.data_dir}. This is deprecated "
                f"and will be removed in a future release: write the snapshot "
                f"as an atmosphere file, as described at {ATMOSPHERE_DOCS}, "
                f"and give it with --atmosphere.",
                FutureWarning, stacklevel=2)
        return
    # The parser notes every layout option that appeared on the command
    # line, so one typed at its default value is caught too.
    noted = getattr(args, "given_options", ())
    given = [name for name in MURAM_LAYOUT_OPTIONS if name in noted]
    if given:
        flags = ", ".join("--" + name.replace("_", "-") for name in given)
        raise ValueError(
            f"--atmosphere carries the cube shape, cell sizes and data itself, "
            f"so {flags} would not be used. Those options describe MURaM's "
            f"own files.")
    if args.slit_rest_time is not None:
        raise ValueError(
            f"Dynamic mode reads its time series from MURaM files and cannot "
            f"take an atmosphere file. A time series of atmosphere files is "
            f"observed by the instrument run instead: see {TIME_SERIES_DOCS}.")
    given = [name for name in DYNAMIC_OPTIONS if name in noted]
    if given:
        flags = ", ".join("--" + name.replace("_", "-") for name in given)
        raise ValueError(
            f"--atmosphere is a static synthesis, so {flags} would not be "
            f"used. Those options only apply to dynamic mode (--slit-rest-time).")


def load_atmosphere_file(
    path: str | Path,
    integration_axis: str,
    downsample: int | bool = False,
    crop_x=None, crop_y=None, crop_z=None,
    velocities=None,
) -> Atmosphere:
    """
    Read an atmosphere file, with the velocity along `integration_axis`.

    Downsampling and cropping are applied in the atmosphere's own
    coordinates. The two image axes must be evenly spaced, for the image
    WCS. The line of sight may be stretched: the emission measure uses each
    cell's true size along it. `velocities` names the velocity components to
    read instead of the one along `integration_axis`, such as `("x", "z")`
    for a slanted line of sight.
    """
    atmosphere = read_atmosphere(path, velocities=velocities or (integration_axis,))
    return _prepare_atmosphere(atmosphere, str(path), integration_axis,
                               downsample, crop_x, crop_y, crop_z)


def _prepare_atmosphere(atmosphere: Atmosphere, name: str, integration_axis: str,
                        downsample, crop_x, crop_y, crop_z) -> Atmosphere:
    """Downsample and crop `atmosphere`, and check its image axes are evenly spaced."""
    if downsample:
        atmosphere = atmosphere.downsampled(downsample)
    if crop_x or crop_y or crop_z:
        atmosphere = atmosphere.cropped(x=crop_x, y=crop_y, z=crop_z)
    for axis in AXES:
        if axis != integration_axis and not atmosphere.is_uniform(axis):
            raise ValueError(
                f"The {axis} axis of {name} is not evenly spaced, and with the "
                f"line of sight along {integration_axis} it would become an "
                f"image axis, whose coordinates must be even. Only the axis "
                f"along the line of sight may be stretched; resample the "
                f"others onto an even grid first.")
    return atmosphere


def resolve_mass_per_electron(args) -> Tuple[float, str]:
    """Mass per free electron in atomic mass units, and where it came from."""
    given = getattr(args, "mass_per_electron", None)
    # Accept the old name, mean_mol_wt, from scripts that set it on `args`.
    if given is None and getattr(args, "mean_mol_wt", None) is not None:
        warnings.warn(
            "mean_mol_wt is the old name of mass_per_electron, the mass of the plasma per "
            "free electron in atomic mass units; it is used, but set mass_per_electron "
            "instead.", FutureWarning, stacklevel=2)
        given = args.mean_mol_wt
    if given is not None:
        try:
            value = require_mass_per_electron(given)
        except ValueError as error:
            raise ValueError(f"--mass-per-electron: {error}") from None
        return value, "given on the command line"
    value = mass_per_electron(args.abundance, getattr(args, "hdf5_dbase_root", None))
    return value, f"fully ionised plasma with {args.abundance} abundances"


def along_line_of_sight(values, integration_axis: str) -> np.ndarray:
    """
    Reshape `values` to broadcast over a (z, y, x) cube.

    `values` holds one value per cell along the line of sight. A single
    value is returned unchanged, so one cell size applies everywhere.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return values
    if values.ndim != 1:
        raise ValueError(f"Expected one value per cell along the line of sight, "
                         f"got an array of shape {values.shape}.")
    shape = [1, 1, 1]
    shape[NUMPY_AXIS[integration_axis]] = values.size
    return values.reshape(shape)


def _disk_view(args, view: str) -> tuple:
    """
    The view from a position on the disk, from --position and its options.

    Returns the position, rotation, projection, curvature, disk geometry,
    frame and heliocentric angle. Without --position all are None and
    curvature is False.
    """
    position = rotation = projection = geometry = frame = heliocentric_angle = None
    if getattr(args, "position", None):
        position = tuple(u.Quantity(value) for value in args.position)
        if not all(value.unit.is_equivalent(u.arcsec) for value in position):
            raise ValueError(f"--position is the box's helioprojective x and y, as angles such "
                             f"as '600 arcsec'; got {args.position}.")
        if not args.atmosphere:
            raise ValueError("--position needs an atmosphere file, given with --atmosphere.")
        if "integration_axis" in getattr(args, "given_options", ()) or view != "z":
            raise ValueError("--position sets the line of sight itself, so --integration-axis "
                             "cannot be given with it.")
        rotation = u.Quantity(getattr(args, "rotation", None) or "0 deg")
        if not rotation.unit.is_equivalent(u.deg):
            raise ValueError(f"--rotation is an angle, such as '30 deg'; got {args.rotation}.")
        curvature = bool(getattr(args, "curvature", False))
        projection = getattr(args, "projection", None)
        if curvature and projection not in (None, "exact"):
            raise ValueError("--curvature shares each cell between sky pixels exactly, so "
                             "it takes no --projection other than 'exact'.")
        projection = "exact" if curvature else (projection or PROJECTIONS[0])
        geometry = disk_geometry(*position, rotation)
        frame = geometry.sky @ np.column_stack([geometry.box_x, geometry.box_y, geometry.up])
        heliocentric_angle = geometry.heliocentric_angle
    else:
        curvature = False
        for option in ("no_wrap", "rotation", "projection", "curvature"):
            if getattr(args, option, None):
                raise ValueError(f"--{option.replace('_', '-')} is for a box seen from a "
                                 f"position on the disk, given with --position.")
    return position, rotation, projection, curvature, geometry, frame, heliocentric_angle


def _writes_pickle(output_name: str) -> bool:
    """Whether `output_name` asks for the deprecated pickle, with a warning if so or if odd."""
    # A name ending in .pkl still gets the deprecated pickle.
    write_pickle = Path(output_name).suffix.lower() in (".pkl", ".pickle", ".dill")
    if write_pickle:
        warnings.warn(
            f"--output-name {output_name} names a pickle, as older versions of ECLIPSE "
            f"wrote the synthesis. Writing one is deprecated and will stop in a future "
            f"release: give the output a name ending in .h5 for a synthesis file.",
            FutureWarning, stacklevel=3)
    elif Path(output_name).suffix.lower() not in (".h5", ".hdf5", ".hdf"):
        warnings.warn(
            f"--output-name {output_name} gets a synthesis file, which is HDF5. Older "
            f"versions of ECLIPSE wrote a pickle under any name; a name ending in .pkl "
            f"still gets one, until that is removed.", UserWarning, stacklevel=3)
    return write_pickle


def main(args=None) -> None:
    """
    Synthesise solar spectra from a 3D MHD simulation.

    Two modes:
    - Static: one timestep, from an atmosphere file (--atmosphere), an
      emission measure file (--emission-measure), or MURaM's own files
      (deprecated).
    - Dynamic (deprecated): time-varying synthesis with raster scanning,
      from MURaM's own files. Observe a time series of atmosphere files in
      the instrument run instead (see eclipse.raster).

    Parameters
    ----------
    args : argparse.Namespace, optional
        Command line arguments. If None, parsed from sys.argv.
    """
    if args is None:
        args = parse_arguments()
    
    # ---------------- Configuration from arguments -----------------
    precision = np.float32 if args.precision == "float32" else np.float64
    if args.downsample < 1:
        raise ValueError(f"--downsample must be 1 or more, got {args.downsample}.")
    downsample = args.downsample if args.downsample > 1 else False
    # Check the lines or the band now, before reading the atmosphere, which
    # can take minutes and gigabytes.
    band = band_grid = None
    if getattr(args, "band", None):
        if args.lines:
            raise ValueError(
                "Give --lines or --band, not both: a run or a frame adds up everything a "
                "synthesis file holds, so a line in both would be counted twice.")
        try:
            band = tuple(u.Quantity(edge).to_value(u.Angstrom) for edge in args.band)
            band_step = u.Quantity(args.band_step).to_value(u.Angstrom)
        except (TypeError, ValueError, u.UnitConversionError):
            raise ValueError(
                f"--band takes two wavelengths, such as '169 Angstrom' '213.5 Angstrom', "
                f"and --band-step one, such as '0.002 Angstrom'; got {args.band} and "
                f"{args.band_step!r}.") from None
        band_grid = band_wavelengths(band, band_step)
    elif not args.lines:
        raise ValueError("Name the --lines to synthesise, or a --band.")
    for name in args.lines or []:
        _parse_line_name(name)
    vel_res = u.Quantity(args.vel_res)
    vel_lim = u.Quantity(args.vel_lim)
    # Check the velocity grid now, before the atmosphere is read or anything
    # computed.
    velocity_grid(vel_res, vel_lim, ("--vel-res", "--vel-lim"))
    if getattr(args, "emission_measure", None):
        _synthesise_emission_measure(args, precision, band, band_grid,
                                     None if band is None else band_step)
        return
    check_source_options(args)

    intensity_unit = u.erg/u.s/u.cm**2/u.sr/u.cm

    print_mem = lambda: f"{psutil.virtual_memory().used/1e9:.2f}/" \
                        f"{psutil.virtual_memory().total/1e9:.2f} GB"

    # The view: the axis to read and sum the atmosphere along, and the side
    # of the box it is seen from.
    view = view_name(args.integration_axis.lower())
    integration_axis, observer_side = view_axis_and_side(view)
    check_atmosphere_options(args)
    # A position on the disk sets the line of sight itself, slanted through
    # the box from above.
    (position, rotation, projection, curvature, geometry, frame,
     heliocentric_angle) = _disk_view(args, view)
    write_pickle = _writes_pickle(args.output_name)

    # Set by whichever route reads the atmosphere. Only an atmosphere file
    # can give the electron or proton density, or a size for each cell along
    # the line of sight.
    ne_values = np_values = None
    sky = None
    los_thickness = None
    atmosphere_metadata = None

    dynamic_mode = args.slit_rest_time is not None

    if dynamic_mode:
        (temp_cube, rho, vel_cube, dynamic_mode_metadata,
         voxel_dx, voxel_dy, voxel_dz) = _dynamic_mode_cubes(
            args, view, integration_axis, band, downsample, precision, print_mem)
        reference_cube = temp_cube

    else:
        # Static mode: from an atmosphere file, or from MURaM's own files
        # (deprecated), read into the atmosphere the converter would write.
        dynamic_mode_metadata = {"enabled": False}

        if args.atmosphere:
            print("STATIC MODE - Synthesis from an atmosphere file")
            print(f"  Atmosphere: {args.atmosphere}")
        else:
            print("STATIC MODE - Synthesis from MURaM files (deprecated)")
            print(f"  Data directory: {args.data_dir}")
        if position is None:
            print(f"  Integration axis: {view}")
        else:
            print(f"  Seen from ({position[0]}, {position[1]}), at "
                  f"{heliocentric_angle:.1f} from the vertical, turned by {rotation}, "
                  f"{projection} projection" + (", on the curved surface" if curvature else ""))
        print(f"  Velocity grid: +/-{vel_lim:.1f} at {vel_res:.1f} resolution")
        print(f"  Precision: {precision}")
        if downsample:
            print(f"  Downsampling: {downsample}x")
        print(f"  Lines: {args.lines}")
        print(f"  Abundance: {args.abundance}")
        if args.crop_x or args.crop_y or args.crop_z:
            print(f"  Cropping: X={args.crop_x}, Y={args.crop_y}, Z={args.crop_z}")
        print()

        print(f"Reading the atmosphere ({print_mem()})")
        crops = dict(crop_x=args.crop_x, crop_y=args.crop_y, crop_z=args.crop_z)
        # A slanted line of sight needs the velocity along each axis it leans
        # along; a part below 1e-9 is rounding error. On the curved surface it
        # leans differently in each column, so all three are needed.
        towards = None if frame is None else frame[2]
        velocities = (None if towards is None else AXES if curvature
                      else tuple(axis for axis, part in zip(AXES, towards) if abs(part) > 1e-9))
        if args.atmosphere:
            atmosphere = load_atmosphere_file(
                args.atmosphere, integration_axis, downsample=downsample, **crops,
                velocities=velocities)
        else:
            atmosphere = load_muram_files(
                args, integration_axis, downsample=downsample, **crops)
        print(atmosphere.describe())

        # Convert to the --precision dtype and to K and cm/s.
        temp_cube = atmosphere.to_ndcube(
            atmosphere.temperature.astype(precision, copy=False).to(u.K, copy=False))
        if towards is None:
            vel_cube = atmosphere.to_ndcube(
                atmosphere.velocity(integration_axis).astype(precision, copy=False)
                .to(u.cm / u.s, copy=False))
        else:
            wrap = not (getattr(args, "no_wrap", False) or args.crop_x or args.crop_y)
            if not wrap and not getattr(args, "no_wrap", False):
                print("  The box is cropped sideways, so it is not taken to repeat: lines of "
                      "sight end at its sides")
            if curvature:
                # The velocity along each of the box's axes; each part of a
                # cell takes the part along its own line of sight.
                vel_cube = None
                velocity_components = tuple(
                    atmosphere.velocity(axis).astype(precision).to_value(u.cm / u.s)
                    for axis in AXES)
                sky = CurvedSkyView(geometry, atmosphere.edges("x"),
                                    atmosphere.edges("y"), atmosphere.edges("z"), wrap)
            else:
                # The velocity away from the observer, along the slanted line
                # of sight.
                away = np.zeros(atmosphere.shape, dtype=precision)
                for axis, part in zip(AXES, towards):
                    if axis in velocities:
                        away -= precision(part) * atmosphere.velocity(axis).astype(
                            precision).to_value(u.cm / u.s)
                vel_cube = atmosphere.to_ndcube(away * u.cm / u.s)
                sky = SkyView(frame, atmosphere.edges("x"), atmosphere.edges("y"),
                              atmosphere.edges("z"), wrap, projection)
        rho = None
        if atmosphere.mass_density is not None:
            rho = atmosphere.mass_density.astype(precision, copy=False)
        if atmosphere.electron_density is not None:
            ne_values = atmosphere.electron_density.astype(precision, copy=False).to_value(
                u.cm**-3)
        elif atmosphere.proton_density is not None:
            np_values = atmosphere.proton_density.astype(precision, copy=False).to_value(
                u.cm**-3)
        los_thickness = atmosphere.cell_thickness(integration_axis)
        reference_cube = temp_cube
        if towards is not None:
            # The image is on the sky's pixels, which are as big as the
            # box's cells across.
            reference_cube = sky.reference_cube(atmosphere.cell_thickness("z").mean(),
                                                meta=temp_cube.meta)

        # Cell sizes, saved as the voxel sizes; a stretched axis has none.
        voxel_dx, voxel_dy, voxel_dz = (
            atmosphere.spacing(axis) if atmosphere.is_uniform(axis) else None
            for axis in AXES)
        atmosphere_metadata = {
            "path": str(Path(args.atmosphere).resolve()) if args.atmosphere else None,
            "source": atmosphere.source,
            "time": atmosphere.time,
            "shape": atmosphere.shape,
            "nonuniform_axes": atmosphere.nonuniform_axes(),
            "electron_density_given": atmosphere.electron_density is not None,
            "proton_density_given": atmosphere.proton_density is not None,
        }

    # ---------------- Common processing (both modes) -----------------
    
    vel_grid = velocity_grid(vel_res, vel_lim, ("--vel-res", "--vel-lim"))

    # Electron density from the atmosphere if given, otherwise from the
    # proton density, otherwise the mass density divided by the mass per
    # free electron.
    if ne_values is not None:
        logT_cube, logN_cube = _log_cubes_by_slab(
            temp_cube.data, ne_values, lambda temperature, electrons: electrons, precision)
        mass_per_electron_amu = None
        mass_per_electron_source = "not needed: the atmosphere gives the electron density"
        print("Electron density taken from the atmosphere")
    elif np_values is not None:
        database = getattr(args, "hdf5_dbase_root", None)
        logT_cube, logN_cube = _log_cubes_by_slab(
            temp_cube.data, np_values,
            lambda temperature, protons: electron_density_from_protons(
                temperature, protons, args.abundance, database), precision)
        mass_per_electron_amu = None
        mass_per_electron_source = "not needed: the atmosphere gives the proton density"
        print(f"Electron density from the proton density, with fiasco's ratio of the two "
              f"for {args.abundance} abundances")
    else:
        mass_per_electron_amu, mass_per_electron_source = resolve_mass_per_electron(args)
        print(f"Electron density from the mass density with "
              f"{mass_per_electron_amu:.4f} u per electron ({mass_per_electron_source})")
        per_electron = mass_per_electron_amu * const.u
        logT_cube, logN_cube = _log_cubes_by_slab(
            temp_cube.data, rho,
            lambda temperature, mass: (mass / per_electron).to_value(u.cm**-3), precision)
    # Free the densities once their logs are taken.
    rho = ne_values = np_values = atmosphere = None

    # The Doppler shift needs the velocity away from the observer, which
    # vel_cube already holds along a slanted line of sight.
    vel_data = (velocity_components if curvature
                else vel_cube.data if position is not None
                else line_of_sight_velocity(vel_cube.data, view))

    # ---------------- Compute contribution functions (fiasco) ---------
    # Only over the densities this atmosphere has.
    logN_min, logN_max, nN = _density_grid_from_points(_density_points_from_logs(
        logT_cube, logN_cube, np.linspace(_LOGT_MIN, _LOGT_MAX, _N_T).astype(precision)))

    # Only a strip of the image, if asked for. The densities above are the
    # whole image's, so one file of a band's lines serves every strip. Along
    # an axis, the strip's cells are cut from the box; seen from a position,
    # the image is cut once made, before the band's spectrum.
    image_columns = columns = None
    if getattr(args, "image_x", None):
        columns = image_columns_between(image_x_edges(reference_cube, view), args.image_x)
        n_columns = image_x_edges(reference_cube, view).size - 1
        image_columns = (columns.start, n_columns)
        if sky is None:
            logT_cube, logN_cube, vel_data = (
                image_strip(cube, view, columns).copy()
                for cube in (logT_cube, logN_cube, vel_data))
        print(f"  Image columns {columns.start} to {columns.stop - 1} of {n_columns} kept")

    band_lines = None
    if band is None:
        print(f"Computing contribution functions via fiasco at 10^{logN_min:.1f} to "
              f"10^{logN_max:.1f} cm^-3 ({print_mem()})")
        goft, logT_goft, logN_grid = compute_goft_fiasco(
            args.lines, abundance=args.abundance, precision=precision,
            logN_min=logN_min, logN_max=logN_max, nN=nN,
            n_workers=args.n_workers,
            hdf5_dbase_root=getattr(args, "hdf5_dbase_root", None),
            temperature_chunk=getattr(args, "goft_temperature_chunk", None),
        )
        # Record the database actually used, so a run that did not choose one
        # can be traced to its atomic data.
        goft_dbase_root = (
            next(iter(goft.values()))["hdf5_dbase_root"] if goft else None
        )
    else:
        # Every line in the band, on the temperatures the lines above would
        # be on.
        logT_band = np.linspace(_LOGT_MIN, _LOGT_MAX, _N_T)
        logN_band = np.linspace(logN_min, logN_max, nN)
        band_lines = band_lines_for(
            args.band_lines or Path(args.output_dir) / "band_lines.h5", band, logT_band,
            logN_band, abundance=args.abundance, n_workers=args.n_workers,
            hdf5_dbase_root=getattr(args, "hdf5_dbase_root", None),
            wider_grid=band_density_grid(logN_min, logN_max))
        goft, logT_goft, logN_grid = {}, logT_band.astype(precision), logN_band.astype(precision)
        goft_dbase_root = band_lines.hdf5_dbase_root
    print(f"  CHIANTI database: {goft_dbase_root}")

    # The DEM uses the G(T) temperature grid.
    logT_grid = logT_goft
    
    # The size of each cell along the line of sight. MURaM's files have one
    # size; an atmosphere file may give every cell its own.
    if los_thickness is None:
        los_thickness = {"x": voxel_dx, "y": voxel_dy, "z": voxel_dz}[integration_axis]
    dh_cm = los_thickness.to_value(u.cm)

    # ---------------- DEM, EM(T,v) and spectra -----------------
    # Continuum over the lines' windows, one entry per group of overlapping
    # windows. Each window is the line's velocity grid around its CHIANTI
    # wavelength, as in synthesise_spectra.
    continuum = None
    if getattr(args, "continuum", False):
        if band is not None:
            # On the band's own wavelengths.
            windows = {f"{CONTINUUM_PREFIX}_{band_grid[0]:.4f}-{band_grid[-1]:.4f}":
                       (band_grid * u.Angstrom).to(u.cm)}
        else:
            windows = continuum_windows([(vel_grid * info["wl0"] / const.c + info["wl0"]).cgs
                                         for info in goft.values()])
        print(f"Computing the continuum over {len(windows)} window"
              f"{'' if len(windows) == 1 else 's'} via fiasco ({print_mem()})")
        continuum = continuum_tables(windows, logT_grid, logN_grid, args.abundance,
                                     args.n_workers, goft_dbase_root)

    print(f"Calculating the DEM, the emission measure in (T,v) and the spectra ({print_mem()})")
    keep_vdem = bool(getattr(args, "keep_vdem", False))
    goft, dem_map, em_tv, electron_density = _synthesise_from_logs(
        logT_cube, logN_cube, vel_data, dh_cm, goft, logT_grid, logN_grid,
        vel_grid, view, precision, continuum=continuum, sky=sky, return_density=True,
        keep_em_tv=keep_vdem or band is not None or write_pickle)
    if sky is not None and columns is not None:
        dem_map, electron_density = (cube[:, columns] for cube in (dem_map, electron_density))
        if em_tv is not None:
            em_tv = em_tv[:, columns]
        for info in goft.values():
            info["si"] = info["si"][:, columns]

    band_name = None
    if band is not None:
        print(f"Calculating the spectrum of the band's {band_lines.n_lines} lines at "
              f"{band_grid.size} wavelengths ({print_mem()})")
        grid = (band_grid * u.Angstrom).to(u.cm)
        spectra = band_spectra(band_lines, em_tv, electron_density, vel_grid, grid,
                               n_workers=args.n_workers)
        band_name = f"band_{band_grid[0]:.4f}-{band_grid[-1]:.4f}"
        goft[band_name] = {"si": spectra.to_value(intensity_unit).astype(precision),
                           "wl_grid": grid, "wl0": grid[grid.size // 2],
                           "atom": None, "ion": None}

    # ---------------- Create output cubes -----------------
    print(f"Creating output cubes ({print_mem()})")
    line_cubes = {}
    for name, info in goft.items():
        line_cubes[name] = create_line_cube(
            name, info, reference_cube, intensity_unit, view, image_columns=image_columns
        )
    
    print(f"Built {len(line_cubes)} line cubes")

    # ---------------- Save results -----------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_name
    
    # The line cubes are the spectra; the other products are saved with them.
    products = {
        "dem_map": dem_map,
        **({"em_tv": em_tv} if keep_vdem else {}),
        # Electron density at each temperature of each pixel (mean over its
        # cells weighted by emission measure), at which G was taken.
        "electron_density": electron_density,
        "logT_grid": logT_grid,
        "vel_grid": vel_grid,
        "logN_grid": logN_grid,
        # Spectra and wavelengths are already in the line cubes.
        "goft": {name: {key: value for key, value in info.items()
                        if key not in ("si", "wl_grid")}
                 for name, info in goft.items()
                 if name not in (continuum or {}) and name != band_name},
        "voxel_sizes": {"dx": voxel_dx, "dy": voxel_dy, "dz": voxel_dz},
        "dynamic_mode": dynamic_mode_metadata,
        "atmosphere": atmosphere_metadata,
        "config": {
            "precision": precision.__name__,
            "keep_vdem": keep_vdem,
            "downsample": downsample,
            "vel_res": vel_res,
            "vel_lim": vel_lim,
            "mass_per_electron": mass_per_electron_amu,
            "mass_per_electron_source": mass_per_electron_source,
            "intensity_unit": str(intensity_unit),
            "atmosphere": args.atmosphere,
            "cube_shape": None if args.atmosphere else args.cube_shape,
            "data_dir": None if args.atmosphere else str(Path(args.data_dir)),
            "lines": args.lines or [],
            "band": None if band is None else list(band),
            "band_step": None if band is None else band_step,
            "image_x": getattr(args, "image_x", None),
            "abundance": args.abundance,
            "continuum": bool(getattr(args, "continuum", False)),
            "hdf5_dbase_root": goft_dbase_root,
            "integration_axis": view,
            "velocity_convention": VELOCITY_CONVENTION,
            "observer_side": observer_side,
            "position": None if position is None else list(position),
            "rotation": rotation,
            "projection": projection,
            "curvature": curvature if position is not None else None,
            "heliocentric_angle": None if position is None else heliocentric_angle,
            "wrap": None if sky is None else sky.wrap,
            "crop_params": {
                "crop_x": args.crop_x,
                "crop_y": args.crop_y,
                "crop_z": args.crop_z
            }
        }
    }
    
    if write_pickle:
        # The deprecated pickle, with the full contribution functions and the
        # emission measure.
        with open(output_file, "wb") as f:
            dill.dump({"line_cubes": line_cubes, "dem_map": dem_map, "em_tv": em_tv, **products,
                       "goft": {name: info for name, info in goft.items()
                                if name not in (continuum or {}) and name != band_name}}, f)
    else:
        # Save the snapshot time, so syntheses of a series of snapshots can be
        # observed as a time series.
        write_line_cubes(line_cubes, output_file,
                         source=(atmosphere_metadata or {}).get("source") or "",
                         products=products,
                         time=(atmosphere_metadata or {}).get("time"))

    print(f"Saved results to {output_file} ({os.path.getsize(output_file) / 1e6:.2f} MB)")
    print("Synthesis complete!")

if __name__ == "__main__":
    main()