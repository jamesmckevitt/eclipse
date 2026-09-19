import os
import re
import argparse
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
import astropy.constants as const
from tqdm import tqdm
import psutil
import dask.array as da
from dask.diagnostics import ProgressBar
from mendeleev import element
import dill
from ndcube import NDCube
from astropy.wcs import WCS
from .utils import (angle_to_distance, require_uniform_grid, require_downsample_divides,
                    velocity_centers_to_edges, VELOCITY_CONVENTION)
from .atmosphere import (AXES, NUMPY_AXIS, Atmosphere, mass_per_electron, read_atmosphere,
                         require_mass_per_electron)

##############################################################################
# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------
##############################################################################

def load_cube(
    file_path: str | Path,
    shape: Tuple[int, int, int] = (512, 768, 256),
    unit: Optional[u.Unit] = None,
    downsample: int | bool = False,
    precision: type = np.float32,
    voxel_dx: Optional[u.Quantity] = None,
    voxel_dy: Optional[u.Quantity] = None,
    voxel_dz: Optional[u.Quantity] = None,
    create_ndcube: bool = False,
) -> np.ndarray | u.Quantity | NDCube:
    """
    Read a Fortran-ordered binary cube (single precision) and optionally return as NDCube.

    The cube is stored (x, z, y) in the file and transposed to (z, y, x)
    upon loading, so that a horizontal slice ``data[k]`` is an image indexed
    ``[y, x]``.

    Parameters
    ----------
    file_path : str | Path
        Path to the binary file.
    shape : Tuple[int, int, int]
        The *full* cube dimensions in the file's own storage order, which is
        ``(nx, nz, ny)`` - the vertical axis comes second, not last.
    unit : astropy.units.Unit, optional
        Astropy unit to attach (e.g. u.K or u.g/u.cm**3). If None, returns
        a plain ndarray.
    downsample : int | bool
        Integer factor; if non-False, keep every *downsample*-th cell along
        each axis (simple stride).
    precision : type
        np.float32 or np.float64 for returned dtype.
    voxel_dx, voxel_dy, voxel_dz : u.Quantity, optional
        Voxel sizes of the file, at full resolution. When *downsample* is
        set, the returned cube's WCS uses them multiplied by it. Required if
        create_ndcube=True.
    create_ndcube : bool, optional
        If True, return an NDCube with proper WCS coordinates.

    Returns
    -------
    ndarray, Quantity, or NDCube
        Array with shape (nz', ny', nx') or NDCube with proper coordinates.
    """
    if downsample:
        require_downsample_divides(shape, downsample)

    data = np.fromfile(file_path, dtype=np.float32).reshape(shape, order="F")
    data = data.transpose(1, 2, 0)  # (z,y,x)

    if downsample:
        data = data[::downsample, ::downsample, ::downsample]
        # Each kept cell now stands for *downsample* cells of the file. New
        # quantities, not *=, which would scale the caller's own voxel sizes
        # and compound across calls.
        voxel_dx, voxel_dy, voxel_dz = (
            None if size is None else size * downsample
            for size in (voxel_dx, voxel_dy, voxel_dz))

    data = data.astype(precision, copy=False)
    
    if unit is not None:
        data = data * unit
        
    if create_ndcube:
        return create_atmosphere_ndcube(data, voxel_dx, voxel_dy, voxel_dz)
    else:
        return data


def create_atmosphere_ndcube(
    data: np.ndarray | u.Quantity,
    voxel_dx: u.Quantity,
    voxel_dy: u.Quantity, 
    voxel_dz: u.Quantity,
) -> NDCube:
    """
    Create an NDCube for atmospheric data with proper heliocentric coordinates.

    Parameters
    ----------
    data : np.ndarray or u.Quantity
        3D data array with shape (nz, ny, nx), so that ``data[k]`` is a
        horizontal slice indexed ``[y, x]``.
    voxel_dx, voxel_dy, voxel_dz : u.Quantity
        Voxel sizes in Mm.

    Returns
    -------
    NDCube
        Cube with proper WCS coordinates.
        X,Y centered at origin, Z starting at 0.
    """
    nz, ny, nx = data.shape

    # Create WCS for heliocentric coordinates
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['SOLX', 'SOLY', 'SOLZ']
    wcs.wcs.cunit = ['Mm', 'Mm', 'Mm']

    # Reference pixels (1-indexed for WCS)
    wcs.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2, 1]  # Z starts at first pixel

    # Reference values
    wcs.wcs.crval = [0, 0, 0]  # X,Y centered at origin, Z starts at 0

    # Pixel scales
    wcs.wcs.cdelt = [
        voxel_dx.to(u.Mm).value,
        voxel_dy.to(u.Mm).value,
        voxel_dz.to(u.Mm).value
    ]

    return NDCube(data.data,
                  wcs=wcs,
                  unit=data.unit)


def read_timestep_time(file_path: Path) -> float:
    """
    Read simulation time from MHD header file.
    
    The header file contains a single line with 9 space-separated values:
    nx ny nz dx dy dz time dt va_max
    
    The time is the 7th value (index 6) in seconds.
    
    Parameters
    ----------
    file_path : Path
        Path to the header file.
        
    Returns
    -------
    float
        Simulation time in seconds.
    """
    with open(file_path, 'r') as f:
        line = f.read().strip()
        values = line.split()
        if len(values) < 7:
            raise ValueError(f"Header file has insufficient values: {file_path} ({len(values)} values)")
        return float(values[6])


def apply_cube_cropping(
    temp_cube: NDCube,
    rho_cube: NDCube,
    vel_cube: NDCube,
    crop_x: Optional[List[str]],
    crop_y: Optional[List[str]],
    crop_z: Optional[List[str]],
) -> Tuple[NDCube, NDCube, NDCube]:
    """
    Apply cropping to temperature, density, and velocity cubes.
    
    Parameters
    ----------
    temp_cube, rho_cube, vel_cube : NDCube
        Input cubes to crop.
    crop_x, crop_y, crop_z : list of str or None
        Crop boundaries for each axis as [min, max] strings with units.
        
    Returns
    -------
    tuple of NDCube
        Cropped (temp_cube, rho_cube, vel_cube).
    """
    # Crop points are given in world axis order, which is (SOLX, SOLY, SOLZ).
    point1 = []
    point2 = []

    if crop_x:
        point1.append(u.Quantity(crop_x[0]))
        point2.append(u.Quantity(crop_x[1]))
    else:
        point1.append(None)
        point2.append(None)

    if crop_y:
        point1.append(u.Quantity(crop_y[0]))
        point2.append(u.Quantity(crop_y[1]))
    else:
        point1.append(None)
        point2.append(None)

    if crop_z:
        point1.append(u.Quantity(crop_z[0]))
        point2.append(u.Quantity(crop_z[1]))
    else:
        point1.append(None)
        point2.append(None)
    
    temp_cube = temp_cube.crop(point1, point2)
    rho_cube = rho_cube.crop(point1, point2)
    vel_cube = vel_cube.crop(point1, point2)
    
    return temp_cube, rho_cube, vel_cube


def _compute_single_ion(args):
    """Worker that computes G(T,N) for one ion.  Imports fiasco locally so
    that each spawned process gets its own HDF5 handles."""

    import fiasco
    import logging

    elem, stage, temperature_K, densities_cm3, abundance, lines, hdf5_dbase_root = args
    temperature = temperature_K * u.K
    densities = densities_cm3 / u.cm**3

    # A spawned process re-imports fiasco from scratch, so it re-reads
    # ~/.fiasco/fiascorc and knows nothing about a database the parent
    # selected in memory. The root therefore has to travel in the arguments.
    ion_kwargs = {} if hdf5_dbase_root is None else {
        'hdf5_dbase_root': hdf5_dbase_root}

    # Suppress repetitive fiasco warnings about missing proton data and
    # autoionization/rrlvl files.  These are CHIANTI database gaps (not all
    # ions have .psplups or .auto/.rrlvl data) and both fiasco and IDL
    # gracefully fall back to excluding proton rates / using the single-ion
    # model.  The warnings fire once per density point, producing hundreds of
    # identical lines.
    fiasco_logger = logging.getLogger('fiasco')
    prev_level = fiasco_logger.level
    fiasco_logger.setLevel(logging.ERROR)

    try:
        ion = fiasco.Ion(f'{elem} {stage}', temperature, abundance=abundance,
                         **ion_kwargs)

        g = ion.contribution_function(densities)
        pe_ratio = ion.proton_electron_ratio
        g = g * pe_ratio[:, np.newaxis, np.newaxis]
    finally:
        fiasco_logger.setLevel(prev_level)

    bb_wl = ion.transitions.wavelength[ion.transitions.is_bound_bound]

    results = {}
    for line_name, target_wl_aa in lines:
        target_wl = target_wl_aa * u.AA
        idx = int(np.argmin(np.abs(bb_wl - target_wl)))
        matched_wl = bb_wl[idx]

        g_tn = g[:, :, idx].to(u.erg * u.cm**3 / u.s).value.T
        np.nan_to_num(g_tn, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        results[line_name] = {
            "g_tn": g_tn,
            "atom": int(ion.atomic_number),
            "ion": stage,
            "target_wl_cm": float(target_wl.to(u.cm).value),
            "matched_wl_aa": float(matched_wl.to(u.AA).value),
            "delta_aa": float(abs(matched_wl - target_wl).to(u.AA).value),
            # The root this Ion was built against.  fiasco resolves it to the
            # fiascorc default when the caller did not choose one, so this is
            # always the database the contribution functions came from.
            "hdf5_dbase_root": str(ion.hdf5_dbase_root),
        }
    return results


def compute_goft_fiasco(
    line_names: List[str],
    abundance: str = "sun_coronal_2021_chianti",
    logT_min: float = 4.0,
    logT_max: float = 9.0,
    nT: int = 101,
    logN_min: float = 7.0,
    logN_max: float = 13.0,
    nN: int = 21,
    precision: type = np.float64,
    n_workers: int = 0,
    hdf5_dbase_root=None,
) -> Tuple[Dict[str, dict], np.ndarray, np.ndarray]:
    """
    Compute G(T,N) contribution functions using fiasco.

    For each line specification (e.g. "Fe12_195.1190"), creates a fiasco Ion,
    computes the contribution function over a (T, n_e) grid, and extracts the
    transition closest to the requested wavelength.

    The CHIANTI contribution function is::

        G_ij = Ab(X) * f_{X,k} * (N_j / N) * A_ij * dE_ij / n_e

    with units of erg cm^3 s^-1.  fiasco's ``contribution_function`` does
    **not** include the n_H / n_e ratio.  However, this function explicitly
    multiplies G by the proton-to-electron ratio so that the result is
    consistent with the n_e^2 * dh emission measure used downstream.

    Parameters
    ----------
    line_names : List[str]
        Line identifiers, e.g. ``["Fe12_195.1190", "Fe09_171.073"]``.
    abundance : str
        CHIANTI abundance dataset name passed to ``fiasco.Ion``.
    logT_min, logT_max : float
        Bounds of the log10(T / K) grid.
    nT : int
        Number of temperature grid points.
    logN_min, logN_max : float
        Bounds of the log10(n_e / cm^-3) grid.
    nN : int
        Number of density grid points.
    precision : type
        Output array dtype (``np.float32`` or ``np.float64``).
    n_workers : int
        Number of parallel processes for multi-ion runs.  Each worker
        spawns a separate process (to avoid HDF5 fork-safety issues) and
        imports fiasco independently, so there is a startup cost per
        worker.  Only useful when computing lines from 2+ distinct ions.
        Defaults to 0, which uses ``os.cpu_count()``.
    hdf5_dbase_root : str or `~pathlib.Path`, optional
        CHIANTI HDF5 database to use.  Defaults to fiasco's own, which comes
        from ``~/.fiasco/fiascorc``.  Pass this to run against a database
        other than the user's default: because each worker is spawned rather
        than forked, it re-imports fiasco and re-reads that file, so setting
        ``fiasco.defaults`` in the parent process has no effect on the
        workers.  Each worker reports back the root its ``Ion`` was built
        with, and that is checked against the request, so a root that fails
        to reach a worker raises rather than letting that worker fall back to
        the fiascorc default.  Note this confirms the argument arrived, not
        that fiasco read the file correctly once pointed at it.

    Returns
    -------
    goft_dict : Dict[str, dict]
        Dictionary keyed by line name, each entry holding:
            ``'wl0'``  -- rest wavelength (Quantity, cm)
            ``'g_tn'`` -- 2-D array G(logN, logT) shape ``(nN, nT)``
            ``'atom'`` -- atomic number
            ``'ion'``  -- ionisation stage
            ``'hdf5_dbase_root'`` -- CHIANTI database these came from,
            resolved to the fiascorc default when none was requested
    logT_grid : np.ndarray
        1-D array of log10(T / K) values.
    logN_grid : np.ndarray
        1-D array of log10(n_e / cm^-3) values.
    """
    logT_grid = np.linspace(logT_min, logT_max, nT)
    logN_grid = np.linspace(logN_min, logN_max, nN)

    temperature_K = 10.0 ** logT_grid
    densities_cm3 = 10.0 ** logN_grid

    # ---- parse line names and group by ion for efficiency ----
    line_pattern = re.compile(r'^([A-Z][a-z]?)(\d+)_(\d+\.?\d*)$')
    ion_lines: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}

    for name in line_names:
        m = line_pattern.match(name)
        if not m:
            raise ValueError(
                f"Cannot parse line name '{name}'. "
                f"Expected format like 'Fe12_195.1190'."
            )
        elem = m.group(1)
        stage = int(m.group(2))
        wl = float(m.group(3))
        ion_lines.setdefault((elem, stage), []).append((name, wl))

    # Build worker arguments (all picklable plain types / numpy arrays)
    dbase_root = None if hdf5_dbase_root is None else str(hdf5_dbase_root)
    worker_args = [
        (elem, stage, temperature_K, densities_cm3, abundance, lines, dbase_root)
        for (elem, stage), lines in ion_lines.items()
    ]

    # ---- dispatch: parallel for 2+ ions, serial otherwise ----
    n_ions = len(worker_args)
    if n_workers <= 0:
        n_workers = os.cpu_count() or 1
    use_parallel = n_workers > 1 and n_ions > 1

    if use_parallel:
        import multiprocessing as mp
        pool_size = min(n_workers, n_ions)
        ctx = mp.get_context("spawn")
        print(f"  Parallel: {pool_size} workers for {n_ions} ions")
        with ctx.Pool(pool_size) as pool:
            all_results = pool.map(_compute_single_ion, worker_args)
    else:
        all_results = [_compute_single_ion(a) for a in tqdm(
            worker_args, desc="Computing G(T,N)", unit="ion"
        )]

    # ---- collect results ----
    goft_dict: Dict[str, dict] = {}
    for result in all_results:
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
                f"matched {info['matched_wl_aa']:.4f} Angstrom "
                f"(delta={info['delta_aa']:.4f} Angstrom)"
            )
            goft_dict[line_name] = {
                "wl0": info["target_wl_cm"] * u.cm,
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the differential emission measure DEM(T) and the emission-measure
    weighted mean electron density <n_e>(T).

    Parameters
    ----------
    logT_cube : np.ndarray
        3D array of log10(T/K) values.
    logN_cube : np.ndarray  
        3D array of log10(n_e/cm^3) values.
    voxel_dh_cm : float or np.ndarray
        Depth of the cells along the integration axis in cm: one value for
        every cell, or one value per cell along that axis.
    logT_grid : np.ndarray
        1D array of temperature bin centers for DEM calculation.
    integration_axis : str
        Axis along which to integrate ("x", "y", or "z").

    Returns
    -------
    dem_map : np.ndarray
        DEM array [cm^-5 per dex]. The two remaining spatial axes come out in
        image order (row, column). Shape depends on integration_axis:
        - "x": (nz, ny, nT)
        - "y": (nz, nx, nT)
        - "z": (ny, nx, nT)
    avg_ne : np.ndarray
        Mean electron density per T-bin [cm^-3]. Same shape as dem_map.
    """
    nT = len(logT_grid)

    # The cubes are (z, y, x), so integrating along a physical axis means
    # summing over the numpy axis it lives on.
    axis_map = {"x": 2, "y": 1, "z": 0}
    if integration_axis not in axis_map:
        raise ValueError(f"integration_axis must be 'x', 'y', or 'z', got {integration_axis}")

    integration_axis_idx = axis_map[integration_axis]

    # Output shape depends on which axis we integrate over
    if integration_axis == "x":
        output_shape = (logT_cube.shape[0], logT_cube.shape[1], nT)  # (nz, ny, nT)
    elif integration_axis == "y":
        output_shape = (logT_cube.shape[0], logT_cube.shape[2], nT)  # (nz, nx, nT)
    else:  # "z"
        output_shape = (logT_cube.shape[1], logT_cube.shape[2], nT)  # (ny, nx, nT)
    
    # Create temperature bin edges from centers
    dlogT = logT_grid[1] - logT_grid[0] if len(logT_grid) > 1 else 0.1
    logT_edges = np.concatenate([
        [logT_grid[0] - dlogT/2],
        logT_grid[:-1] + dlogT/2,
        [logT_grid[-1] + dlogT/2]
    ])

    ne = 10.0 ** logN_cube.astype(np.float64)
    dh = along_line_of_sight(voxel_dh_cm, integration_axis)
    w2 = ne**2 * dh  # weights for EM
    w3 = ne**3 * dh  # weights for EM*n_e

    dem = np.zeros(output_shape)
    avg_ne = np.zeros_like(dem)

    for idx in tqdm(range(nT), desc="DEM bins", unit="bin", leave=False):
        lo, hi = logT_edges[idx], logT_edges[idx + 1]
        mask = (logT_cube >= lo) & (logT_cube < hi)  # (nz,ny,nx)

        # Integrate along the specified axis
        em = np.sum(w2 * mask, axis=integration_axis_idx)    # cm^-5
        em_n = np.sum(w3 * mask, axis=integration_axis_idx)  # cm^-5 * n_e

        dem[..., idx] = em / dlogT
        avg_ne[..., idx] = np.divide(em_n, em, where=em > 0.0)

    return dem, avg_ne


def interpolate_g_on_dem(
    goft: Dict[str, dict],
    avg_ne: np.ndarray,
    logT_grid: np.ndarray,
    logN_grid: np.ndarray,
    logT_goft: np.ndarray,
    precision: type = np.float32,
) -> None:
    """
    For every spectral line, interpolate G(T,N) onto the DEM grid.
    
    Parameters
    ----------
    goft : Dict[str, dict]
        Dictionary of line data, modified in place.
    avg_ne : np.ndarray
        Emission-measure weighted electron density (n_rows, n_cols, nT), in
        the spatial layout compute_dem produces.
    logT_grid : np.ndarray
        Temperature grid for DEM (nT,).
    logN_grid : np.ndarray
        Density grid for GOFT interpolation.
    logT_goft : np.ndarray
        Temperature grid for GOFT interpolation.
    precision : type
        Output precision for interpolated G values.
    """
    nT, n_rows, n_cols = len(logT_grid), *avg_ne.shape[:2]

    # Build query points for interpolation
    logNe_flat = np.log10(avg_ne, where=avg_ne > 0.0,
                         out=np.zeros_like(avg_ne)).transpose(2, 0, 1).ravel()
    logT_flat = np.broadcast_to(logT_grid[:, None, None],
                               (nT, n_rows, n_cols)).ravel()
    query_pts = np.column_stack((logNe_flat, logT_flat))

    for name, info in tqdm(goft.items(), desc="interpolating G", unit="line", leave=False):
        rgi = RegularGridInterpolator(
            (logN_grid, logT_goft), info["g_tn"],
            method="linear", bounds_error=False, fill_value=0.0
        )
        g_flat = rgi(query_pts)
        info["g"] = g_flat.reshape(nT, n_rows, n_cols).transpose(1, 2, 0).astype(precision)


##############################################################################
# ---------------------------------------------------------------------------
#  Build EM(T,v) and synthesise spectra
# ---------------------------------------------------------------------------
##############################################################################

# Which side of the box the observer is on, for each integration axis: +1 on
# the side of increasing coordinate, -1 on the other.  It is the side from
# which the line cube, with its rows and columns as create_line_cube lays them
# out, is seen the right way round, so the column axis crossed with the row
# axis points at the observer: above the box (+z) for the top-down view, and
# at +x and at -y for the two side views.
OBSERVER_SIDE = {"x": +1, "y": -1, "z": +1}


def line_of_sight_velocity(velocity, integration_axis: str):
    """
    Turn the velocity along the integration axis into velocity away from the observer.

    Simulation velocities are positive towards increasing coordinate, so an
    upflow has a positive z velocity.  Seen from above, that upflow is moving
    towards the observer and is blueshifted, so its line-of-sight velocity is
    negative.  Positive line-of-sight velocity is a redshift: each line is
    placed at ``lambda_0 (1 + v / c)``.

    Parameters
    ----------
    velocity : np.ndarray or u.Quantity
        Velocity along the integration axis, positive towards increasing
        coordinate.
    integration_axis : str
        ``"x"``, ``"y"`` or ``"z"``.  The observer is on the side given by
        :data:`OBSERVER_SIDE`.

    Returns
    -------
    np.ndarray or u.Quantity
        Velocity away from the observer, in the same units.
    """
    if integration_axis not in OBSERVER_SIDE:
        raise ValueError(
            f"integration_axis must be 'x', 'y', or 'z', got {integration_axis}"
        )
    return -OBSERVER_SIDE[integration_axis] * velocity


def build_em_tv(
    logT_cube: np.ndarray,
    vel_cube: np.ndarray,
    logT_grid: np.ndarray,
    vel_grid: np.ndarray,
    ne_sq_dh: np.ndarray,
    integration_axis: str = "z",
) -> np.ndarray:
    """
    Construct the 4-D emission-measure cube EM(row, column, T, v) [cm^-5].

    Parameters
    ----------
    logT_cube : np.ndarray
        3D temperature cube, shape (nz, ny, nx).
    vel_cube : np.ndarray
        3D velocity cube along the integration axis.
    logT_grid : np.ndarray
        Temperature bin centers.
    vel_grid : np.ndarray
        Velocity bin centers.
    ne_sq_dh : np.ndarray
        n_e^2 * dh for each voxel.
    integration_axis : str
        Axis along which to integrate ("x", "y", or "z").

    Returns
    -------
    em_tv : np.ndarray
        4D emission measure cube. The two remaining spatial axes come out in
        image order (row, column). Shape depends on integration_axis:
        - "x": (nz, ny, nT, nv)
        - "y": (nz, nx, nT, nv)
        - "z": (ny, nx, nT, nv)
    """
    print(f"  Building 4-D emission-measure cube along {integration_axis}-axis...")

    # The cubes are (z, y, x); see compute_dem.
    axis_map = {"x": 2, "y": 1, "z": 0}
    if integration_axis not in axis_map:
        raise ValueError(f"integration_axis must be 'x', 'y', or 'z', got {integration_axis}")

    integration_axis_idx = axis_map[integration_axis]
    
    # Create temperature bin edges from centers
    dlogT = logT_grid[1] - logT_grid[0] if len(logT_grid) > 1 else 0.1
    logT_edges = np.concatenate([
        [logT_grid[0] - dlogT/2],
        logT_grid[:-1] + dlogT/2,
        [logT_grid[-1] + dlogT/2]
    ])
    
    # Compute velocity bin edges from centers
    v_edges = velocity_centers_to_edges(vel_grid.value)
    
    mask_T = (logT_cube[..., None] >= logT_edges[:-1]) & \
             (logT_cube[..., None] <  logT_edges[1:])
    mask_V = (vel_cube[..., None] >= v_edges[:-1]) & \
             (vel_cube[..., None] <  v_edges[1:])

    # Build the 4-D emission-measure cube EM(spatial,T,v) by summing over the integration axis
    ne_sq_dh_d = da.from_array(ne_sq_dh, chunks='auto')
    mask_T_d   = da.from_array(mask_T,   chunks='auto')
    mask_V_d   = da.from_array(mask_V,   chunks='auto')
    
    # Sum along the specified integration axis. The cube subscripts are
    # i=z, j=y, k=x, so the surviving pair is always (row, column).
    if integration_axis == "x":
        em_tv_d = da.einsum("ijk,ijkl,ijkm->ijlm", ne_sq_dh_d, mask_T_d, mask_V_d, optimize=True)
    elif integration_axis == "y":
        em_tv_d = da.einsum("ijk,ijkl,ijkm->iklm", ne_sq_dh_d, mask_T_d, mask_V_d, optimize=True)
    else:  # "z"
        em_tv_d = da.einsum("ijk,ijkl,ijkm->jklm", ne_sq_dh_d, mask_T_d, mask_V_d, optimize=True)
        
    with ProgressBar():
        em_tv = em_tv_d.compute()

    return em_tv


def synthesise_spectra(
    goft: Dict[str, dict],
    em_tv: np.ndarray,
    vel_grid: np.ndarray,
    logT_grid: np.ndarray,
) -> None:
    """
    Convolve EM(T,v) with thermal Gaussians plus Doppler shift to obtain the
    specific intensity cube I(row, column, lambda) for every line.

    Parameters
    ----------
    goft : Dict[str, dict]
        Dictionary of line data, modified in place with 'si' and 'wl_grid'.
    em_tv : np.ndarray
        4D emission measure cube (n_rows, n_cols, nT, nv), in the spatial
        layout build_em_tv produces.
    vel_grid : np.ndarray
        Velocity grid centers for wavelength calculation.
    logT_grid : np.ndarray
        Temperature bin centers.
    """
    kb = const.k_B.cgs.value
    c_cm_s = const.c.cgs.value

    # The wavelength grid built below is the velocity grid mapped through
    # lambda_0 (1 + v/c), and create_line_cube writes its CDELT from the first
    # step alone, so an uneven velocity grid becomes a wrong wavelength axis.
    require_uniform_grid(vel_grid, "vel_grid")

    for line, data in tqdm(goft.items(), desc="spectra", unit="line", leave=False):
        wl0 = data["wl0"].cgs.value  # cm
        
        # Create wavelength grid for this line
        data["wl_grid"] = (vel_grid * data["wl0"] / const.c + data["wl0"]).cgs
        wl_grid = data["wl_grid"].cgs.value  # (n_lambda,)

        atom = element(int(data["atom"]))
        atom_weight_g = (atom.atomic_weight * u.u).cgs.value

        # Thermal width per T-bin: sigma_T (nT,)
        sigma_T = wl0 * np.sqrt(kb * (10 ** logT_grid) / atom_weight_g) / c_cm_s

        # Doppler-shifted center for each v-bin: (nv,)
        lam_cent = wl0 * (1 + vel_grid.value / c_cm_s)

        # Build phi(T,v,lambda) as (nT,nv,n_lambda)
        delta = wl_grid[None, None, :] - lam_cent[None, :, None]
        phi = np.exp(-0.5 * (delta / sigma_T[:, None, None]) ** 2)
        phi /= sigma_T[:, None, None] * np.sqrt(2 * np.pi)

        # EM(row,col,T,v) * G(T)  ->  (n_rows,n_cols,nT,nv)
        weighted = em_tv * data["g"][..., None]

        # Collapse T and v: dot ((nT,nv) , (nT,nv)) -> (n_rows,n_cols,n_lambda)
        spec_map = np.tensordot(weighted, phi, axes=([2, 3], [0, 1]))

        data["si"] = spec_map / (4 * np.pi)


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
) -> Tuple[Dict[str, dict], np.ndarray, np.ndarray]:
    """
    The spectra of every line from one set of cubes: a whole box or a strip of it.

    Parameters
    ----------
    temperature : np.ndarray
        Temperature of every cell in K, ``(nz, ny, nx)``.
    electron_density : np.ndarray
        Electron density of every cell in cm^-3, the same shape.
    los_velocity : np.ndarray
        Velocity away from the observer in cm/s, the same shape.
    dh_cm : float or np.ndarray
        Depth of the cells along the line of sight in cm, one value for all
        or one per cell along that axis.
    goft : dict
        Contribution functions from :func:`compute_goft_fiasco`, on the
        temperature grid *logT_grid* and density grid *logN_grid*. Not
        modified: the entries are copied before the interpolated function
        and the spectra are added to them.
    logT_grid, logN_grid : np.ndarray
        The grids the contribution functions are tabulated on; the DEM uses
        the same temperature grid.
    vel_grid : u.Quantity
        Velocity bin centres, evenly spaced.
    integration_axis : str
        ``"x"``, ``"y"`` or ``"z"``.
    precision : type
        np.float32 or np.float64.

    Returns
    -------
    lines : dict
        A copy of *goft* whose entries also hold ``"g"``, the contribution
        function on the DEM, ``"wl_grid"`` and ``"si"``, the specific
        intensity ``(rows, columns, wavelength)``.
    dem_map : np.ndarray
        As :func:`compute_dem` returns it.
    em_tv : np.ndarray
        As :func:`build_em_tv` returns it.
    """
    logN_cube = np.log10(electron_density, where=electron_density > 0.0,
                         out=np.zeros_like(electron_density)).astype(precision)
    logT_cube = np.log10(temperature, where=temperature > 0.0,
                         out=np.zeros_like(temperature)).astype(precision)

    dem_map, avg_ne_map = compute_dem(logT_cube, logN_cube, dh_cm, logT_grid, integration_axis)

    lines = {name: dict(info) for name, info in goft.items()}
    interpolate_g_on_dem(lines, avg_ne_map, logT_grid, logN_grid, logT_grid, precision)

    ne_sq_dh = ((10.0 ** logN_cube.astype(np.float64)) ** 2
                * along_line_of_sight(dh_cm, integration_axis))
    em_tv = build_em_tv(logT_cube, los_velocity, logT_grid, vel_grid, ne_sq_dh, integration_axis)

    synthesise_spectra(lines, em_tv, vel_grid, logT_grid)
    return lines, dem_map, em_tv


def _world_at(coords: u.Quantity, crpix: float) -> float:
    """
    The value an even grid *coords* has at 1-based pixel *crpix*, as a plain number.

    A reference pixel at the middle of an axis falls between two pixels when
    there is an even number of them, so the reference value has to be read
    off the grid there rather than taken from the pixel below.
    """
    if coords.size == 1:
        return coords[0].value
    step = (coords[1] - coords[0]).value
    return coords[0].value + (crpix - 1) * step


def create_line_cube(
    line_name: str,
    line_data: dict,
    spatial_cube: NDCube,
    intensity_unit: u.Unit,
    integration_axis: str = "z",
) -> NDCube:
    """
    Create an NDCube for a single spectral line using spatial coordinates from existing cube.
    
    Parameters
    ----------
    line_name : str
        Name of the spectral line.
    line_data : dict
        Dictionary containing line data with 'si', 'wl_grid', 'wl0'.
    spatial_cube : NDCube
        Reference cube for spatial coordinates.
    intensity_unit : u.Unit
        Unit for the intensity data.
    integration_axis : str
        Axis along which integration was performed ("x", "y", or "z").

    Returns
    -------
    NDCube
        Cube with proper WCS and metadata, indexed ``[row, column, wavelength]``
        like an image: the first axis is the vertical direction of the scene
        and the second the horizontal.  Summing over the last axis gives an
        array that plots the right way up, and slicing out the celestial WCS
        gives one a SunPy map accepts directly.
    """
    # An axis whose cells differ in size has no one CDELT. Only the line of
    # sight may be such an axis, and that is the one integrated out here.
    nonuniform = (spatial_cube.meta or {}).get("nonuniform_axes", [])
    stretched = [axis for axis in AXES
                 if axis != integration_axis and axis in nonuniform]
    if stretched:
        raise ValueError(
            f"The {', '.join(stretched)} axis of the atmosphere is not evenly "
            f"spaced, so it cannot be an image axis of a view along "
            f"{integration_axis}. Only the line of sight may be stretched.")

    # The simulation cubes are (z, y, x), so integrating one axis out leaves
    # 'si' already in (row, column, wavelength) order for every view.
    cube_data = line_data["si"]

    # The cell size of each axis comes from the reference cube's own WCS, so
    # that an axis a single cell wide has one too.
    reference_wcs = spatial_cube.wcs.wcs
    cell_size = [(reference_wcs.cdelt[i] * u.Unit(reference_wcs.cunit[i])).to_value(u.Mm)
                 for i in range(3)]

    # The WCS below carries a single linear CDELT taken from the first
    # wavelength step, so the grid has to be uniform for that to describe it.
    # Checked here as well as in synthesise_spectra because this is a public
    # entry point: the DEM and VDEM routes call it directly.
    require_uniform_grid(line_data["wl_grid"], "wl_grid")

    # Get spatial coordinate information from the reference cube,
    # whose array axes are (z, y, x)
    if integration_axis == "x":
        # Integration along X -> data shape (nz, ny, n_lambda): rows are Z, columns are Y
        nz, ny, nl = cube_data.shape
        y_coords = spatial_cube.axis_world_coords(1)[0]  # Y coordinates
        z_coords = spatial_cube.axis_world_coords(0)[0]  # Z coordinates

        spatial_axes = ['WAVE', 'SOLY', 'SOLZ']  # Wavelength, Y, Z
        spatial_units = ['cm', 'Mm', 'Mm']
        spatial_cdelt = [
            np.diff(line_data["wl_grid"].to(u.cm).value)[0],
            cell_size[1],
            cell_size[2],
        ]
        spatial_crpix = [(nl + 1) / 2, (ny + 1) / 2, 1]  # Wavelength centered, Y centered, Z at first pixel
        spatial_crval = [
            _world_at(line_data["wl_grid"].to(u.cm), spatial_crpix[0]),
            _world_at(y_coords.to(u.Mm), spatial_crpix[1]),
            z_coords[0].to(u.Mm).value  # Z starts where original cube starts
        ]

    elif integration_axis == "y":
        # Integration along Y -> data shape (nz, nx, n_lambda): rows are Z, columns are X
        nz, nx, nl = cube_data.shape
        x_coords = spatial_cube.axis_world_coords(2)[0]  # X coordinates
        z_coords = spatial_cube.axis_world_coords(0)[0]  # Z coordinates

        spatial_axes = ['WAVE', 'SOLX', 'SOLZ']  # Wavelength, X, Z
        spatial_units = ['cm', 'Mm', 'Mm']
        spatial_cdelt = [
            np.diff(line_data["wl_grid"].to(u.cm).value)[0],
            cell_size[0],
            cell_size[2],
        ]
        spatial_crpix = [(nl + 1) / 2, (nx + 1) / 2, 1]  # Wavelength centered, X centered, Z at first pixel
        spatial_crval = [
            _world_at(line_data["wl_grid"].to(u.cm), spatial_crpix[0]),
            _world_at(x_coords.to(u.Mm), spatial_crpix[1]),
            z_coords[0].to(u.Mm).value  # Z starts where original cube starts
        ]

    else:  # integration_axis == "z"
        # Integration along Z -> data shape (ny, nx, n_lambda): rows are Y, columns are X
        ny, nx, nl = cube_data.shape
        x_coords = spatial_cube.axis_world_coords(2)[0]  # X coordinates
        y_coords = spatial_cube.axis_world_coords(1)[0]  # Y coordinates

        spatial_axes = ['WAVE', 'SOLX', 'SOLY']  # Wavelength, X, Y
        spatial_units = ['cm', 'Mm', 'Mm']
        spatial_cdelt = [
            np.diff(line_data["wl_grid"].to(u.cm).value)[0],
            cell_size[0],
            cell_size[1],
        ]
        spatial_crpix = [(nl + 1) / 2, (nx + 1) / 2, (ny + 1) / 2]  # All centered
        spatial_crval = [
            _world_at(line_data["wl_grid"].to(u.cm), spatial_crpix[0]),
            _world_at(x_coords.to(u.Mm), spatial_crpix[1]),
            _world_at(y_coords.to(u.Mm), spatial_crpix[2]),
        ]

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
            "integration_axis": integration_axis,
            "velocity_convention": VELOCITY_CONVENTION,
            "spatial_reference": spatial_cube.meta if hasattr(spatial_cube, 'meta') else None
        }
    )



##############################################################################
# ---------------------------------------------------------------------------
#                 M A I N   W O R K F L O W
# ---------------------------------------------------------------------------
##############################################################################

# The options that say where MURaM's files are and how they are laid out. An
# atmosphere file carries all of this itself, so giving both is a
# contradiction rather than a choice.
MURAM_LAYOUT_OPTIONS = ("data_dir", "temp_file", "rho_file", "vx_file", "vy_file",
                        "vz_file", "cube_shape", "voxel_dx", "voxel_dy", "voxel_dz")


class _NotedOption(argparse.Action):
    """Stores the value and records that the option was given, at its default or not."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        given = getattr(namespace, "given_options", None)
        if given is None:
            given = set()
            namespace.given_options = given
        given.add(self.dest)


def build_parser() -> argparse.ArgumentParser:
    """The command line options of synthesise-spectra."""
    parser = argparse.ArgumentParser(
        description="Synthesise solar spectra from 3D MHD simulation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output paths
    parser.add_argument("--atmosphere", type=str, default=None,
                       help="An ECLIPSE atmosphere file (HDF5) to synthesise from, "
                            "written by eclipse-atmosphere or by your own code. "
                            "It carries the cube shape and cell sizes, so it "
                            "replaces the MURaM file options.")
    parser.add_argument("--data-dir", type=str, default="data/atmosphere",
                       action=_NotedOption,
                       help="Directory containing simulation data")
    parser.add_argument("--output-dir", type=str, default="./run/input",
                       help="Output directory for results")
    parser.add_argument("--output-name", type=str, default="synthesised_spectra.pkl",
                       help="Output filename")
    
    # Line / abundance specification (fiasco)
    parser.add_argument("--lines", nargs="+", required=True,
                       help="Line specifications (e.g. Fe12_195.1190 Fe09_171.073)")
    parser.add_argument("--abundance", type=str, default="sun_coronal_2021_chianti",
                       help="CHIANTI abundance dataset name for fiasco")
    parser.add_argument("--n-workers", type=int, default=0,
                       help="Number of parallel workers for fiasco G(T,N) "
                            "computation (0 = all CPUs, default: 0)")
    parser.add_argument("--hdf5-dbase-root", type=str, default=None,
                       help="CHIANTI HDF5 database to use, overriding the one "
                            "in ~/.fiasco/fiascorc. Use this to run against a "
                            "second CHIANTI version without changing the "
                            "default for other work.")
    
    # Simulation files
    parser.add_argument("--temp-file", type=str, default="temp/eosT.0270000",
                       action=_NotedOption,
                       help="Temperature file relative to data-dir")
    parser.add_argument("--rho-file", type=str, default="rho/result_prim_0.0270000",
                       action=_NotedOption,
                       help="Density file relative to data-dir")
    parser.add_argument("--vx-file", type=str, default="vx/result_prim_1.0270000",
                       action=_NotedOption,
                       help="Velocity x file relative to data-dir")
    parser.add_argument("--vy-file", type=str, default="vy/result_prim_3.0270000",
                       action=_NotedOption,
                       help="Velocity y file relative to data-dir")
    parser.add_argument("--vz-file", type=str, default="vz/result_prim_2.0270000",
                       action=_NotedOption,
                       help="Velocity z file relative to data-dir")

    # Grid parameters
    parser.add_argument("--cube-shape", nargs=3, type=int, default=[512, 768, 256],
                       action=_NotedOption,
                       help="Cube dimensions in the file's storage order (nx nz ny)")
    parser.add_argument("--voxel-dx", type=str, default="0.192 Mm",
                       action=_NotedOption,
                       help="Voxel size in x (e.g. '0.192 Mm')")
    parser.add_argument("--voxel-dy", type=str, default="0.192 Mm",
                       action=_NotedOption,
                       help="Voxel size in y (e.g. '0.192 Mm')")
    parser.add_argument("--voxel-dz", type=str, default="0.064 Mm",
                       action=_NotedOption,
                       help="Voxel size in z (e.g. '0.064 Mm')")
    
    # Integration direction
    parser.add_argument("--integration-axis", choices=["x", "y", "z"], default="z",
                       help="Axis along which to integrate (x, y, or z)")
    
    # Cropping parameters (in Heliocentric coordinates)
    parser.add_argument("--crop-x", nargs=2, type=str, default=None,
                       help="Crop in x direction: x_min x_max (e.g. '-50 Mm' '50 Mm')")
    parser.add_argument("--crop-y", nargs=2, type=str, default=None,
                       help="Crop in y direction: y_min y_max (e.g. '-50 Mm' '50 Mm')")
    parser.add_argument("--crop-z", nargs=2, type=str, default=None,
                       help="Crop in z direction: z_min z_max (e.g. '0 Mm' '20 Mm')")
    
    # Velocity grid
    parser.add_argument("--vel-res", type=str, default="5.0 km/s",
                       help="Velocity resolution (e.g. '5.0 km/s')")
    parser.add_argument("--vel-lim", type=str, default="300.0 km/s",
                       help="Velocity limit +/- (e.g. '300.0 km/s')")
    
    # Processing options
    parser.add_argument("--downsample", type=int, default=1,
                       help="Downsampling factor (1 = no downsampling)")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64",
                       help="Numerical precision")
    parser.add_argument("--mass-per-electron", "--mean-mol-wt",
                       dest="mass_per_electron", type=float, default=None,
                       help="Mass of the plasma per free electron, in atomic mass "
                            "units, which turns a mass density into an electron "
                            "density. By default it is worked out from --abundance "
                            "for a fully ionised plasma, about 1.16 for coronal "
                            "abundances. --mean-mol-wt is the old name; ECLIPSE "
                            "0.8.0 and earlier used 1.29, the value for a neutral gas. "
                            "Not used when the atmosphere gives an electron density.")
    
    return parser


def parse_arguments(argv=None):
    """Parse command line arguments for spectrum synthesis."""
    return build_parser().parse_args(argv)


def check_atmosphere_options(args) -> None:
    """
    Refuse --atmosphere alongside options it makes meaningless.

    The MURaM layout options describe files the atmosphere route never
    reads, so one given explicitly would be ignored without a word.
    """
    if not args.atmosphere:
        return
    # The parser notes every layout option that appeared on the command
    # line, so one typed at its default value is caught too.
    noted = getattr(args, "given_options", ())
    given = [name for name in MURAM_LAYOUT_OPTIONS if name in noted]
    if given:
        flags = ", ".join("--" + name.replace("_", "-") for name in given)
        raise ValueError(
            f"--atmosphere carries the cube shape, cell sizes and data itself, "
            f"so {flags} would not be used. Give one or the other.")


def load_atmosphere_file(
    path: str | Path,
    integration_axis: str,
    downsample: int | bool = False,
    crop_x=None, crop_y=None, crop_z=None,
) -> Atmosphere:
    """
    Read an atmosphere file with the velocity a view along *integration_axis* needs.

    Downsampling and cropping are applied here, in the atmosphere's own
    coordinates, and the two axes that become the image are checked to be
    evenly spaced, since the image WCS can only describe an even grid. The
    line of sight may be stretched: the emission measure uses the true size
    of each cell along it.
    """
    atmosphere = read_atmosphere(path, velocities=(integration_axis,))
    if downsample:
        atmosphere = atmosphere.downsampled(downsample)
    if crop_x or crop_y or crop_z:
        atmosphere = atmosphere.cropped(x=crop_x, y=crop_y, z=crop_z)
    for axis in AXES:
        if axis != integration_axis and not atmosphere.is_uniform(axis):
            raise ValueError(
                f"The {axis} axis of {path} is not evenly spaced, and with the "
                f"line of sight along {integration_axis} it would become an "
                f"image axis, whose coordinates must be even. Only the axis "
                f"along the line of sight may be stretched; resample the "
                f"others onto an even grid first.")
    return atmosphere


def resolve_mass_per_electron(args) -> Tuple[float, str]:
    """The mass per free electron to use, in atomic mass units, and where it came from."""
    if args.mass_per_electron is not None:
        try:
            value = require_mass_per_electron(args.mass_per_electron)
        except ValueError as error:
            raise ValueError(f"--mass-per-electron: {error}") from None
        return value, "given on the command line"
    value = mass_per_electron(args.abundance, getattr(args, "hdf5_dbase_root", None))
    return value, f"fully ionised plasma with {args.abundance} abundances"


def along_line_of_sight(values, integration_axis: str) -> np.ndarray:
    """
    *values*, one per cell along the line of sight, shaped to broadcast over a (z, y, x) cube.

    A single value comes back as it is, so one cell size applies everywhere.
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


def main(args=None) -> None:
    """
    Main workflow for synthesising solar spectra from 3D MHD simulations.

    Synthesises one snapshot, from an atmosphere file (--atmosphere) or from
    MURaM's own files. A time series is observed by the instrument run
    instead, from a series of atmosphere files (see euvst_response.raster).
    
    Parameters
    ----------
    args : argparse.Namespace, optional
        Command line arguments. If None, will parse from sys.argv.
    """
    if args is None:
        args = parse_arguments()
    
    # ---------------- Configuration from arguments -----------------
    precision = np.float32 if args.precision == "float32" else np.float64
    if args.downsample < 1:
        raise ValueError(f"--downsample must be 1 or more, got {args.downsample}.")
    downsample = args.downsample if args.downsample > 1 else False
    vel_res = u.Quantity(args.vel_res)
    vel_lim = u.Quantity(args.vel_lim)
    # Voxel sizes of the simulation files. load_cube scales these itself when
    # it downsamples, so they are passed to it as given.
    file_voxel_dz = u.Quantity(args.voxel_dz)
    file_voxel_dx = u.Quantity(args.voxel_dx)
    file_voxel_dy = u.Quantity(args.voxel_dy)

    # Voxel sizes of the cubes as synthesised, for the path length along the
    # line of sight and the saved metadata.
    voxel_dz = file_voxel_dz * (downsample or 1)
    voxel_dx = file_voxel_dx * (downsample or 1)
    voxel_dy = file_voxel_dy * (downsample or 1)

    intensity_unit = u.erg/u.s/u.cm**2/u.sr/u.cm
    
    print_mem = lambda: f"{psutil.virtual_memory().used/1e9:.2f}/" \
                        f"{psutil.virtual_memory().total/1e9:.2f} GB"

    base_dir = Path(args.data_dir)
    integration_axis = args.integration_axis.lower()
    check_atmosphere_options(args)

    # What the common processing below needs from whichever route reads the
    # atmosphere. Only an atmosphere file can give the electron density
    # directly or a different size for every cell along the line of sight;
    # the MURaM routes give a mass density and one cell size.
    ne_values = None
    los_thickness = None
    atmosphere_metadata = None

    if args.atmosphere:
        # Static mode from an atmosphere file, which brings its own layout
        dynamic_mode_metadata = {"enabled": False}

        print("STATIC MODE - Synthesis from an atmosphere file")
        print(f"  Atmosphere: {args.atmosphere}")
        print(f"  Integration axis: {integration_axis}")
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
        atmosphere = load_atmosphere_file(
            args.atmosphere, integration_axis, downsample=downsample,
            crop_x=args.crop_x, crop_y=args.crop_y, crop_z=args.crop_z)
        print(atmosphere.describe())

        # The file may hold float32 in any units; the run works in its own
        # precision, as the MURaM route does from the moment it reads its
        # files, and the processing below takes the cubes' values as K and
        # cm/s, so they are converted here.
        temp_cube = atmosphere.to_ndcube(
            atmosphere.temperature.astype(precision).to(u.K))
        vel_cube = atmosphere.to_ndcube(
            atmosphere.velocity(integration_axis).astype(precision).to(u.cm / u.s))
        rho = None
        if atmosphere.mass_density is not None:
            rho = atmosphere.mass_density.astype(precision)
        if atmosphere.electron_density is not None:
            ne_values = atmosphere.electron_density.astype(precision).to_value(u.cm**-3)
        los_thickness = atmosphere.cell_thickness(integration_axis)
        reference_cube = temp_cube

        # The cell sizes stand in for the MURaM voxel sizes in what is saved;
        # a stretched axis has no single size.
        voxel_dx, voxel_dy, voxel_dz = (
            atmosphere.spacing(axis) if atmosphere.is_uniform(axis) else None
            for axis in AXES)
        atmosphere_metadata = {
            "path": str(Path(args.atmosphere).resolve()),
            "source": atmosphere.source,
            "time": atmosphere.time,
            "shape": atmosphere.shape,
            "nonuniform_axes": atmosphere.nonuniform_axes(),
            "electron_density_given": atmosphere.electron_density is not None,
        }

    else:
        # Static mode from MURaM files
        dynamic_mode_metadata = {"enabled": False}

        files = {
            "T": args.temp_file,
            "rho": args.rho_file,
        }
        
        # Determine velocity file based on integration axis
        if integration_axis == "x":
            files["vel"] = args.vx_file
            voxel_dh = voxel_dx
        elif integration_axis == "y":
            files["vel"] = args.vy_file
            voxel_dh = voxel_dy
        else:  # "z"
            files["vel"] = args.vz_file
            voxel_dh = voxel_dz
        
        paths = {k: base_dir / fname for k, fname in files.items()}
        
        # Validate input files exist
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} file not found: {path}")
        
        print(f"STATIC MODE - Single timestep synthesis")
        print(f"  Data directory: {base_dir}")
        print(f"  Cube shape: {args.cube_shape}")
        print(f"  Voxel sizes: {voxel_dx:.3f} x {voxel_dy:.3f} x {voxel_dz:.3f}")
        print(f"  Integration axis: {integration_axis}")
        print(f"  Velocity grid: +/-{vel_lim:.1f} at {vel_res:.1f} resolution")
        print(f"  Precision: {precision}")
        if downsample:
            print(f"  Downsampling: {downsample}x")
        print(f"  Lines: {args.lines}")
        print(f"  Abundance: {args.abundance}")
        if args.crop_x or args.crop_y or args.crop_z:
            print(f"  Cropping: X={args.crop_x}, Y={args.crop_y}, Z={args.crop_z}")
        print()
        
        # Load simulation data as NDCubes
        print(f"Loading cubes ({print_mem()})")
        temp_cube = load_cube(
            paths["T"], shape=tuple(args.cube_shape), unit=u.K, 
            downsample=downsample, precision=precision,
            voxel_dx=file_voxel_dx, voxel_dy=file_voxel_dy,
            voxel_dz=file_voxel_dz, create_ndcube=True
        )
        rho_cube = load_cube(
            paths["rho"], shape=tuple(args.cube_shape), unit=u.g/u.cm**3, 
            downsample=downsample, precision=precision,
            voxel_dx=file_voxel_dx, voxel_dy=file_voxel_dy,
            voxel_dz=file_voxel_dz, create_ndcube=True
        )
        vel_cube = load_cube(
            paths["vel"], shape=tuple(args.cube_shape), unit=u.cm/u.s, 
            downsample=downsample, precision=precision,
            voxel_dx=file_voxel_dx, voxel_dy=file_voxel_dy,
            voxel_dz=file_voxel_dz, create_ndcube=True
        )

        # Apply cropping if requested
        if args.crop_x or args.crop_y or args.crop_z:
            print(f"Applying cropping ({print_mem()})")
            temp_cube, rho_cube, vel_cube = apply_cube_cropping(
                temp_cube, rho_cube, vel_cube,
                args.crop_x, args.crop_y, args.crop_z
            )
            print(f"Cropped cubes to shape: {temp_cube.data.shape}")

        reference_cube = temp_cube
        rho = u.Quantity(rho_cube.data, rho_cube.unit)

    # ---------------- Common processing (both modes) -----------------
    
    # Build velocity grid
    vel_grid = np.arange(
        -vel_lim.to(u.cm / u.s).value,
        vel_lim.to(u.cm / u.s).value + vel_res.to(u.cm / u.s).value,
        vel_res.to(u.cm / u.s).value
    ) * (u.cm / u.s)

    # The electron density: the atmosphere's own where it gives one, otherwise
    # the mass density over the mass per free electron.
    if ne_values is None:
        mass_per_electron_amu, mass_per_electron_source = resolve_mass_per_electron(args)
        print(f"Electron density from the mass density with "
              f"{mass_per_electron_amu:.4f} u per electron ({mass_per_electron_source})")
        ne_values = (rho / (mass_per_electron_amu * const.u)).to_value(u.cm**-3)
    else:
        mass_per_electron_amu = None
        mass_per_electron_source = "not needed: the atmosphere gives the electron density"
        print("Electron density taken from the atmosphere")

    # The velocity files hold the velocity along each axis; the Doppler shift
    # needs the velocity away from the observer.
    vel_data = line_of_sight_velocity(vel_cube.data, integration_axis)

    # ---------------- Compute contribution functions (fiasco) ---------
    print(f"Computing contribution functions via fiasco ({print_mem()})")
    goft, logT_goft, logN_grid = compute_goft_fiasco(
        args.lines, abundance=args.abundance, precision=precision,
        n_workers=args.n_workers,
        hdf5_dbase_root=getattr(args, "hdf5_dbase_root", None),
    )

    # Record the database the contribution functions actually came from, not
    # the request, so that a run which did not choose one is still traceable
    # to the atomic data it used.
    goft_dbase_root = (
        next(iter(goft.values()))["hdf5_dbase_root"] if goft else None
    )
    print(f"  CHIANTI database: {goft_dbase_root}")

    # Use the GOFT temperature grid as our DEM temperature grid
    logT_grid = logT_goft

    # The size of each cell along the line of sight. MURaM's files have one
    # size; an atmosphere file may give every cell its own.
    if los_thickness is None:
        los_thickness = {"x": voxel_dx, "y": voxel_dy, "z": voxel_dz}[integration_axis]
    dh_cm = los_thickness.to_value(u.cm)

    # ---------------- DEM, EM(T,v) and spectra -----------------
    print(f"Calculating the DEM, the emission measure in (T,v) and the spectra ({print_mem()})")
    goft, dem_map, em_tv = synthesise_cubes(
        temp_cube.data, ne_values, vel_data, dh_cm, goft, logT_grid, logN_grid,
        vel_grid, integration_axis, precision)

    # ---------------- Create output cubes -----------------
    print(f"Creating output cubes ({print_mem()})")
    line_cubes = {}
    for name, info in goft.items():
        line_cubes[name] = create_line_cube(
            name, info, reference_cube, intensity_unit, integration_axis
        )

    print(f"Built {len(line_cubes)} line cubes")

    # ---------------- Save results -----------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_name
    
    # Save main results
    results_data = {
        "line_cubes": line_cubes,
        "dem_map": dem_map,
        "em_tv": em_tv,
        "logT_grid": logT_grid,
        "vel_grid": vel_grid,
        "logN_grid": logN_grid,
        "goft": goft,
        "voxel_sizes": {"dx": voxel_dx, "dy": voxel_dy, "dz": voxel_dz},
        "dynamic_mode": dynamic_mode_metadata,
        "atmosphere": atmosphere_metadata,
        "config": {
            "precision": precision.__name__,
            "downsample": downsample,
            "vel_res": vel_res,
            "vel_lim": vel_lim,
            "mass_per_electron": mass_per_electron_amu,
            "mass_per_electron_source": mass_per_electron_source,
            "intensity_unit": str(intensity_unit),
            "atmosphere": args.atmosphere,
            "cube_shape": None if args.atmosphere else args.cube_shape,
            "data_dir": None if args.atmosphere else str(base_dir),
            "lines": args.lines,
            "abundance": args.abundance,
            "hdf5_dbase_root": goft_dbase_root,
            "integration_axis": integration_axis,
            "velocity_convention": VELOCITY_CONVENTION,
            "observer_side": OBSERVER_SIDE[integration_axis],
            "crop_params": {
                "crop_x": args.crop_x,
                "crop_y": args.crop_y,
                "crop_z": args.crop_z
            }
        }
    }
    
    with open(output_file, "wb") as f:
        dill.dump(results_data, f)
    
    print(f"Saved results to {output_file} ({os.path.getsize(output_file) / 1e6:.2f} MB)")
    print("Synthesis complete!")

if __name__ == "__main__":
    main()