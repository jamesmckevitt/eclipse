import os
import sys
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
import dill
from ndcube import NDCube
from astropy.wcs import WCS
from .utils import (angle_to_distance, element_data, require_uniform_grid,
                    require_downsample_divides, velocity_centers_to_edges, velocity_grid,
                    view_axis_and_side, view_name, OBSERVER_SIDE, VELOCITY_CONVENTION)
from .synthesis_file import write_line_cubes
from .continuum import compute_continuum_fiasco, continuum_spectra, continuum_windows
from .atmosphere import (AXES, NUMPY_AXIS, Atmosphere, _offer_database_build,
                         mass_per_electron, read_atmosphere, require_mass_per_electron)

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
    Give a cube of a simulation's cells coordinates on the Sun, from the cells' sizes.

    x and y are centred on zero, and z starts from zero.

    Parameters
    ----------
    data : np.ndarray or u.Quantity
        The cube, shaped (nz, ny, nx), so that ``data[k]`` is a horizontal
        slice indexed ``[y, x]``.
    voxel_dx, voxel_dy, voxel_dz : u.Quantity
        The size of a cell along each axis.

    Returns
    -------
    NDCube
        The cube, with its coordinates in Mm.
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

    # A plain array, as the docstring allows, has no unit.
    return NDCube(np.asarray(getattr(data, "value", data)),
                  wcs=wcs,
                  unit=getattr(data, "unit", None))


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


def discover_timesteps(
    time_dir: Path,
    time_filename: str,
) -> Dict[str, float]:
    """
    Discover all available timesteps and their simulation times.
    
    Parameters
    ----------
    time_dir : Path
        Directory containing header files.
    time_filename : str
        Filename prefix before the timestep suffix (e.g., "Header").
        
    Returns
    -------
    dict
        Mapping of timestep suffix to simulation time in seconds.
        E.g., {"0270000": 26729.535, "0280000": 27571.395, ...}
    """
    if not time_dir.is_dir():
        raise FileNotFoundError(f"Time directory not found: {time_dir}")
    
    timesteps = {}
    
    for file_path in sorted(time_dir.iterdir()):
        if file_path.is_file() and file_path.name.startswith(time_filename):
            # Extract suffix after the filename prefix
            suffix = file_path.name[len(time_filename):]
            if suffix.startswith('.'):
                suffix = suffix[1:]  # Remove leading dot
            
            try:
                sim_time = read_timestep_time(file_path)
                timesteps[suffix] = sim_time
            except Exception as e:
                warnings.warn(f"Could not read time from {file_path}: {e}")
    
    if not timesteps:
        raise ValueError(f"No valid timestep files found in {time_dir} with prefix '{time_filename}'")
    
    return timesteps


def get_file_for_timestep(
    directory: Path,
    filename: str,
    suffix: str,
) -> Path:
    """
    Get the file path for a specific timestep.
    
    Parameters
    ----------
    directory : Path
        Directory containing the data files.
    filename : str
        Filename prefix before the suffix (e.g., "eosT").
    suffix : str
        Timestep suffix (e.g., "0270000").
        
    Returns
    -------
    Path
        Full path to the file.
    """
    file_path = directory / f"{filename}.{suffix}"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path


def compute_slice_timestep_mapping_mhd(
    nx_mhd: int,
    voxel_dx: u.Quantity,
    slit_width: u.Quantity,
    slit_rest_time: u.Quantity,
    timestep_times: Dict[str, float],
    crop_x: Optional[Tuple[u.Quantity, u.Quantity]] = None,
) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Compute which timestep suffix to use for each MHD X-slice.
    
    The spectrometer scans from right to left (high X to low X).
    Each slit position covers a physical width (slit_width converted to Mm).
    Determine which slit position each MHD slice belongs to, then find
    the appropriate timestep based on observation time.
    
    If crop_x is specified, the observation timing is calculated as if the
    scan starts at the right edge of the crop region. Slices to the right
    of the crop region are filled with the first timestep.
    
    To avoid error accumulation, calculate the observation time for each
    MHD slice based on its absolute physical position.
    
    Parameters
    ----------
    nx_mhd : int
        Number of MHD X-slices.
    voxel_dx : u.Quantity
        MHD voxel size in X direction (physical units, e.g., Mm).
    slit_width : u.Quantity
        Slit width in angular units (e.g., arcsec).
    slit_rest_time : u.Quantity
        Slit rest time per position.
    timestep_times : dict
        Mapping of timestep suffix to simulation time in seconds.
    crop_x : tuple of u.Quantity, optional
        If provided, (x_min, x_max) crop boundaries. Observation timing starts
        at x_max (right edge of crop). Slices outside crop on the right use
        the first timestep.
        
    Returns
    -------
    slice_mapping : list
        List of timestep suffixes, one per MHD X-slice.
    grouped : dict
        Dictionary mapping each unique timestep suffix to list of MHD slice indices.
    """
    rest_time_sec = slit_rest_time.to_value(u.s)
    
    # Convert slit width from angular to physical distance
    slit_physical = angle_to_distance(slit_width).to(u.Mm)
    voxel_physical = voxel_dx.to(u.Mm)
    
    # Sort timesteps by simulation time
    sorted_timesteps = sorted(timestep_times.items(), key=lambda x: x[1])
    suffixes = [s for s, t in sorted_timesteps]
    times = np.array([t for s, t in sorted_timesteps])
    first_suffix = suffixes[0]
    
    # Check if observation time range extends beyond available MHD time range
    t0 = times[0]
    t_final = times[-1]
    mhd_duration = t_final - t0
    
    # Total physical extent of the full domain
    total_extent = nx_mhd * voxel_physical
    
    # Determine the effective scan start position (right edge)
    # If crop_x is specified, scan starts at the crop boundary
    if crop_x is not None:
        x_min_crop = crop_x[0].to(u.Mm)
        x_max_crop = crop_x[1].to(u.Mm)
        
        # The domain is centered at 0, so convert to absolute position
        # WCS has X centered, so x=0 is at nx/2
        domain_center = total_extent / 2
        scan_start_position = domain_center + x_max_crop  # Right edge of crop in absolute coords
        scan_end_position = domain_center + x_min_crop    # Left edge of crop in absolute coords
        
        # Calculate observation duration for just the cropped region
        crop_extent = x_max_crop - x_min_crop
        n_slit_positions_crop = int(np.ceil((crop_extent / slit_physical).decompose().value))
        observation_duration = n_slit_positions_crop * rest_time_sec
        
        print(f"  Crop region: X = [{x_min_crop:.3f}, {x_max_crop:.3f}]")
        print(f"  Observation starts at X = {x_max_crop:.3f} (right edge of crop)")
    else:
        scan_start_position = total_extent  # Right edge of full domain
        scan_end_position = 0 * u.Mm
        n_slit_positions_crop = int(np.ceil((total_extent / slit_physical).decompose().value))
        observation_duration = n_slit_positions_crop * rest_time_sec
    
    if observation_duration > mhd_duration:
        print(f"  WARNING: Observation duration ({observation_duration:.1f} s) exceeds MHD time range ({mhd_duration:.1f} s)")
        print(f"  Slices observed after t={t_final:.1f} s will use the last available timestep")
    
    slice_mapping = []
    
    # For each MHD X-slice, calculate which slit position it belongs to
    # Scanning right to left: scan_start_position is observed first (at t=t0)
    for mhd_slice_idx in range(nx_mhd):
        # Physical position of this slice (center of voxel) in absolute coords
        x_physical = (mhd_slice_idx + 0.5) * voxel_physical
        
        # Distance from the scan start position (right edge of observation region)
        distance_from_scan_start = scan_start_position - x_physical
        
        if distance_from_scan_start < 0:
            # This slice is to the right of the scan start (outside crop on right)
            # Use the first timestep for these slices
            slice_mapping.append(first_suffix)
            continue
        
        # Which slit position does this belong to?
        # slit_position = 0 is at scan_start_position
        slit_position = int(np.floor((distance_from_scan_start / slit_physical).decompose().value))
        
        # Observation time for this slit position
        observation_time = t0 + slit_position * rest_time_sec
        
        # Find the latest timestep that doesn't exceed observation_time
        idx = np.searchsorted(times, observation_time, side='right') - 1
        # Clamp to valid range (0 to len(times)-1)
        idx = max(0, min(idx, len(times) - 1))
        slice_mapping.append(suffixes[idx])
    
    # Group slices by timestep for efficient processing
    grouped = {}
    for slice_idx, suffix in enumerate(slice_mapping):
        if suffix not in grouped:
            grouped[suffix] = []
        grouped[suffix].append(slice_idx)
    
    # Print statistics
    if crop_x is not None:
        print(f"  Cropped region extent: {crop_extent:.3f}")
    else:
        print(f"  Physical domain extent: {total_extent:.3f}")
    print(f"  Slit physical width: {slit_physical:.3f}")
    print(f"  Number of slit positions: {n_slit_positions_crop}")
    print(f"  Total raster time: {observation_duration:.1f} s")
    
    return slice_mapping, grouped


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
    
    # A crop that keeps one cell along an axis keeps the axis, which the
    # synthesis needs all three of.
    temp_cube = temp_cube.crop(point1, point2, keepdims=True)
    rho_cube = rho_cube.crop(point1, point2, keepdims=True)
    vel_cube = vel_cube.crop(point1, point2, keepdims=True)
    
    return temp_cube, rho_cube, vel_cube


def build_composite_cubes_mhd(
    base_dir: Path,
    temp_dir: str,
    temp_filename: str,
    rho_dir: str,
    rho_filename: str,
    vel_dir: str,
    vel_filename: str,
    slice_mapping: List[str],
    grouped_slices: Dict[str, List[int]],
    cube_shape: Tuple[int, int, int],
    voxel_dx: u.Quantity,
    voxel_dy: u.Quantity,
    voxel_dz: u.Quantity,
    downsample: int | bool,
    precision: type,
) -> Tuple[NDCube, NDCube, NDCube]:
    """
    Build composite temp, rho, vel cubes from multiple timesteps at MHD resolution.
    
    Each MHD X-slice comes from the appropriate timestep per slice_mapping.
    No spatial rebinning is performed - output is at full MHD resolution.
    
    Parameters
    ----------
    base_dir : Path
        Base directory for atmosphere data.
    temp_dir, temp_filename : str
        Directory and filename prefix for temperature files.
    rho_dir, rho_filename : str
        Directory and filename prefix for density files.
    vel_dir, vel_filename : str
        Directory and filename prefix for velocity files.
    slice_mapping : list
        Timestep suffix for each MHD X-slice.
    grouped_slices : dict
        Slices grouped by timestep for efficient processing.
    cube_shape : tuple
        Original cube dimensions in the file's storage order.
    voxel_dx, voxel_dy, voxel_dz : u.Quantity
        Voxel sizes of the files, at full resolution; load_cube applies the
        downsampling to them.
    downsample : int or bool
        Downsampling factor.
    precision : type
        Numerical precision.
        
    Returns
    -------
    tuple
        (temp_composite, rho_composite, vel_composite) NDCubes at MHD resolution.
    """
    nx_mhd = len(slice_mapping)
    
    # Load first timestep to determine dimensions and get reference WCS
    first_suffix = list(grouped_slices.keys())[0]
    temp_file = get_file_for_timestep(base_dir / temp_dir, temp_filename, first_suffix)
    temp_cube_ref = load_cube(
        temp_file, shape=cube_shape, unit=u.K,
        downsample=downsample, precision=precision,
        voxel_dx=voxel_dx, voxel_dy=voxel_dy, voxel_dz=voxel_dz,
        create_ndcube=True
    )
    
    nz, ny, nx = temp_cube_ref.data.shape
    reference_wcs = temp_cube_ref.wcs

    # Verify dimensions match slice mapping
    if nx != nx_mhd:
        raise ValueError(f"Cube X dimension ({nx}) doesn't match slice mapping ({nx_mhd})")

    # Initialise composite arrays
    temp_composite = np.zeros((nz, ny, nx), dtype=precision)
    rho_composite = np.zeros((nz, ny, nx), dtype=precision)
    vel_composite = np.zeros((nz, ny, nx), dtype=precision)
    
    # Process each timestep
    for suffix, slice_indices in tqdm(grouped_slices.items(), desc="Loading timesteps", unit="timestep"):
        # Load cubes for this timestep
        temp_file = get_file_for_timestep(base_dir / temp_dir, temp_filename, suffix)
        rho_file = get_file_for_timestep(base_dir / rho_dir, rho_filename, suffix)
        vel_file = get_file_for_timestep(base_dir / vel_dir, vel_filename, suffix)
        
        temp_cube = load_cube(
            temp_file, shape=cube_shape, unit=u.K,
            downsample=downsample, precision=precision,
            voxel_dx=voxel_dx, voxel_dy=voxel_dy, voxel_dz=voxel_dz,
            create_ndcube=True
        )
        rho_cube = load_cube(
            rho_file, shape=cube_shape, unit=u.g/u.cm**3,
            downsample=downsample, precision=precision,
            voxel_dx=voxel_dx, voxel_dy=voxel_dy, voxel_dz=voxel_dz,
            create_ndcube=True
        )
        vel_cube = load_cube(
            vel_file, shape=cube_shape, unit=u.cm/u.s,
            downsample=downsample, precision=precision,
            voxel_dx=voxel_dx, voxel_dy=voxel_dy, voxel_dz=voxel_dz,
            create_ndcube=True
        )
        
        # Copy relevant slices to composite (no rebinning - direct copy)
        for slice_idx in slice_indices:
            temp_composite[:, :, slice_idx] = temp_cube.data[:, :, slice_idx]
            rho_composite[:, :, slice_idx] = rho_cube.data[:, :, slice_idx]
            vel_composite[:, :, slice_idx] = vel_cube.data[:, :, slice_idx]
    
    # Create NDCubes with proper WCS (at MHD resolution)
    temp_ndcube = NDCube(temp_composite * u.K, wcs=reference_wcs, meta={"source": "composite_dynamic"})
    rho_ndcube = NDCube(rho_composite * (u.g/u.cm**3), wcs=reference_wcs, meta={"source": "composite_dynamic"})
    vel_ndcube = NDCube(vel_composite * (u.cm/u.s), wcs=reference_wcs, meta={"source": "composite_dynamic"})
    
    return temp_ndcube, rho_ndcube, vel_ndcube


def _match_candidates(line_name: str, wavelengths_aa: np.ndarray, observed: np.ndarray) -> tuple:
    """
    The transitions *line_name* names, by index, and how they were found.

    First those whose wavelength, written to as many decimals as the name
    gives, is the name's: observed ones if any are, else theoretical ones, so
    that a line CHIANTI has only worked out can be named at its wavelength.
    With none, the nearest line CHIANTI has observed: its theoretical
    wavelengths include many weak transitions within a few mA of strong
    lines, and a name a little off an observed wavelength, as another line
    list may give it, would otherwise pick one of those, up to 1e12 times
    fainter. An ion with no observed lines has only its theoretical ones.
    All the transitions at the chosen wavelength are given, for the
    brightest to be taken.
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


def _compute_single_ion(args):
    """Worker that computes G(T,N) for one ion.  Imports fiasco locally so
    that each spawned process gets its own HDF5 handles.

    With a temperature chunk, fiasco is called on that many temperatures at a
    time, one call after another, and only the requested lines are kept from
    each call, so the memory needed is that of one chunk rather than of the
    whole temperature grid."""

    import fiasco
    import logging

    (elem, stage, temperature_K, densities_cm3, abundance, lines,
     hdf5_dbase_root, temperature_chunk) = args
    densities = densities_cm3 / u.cm**3
    n_temperatures = len(temperature_K)
    step = n_temperatures if temperature_chunk is None else temperature_chunk

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

    g_parts = {}
    try:
        for start in range(0, n_temperatures, step):
            ion = fiasco.Ion(f'{elem} {stage}',
                             temperature_K[start:start + step] * u.K,
                             abundance=abundance, **ion_kwargs)

            g = ion.contribution_function(densities)
            pe_ratio = ion.proton_electron_ratio
            g = g * pe_ratio[:, np.newaxis, np.newaxis]

            # The transitions do not depend on temperature, so the lines are
            # matched once, on the first chunk.
            if start == 0:
                bound = ion.transitions.is_bound_bound
                bb_wl = ion.transitions.wavelength[bound]
                # CHIANTI stores its theoretical wavelengths negative, and
                # fiasco gives them positive with this flag.
                observed = np.asarray(ion.transitions.is_observed[bound])
                matches = {line_name: _match_candidates(line_name, bb_wl.to_value(u.AA), observed)
                           for line_name, _ in lines}
                g_parts = {(line_name, int(idx)): [] for line_name, (indices, _) in matches.items()
                           for idx in indices}
            for (line_name, idx), parts in g_parts.items():
                parts.append(g[:, :, idx].to(u.erg * u.cm**3 / u.s).value)
            del g
    finally:
        fiasco_logger.setLevel(prev_level)

    results = {}
    for line_name, target_wl_aa in lines:
        target_wl = target_wl_aa * u.AA
        indices, how = matches[line_name]
        tables = {}
        for idx in indices:
            g_tn = np.concatenate(g_parts[(line_name, int(idx))]).T
            tables[int(idx)] = np.nan_to_num(g_tn, nan=0.0, posinf=0.0, neginf=0.0)
        # Transitions CHIANTI lists at one wavelength are told apart by their brightness.
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
            # The root this Ion was built against.  fiasco resolves it to the
            # fiascorc default when the caller did not choose one, so this is
            # always the database the contribution functions came from.
            "hdf5_dbase_root": str(ion.hdf5_dbase_root),
        }
    return results


# A line name: element, ionisation stage, and a wavelength in Angstrom, such
# as Fe12_195.1190.
_LINE_NAME = re.compile(r'^([A-Z][a-z]?)(\d+)_(\d+\.?\d*)$')


def _parse_line_name(name: str) -> Tuple[str, int, float]:
    """
    The element, ionisation stage and wavelength in Angstrom of a line name
    such as ``Fe12_195.1190``.

    The element must be one, the stage one it has, from 1, neutral, to its
    atomic number plus one, and the wavelength above zero: an element or
    stage that is not was found only when its ion was made, after the
    atmosphere had been read, and a wavelength of zero matched whichever
    transition was shortest.
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

# The grids G(T, n_e) is worked out on unless given others: log10 T from 4 to 9
# every 0.05, and log10 n_e from 7 to 13 every 0.3. density_grid carries the
# density grid on, on the same points, as far as an atmosphere needs.
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
    Compute each line's contribution function, G(T, n_e), from CHIANTI through fiasco.

    Each line is matched to a transition in CHIANTI as the docs page on
    naming spectral lines describes, and placed at CHIANTI's wavelength,
    ``wl0``, not the name's. G includes the ratio of protons to electrons, so
    that it multiplies an emission measure of n_e^2 dh. With two or more ions
    and more than one worker, the ions are computed in separate processes,
    so a script calling this needs an ``if __name__ == "__main__":`` guard.

    Parameters
    ----------
    line_names : list of str
        The lines, such as ``["Fe12_195.1190", "Fe09_171.073"]``.
    abundance : str, optional
        The CHIANTI abundance set. Default ``"sun_coronal_2021_chianti"``.
    logT_min, logT_max : float, optional
        The range of log10(T / K).
    nT : int, optional
        How many temperatures, evenly spaced in log10(T), across that range.
    logN_min, logN_max : float, optional
        The range of log10(n_e / cm-3).
    nN : int, optional
        How many densities, evenly spaced in log10(n_e), across that range.
    precision : type, optional
        ``np.float32`` or ``np.float64`` (default).
    n_workers : int, optional
        How many processes to compute the ions with, one ion each at a time,
        so never more than there are ions. Default 0, which uses every CPU
        this process may use, as SLURM allocates them, up to that limit.
    hdf5_dbase_root : str or Path, optional
        The CHIANTI database to read. Default fiasco's own, set in
        ``~/.fiasco/fiascorc``.
    temperature_chunk : int, optional
        Give fiasco this many temperatures at a time, rather than the whole
        grid, to use less memory. The result is the same. Default None, for
        the whole grid.

    Returns
    -------
    goft_dict : dict
        For each line, by name: ``wl0``, its wavelength in CHIANTI; ``g_tn``,
        G on the grid of densities and temperatures, shaped (nN, nT), in
        erg cm3 / s; ``atom`` and ``ion``, its atomic number and ionisation
        stage; and ``hdf5_dbase_root``, the database it came from.
    logT_grid : np.ndarray
        The temperatures, as log10(T / K).
    logN_grid : np.ndarray
        The densities, as log10(n_e / cm-3).
    """
    # fiasco's contribution_function, G = Ab(X) f_{X,k} (N_j / N) A_ij dE_ij
    # / n_e, leaves out n_H / n_e, which is multiplied in here. Each worker is
    # spawned rather than forked, to keep HDF5 safe, so it imports fiasco and
    # reads fiascorc afresh: setting fiasco.defaults in the parent does not
    # reach it, which is why the database is passed down, and each worker
    # reports the root its Ion was built with, checked against the request.
    # fiasco solves the level populations for every temperature it is given
    # at once, so its memory grows with them; chunks of an ion are computed
    # one after another, at the cost of reading its atomic data again.
    if temperature_chunk is not None and temperature_chunk < 1:
        raise ValueError(
            f"temperature_chunk must be a positive number of temperatures, "
            f"not {temperature_chunk}."
        )

    logT_grid = np.linspace(logT_min, logT_max, nT)
    logN_grid = np.linspace(logN_min, logN_max, nN)

    temperature_K = 10.0 ** logT_grid
    densities_cm3 = 10.0 ** logN_grid

    # ---- parse line names and group by ion for efficiency ----
    ion_lines: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}

    for name in line_names:
        elem, stage, wl = _parse_line_name(name)
        ion_lines.setdefault((elem, stage), []).append((name, wl))

    # Build worker arguments (all picklable plain types / numpy arrays)
    dbase_root = None if hdf5_dbase_root is None else str(hdf5_dbase_root)
    worker_args = [
        (elem, stage, temperature_K, densities_cm3, abundance, lines, dbase_root,
         temperature_chunk)
        for (elem, stage), lines in ion_lines.items()
    ]
    if temperature_chunk is not None and temperature_chunk < nT:
        print(f"  {nT} temperatures in chunks of {temperature_chunk}")

    # Here, since spawned workers cannot ask whether to build it.
    _offer_database_build(dbase_root)

    # ---- dispatch: parallel for 2+ ions, serial otherwise ----
    n_ions = len(worker_args)
    if n_workers <= 0:
        # The CPUs this process may run on, as a SLURM step gives it, not
        # the whole node's, which would start a worker per CPU of the node.
        n_workers = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                     else os.cpu_count() or 1)
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
                f"matched {info['matched_wl_aa']:.4f} Angstrom, "
                f"{'observed' if info['observed'] else 'theoretical'} "
                f"(delta={info['delta_aa']:.4f} Angstrom)"
            )
            # No line of the ion is at the name's wavelength to its digits.
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
                # The line is where CHIANTI has observed it, whatever
                # digits the name gave.
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
        The view: the axis to integrate along, ``"x"``, ``"y"`` or ``"z"``,
        optionally with a sign for the side of the box to look from, such as
        ``"-x"``; see `line_of_sight_velocity`.

    Returns
    -------
    dem_map : np.ndarray
        DEM array [cm^-5 per dex]. The two remaining spatial axes come out in
        image order (row, column). Shape depends on the axis:
        - "x": (nz, ny, nT)
        - "y": (nz, nx, nT)
        - "z": (ny, nx, nT)
    avg_ne : np.ndarray
        Mean electron density per T-bin [cm^-3]. Same shape as dem_map.
    """
    nT = len(logT_grid)
    axis, _ = view_axis_and_side(integration_axis)

    dlogT, logT_edges = _temperature_bins(logT_grid)

    ne = 10.0 ** logN_cube.astype(np.float64)
    dh = along_line_of_sight(voxel_dh_cm, axis)
    w2 = np.broadcast_to(ne**2 * dh, logT_cube.shape)  # weights for EM
    w3 = np.broadcast_to(ne**3 * dh, logT_cube.shape)  # weights for EM*n_e

    # Each cell is shared between the two temperature bins about it, as in
    # build_em_tv, so that the DEM is the emission measure it spreads over
    # velocity, and each bin's density that of the plasma it holds.
    lower, share, inside = _neighbouring_bins(logT_cube, logT_grid, logT_edges)
    pixel, (n_rows, n_cols) = _image_pixels(logT_cube.shape, integration_axis)
    em = np.zeros(n_rows * n_cols * nT)   # cm^-5
    em_n = np.zeros_like(em)              # cm^-5 * n_e
    for step, of_T in ((0, 1.0 - share), (1, share)):
        in_bin = (pixel * nT + np.minimum(lower + step, nT - 1)).ravel()
        of_T = np.where(inside, of_T, 0.0)
        np.add.at(em, in_bin, (w2 * of_T).ravel())
        np.add.at(em_n, in_bin, (w3 * of_T).ravel())
    em, em_n = em.reshape(n_rows, n_cols, nT), em_n.reshape(n_rows, n_cols, nT)

    dem = em / dlogT
    # Zero where no plasma is at a temperature, rather than whatever memory
    # the division left there, which reached the saved G diagnostic.
    avg_ne = np.divide(em_n, em, out=np.zeros_like(em), where=em > 0.0)
    return dem, avg_ne


def _temperature_bins(logT_grid: np.ndarray) -> Tuple[float, np.ndarray]:
    """The width and the edges of the DEM's temperature bins, centred on *logT_grid*."""
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
    Each value's shares of the two bins about it: the bin whose centre is at
    or below it, the share the next bin up takes, and whether it is within
    the bins' edges at all.

    The share is in proportion to how far the value is from the one centre
    towards the next, so that the two centres, weighted by their shares,
    average to the value itself: a flow of 2.4 km/s on a 5 km/s grid is 0.52
    of the 0 km/s bin and 0.48 of the 5 km/s one, where it went wholly to the
    0 km/s bin. A value between the first or last centre and that bin's outer
    edge is that bin's alone, as before, so the grid reaches as far as it
    did; one beyond the edges is no bin's, and its share is zero.
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
    For a (z, y, x) cube seen along *integration_axis*, the image pixel each
    cell is seen in, as row times the number of columns plus column, and the
    image's (rows, columns), in the order compute_dem and build_em_tv give.

    Seen from the side of the box opposite the one :data:`OBSERVER_SIDE`
    gives, the image is mirrored left to right, as walking round to the
    other side of the box mirrors it, so that it is the right way round for
    that observer too.
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
    """log10 of the temperature and the electron density, as the synthesis works with them.

    A cell with no electrons has a log density of minus infinity, so that it
    adds nothing to the emission measure; one with no temperature is put at
    log T = 0, below every temperature grid.
    """
    logN_cube = np.log10(electron_density, where=electron_density > 0.0,
                         out=np.full_like(electron_density, -np.inf)).astype(precision)
    logT_cube = np.log10(temperature, where=temperature > 0.0,
                         out=np.zeros_like(temperature)).astype(precision)
    return logT_cube, logN_cube


def density_grid(
    temperature: np.ndarray,
    electron_density: np.ndarray,
    logT_grid: np.ndarray,
    precision: type = np.float64,
) -> Tuple[float, float, int]:
    """The density grid to work G(T, n_e) out on for an atmosphere.

    The contribution functions are read at the mean density of each
    temperature along each line of sight, and taken as zero off their grid.
    This grid has the points of the default one, every 0.3 in log10 n_e
    through 10**7 cm^-3, and goes each way until it is past the density of
    every cell whose temperature is on *logT_grid*, and no further. No mean
    density falls off it, and an atmosphere with a narrower range of
    densities than the default grid needs fewer points.

    Parameters
    ----------
    temperature, electron_density : np.ndarray
        The temperature of every cell in K and its electron density in cm^-3.
    logT_grid : np.ndarray
        The temperature grid of the DEM, as :func:`compute_goft_fiasco`
        returns it.
    precision : type
        The precision of the synthesis, which the densities are rounded to as
        :func:`synthesise_cubes` rounds them.

    Returns
    -------
    logN_min, logN_max : float
    nN : int
        The grid, for :func:`compute_goft_fiasco`. It is the default grid if
        no cell has electrons at a temperature on *logT_grid*.
    """
    points = _density_points(temperature, electron_density, logT_grid, precision)
    if points is None:
        return _LOGN_MIN, _LOGN_MAX, _N_N
    first, last = points
    return _density_point(first), _density_point(last), last - first + 1


def _density_point(index: int) -> float:
    """log10 n_e of a point of the default density grid, counted from its first."""
    return _LOGN_MIN + index * (_LOGN_MAX - _LOGN_MIN) / (_N_N - 1)


def _density_points(temperature: np.ndarray, electron_density: np.ndarray,
                    logT_grid: np.ndarray, precision: type) -> Optional[Tuple[int, int]]:
    """The first and last point of :func:`density_grid`, counted as :func:`_density_point` counts them.

    None if no cell has electrons at a temperature on *logT_grid*.
    """
    logT_cube, logN_cube = _log_cubes(temperature, electron_density, precision)
    _, logT_edges = _temperature_bins(logT_grid)
    counted = ((logT_cube >= logT_edges[0]) & (logT_cube < logT_edges[-1])
               & np.isfinite(logN_cube))
    if not counted.any():
        return None

    step = (_LOGN_MAX - _LOGN_MIN) / (_N_N - 1)
    # Where the lowest and highest densities fall among the default grid's
    # points, counted in steps from its first; rounded, so that a density on a
    # point, but for the arithmetic that put it there, counts as on it.
    lowest, highest = (
        round((float(extreme) - _LOGN_MIN) / step, 9) for extreme in (
            np.min(logN_cube, where=counted, initial=np.inf),
            np.max(logN_cube, where=counted, initial=-np.inf)))
    # The points just past them, so that a mean density a rounding error
    # beyond the lowest or the highest is still on the grid.
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
    Take each line's contribution function at the density of each pixel and temperature.

    G is interpolated linearly from the grid `compute_goft_fiasco` computed
    it on, and is zero outside it. The result is added to each line's entry
    in *goft* as ``g``, shaped (rows, columns, temperatures), ready for
    `synthesise_spectra`.

    Parameters
    ----------
    goft : dict
        The lines, as `compute_goft_fiasco` gives them, changed in place.
    avg_ne : np.ndarray
        The electron density in each pixel at each temperature, in cm-3,
        shaped (rows, columns, temperatures). Where the cells along the line
        of sight have different densities, give their mean weighted by
        emission measure, as ECLIPSE's own synthesis does. For a DEM, give
        the one density you choose.
    logT_grid : np.ndarray
        The temperatures to take G at, as log10(T / K).
    logN_grid : np.ndarray
        The densities G was computed on, from `compute_goft_fiasco`.
    logT_goft : np.ndarray
        The temperatures G was computed on, from `compute_goft_fiasco`.
    precision : type, optional
        ``np.float32`` (default) or ``np.float64``.
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

# A view looks along one axis of the box, from one side of it. Named by its
# axis alone, it is from the side OBSERVER_SIDE (from utils) gives: +1 on the
# side of increasing coordinate, -1 on the other. That is the side from which
# the line cube, with its rows and columns as create_line_cube lays them out,
# is seen the right way round, so the column axis crossed with the row axis
# points at the observer: above the box (+z) for the top-down view, and at +x
# and at -y for the two side views. Named with a sign, such as "-x", it is
# from that side; from the other side the image is mirrored left to right
# (_image_pixels), so that it is the right way round for that observer too.
SIGNED_VIEWS = ("+x", "-x", "+y", "-y", "+z", "-z")


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
        The view: ``"x"``, ``"y"`` or ``"z"``, whose observer is on the side
        given by :data:`OBSERVER_SIDE`, or an axis with a sign for the side,
        such as ``"-x"`` for the view from -x.

    Returns
    -------
    np.ndarray or u.Quantity
        Velocity away from the observer, in the same units.
    """
    _, side = view_axis_and_side(integration_axis)
    return -side * velocity


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

    Each cell's emission measure is shared between the two temperature bins
    and the two velocity bins about it, in proportion to how near it is to
    each (:func:`_neighbouring_bins`), so that its mean velocity over the bins
    is the cell's and its contribution function is interpolated in
    temperature. A value beyond the outermost bin centre, within that bin,
    goes to it alone.

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
        The view: the axis to integrate along, ``"x"``, ``"y"`` or ``"z"``,
        optionally with a sign for the side of the box to look from, such as
        ``"-x"``; see `line_of_sight_velocity`.

    Returns
    -------
    em_tv : np.ndarray
        4D emission measure cube. The two remaining spatial axes come out in
        image order (row, column). Shape depends on the axis:
        - "x": (nz, ny, nT, nv)
        - "y": (nz, nx, nT, nv)
        - "z": (ny, nx, nT, nv)
    """
    axis, _ = view_axis_and_side(integration_axis)
    print(f"  Building 4-D emission-measure cube along {axis}-axis...")

    _, logT_edges = _temperature_bins(logT_grid)
    v_centres = (vel_grid.to_value(u.cm / u.s) if isinstance(vel_grid, u.Quantity)
                 else np.asarray(vel_grid, dtype=float))
    v_edges = velocity_centers_to_edges(v_centres)

    # Each cell's emission measure is shared between the two temperature bins
    # and the two velocity bins about it, in proportion to how near it is to
    # each: put wholly in the nearest, a uniform 2.4 km/s flow on the default
    # 5 km/s grid was synthesised at rest and a 2.6 km/s one at 5 km/s, and a
    # cell's contribution function was that of the nearest temperature bin.
    # Shared, its mean velocity over the bins is the cell's, its line's width
    # grows by the spread of the two bins' velocities, at most a quarter of a
    # bin squared in variance, and the contribution function is interpolated
    # between the two temperatures. Past the outermost centres, within the
    # outermost bins, a cell goes to that bin alone, as before.
    lower_T, share_T, in_T = _neighbouring_bins(logT_cube, logT_grid, logT_edges)
    lower_v, share_v, in_v = _neighbouring_bins(vel_cube, v_centres, v_edges)

    # Plasma faster than the grid reaches emits beyond the synthesised
    # wavelengths, so its emission is not in the spectra, though the DEM keeps
    # it. Said, since the default grid misses the fastest flows of a flare.
    in_temperature = ne_sq_dh * in_T
    beyond = in_temperature[~in_v].sum()
    if beyond > 0:
        warnings.warn(
            f"{100 * beyond / in_temperature.sum():.3g} per cent of the emission measure is "
            f"from plasma faster than the velocity grid reaches, "
            f"{v_edges[0] / 1e5:.1f} to {v_edges[-1] / 1e5:.1f} km/s, and emits beyond the "
            f"synthesised wavelengths, so it is not in the spectra. Widen --vel-lim to keep "
            f"it.", UserWarning, stacklevel=2)

    # Summed along the line of sight into each image pixel's bins.
    nT, nv = len(logT_grid), len(v_centres)
    pixel, (n_rows, n_cols) = _image_pixels(logT_cube.shape, integration_axis)
    em = np.where(in_T & in_v, ne_sq_dh, 0.0)
    em_tv = np.zeros(n_rows * n_cols * nT * nv)
    for step_T, of_T in ((0, 1.0 - share_T), (1, share_T)):
        in_bin_T = pixel * nT + np.minimum(lower_T + step_T, nT - 1)
        for step_v, of_v in ((0, 1.0 - share_v), (1, share_v)):
            in_bin = in_bin_T * nv + np.minimum(lower_v + step_v, nv - 1)
            np.add.at(em_tv, in_bin.ravel(), (em * of_T * of_v).ravel())
    return em_tv.reshape(n_rows, n_cols, nT, nv)


def synthesise_spectra(
    goft: Dict[str, dict],
    em_tv: np.ndarray,
    vel_grid: u.Quantity | np.ndarray,
    logT_grid: np.ndarray,
) -> None:
    """
    Synthesise each line's spectrum in every pixel from the emission measure by temperature and velocity.

    Each temperature and velocity bin gives the line a Gaussian of that
    temperature's thermal width, Doppler shifted by that velocity, with the
    emission measure times G. The spectra are added to each line's entry in
    *goft* as ``si``, in erg / (s cm2 sr cm), on the wavelengths ``wl_grid``:
    the velocity grid converted with ``lambda_0 (1 + v / c)``.

    Parameters
    ----------
    goft : dict
        The lines, with ``g`` from `interpolate_g_on_dem`, changed in place.
    em_tv : np.ndarray
        The emission measure in each bin, in cm-5, shaped (rows, columns,
        temperatures, velocities).
    vel_grid : u.Quantity or np.ndarray
        The velocities of the bins' centres, evenly spaced and increasing. A
        plain array is taken to be in cm/s.
    logT_grid : np.ndarray
        The temperatures of the bins' centres, as log10(T / K).
    """
    kb = const.k_B.cgs.value
    c_cm_s = const.c.cgs.value
    # The Doppler shifts below are worked out in cm/s, which a grid in km/s
    # was taken to be, moving every line by a factor of 1e5 too little.
    vel_grid = (vel_grid.to(u.cm / u.s) if isinstance(vel_grid, u.Quantity)
                else np.asarray(vel_grid, dtype=float) * (u.cm / u.s))

    # The wavelength grid built below is the velocity grid mapped through
    # lambda_0 (1 + v/c), and create_line_cube writes its CDELT from the first
    # step alone, so an uneven velocity grid becomes a wrong wavelength axis.
    require_uniform_grid(vel_grid, "vel_grid")

    for line, data in tqdm(goft.items(), desc="spectra", unit="line", leave=False):
        wl0 = data["wl0"].cgs.value  # cm
        
        # Create wavelength grid for this line
        data["wl_grid"] = (vel_grid * data["wl0"] / const.c + data["wl0"]).cgs
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
    *,
    continuum: Optional[Dict[str, dict]] = None,
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
        The view: ``"x"``, ``"y"`` or ``"z"``, optionally with a sign for the side of
        the box to look from, such as ``"-x"``; see `line_of_sight_velocity`.
    precision : type
        np.float32 or np.float64.
    continuum : dict, optional
        The continuum to add, by entry name: ``"wl_grid"``, its
        wavelengths, and ``"free"`` and ``"two_photon"``, as
        `continuum.compute_continuum_fiasco` gives them on these
        temperatures and densities. Keyword only. Default None, for none.

    Returns
    -------
    lines : dict
        A copy of *goft* whose entries also hold ``"g"``, the contribution
        function on the DEM, ``"wl_grid"`` and ``"si"``, the specific
        intensity ``(rows, columns, wavelength)``. With *continuum*, it also
        holds an entry for each of its names, with ``"wl_grid"``, ``"si"``,
        ``"wl0"``, the middle of its wavelengths, and no ``"atom"`` or
        ``"ion"``.
    dem_map : np.ndarray
        As :func:`compute_dem` returns it.
    em_tv : np.ndarray
        As :func:`build_em_tv` returns it.
    """
    logT_cube, logN_cube = _log_cubes(temperature, electron_density, precision)

    dem_map, avg_ne_map = compute_dem(logT_cube, logN_cube, dh_cm, logT_grid, integration_axis)

    # The contribution functions are worked out on a grid of densities and
    # taken as zero off it; said, with how much of the emission that is.
    occupied = dem_map > 0
    log_ne = np.log10(avg_ne_map, where=avg_ne_map > 0,
                      out=np.full(avg_ne_map.shape, -np.inf))
    off_grid = occupied & ((log_ne < logN_grid[0]) | (log_ne > logN_grid[-1]))
    if off_grid.any():
        warnings.warn(
            f"{100 * dem_map[off_grid].sum() / dem_map[occupied].sum():.3g} per cent of the "
            f"emission measure is at electron densities outside the {10 ** logN_grid[0]:.0e} "
            f"to {10 ** logN_grid[-1]:.0e} cm^-3 the contribution functions are worked out "
            f"for, where they are taken as zero.", UserWarning, stacklevel=2)

    lines = {name: dict(info) for name, info in goft.items()}
    interpolate_g_on_dem(lines, avg_ne_map, logT_grid, logN_grid, logT_grid, precision)

    ne_sq_dh = ((10.0 ** logN_cube.astype(np.float64)) ** 2
                * along_line_of_sight(dh_cm, view_axis_and_side(integration_axis)[0]))
    em_tv = build_em_tv(logT_cube, los_velocity, logT_grid, vel_grid, ne_sq_dh, integration_axis)

    synthesise_spectra(lines, em_tv, vel_grid, logT_grid)

    # The continuum takes the emission measure in each temperature bin, from
    # the DEM, which keeps the cells moving faster than the velocity grid
    # reaches, as the continuum is not Doppler shifted, and each pixel's
    # density at that temperature.
    if continuum:
        emission_measure = dem_map * _temperature_bins(logT_grid)[0]
        for name, table in continuum.items():
            grid = u.Quantity(table["wl_grid"]).to(u.cm)
            spectra = continuum_spectra(emission_measure, avg_ne_map, logN_grid,
                                        table["free"], table["two_photon"])
            lines[name] = {"si": spectra.astype(precision), "wl_grid": grid,
                           "wl0": grid[grid.size // 2], "atom": None, "ion": None}
    return lines, dem_map, em_tv


def _cell_size_mm(cube: NDCube, pixel_axis: int) -> float:
    """The world distance across one pixel of *cube* along *pixel_axis*, in Mm.

    Measured between the two edges of the first pixel, so it needs no second
    pixel and no access to a CDELT, which a cropped cube's WCS does not have.
    """
    # A cropped cube wraps its WCS in a high-level object; the pixel-to-world
    # conversion of plain numbers is on the low-level one underneath.
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
    The value an even grid *coords* has at 1-based pixel *crpix*, as a plain number.

    A reference pixel at the middle of an axis falls between two pixels when
    there is an even number of them, so the reference value has to be read
    off the grid there rather than taken from the pixel below.
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
) -> NDCube:
    """
    Make one line's synthesised spectra into an NDCube, with the image's coordinates.

    Parameters
    ----------
    line_name : str
        The line's name.
    line_data : dict
        The line's entry from `compute_goft_fiasco`, after
        `synthesise_spectra` has added its spectra, ``si``, and their
        wavelengths, ``wl_grid``.
    spatial_cube : NDCube
        A cube of the atmosphere, such as `create_atmosphere_ndcube` makes,
        whose coordinates the image takes.
    intensity_unit : u.Unit
        The unit of the spectra.
    integration_axis : str, optional
        The view the synthesis had: the axis it looked along, ``"x"``,
        ``"y"`` or ``"z"`` (default), optionally with a sign for the side of
        the box it looked from, such as ``"-x"``; see
        `line_of_sight_velocity`. From the side opposite the one
        :data:`OBSERVER_SIDE` gives, the image is mirrored left to right, so
        its column coordinate is the box's coordinate with its sign changed,
        which increases to that observer's right.

    Returns
    -------
    NDCube
        The spectra, indexed ``[row, column, wavelength]`` like an image, so
        that summing over wavelength gives an array that plots the right way
        up.
    """
    view = view_name(integration_axis)
    los_axis, side = view_axis_and_side(view)
    # An axis whose cells differ in size has no one CDELT. Only the line of
    # sight may be such an axis, and that is the one integrated out here.
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

    # The cell size of each axis comes from the reference cube's own WCS, so
    # that an axis a single cell wide has one too. It is read as the world
    # distance across one pixel, which any WCS answers, cropped ones included.
    cell_size = [_cell_size_mm(spatial_cube, pixel_axis) for pixel_axis in range(3)]

    # The WCS below carries a single linear CDELT, the grid's one spacing, so
    # the grid has to be uniform for that to describe it. Checked here as
    # well as in synthesise_spectra because this is a public entry point: the
    # DEM and VDEM routes call it directly.
    wl_step = require_uniform_grid(line_data["wl_grid"].to(u.cm), "wl_grid")

    # Get spatial coordinate information from the reference cube,
    # whose array axes are (z, y, x)
    if los_axis == "x":
        # Integration along X -> data shape (nz, ny, n_lambda): rows are Z, columns are Y
        nz, ny, nl = cube_data.shape
        y_coords = spatial_cube.axis_world_coords(1)[0]  # Y coordinates
        z_coords = spatial_cube.axis_world_coords(0)[0]  # Z coordinates

        spatial_axes = ['WAVE', 'SOLY', 'SOLZ']  # Wavelength, Y, Z
        spatial_units = ['cm', 'Mm', 'Mm']
        spatial_cdelt = [
            wl_step,
            cell_size[1],
            cell_size[2],
        ]
        spatial_crpix = [(nl + 1) / 2, (ny + 1) / 2, 1]  # Wavelength centered, Y centered, Z at first pixel
        spatial_crval = [
            _world_at(line_data["wl_grid"].to(u.cm), spatial_crpix[0]),
            _world_at(y_coords.to(u.Mm), spatial_crpix[1]),
            z_coords[0].to(u.Mm).value  # Z starts where original cube starts
        ]

    elif los_axis == "y":
        # Integration along Y -> data shape (nz, nx, n_lambda): rows are Z, columns are X
        nz, nx, nl = cube_data.shape
        x_coords = spatial_cube.axis_world_coords(2)[0]  # X coordinates
        z_coords = spatial_cube.axis_world_coords(0)[0]  # Z coordinates

        spatial_axes = ['WAVE', 'SOLX', 'SOLZ']  # Wavelength, X, Z
        spatial_units = ['cm', 'Mm', 'Mm']
        spatial_cdelt = [
            wl_step,
            cell_size[0],
            cell_size[2],
        ]
        spatial_crpix = [(nl + 1) / 2, (nx + 1) / 2, 1]  # Wavelength centered, X centered, Z at first pixel
        spatial_crval = [
            _world_at(line_data["wl_grid"].to(u.cm), spatial_crpix[0]),
            _world_at(x_coords.to(u.Mm), spatial_crpix[1]),
            z_coords[0].to(u.Mm).value  # Z starts where original cube starts
        ]

    else:  # los_axis == "z"
        # Integration along Z -> data shape (ny, nx, n_lambda): rows are Y, columns are X
        ny, nx, nl = cube_data.shape
        x_coords = spatial_cube.axis_world_coords(2)[0]  # X coordinates
        y_coords = spatial_cube.axis_world_coords(1)[0]  # Y coordinates

        spatial_axes = ['WAVE', 'SOLX', 'SOLY']  # Wavelength, X, Y
        spatial_units = ['cm', 'Mm', 'Mm']
        spatial_cdelt = [
            wl_step,
            cell_size[0],
            cell_size[1],
        ]
        spatial_crpix = [(nl + 1) / 2, (nx + 1) / 2, (ny + 1) / 2]  # All centered
        spatial_crval = [
            _world_at(line_data["wl_grid"].to(u.cm), spatial_crpix[0]),
            _world_at(x_coords.to(u.Mm), spatial_crpix[1]),
            _world_at(y_coords.to(u.Mm), spatial_crpix[2]),
        ]

    # Seen from the other side, the image is mirrored left to right
    # (_image_pixels), so its columns run along the box's coordinate with its
    # sign changed. They are centred, so the middle column keeps its place.
    if side != OBSERVER_SIDE[los_axis]:
        spatial_crval[1] = -spatial_crval[1]

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



##############################################################################
# ---------------------------------------------------------------------------
#                 M A I N   W O R K F L O W
# ---------------------------------------------------------------------------
##############################################################################

# The options that say where MURaM's own files are and how they are laid out,
# for dynamic mode and for the deprecated static route that reads them
# without an atmosphere file. An atmosphere file carries all of this itself,
# so giving both is a contradiction rather than a choice.
MURAM_LAYOUT_OPTIONS = ("data_dir", "temp_file", "rho_file", "vx_file", "vy_file",
                        "vz_file", "cube_shape", "voxel_dx", "voxel_dy", "voxel_dz")

# The options only dynamic mode reads, which a static synthesis from an
# atmosphere file would ignore.
DYNAMIC_OPTIONS = ("slit_width", "temp_dir", "temp_filename", "rho_dir", "rho_filename",
                   "vx_dir", "vx_filename", "vy_dir", "vy_filename", "vz_dir",
                   "vz_filename", "time_dir", "time_filename")

# Where the documentation describes the atmosphere file and how to write one.
ATMOSPHERE_DOCS = "https://solarc-eclipse.readthedocs.io/en/stable/synthesis/#atmosphere-files"
# Where it describes observing a time series of atmosphere files, which
# replaces the deprecated dynamic mode.
TIME_SERIES_DOCS = "https://solarc-eclipse.readthedocs.io/en/stable/time-series/"


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
                       help="The ECLIPSE atmosphere file (HDF5) to synthesise "
                            f"from; see {ATMOSPHERE_DOCS} for how to write one. "
                            "Required, except in dynamic mode.")
    # The MURaM files static mode read before atmosphere files, kept out of
    # the help so that old command lines still run, with a warning, until
    # the route is removed.
    for flag, default in (("--temp-file", "temp/eosT.0270000"),
                          ("--rho-file", "rho/result_prim_0.0270000"),
                          ("--vx-file", "vx/result_prim_1.0270000"),
                          ("--vy-file", "vy/result_prim_3.0270000"),
                          ("--vz-file", "vz/result_prim_2.0270000")):
        parser.add_argument(flag, type=str, default=default, action=_NotedOption,
                            help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=str, default="./run/input",
                       help="Output directory for results")
    parser.add_argument("--output-name", type=str, default="synthesised_spectra.h5",
                       help="Output filename, an HDF5 synthesis file; a name ending in .pkl "
                            "writes the deprecated pickle instead")
    
    # Line / abundance specification (fiasco)
    parser.add_argument("--lines", nargs="+", required=True,
                       help="Line specifications (e.g. Fe12_195.1190 Fe09_171.073)")
    parser.add_argument("--abundance", type=str, default="sun_coronal_2021_chianti",
                       help="CHIANTI abundance dataset name for fiasco")
    parser.add_argument("--continuum", action="store_true",
                       help="Also synthesise the free-free, free-bound and two-photon "
                            "continuum under the lines' windows, as entries of its own "
                            "beside the lines")
    parser.add_argument("--n-workers", type=int, default=0,
                       help="Number of parallel workers for fiasco G(T,N) "
                            "computation (0 = all CPUs, default: 0)")
    parser.add_argument("--hdf5-dbase-root", type=str, default=None,
                       help="CHIANTI HDF5 database to use, overriding the one "
                            "in ~/.fiasco/fiascorc. Use this to run against a "
                            "second CHIANTI version without changing the "
                            "default for other work.")
    parser.add_argument("--goft-temperature-chunk", type=int, default=None,
                       help="Compute G(T,N) this many temperatures at a time "
                            "rather than the whole grid at once. This lowers "
                            "fiasco's peak memory, which for an ion with many "
                            "levels can be several GB, at the cost of reading "
                            "the atomic data again for each chunk. The result "
                            "is the same.")

    # Integration direction
    parser.add_argument("--integration-axis", choices=["x", "y", "z", *SIGNED_VIEWS],
                       default="z",
                       help="The axis to look along, x, y or z, from +x, -y or above; or the "
                            "axis with a sign for the side to look from, such as -x")
    
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
                            "0.11.0 and earlier used 1.29, the value for a neutral gas. "
                            "Not used when the atmosphere gives an electron density.")
    
    # Dynamic atmosphere mode (time-varying synthesis), which reads MURaM's
    # own files and so carries the options describing their layout.
    dynamic_group = parser.add_argument_group("Dynamic atmosphere mode (deprecated)",
        "Options for synthesising with time-varying atmosphere (raster scanning). "
        "Deprecated: observe a time series of atmosphere files with the "
        "instrument run instead.")
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

    return parser


def parse_arguments(argv=None):
    """Parse command line arguments for spectrum synthesis."""
    words = list(sys.argv[1:] if argv is None else argv)
    # argparse takes a word that starts with a dash, such as the -x of
    # "--integration-axis -x", for an option, and stops. Joined into
    # "--integration-axis=-x", it is read as the value.
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
    Refuse an atmosphere given alongside options it makes meaningless, and warn when there is none.

    The synthesis reads its atmosphere from an atmosphere file. Without one,
    static mode still reads MURaM's own files, and dynamic mode reads a
    time series of them; both are deprecated. The MURaM layout options
    describe those files, and the dynamic mode options only apply to dynamic
    mode, so one of either given with --atmosphere would be ignored without
    a word.
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
    return _prepare_atmosphere(atmosphere, str(path), integration_axis,
                               downsample, crop_x, crop_y, crop_z)


def load_muram_files(
    args,
    integration_axis: str,
    downsample: int | bool = False,
    crop_x=None, crop_y=None, crop_z=None,
) -> Atmosphere:
    """
    Read the MURaM files the deprecated static options name, as :func:`load_atmosphere_file` reads a file.

    The box is placed where ECLIPSE has always placed a MURaM box, x and y
    centred on zero and z = 0 at the centre of the bottom cell, so the crop
    options mean what they did before atmosphere files.
    """
    data_dir = Path(args.data_dir)
    files = {
        "temperature": (args.temp_file, u.K),
        "mass_density": (args.rho_file, u.g / u.cm**3),
        f"velocity_{integration_axis}": (getattr(args, f"v{integration_axis}_file"),
                                         u.cm / u.s),
    }
    cubes = {}
    for name, (file_name, unit) in files.items():
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        cubes[name] = load_cube(path, shape=tuple(args.cube_shape), unit=unit)

    nz, ny, nx = cubes["temperature"].shape
    voxel = {axis: u.Quantity(getattr(args, f"voxel_d{axis}")) for axis in AXES}
    atmosphere = Atmosphere(
        x_edges=(np.arange(nx + 1) - nx / 2) * voxel["x"],
        y_edges=(np.arange(ny + 1) - ny / 2) * voxel["y"],
        z_edges=(np.arange(nz + 1) - 0.5) * voxel["z"],
        source="MURaM", **cubes)
    return _prepare_atmosphere(atmosphere, f"the MURaM files in {data_dir}",
                               integration_axis, downsample, crop_x, crop_y, crop_z)


def _prepare_atmosphere(atmosphere: Atmosphere, name: str, integration_axis: str,
                        downsample, crop_x, crop_y, crop_z) -> Atmosphere:
    """Downsample and crop *atmosphere*, and check that its image axes are even."""
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
    """The mass per free electron to use, in atomic mass units, and where it came from."""
    given = getattr(args, "mass_per_electron", None)
    # A script that sets the option's old name on its arguments still gets it.
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

    Supports two modes:
    - Static mode: Single timestep synthesis, from an atmosphere file
      (--atmosphere), or from MURaM's own files, which is deprecated
    - Dynamic mode: Time-varying synthesis with raster scanning, from MURaM's
      own files, which is deprecated: the instrument run observes a time
      series of atmosphere files instead (see euvst_response.raster)

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
    # Checked now, where a name that could not be read was found only once
    # the atmosphere had been read, which can take minutes and gigabytes.
    for name in args.lines:
        _parse_line_name(name)
    vel_res = u.Quantity(args.vel_res)
    vel_lim = u.Quantity(args.vel_lim)
    # Checked now, before the atmosphere is read or anything computed.
    velocity_grid(vel_res, vel_lim, ("--vel-res", "--vel-lim"))

    intensity_unit = u.erg/u.s/u.cm**2/u.sr/u.cm

    print_mem = lambda: f"{psutil.virtual_memory().used/1e9:.2f}/" \
                        f"{psutil.virtual_memory().total/1e9:.2f} GB"

    # The view: the axis looked along, which the atmosphere is read and
    # summed along, and the side of the box it is seen from.
    view = view_name(args.integration_axis.lower())
    integration_axis, observer_side = view_axis_and_side(view)
    check_atmosphere_options(args)
    # A name ending in .pkl still gets the pickle older versions wrote, so
    # that scripts written for them keep working until it is removed.
    write_pickle = Path(args.output_name).suffix.lower() in (".pkl", ".pickle", ".dill")
    if write_pickle:
        warnings.warn(
            f"--output-name {args.output_name} names a pickle, as older versions of ECLIPSE "
            f"wrote the synthesis. Writing one is deprecated and will stop in a future "
            f"release: give the output a name ending in .h5 for a synthesis file.",
            FutureWarning, stacklevel=2)
    elif Path(args.output_name).suffix.lower() not in (".h5", ".hdf5", ".hdf"):
        warnings.warn(
            f"--output-name {args.output_name} gets a synthesis file, which is HDF5. Older "
            f"versions of ECLIPSE wrote a pickle under any name; a name ending in .pkl "
            f"still gets one, until that is removed.", UserWarning, stacklevel=2)

    # What the common processing below needs from whichever route reads the
    # atmosphere. Only an atmosphere file can give the electron density
    # directly or a different size for every cell along the line of sight;
    # dynamic mode gives a mass density and one cell size.
    ne_values = None
    los_thickness = None
    atmosphere_metadata = None

    # Determine if we're in dynamic mode
    dynamic_mode = args.slit_rest_time is not None

    if dynamic_mode:
        # Validate dynamic mode requirements
        if args.slit_width is None:
            raise ValueError("--slit-width is required for dynamic mode (when --slit-rest-time is specified)")
        # The snapshots are laid across x, which the slit steps across in a
        # view along z or y. Seen along x, x is the line of sight, and every
        # pixel would add up cells from all the snapshots.
        if integration_axis == "x":
            raise ValueError(
                "Dynamic mode lays its snapshots across x, which is the line of sight "
                "of a view along x, so every pixel would add up all of them. View along "
                "z or y.")
        if view != integration_axis:
            raise ValueError(
                f"Dynamic mode, which is deprecated, looks only from the side of the box "
                f"that --integration-axis {integration_axis} names. To look from "
                f"{view}, synthesise from an atmosphere file with --atmosphere.")

        base_dir = Path(args.data_dir)
        # Voxel sizes of the MURaM files. load_cube scales these itself when
        # it downsamples, so they are passed to it as given.
        file_voxel_dz = u.Quantity(args.voxel_dz)
        file_voxel_dx = u.Quantity(args.voxel_dx)
        file_voxel_dy = u.Quantity(args.voxel_dy)

        # Voxel sizes of the cubes as synthesised, for the path length along
        # the line of sight, the slice timing and the saved metadata.
        voxel_dz = file_voxel_dz * (downsample or 1)
        voxel_dx = file_voxel_dx * (downsample or 1)
        voxel_dy = file_voxel_dy * (downsample or 1)

        # Parse slit rest time and slit width
        slit_rest_time = u.Quantity(args.slit_rest_time)
        slit_width = u.Quantity(args.slit_width)
        
        # Determine which velocity direction to use
        if integration_axis == "x":
            vel_dir = args.vx_dir or "vx"
            vel_filename = args.vx_filename
            voxel_dh = voxel_dx
        elif integration_axis == "y":
            vel_dir = args.vy_dir or "vy"
            vel_filename = args.vy_filename
            voxel_dh = voxel_dy
        else:  # "z"
            vel_dir = args.vz_dir or "vz"
            vel_filename = args.vz_filename
            voxel_dh = voxel_dz
        
        # Set directory defaults
        temp_dir = args.temp_dir or "temp"
        rho_dir = args.rho_dir or "rho"
        
        print(f"DYNAMIC MODE - Time-varying synthesis at MHD resolution (deprecated)")
        print(f"  Slit width: {slit_width}")
        print(f"  Slit rest time: {slit_rest_time}")
        print(f"  Voxel dx: {voxel_dx}")
        print()
        
        # Discover available timesteps
        time_dir = base_dir / args.time_dir
        print(f"Discovering timesteps from {time_dir}...")
        timestep_times = discover_timesteps(time_dir, args.time_filename)
        print(f"  Found {len(timestep_times)} timesteps")
        for suffix, sim_time in sorted(timestep_times.items(), key=lambda x: x[1]):
            print(f"    {suffix}: {sim_time:.3f} s")
        print()
        
        # Calculate MHD cube dimensions
        cube_shape_tuple = tuple(args.cube_shape)
        nx_mhd = cube_shape_tuple[0]
        if downsample:
            nx_mhd = nx_mhd // downsample
        
        # Prepare crop_x for slice mapping if specified
        crop_x_for_mapping = None
        if args.crop_x:
            crop_x_for_mapping = (u.Quantity(args.crop_x[0]), u.Quantity(args.crop_x[1]))
        
        # Compute slice-to-timestep mapping at MHD resolution
        print(f"Computing slice-to-timestep mapping at MHD resolution...")
        slice_mapping, grouped_slices = compute_slice_timestep_mapping_mhd(
            nx_mhd, voxel_dx, slit_width, slit_rest_time, timestep_times,
            crop_x=crop_x_for_mapping
        )
        print(f"  MHD slices per timestep:")
        for suffix, indices in sorted(grouped_slices.items(), key=lambda x: min(x[1])):
            print(f"    {suffix}: {len(indices)} slices (indices {min(indices)}-{max(indices)})")
        print()
        
        # Build composite cubes at MHD resolution
        print(f"Building composite atmosphere cubes at MHD resolution ({print_mem()})...")
        temp_cube, rho_cube, vel_cube = build_composite_cubes_mhd(
            base_dir=base_dir,
            temp_dir=temp_dir,
            temp_filename=args.temp_filename,
            rho_dir=rho_dir,
            rho_filename=args.rho_filename,
            vel_dir=vel_dir,
            vel_filename=vel_filename,
            slice_mapping=slice_mapping,
            grouped_slices=grouped_slices,
            cube_shape=cube_shape_tuple,
            voxel_dx=file_voxel_dx,
            voxel_dy=file_voxel_dy,
            voxel_dz=file_voxel_dz,
            downsample=downsample,
            precision=precision,
        )
        print(f"  Composite cube shape: {temp_cube.data.shape}")
        
        # Apply cropping if requested in dynamic mode
        if args.crop_x or args.crop_y or args.crop_z:
            print(f"Applying cropping ({print_mem()})")
            temp_cube, rho_cube, vel_cube = apply_cube_cropping(
                temp_cube, rho_cube, vel_cube,
                args.crop_x, args.crop_y, args.crop_z
            )
            print(f"  Cropped cubes to shape: {temp_cube.data.shape}")
        
        reference_cube = temp_cube
        rho = u.Quantity(rho_cube.data, rho_cube.unit)

        # Dynamic mode metadata for output (no spatial rebinning in synthesis)
        dynamic_mode_metadata = {
            "enabled": True,
            "slit_width": slit_width,
            "slit_rest_time": slit_rest_time,
            "scan_direction": "right_to_left",
            "slice_timesteps": slice_mapping,
            "available_timesteps": timestep_times,
            "spatially_rebinned": False,  # Output is at MHD resolution
        }
        
    else:
        # Static mode from an atmosphere file, which brings its own layout, or
        # by the deprecated route from MURaM's own files, read into the same
        # atmosphere the converter would write
        dynamic_mode_metadata = {"enabled": False}

        if args.atmosphere:
            print("STATIC MODE - Synthesis from an atmosphere file")
            print(f"  Atmosphere: {args.atmosphere}")
        else:
            print("STATIC MODE - Synthesis from MURaM files (deprecated)")
            print(f"  Data directory: {args.data_dir}")
        print(f"  Integration axis: {view}")
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
        if args.atmosphere:
            atmosphere = load_atmosphere_file(
                args.atmosphere, integration_axis, downsample=downsample, **crops)
        else:
            atmosphere = load_muram_files(
                args, integration_axis, downsample=downsample, **crops)
        print(atmosphere.describe())

        # The file may hold float32 in any units; the run works in the
        # precision --precision asks for, and the processing below takes the
        # cubes' values as K and cm/s, so they are converted here.
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
            "path": str(Path(args.atmosphere).resolve()) if args.atmosphere else None,
            "source": atmosphere.source,
            "time": atmosphere.time,
            "shape": atmosphere.shape,
            "nonuniform_axes": atmosphere.nonuniform_axes(),
            "electron_density_given": atmosphere.electron_density is not None,
        }

    # ---------------- Common processing (both modes) -----------------
    
    # Build velocity grid
    vel_grid = velocity_grid(vel_res, vel_lim, ("--vel-res", "--vel-lim"))

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
    vel_data = line_of_sight_velocity(vel_cube.data, view)

    # ---------------- Compute contribution functions (fiasco) ---------
    # At the densities this atmosphere has, and no others.
    logN_min, logN_max, nN = density_grid(
        temp_cube.data, ne_values,
        np.linspace(_LOGT_MIN, _LOGT_MAX, _N_T).astype(precision), precision)
    print(f"Computing contribution functions via fiasco at 10^{logN_min:.1f} to "
          f"10^{logN_max:.1f} cm^-3 ({print_mem()})")
    goft, logT_goft, logN_grid = compute_goft_fiasco(
        args.lines, abundance=args.abundance, precision=precision,
        logN_min=logN_min, logN_max=logN_max, nN=nN,
        n_workers=args.n_workers,
        hdf5_dbase_root=getattr(args, "hdf5_dbase_root", None),
        temperature_chunk=getattr(args, "goft_temperature_chunk", None),
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
    # The continuum over the lines' windows, each group of overlapping ones
    # an entry of its own: the windows are each line's velocity grid about
    # its wavelength in CHIANTI, as synthesise_spectra lays them out.
    continuum = None
    if getattr(args, "continuum", False):
        windows = continuum_windows([(vel_grid * info["wl0"] / const.c + info["wl0"]).cgs
                                     for info in goft.values()])
        print(f"Computing the continuum over {len(windows)} window"
              f"{'' if len(windows) == 1 else 's'} via fiasco ({print_mem()})")
        wavelength = np.concatenate([grid.to_value(u.cm) for grid in windows.values()]) * u.cm
        free, two_photon = compute_continuum_fiasco(
            wavelength, logT_grid, logN_grid, abundance=args.abundance,
            n_workers=args.n_workers, hdf5_dbase_root=goft_dbase_root)
        continuum, start = {}, 0
        for name, grid in windows.items():
            stop = start + grid.size
            continuum[name] = {"wl_grid": grid, "free": free[:, start:stop],
                               "two_photon": two_photon[:, :, start:stop]}
            start = stop

    print(f"Calculating the DEM, the emission measure in (T,v) and the spectra ({print_mem()})")
    goft, dem_map, em_tv = synthesise_cubes(
        temp_cube.data, ne_values, vel_data, dh_cm, goft, logT_grid, logN_grid,
        vel_grid, view, precision, continuum=continuum)

    # ---------------- Create output cubes -----------------
    print(f"Creating output cubes ({print_mem()})")
    line_cubes = {}
    for name, info in goft.items():
        line_cubes[name] = create_line_cube(
            name, info, reference_cube, intensity_unit, view
        )
    
    print(f"Built {len(line_cubes)} line cubes")

    # ---------------- Save results -----------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_name
    
    # The line cubes are the file's spectra; everything else the synthesis
    # worked out goes with them, so that the file is the whole synthesis.
    products = {
        "dem_map": dem_map,
        "em_tv": em_tv,
        "logT_grid": logT_grid,
        "vel_grid": vel_grid,
        "logN_grid": logN_grid,
        # Each line's spectra and wavelengths are the file's lines, so they
        # are not kept a second time here.
        "goft": {name: {key: value for key, value in info.items()
                        if key not in ("si", "wl_grid")}
                 for name, info in goft.items() if name not in (continuum or {})},
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
            "data_dir": None if args.atmosphere else str(Path(args.data_dir)),
            "lines": args.lines,
            "abundance": args.abundance,
            "continuum": bool(getattr(args, "continuum", False)),
            "hdf5_dbase_root": goft_dbase_root,
            "integration_axis": view,
            "velocity_convention": VELOCITY_CONVENTION,
            "observer_side": observer_side,
            "crop_params": {
                "crop_x": args.crop_x,
                "crop_y": args.crop_y,
                "crop_z": args.crop_z
            }
        }
    }
    
    if write_pickle:
        # As older versions wrote it, contribution functions and all.
        with open(output_file, "wb") as f:
            dill.dump({"line_cubes": line_cubes, **products,
                       "goft": {name: info for name, info in goft.items()
                                if name not in (continuum or {})}}, f)
    else:
        # The snapshot's time goes with the spectra, so that syntheses of a
        # series of snapshots can be observed as a time series.
        write_line_cubes(line_cubes, output_file,
                         source=(atmosphere_metadata or {}).get("source") or "",
                         products=products,
                         time=(atmosphere_metadata or {}).get("time"))

    print(f"Saved results to {output_file} ({os.path.getsize(output_file) / 1e6:.2f} MB)")
    print("Synthesis complete!")

if __name__ == "__main__":
    main()