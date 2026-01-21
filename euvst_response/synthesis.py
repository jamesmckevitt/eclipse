import os
import argparse
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
from scipy.io import readsav
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
from .utils import angle_to_distance

##############################################################################
# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------
##############################################################################

def velocity_centers_to_edges(vel_grid: np.ndarray) -> np.ndarray:
    """
    Convert velocity grid centers to bin edges.
    
    Parameters
    ----------
    vel_grid : np.ndarray
        1D array of velocity centers.
        
    Returns
    -------
    np.ndarray
        1D array of velocity bin edges (length = len(vel_grid) + 1).
    """
    if len(vel_grid) < 2:
        raise ValueError("vel_grid must have at least 2 elements")
    
    dv = vel_grid[1] - vel_grid[0]
    return np.concatenate([
        [vel_grid[0] - 0.5 * dv],
        vel_grid[:-1] + 0.5 * dv,
        [vel_grid[-1] + 0.5 * dv]
    ])

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

    The cube is stored (x, z, y) in the file and transposed to (x, y, z)
    upon loading.

    Parameters
    ----------
    file_path : str | Path
        Path to the binary file.
    shape : Tuple[int, int, int]
        Tuple (nx, ny, nz) describing the *full* cube dimensions.
    unit : astropy.units.Unit, optional
        Astropy unit to attach (e.g. u.K or u.g/u.cm**3). If None, returns
        a plain ndarray.
    downsample : int | bool
        Integer factor; if non-False, keep every *downsample*-th cell along
        each axis (simple stride).
    precision : type
        np.float32 or np.float64 for returned dtype.
    voxel_dx, voxel_dy, voxel_dz : u.Quantity, optional
        Voxel sizes for creating proper WCS coordinates. Required if create_ndcube=True.
    create_ndcube : bool, optional
        If True, return an NDCube with proper WCS coordinates.

    Returns
    -------
    ndarray, Quantity, or NDCube
        Array with shape (nx', ny', nz') or NDCube with proper coordinates.
    """
    data = np.fromfile(file_path, dtype=np.float32).reshape(shape, order="F")
    data = data.transpose(0, 2, 1)  # (x,y,z)

    if downsample:
        data = data[::downsample, ::downsample, ::downsample]
        voxel_dx *= downsample
        voxel_dy *= downsample
        voxel_dz *= downsample

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
        3D data array with shape (nx, ny, nz).
    voxel_dx, voxel_dy, voxel_dz : u.Quantity
        Voxel sizes in Mm.
        
    Returns
    -------
    NDCube
        Cube with proper WCS coordinates.
        X,Y centered at origin, Z starting at 0.
    """
    nx, ny, nz = data.shape
    
    # Create WCS for heliocentric coordinates
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['SOLZ', 'SOLY', 'SOLX']
    wcs.wcs.cunit = ['Mm', 'Mm', 'Mm']
    
    # Reference pixels (1-indexed for WCS)
    wcs.wcs.crpix = [1, (ny + 1) / 2, (nx + 1) / 2]  # Z starts at first pixel
    
    # Reference values
    wcs.wcs.crval = [0, 0, 0]  # X,Y centered at origin, Z starts at 0
    
    # Pixel scales
    wcs.wcs.cdelt = [
        voxel_dz.to(u.Mm).value,
        voxel_dy.to(u.Mm).value,  
        voxel_dx.to(u.Mm).value
    ]
    
    return NDCube(data.data,
                  wcs=wcs,
                  unit=data.unit)


def read_timestep_time(file_path: Path) -> float:
    """
    Read simulation time from MHD time file header.
    
    The time is stored in the 4th element (index 3) of the file header 
    as a float32 value in seconds.
    
    Parameters
    ----------
    file_path : Path
        Path to the time file.
        
    Returns
    -------
    float
        Simulation time in seconds.
    """
    with open(file_path, 'rb') as f:
        header = np.fromfile(f, dtype=np.float32, count=10)
        if header.size < 4:
            raise ValueError(f"Time file header too short: {file_path}")
        return float(header[3])


def discover_timesteps(
    time_dir: Path,
    time_filename: str,
) -> Dict[str, float]:
    """
    Discover all available timesteps and their simulation times.
    
    Parameters
    ----------
    time_dir : Path
        Directory containing time files.
    time_filename : str
        Filename prefix before the timestep suffix (e.g., "tau_slice_0.100").
        
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
) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Compute which timestep suffix to use for each MHD X-slice.
    
    The spectrometer scans from right to left (high X to low X).
    Each slit position covers a physical width (slit_width converted to Mm).
    Determine which slit position each MHD slice belongs to, then find
    the appropriate timestep based on observation time.
    
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
    
    # Reference time is the first available timestep
    t0 = times[0]
    
    # Total physical extent of the domain
    total_extent = nx_mhd * voxel_physical
    
    slice_mapping = []
    
    # For each MHD X-slice, calculate which slit position it belongs to
    # Scanning right to left: rightmost (x = nx_mhd-1) is observed first
    for mhd_slice_idx in range(nx_mhd):
        # Physical position of this slice (center of voxel)
        x_physical = (mhd_slice_idx + 0.5) * voxel_physical
        
        # Distance from the right edge (scanning starts from right)
        distance_from_right = total_extent - x_physical
        
        # Which slit position does this belong to?
        # slit_position = 0 is the rightmost, slit_position = N is leftmost
        slit_position = int(np.floor((distance_from_right / slit_physical).decompose().value))
        
        # Observation time for this slit position
        observation_time = t0 + slit_position * rest_time_sec
        
        # Find nearest timestep
        idx = np.argmin(np.abs(times - observation_time))
        slice_mapping.append(suffixes[idx])
    
    # Group slices by timestep for efficient processing
    grouped = {}
    for slice_idx, suffix in enumerate(slice_mapping):
        if suffix not in grouped:
            grouped[suffix] = []
        grouped[suffix].append(slice_idx)
    
    # Print statistics
    n_slit_positions = int(np.ceil((total_extent / slit_physical).decompose().value))
    print(f"  Physical domain extent: {total_extent:.3f}")
    print(f"  Slit physical width: {slit_physical:.3f}")
    print(f"  Number of slit positions: {n_slit_positions}")
    print(f"  Total raster time: {n_slit_positions * rest_time_sec:.1f} s")
    
    return slice_mapping, grouped


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
        Original cube dimensions (nx, ny, nz).
    voxel_dx, voxel_dy, voxel_dz : u.Quantity
        Voxel sizes.
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
    
    nx, ny, nz = temp_cube_ref.data.shape
    reference_wcs = temp_cube_ref.wcs
    
    # Verify dimensions match slice mapping
    if nx != nx_mhd:
        raise ValueError(f"Cube X dimension ({nx}) doesn't match slice mapping ({nx_mhd})")
    
    # Initialise composite arrays
    temp_composite = np.zeros((nx, ny, nz), dtype=precision)
    rho_composite = np.zeros((nx, ny, nz), dtype=precision)
    vel_composite = np.zeros((nx, ny, nz), dtype=precision)
    
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
            temp_composite[slice_idx, :, :] = temp_cube.data[slice_idx, :, :]
            rho_composite[slice_idx, :, :] = rho_cube.data[slice_idx, :, :]
            vel_composite[slice_idx, :, :] = vel_cube.data[slice_idx, :, :]
    
    # Create NDCubes with proper WCS (at MHD resolution)
    temp_ndcube = NDCube(temp_composite * u.K, wcs=reference_wcs, meta={"source": "composite_dynamic"})
    rho_ndcube = NDCube(rho_composite * (u.g/u.cm**3), wcs=reference_wcs, meta={"source": "composite_dynamic"})
    vel_ndcube = NDCube(vel_composite * (u.cm/u.s), wcs=reference_wcs, meta={"source": "composite_dynamic"})
    
    return temp_ndcube, rho_ndcube, vel_ndcube


def read_goft(
    sav_file: str | Path,
    limit_lines: Optional[List[str]] = None,
    precision: type = np.float64,
) -> Tuple[Dict[str, dict], np.ndarray, np.ndarray]:
    """
    Read a CHIANTI G(T,N) .sav file produced by IDL.

    Parameters
    ----------
    sav_file : str | Path
        Path to the IDL save file containing GOFT data.
    limit_lines : List[str], optional
        If provided, only load these specific lines.
    precision : type
        Precision for arrays (np.float32 or np.float64).

    Returns
    -------
    goft_dict : Dict[str, dict]
        Dictionary keyed by line name, each entry holding:
            'wl0'      - rest wavelength (Quantity, cm)
            'g_tn'     - 2-D array G(logT, logN)  [erg cm^3 s^-1]
            'atom'     - atomic number
            'ion'      - ionisation stage
    logT_grid : np.ndarray
        1-D array of log10(T/K) values.
    logN_grid : np.ndarray
        1-D array of log10(N_e/cm^3) values.
    """
    raw = readsav(sav_file)
    goft_dict: Dict[str, dict] = {}

    logT_grid = raw["logTarr"].astype(precision)
    logN_grid = raw["logNarr"].astype(precision)

    for entry in raw["goftarr"]:
        # Handle both string and bytes for line names (different IDL save versions)
        line_name = entry[0]  # This is the 'name' field from the IDL structure
        if hasattr(line_name, 'decode'):
            line_name = line_name.decode()  # bytes -> string
        # line_name is now a string, e.g. "Fe12_195.1190"
        
        if limit_lines and line_name not in limit_lines:
            continue

        rest_wl = float(line_name.split("_")[1]) * u.AA  # A -> Quantity
        goft_dict[line_name] = {
            "wl0": rest_wl.to(u.cm),
            "g_tn": entry[4].astype(precision),  # This is the 'goft' field [nT, nN]
            "atom": entry[1],  # This is the 'atom' field
            "ion": entry[2],   # This is the 'ion' field
        }

    return goft_dict, logT_grid, logN_grid


##############################################################################
# ---------------------------------------------------------------------------
#  DEM and G(T) helpers
# ---------------------------------------------------------------------------
##############################################################################

def compute_dem(
    logT_cube: np.ndarray,
    logN_cube: np.ndarray,
    voxel_dh_cm: float,
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
    voxel_dh_cm : float
        Voxel depth in cm along integration axis.
    logT_grid : np.ndarray
        1D array of temperature bin centers for DEM calculation.
    integration_axis : str
        Axis along which to integrate ("x", "y", or "z").

    Returns
    -------
    dem_map : np.ndarray
        DEM array [cm^-5 per dex]. Shape depends on integration_axis:
        - "x": (ny, nz, nT) 
        - "y": (nx, nz, nT)
        - "z": (nx, ny, nT)
    avg_ne : np.ndarray
        Mean electron density per T-bin [cm^-3]. Same shape as dem_map.
    """
    nT = len(logT_grid)
    
    # Determine integration axis and output shape
    axis_map = {"x": 0, "y": 1, "z": 2}
    if integration_axis not in axis_map:
        raise ValueError(f"integration_axis must be 'x', 'y', or 'z', got {integration_axis}")
    
    integration_axis_idx = axis_map[integration_axis]
    
    # Output shape depends on which axis we integrate over
    if integration_axis == "x":
        output_shape = (logT_cube.shape[1], logT_cube.shape[2], nT)  # (ny, nz, nT)
    elif integration_axis == "y":
        output_shape = (logT_cube.shape[0], logT_cube.shape[2], nT)  # (nx, nz, nT)
    else:  # "z"
        output_shape = (logT_cube.shape[0], logT_cube.shape[1], nT)  # (nx, ny, nT)
    
    # Create temperature bin edges from centers
    dlogT = logT_grid[1] - logT_grid[0] if len(logT_grid) > 1 else 0.1
    logT_edges = np.concatenate([
        [logT_grid[0] - dlogT/2],
        logT_grid[:-1] + dlogT/2,
        [logT_grid[-1] + dlogT/2]
    ])

    ne = 10.0 ** logN_cube.astype(np.float64)
    w2 = ne**2  # weights for EM
    w3 = ne**3  # weights for EM*n_e

    dem = np.zeros(output_shape)
    avg_ne = np.zeros_like(dem)

    for idx in tqdm(range(nT), desc="DEM bins", unit="bin", leave=False):
        lo, hi = logT_edges[idx], logT_edges[idx + 1]
        mask = (logT_cube >= lo) & (logT_cube < hi)  # (nx,ny,nz)

        # Integrate along the specified axis
        em = np.sum(w2 * mask, axis=integration_axis_idx) * voxel_dh_cm    # cm^-5
        em_n = np.sum(w3 * mask, axis=integration_axis_idx) * voxel_dh_cm  # cm^-5 * n_e

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
        Emission-measure weighted electron density (nx, ny, nT).
    logT_grid : np.ndarray
        Temperature grid for DEM (nT,).
    logN_grid : np.ndarray
        Density grid for GOFT interpolation.
    logT_goft : np.ndarray
        Temperature grid for GOFT interpolation.
    precision : type
        Output precision for interpolated G values.
    """
    nT, nx, ny = len(logT_grid), *avg_ne.shape[:2]

    # Build query points for interpolation
    logNe_flat = np.log10(avg_ne, where=avg_ne > 0.0, 
                         out=np.zeros_like(avg_ne)).transpose(2, 0, 1).ravel()
    logT_flat = np.broadcast_to(logT_grid[:, None, None],
                               (nT, nx, ny)).ravel()
    query_pts = np.column_stack((logNe_flat, logT_flat))

    for name, info in tqdm(goft.items(), desc="interpolating G", unit="line", leave=False):
        rgi = RegularGridInterpolator(
            (logN_grid, logT_goft), info["g_tn"],
            method="linear", bounds_error=False, fill_value=0.0
        )
        g_flat = rgi(query_pts)
        info["g"] = g_flat.reshape(nT, nx, ny).transpose(1, 2, 0).astype(precision)


##############################################################################
# ---------------------------------------------------------------------------
#  Build EM(T,v) and synthesise spectra
# ---------------------------------------------------------------------------
##############################################################################

def build_em_tv(
    logT_cube: np.ndarray,
    vel_cube: np.ndarray,
    logT_grid: np.ndarray,
    vel_grid: np.ndarray,
    ne_sq_dh: np.ndarray,
    integration_axis: str = "z",
) -> np.ndarray:
    """
    Construct 4-D emission-measure cube EM(x,y,T,v) [cm^-5].
    
    Parameters
    ----------
    logT_cube : np.ndarray
        3D temperature cube.
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
        4D emission measure cube. Shape depends on integration_axis:
        - "x": (ny, nz, nT, nv)
        - "y": (nx, nz, nT, nv)
        - "z": (nx, ny, nT, nv)
    """
    print(f"  Building 4-D emission-measure cube along {integration_axis}-axis...")
    
    # Determine integration axis and output shape
    axis_map = {"x": 0, "y": 1, "z": 2}
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
    
    # Sum along the specified integration axis
    if integration_axis == "x":
        em_tv_d = da.einsum("ijk,ijkl,ijkm->jklm", ne_sq_dh_d, mask_T_d, mask_V_d, optimize=True)
    elif integration_axis == "y":
        em_tv_d = da.einsum("ijk,ijkl,ijkm->iklm", ne_sq_dh_d, mask_T_d, mask_V_d, optimize=True)
    else:  # "z"
        em_tv_d = da.einsum("ijk,ijkl,ijkm->ijlm", ne_sq_dh_d, mask_T_d, mask_V_d, optimize=True)
        
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
    specific intensity cube I(x,y,lambda) for every line.
    
    Parameters
    ----------
    goft : Dict[str, dict]
        Dictionary of line data, modified in place with 'si' and 'wl_grid'.
    em_tv : np.ndarray
        4D emission measure cube (nx, ny, nT, nv).
    vel_grid : np.ndarray
        Velocity grid centers for wavelength calculation.
    logT_grid : np.ndarray
        Temperature bin centers.
    """
    kb = const.k_B.cgs.value
    c_cm_s = const.c.cgs.value

    for line, data in tqdm(goft.items(), desc="spectra", unit="line", leave=False):
        wl0 = data["wl0"].cgs.value  # cm
        
        # Create wavelength grid for this line
        data["wl_grid"] = (vel_grid * data["wl0"] / const.c + data["wl0"]).cgs
        wl_grid = data["wl_grid"].cgs.value  # (n_lambda,)

        atom = element(int(data["atom"]))
        atom_weight_g = (atom.atomic_weight * u.u).cgs.value

        # Thermal width per T-bin: sigma_T (nT,)
        sigma_T = wl0 * np.sqrt(2 * kb * (10 ** logT_grid) / atom_weight_g) / c_cm_s

        # Doppler-shifted center for each v-bin: (nv,)
        lam_cent = wl0 * (1 + vel_grid.value / c_cm_s)

        # Build phi(T,v,lambda) as (nT,nv,n_lambda)
        delta = wl_grid[None, None, :] - lam_cent[None, :, None]
        phi = np.exp(-0.5 * (delta / sigma_T[:, None, None]) ** 2)
        phi /= sigma_T[:, None, None] * np.sqrt(2 * np.pi)

        # EM(x,y,T,v) * G(T)  ->  (nx,ny,nT,nv)
        weighted = em_tv * data["g"][..., None]

        # Collapse T and v: dot ((nT,nv) , (nT,nv)) -> (nx,ny,n_lambda)
        spec_map = np.tensordot(weighted, phi, axes=([2, 3], [0, 1]))

        data["si"] = spec_map / (4 * np.pi)


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
        Cube with proper WCS and metadata.
    """
    cube_data = line_data["si"]  # Shape depends on integration_axis
    
    # Get spatial coordinate information from the reference cube
    if integration_axis == "x":
        # Integration along X -> data shape (ny, nz, n_lambda), spatial axes: Y, Z
        ny, nz, nl = cube_data.shape
        y_coords = spatial_cube.axis_world_coords(1)[0]  # Y coordinates  
        z_coords = spatial_cube.axis_world_coords(2)[0]  # Z coordinates
        
        spatial_axes = ['WAVE', 'SOLZ', 'SOLY']  # Wavelength, Z, Y
        spatial_units = ['cm', 'Mm', 'Mm']
        spatial_cdelt = [
            np.diff(line_data["wl_grid"].to(u.cm).value)[0],
            z_coords[1].to(u.Mm).value - z_coords[0].to(u.Mm).value,
            y_coords[1].to(u.Mm).value - y_coords[0].to(u.Mm).value
        ]
        spatial_crpix = [(nl + 1) / 2, 1, (ny + 1) / 2]  # Wavelength centered, Z at first pixel, Y centered
        spatial_crval = [
            line_data["wl0"].to(u.cm).value, 
            z_coords[0].to(u.Mm).value,  # Z starts where original cube starts
            y_coords[ny//2].to(u.Mm).value  # Y centered
        ]
            
    elif integration_axis == "y":
        # Integration along Y -> data shape (nx, nz, n_lambda), spatial axes: X, Z
        nx, nz, nl = cube_data.shape
        x_coords = spatial_cube.axis_world_coords(0)[0]  # X coordinates
        z_coords = spatial_cube.axis_world_coords(2)[0]  # Z coordinates
        
        spatial_axes = ['WAVE', 'SOLZ', 'SOLX']  # Wavelength, Z, X
        spatial_units = ['cm', 'Mm', 'Mm']
        spatial_cdelt = [
            np.diff(line_data["wl_grid"].to(u.cm).value)[0],
            z_coords[1].to(u.Mm).value - z_coords[0].to(u.Mm).value,
            x_coords[1].to(u.Mm).value - x_coords[0].to(u.Mm).value
        ]
        spatial_crpix = [(nl + 1) / 2, 1, (nx + 1) / 2]  # Wavelength centered, Z at first pixel, X centered
        spatial_crval = [
            line_data["wl0"].to(u.cm).value,
            z_coords[0].to(u.Mm).value,  # Z starts where original cube starts  
            x_coords[nx//2].to(u.Mm).value  # X centered
        ]
            
    else:  # integration_axis == "z"
        # Integration along Z -> data shape (nx, ny, n_lambda), spatial axes: X, Y
        nx, ny, nl = cube_data.shape
        x_coords = spatial_cube.axis_world_coords(0)[0]  # X coordinates
        y_coords = spatial_cube.axis_world_coords(1)[0]  # Y coordinates
        
        spatial_axes = ['WAVE', 'SOLY', 'SOLX']  # Wavelength, Y, X
        spatial_units = ['cm', 'Mm', 'Mm']
        spatial_cdelt = [
            np.diff(line_data["wl_grid"].to(u.cm).value)[0],
            y_coords[1].to(u.Mm).value - y_coords[0].to(u.Mm).value,
            x_coords[1].to(u.Mm).value - x_coords[0].to(u.Mm).value
        ]
        spatial_crpix = [(nl + 1) / 2, (ny + 1) / 2, (nx + 1) / 2]  # All centered
        spatial_crval = [
            line_data["wl0"].to(u.cm).value,
            y_coords[ny//2].to(u.Mm).value,  # Y centered
            x_coords[nx//2].to(u.Mm).value   # X centered
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
            "spatial_reference": spatial_cube.meta if hasattr(spatial_cube, 'meta') else None
        }
    )



##############################################################################
# ---------------------------------------------------------------------------
#                 M A I N   W O R K F L O W
# ---------------------------------------------------------------------------
##############################################################################

def parse_arguments():
    """Parse command line arguments for spectrum synthesis."""
    parser = argparse.ArgumentParser(
        description="Synthesise solar spectra from 3D MHD simulation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output paths
    parser.add_argument("--data-dir", type=str, default="data/atmosphere",
                       help="Directory containing simulation data")
    parser.add_argument("--goft-file", type=str, default="./data/gofnt.sav",
                       help="Path to CHIANTI G(T,N) save file")
    parser.add_argument("--output-dir", type=str, default="./run/input",
                       help="Output directory for results")
    parser.add_argument("--output-name", type=str, default="synthesised_spectra.pkl",
                       help="Output filename")
    
    # Simulation files
    parser.add_argument("--temp-file", type=str, default="temp/eosT.0270000",
                       help="Temperature file relative to data-dir")
    parser.add_argument("--rho-file", type=str, default="rho/result_prim_0.0270000",
                       help="Density file relative to data-dir")
    parser.add_argument("--vx-file", type=str, default="vx/result_prim_1.0270000",
                       help="Velocity x file relative to data-dir")
    parser.add_argument("--vy-file", type=str, default="vy/result_prim_3.0270000",
                       help="Velocity y file relative to data-dir")
    parser.add_argument("--vz-file", type=str, default="vz/result_prim_2.0270000",
                       help="Velocity z file relative to data-dir")
    
    # Grid parameters
    parser.add_argument("--cube-shape", nargs=3, type=int, default=[512, 768, 256],
                       help="Cube dimensions (nx ny nz)")
    parser.add_argument("--voxel-dx", type=str, default="0.192 Mm",
                       help="Voxel size in x (e.g. '0.192 Mm')")
    parser.add_argument("--voxel-dy", type=str, default="0.192 Mm",
                       help="Voxel size in y (e.g. '0.192 Mm')")
    parser.add_argument("--voxel-dz", type=str, default="0.064 Mm",
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
    parser.add_argument("--mean-mol-wt", type=float, default=1.29,
                       help="Mean molecular weight")
    
    # Line selection
    parser.add_argument("--limit-lines", nargs="*", default=None,
                       help="Limit to specific lines (e.g. Fe12_195.1190)")
    
    # Dynamic atmosphere mode (time-varying synthesis)
    dynamic_group = parser.add_argument_group("Dynamic atmosphere mode",
        "Options for synthesising with time-varying atmosphere (raster scanning)")
    dynamic_group.add_argument("--slit-rest-time", type=str, default=None,
                       help="Slit rest time per position (e.g. '40 s'). "
                            "Enables dynamic mode when specified.")
    dynamic_group.add_argument("--slit-width", type=str, default=None,
                       help="Slit width (e.g. '0.2 arcsec', required for dynamic mode)")
    
    # Directory arguments for dynamic mode
    dynamic_group.add_argument("--temp-dir", type=str, default=None,
                       help="Directory containing temperature files (for dynamic mode)")
    dynamic_group.add_argument("--temp-filename", type=str, default="eosT",
                       help="Temperature filename prefix before timestep suffix")
    dynamic_group.add_argument("--rho-dir", type=str, default=None,
                       help="Directory containing density files (for dynamic mode)")
    dynamic_group.add_argument("--rho-filename", type=str, default="result_prim_0",
                       help="Density filename prefix before timestep suffix")
    dynamic_group.add_argument("--vx-dir", type=str, default=None,
                       help="Directory containing vx files (for dynamic mode)")
    dynamic_group.add_argument("--vx-filename", type=str, default="result_prim_1",
                       help="Vx filename prefix before timestep suffix")
    dynamic_group.add_argument("--vy-dir", type=str, default=None,
                       help="Directory containing vy files (for dynamic mode)")
    dynamic_group.add_argument("--vy-filename", type=str, default="result_prim_3",
                       help="Vy filename prefix before timestep suffix")
    dynamic_group.add_argument("--vz-dir", type=str, default=None,
                       help="Directory containing vz files (for dynamic mode)")
    dynamic_group.add_argument("--vz-filename", type=str, default="result_prim_2",
                       help="Vz filename prefix before timestep suffix")
    dynamic_group.add_argument("--time-dir", type=str, default="time",
                       help="Directory containing time files (for dynamic mode)")
    dynamic_group.add_argument("--time-filename", type=str, default="tau_slice_0.100",
                       help="Time filename prefix before timestep suffix")
    
    return parser.parse_args()


def main(args=None) -> None:
    """
    Main workflow for synthesising solar spectra from 3D MHD simulations.
    
    Supports two modes:
    - Static mode: Single timestep synthesis
    - Dynamic mode: Time-varying synthesis with raster scanning
    
    Parameters
    ----------
    args : argparse.Namespace, optional
        Command line arguments. If None, will parse from sys.argv.
    """
    if args is None:
        args = parse_arguments()
    
    # ---------------- Configuration from arguments -----------------
    precision = np.float32 if args.precision == "float32" else np.float64
    downsample = args.downsample if args.downsample > 1 else False
    limit_lines = args.limit_lines
    vel_res = u.Quantity(args.vel_res)
    vel_lim = u.Quantity(args.vel_lim)
    voxel_dz = u.Quantity(args.voxel_dz)
    voxel_dx = u.Quantity(args.voxel_dx)
    voxel_dy = u.Quantity(args.voxel_dy)
    
    if downsample:
        voxel_dz *= downsample
        voxel_dx *= downsample
        voxel_dy *= downsample
        
    mean_mol_wt = args.mean_mol_wt
    intensity_unit = u.erg/u.s/u.cm**2/u.sr/u.cm
    
    print_mem = lambda: f"{psutil.virtual_memory().used/1e9:.2f}/" \
                        f"{psutil.virtual_memory().total/1e9:.2f} GB"

    base_dir = Path(args.data_dir)
    integration_axis = args.integration_axis.lower()
    
    # Determine if we're in dynamic mode
    dynamic_mode = args.slit_rest_time is not None
    
    if dynamic_mode:
        # Validate dynamic mode requirements
        if args.slit_width is None:
            raise ValueError("--slit-width is required for dynamic mode (when --slit-rest-time is specified)")
        
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
        
        print(f"DYNAMIC MODE - Time-varying synthesis at MHD resolution")
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
        
        # Compute slice-to-timestep mapping at MHD resolution
        print(f"Computing slice-to-timestep mapping at MHD resolution...")
        slice_mapping, grouped_slices = compute_slice_timestep_mapping_mhd(
            nx_mhd, voxel_dx, slit_width, slit_rest_time, timestep_times
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
            voxel_dx=voxel_dx,
            voxel_dy=voxel_dy,
            voxel_dz=voxel_dz,
            downsample=downsample,
            precision=precision,
        )
        print(f"  Composite cube shape: {temp_cube.data.shape}")
        
        reference_cube = temp_cube
        
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
        # Static mode (original behavior)
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
        print(f"  Velocity grid: ±{vel_lim:.1f} at {vel_res:.1f} resolution")
        print(f"  Precision: {precision}")
        if downsample:
            print(f"  Downsampling: {downsample}x")
        if limit_lines:
            print(f"  Limited to lines: {limit_lines}")
        if args.crop_x or args.crop_y or args.crop_z:
            print(f"  Cropping: X={args.crop_x}, Y={args.crop_y}, Z={args.crop_z}")
        print()
        
        # Load simulation data as NDCubes
        print(f"Loading cubes ({print_mem()})")
        temp_cube = load_cube(
            paths["T"], shape=tuple(args.cube_shape), unit=u.K, 
            downsample=downsample, precision=precision,
            voxel_dx=voxel_dx, voxel_dy=voxel_dy, voxel_dz=voxel_dz, 
            create_ndcube=True
        )
        rho_cube = load_cube(
            paths["rho"], shape=tuple(args.cube_shape), unit=u.g/u.cm**3, 
            downsample=downsample, precision=precision,
            voxel_dx=voxel_dx, voxel_dy=voxel_dy, voxel_dz=voxel_dz, 
            create_ndcube=True
        )
        vel_cube = load_cube(
            paths["vel"], shape=tuple(args.cube_shape), unit=u.cm/u.s, 
            downsample=downsample, precision=precision,
            voxel_dx=voxel_dx, voxel_dy=voxel_dy, voxel_dz=voxel_dz, 
            create_ndcube=True
        )

        # Apply cropping if requested
        if args.crop_x or args.crop_y or args.crop_z:
            print(f"Applying cropping ({print_mem()})")
            
            point1 = []
            point2 = []
            
            if args.crop_z:
                point1.append(u.Quantity(args.crop_z[0]))
                point2.append(u.Quantity(args.crop_z[1]))
            else:
                point1.append(None)
                point2.append(None)
                
            if args.crop_y:
                point1.append(u.Quantity(args.crop_y[0]))
                point2.append(u.Quantity(args.crop_y[1]))
            else:
                point1.append(None)
                point2.append(None)
                
            if args.crop_x:
                point1.append(u.Quantity(args.crop_x[0]))
                point2.append(u.Quantity(args.crop_x[1]))
            else:
                point1.append(None)
                point2.append(None)
            
            temp_cube = temp_cube.crop(point1, point2)
            rho_cube = rho_cube.crop(point1, point2)
            vel_cube = vel_cube.crop(point1, point2)
            
            print(f"Cropped cubes to shape: {temp_cube.data.shape}")

        reference_cube = temp_cube
    
    # ---------------- Common processing (both modes) -----------------
    
    goft_path = Path(args.goft_file)
    if not goft_path.exists():
        raise FileNotFoundError(f"GOFT file not found: {goft_path}")
    
    # Build velocity grid
    vel_grid = np.arange(
        -vel_lim.to(u.cm / u.s).value,
        vel_lim.to(u.cm / u.s).value + vel_res.to(u.cm / u.s).value,
        vel_res.to(u.cm / u.s).value
    ) * (u.cm / u.s)

    # Convert to log10 temperature and density
    ne_arr = (rho_cube / (mean_mol_wt * const.u.cgs.to(u.g))).to(1/u.cm**3)
    logN_cube = np.log10(ne_arr.data, where=ne_arr.data > 0.0, 
                        out=np.zeros_like(ne_arr.data)).astype(precision)
    logT_cube = np.log10(temp_cube.data, where=temp_cube.data > 0.0, 
                        out=np.zeros_like(temp_cube.data)).astype(precision)
    
    vel_data = vel_cube.data

    # ---------------- Load contribution functions -----------------
    print(f"Loading contribution functions ({print_mem()})")
    goft, logT_goft, logN_grid = read_goft(goft_path, limit_lines, precision)

    # Use the GOFT temperature grid as our DEM temperature grid
    logT_grid = logT_goft
    
    # Determine voxel_dh based on integration axis
    if integration_axis == "x":
        voxel_dh = voxel_dx
    elif integration_axis == "y":
        voxel_dh = voxel_dy
    else:
        voxel_dh = voxel_dz
    dh_cm = voxel_dh.to(u.cm).value

    # ---------------- Calculate DEM -----------------
    print(f"Calculating DEM and average density per bin ({print_mem()})")
    dem_map, avg_ne_map = compute_dem(logT_cube, logN_cube, dh_cm, logT_grid, integration_axis)

    print(f"Interpolating contribution function on the DEM ({print_mem()})")
    interpolate_g_on_dem(goft, avg_ne_map, logT_grid, logN_grid, logT_goft, precision)

    # ---------------- Build EM(T,v) cube -----------------
    ne_sq_dh = (10.0 ** logN_cube.astype(np.float64)) ** 2 * dh_cm
    print(f"Calculating emission measure cube in (T,v) space ({print_mem()})")
    em_tv = build_em_tv(logT_cube, vel_data, logT_grid, vel_grid, ne_sq_dh, integration_axis)

    # ---------------- Synthesise spectra -----------------
    print(f"Synthesising spectra ({print_mem()})")
    synthesise_spectra(goft, em_tv, vel_grid, logT_grid)

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
        "config": {
            "precision": precision.__name__,
            "downsample": downsample,
            "vel_res": vel_res,
            "vel_lim": vel_lim,
            "mean_mol_wt": mean_mol_wt,
            "intensity_unit": str(intensity_unit),
            "cube_shape": args.cube_shape,
            "data_dir": str(base_dir),
            "goft_file": str(goft_path),
            "integration_axis": integration_axis,
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