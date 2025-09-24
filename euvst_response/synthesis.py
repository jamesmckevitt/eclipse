import os
import argparse
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
from datetime import datetime
import shutil

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
            'ion'      - ionization stage
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
        description="Synthesize solar spectra from 3D MHD simulation data",
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
    parser.add_argument("--voxel-dx", type=float, default=0.192,
                       help="Voxel size in x (Mm)")
    parser.add_argument("--voxel-dy", type=float, default=0.192,
                       help="Voxel size in y (Mm)")
    parser.add_argument("--voxel-dz", type=float, default=0.064,
                       help="Voxel size in z (Mm)")
    
    # Integration direction
    parser.add_argument("--integration-axis", choices=["x", "y", "z"], default="z",
                       help="Axis along which to integrate (x, y, or z)")
    
    # Cropping parameters (in Heliocentric coordinates, Mm)
    parser.add_argument("--crop-x", nargs=2, type=float, default=None,
                       help="Crop in x direction: x_min x_max (Mm, None for no cropping)")
    parser.add_argument("--crop-y", nargs=2, type=float, default=None,
                       help="Crop in y direction: y_min y_max (Mm, None for no cropping)")
    parser.add_argument("--crop-z", nargs=2, type=float, default=None,
                       help="Crop in z direction: z_min z_max (Mm, None for no cropping)")
    
    # Velocity grid
    parser.add_argument("--vel-res", type=float, default=5.0,
                       help="Velocity resolution (km/s)")
    parser.add_argument("--vel-lim", type=float, default=300.0,
                       help="Velocity limit +/- (km/s)")
    
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
    
    return parser.parse_args()


def main(args=None) -> None:
    """
    Main workflow for synthesizing solar spectra from 3D MHD simulations.
    
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
    vel_res = args.vel_res * u.km / u.s
    vel_lim = args.vel_lim * u.km / u.s
    voxel_dz = args.voxel_dz * u.Mm
    voxel_dx = args.voxel_dx * u.Mm
    voxel_dy = args.voxel_dy * u.Mm
    
    if downsample:
        voxel_dz *= downsample
        voxel_dx *= downsample
        voxel_dy *= downsample
        
    mean_mol_wt = args.mean_mol_wt
    intensity_unit = u.erg/u.s/u.cm**2/u.sr/u.cm
    
    print_mem = lambda: f"{psutil.virtual_memory().used/1e9:.2f}/" \
                        f"{psutil.virtual_memory().total/1e9:.2f} GB"

    # File paths from arguments
    base_dir = Path(args.data_dir)
    files = {
        "T": args.temp_file,
        "rho": args.rho_file,
    }
    
    # Determine velocity file based on integration axis
    integration_axis = args.integration_axis.lower()
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
    
    goft_path = Path(args.goft_file)
    if not goft_path.exists():
        raise FileNotFoundError(f"GOFT file not found: {goft_path}")

    print(f"Synthesis configuration:")
    print(f"  Data directory: {base_dir}")
    print(f"  GOFT file: {goft_path}")
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

    # ---------------- Build grids -----------------
    # Velocity grid (symmetric about zero, inclusive)
    vel_grid = np.arange(
        -vel_lim.to(u.cm / u.s).value,
        vel_lim.to(u.cm / u.s).value + vel_res.to(u.cm / u.s).value,
        vel_res.to(u.cm / u.s).value
    ) * (u.cm / u.s)

    # ---------------- Load simulation data as NDCubes -----------------
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
        
        # Create coordinate points for cropping
        # NDCube expects coordinates as [point1, point2] where each point is [coord1, coord2, coord3]
        point1 = []
        point2 = []
        
        # Z coordinate (first axis)
        if args.crop_z:
            point1.append(args.crop_z[0] * u.Mm)
            point2.append(args.crop_z[1] * u.Mm)
        else:
            point1.append(None)
            point2.append(None)
            
        # Y coordinate (second axis)  
        if args.crop_y:
            point1.append(args.crop_y[0] * u.Mm)
            point2.append(args.crop_y[1] * u.Mm)
        else:
            point1.append(None)
            point2.append(None)
            
        # X coordinate (third axis)
        if args.crop_x:
            point1.append(args.crop_x[0] * u.Mm)
            point2.append(args.crop_x[1] * u.Mm)
        else:
            point1.append(None)
            point2.append(None)
        
        # Crop all cubes
        temp_cube = temp_cube.crop(point1, point2)
        rho_cube = rho_cube.crop(point1, point2)
        vel_cube = vel_cube.crop(point1, point2)
        
        print(f"Cropped cubes to shape: {temp_cube.data.shape}")

    # Convert to log10 temperature and density
    ne_arr = (rho_cube / (mean_mol_wt * const.u.cgs.to(u.g))).to(1/u.cm**3)
    logN_cube = np.log10(ne_arr.data, where=ne_arr.data > 0.0, 
                        out=np.zeros_like(ne_arr.data)).astype(precision)
    logT_cube = np.log10(temp_cube.data, where=temp_cube.data > 0.0, 
                        out=np.zeros_like(temp_cube.data)).astype(precision)
    
    # Extract data arrays for calculations but keep reference cube for coordinates
    temp_data = temp_cube.data
    rho_data = rho_cube.data  
    vel_data = vel_cube.data
    reference_cube = temp_cube  # Keep this for coordinate reference

    # ---------------- Load contribution functions -----------------
    print(f"Loading contribution functions ({print_mem()})")
    goft, logT_goft, logN_grid = read_goft(goft_path, limit_lines, precision)

    # Use the GOFT temperature grid as our DEM temperature grid
    logT_grid = logT_goft
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

    # ---------------- Synthesize spectra -----------------
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