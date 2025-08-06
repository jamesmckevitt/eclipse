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

def load_cube(
    file_path: str | Path,
    shape: Tuple[int, int, int] = (512, 768, 256),
    unit: Optional[u.Unit] = None,
    downsample: int | bool = False,
    precision: type = np.float32,
) -> np.ndarray | u.Quantity:
    """
    Read a Fortran-ordered binary cube (single precision) and return as a
    NumPy array (or Quantity if *unit* is given).

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

    Returns
    -------
    ndarray or Quantity
        Array with shape (nx', ny', nz').
    """
    data = np.fromfile(file_path, dtype=np.float32).reshape(shape, order="F")
    data = data.transpose(0, 2, 1)  # (x,y,z)

    if downsample:
        data = data[::downsample, ::downsample, ::downsample]

    data = data.astype(precision, copy=False)
    return data * unit if unit is not None else data


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
    voxel_dz_cm: float,
    logT_grid: np.ndarray,
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
    voxel_dz_cm : float
        Voxel depth in cm.
    logT_grid : np.ndarray
        1D array of temperature bin centers for DEM calculation.

    Returns
    -------
    dem_map : np.ndarray
        DEM array with shape (nx, ny, nT) [cm^-5 per dex].
    avg_ne : np.ndarray
        Mean electron density per T-bin with shape (nx, ny, nT) [cm^-3].
    """
    nx, ny, _ = logT_cube.shape
    nT = len(logT_grid)
    
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

    dem = np.zeros((nx, ny, nT))
    avg_ne = np.zeros_like(dem)

    for idx in tqdm(range(nT), desc="DEM bins", unit="bin", leave=False):
        lo, hi = logT_edges[idx], logT_edges[idx + 1]
        mask = (logT_cube >= lo) & (logT_cube < hi)  # (nx,ny,nz)

        em = np.sum(w2 * mask, axis=2) * voxel_dz_cm    # cm^-5
        em_n = np.sum(w3 * mask, axis=2) * voxel_dz_cm  # cm^-5 * n_e

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
    vz_cube: np.ndarray,
    logT_grid: np.ndarray,
    v_edges: np.ndarray,
    ne_sq_dh: np.ndarray,
) -> np.ndarray:
    """
    Construct 4-D emission-measure cube EM(x,y,T,v) [cm^-5].
    
    Parameters
    ----------
    logT_cube : np.ndarray
        3D temperature cube.
    vz_cube : np.ndarray
        3D velocity cube.
    logT_grid : np.ndarray
        Temperature bin centers.
    v_edges : np.ndarray
        Velocity bin edges.
    ne_sq_dh : np.ndarray
        n_e^2 * dh for each voxel.
        
    Returns
    -------
    em_tv : np.ndarray
        4D emission measure cube (nx, ny, nT, nv).
    """
    print("  Building 4-D emission-measure cube...")
    
    # Create temperature bin edges from centers
    dlogT = logT_grid[1] - logT_grid[0] if len(logT_grid) > 1 else 0.1
    logT_edges = np.concatenate([
        [logT_grid[0] - dlogT/2],
        logT_grid[:-1] + dlogT/2,
        [logT_grid[-1] + dlogT/2]
    ])
    
    mask_T = (logT_cube[..., None] >= logT_edges[:-1]) & \
             (logT_cube[..., None] <  logT_edges[1:])
    mask_V = (vz_cube.value[..., None] >= v_edges[:-1]) & \
             (vz_cube.value[..., None] <  v_edges[1:])

    # Build the 4-D emission-measure cube EM(x,y,T,v) by summing over the z-axis
    ne_sq_dh_d = da.from_array(ne_sq_dh, chunks='auto')
    mask_T_d   = da.from_array(mask_T,   chunks='auto')
    mask_V_d   = da.from_array(mask_V,   chunks='auto')
    em_tv_d = da.einsum("ijk,ijkl,ijkm->ijlm",
                        ne_sq_dh_d, mask_T_d, mask_V_d, optimize=True)
    with ProgressBar():
        em_tv = em_tv_d.compute()

    return em_tv


def synthesise_spectra(
    goft: Dict[str, dict],
    em_tv: np.ndarray,
    v_edges: np.ndarray,
    logT_grid: np.ndarray,
    vel_grid: np.ndarray,
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
    v_edges : np.ndarray
        Velocity bin edges.
    logT_grid : np.ndarray
        Temperature bin centers.
    vel_grid : np.ndarray
        Velocity grid for wavelength calculation.
    """
    kb = const.k_B.cgs.value
    c_cm_s = const.c.cgs.value
    v_centres = 0.5 * (v_edges[:-1] + v_edges[1:])  # (nv,)

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
        lam_cent = wl0 * (1 + v_centres / c_cm_s)

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
    voxel_dx: u.Quantity,
    voxel_dy: u.Quantity,
    intensity_unit: u.Unit,
) -> NDCube:
    """
    Create an NDCube for a single spectral line.
    
    Parameters
    ----------
    line_name : str
        Name of the spectral line.
    line_data : dict
        Dictionary containing line data with 'si', 'wl_grid', 'wl0'.
    voxel_dx, voxel_dy : u.Quantity
        Spatial pixel sizes.
    intensity_unit : u.Unit
        Unit for the intensity data.
        
    Returns
    -------
    NDCube
        Cube with proper WCS and metadata.
    """
    cube_data = line_data["si"]  # (nx,ny,n_lambda)
    nx, ny, nl = cube_data.shape

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['WAVE', 'SOLY', 'SOLX']
    wcs.wcs.cunit = ['cm', 'Mm', 'Mm']
    wcs.wcs.crpix = [(nl + 1) / 2, (ny + 1) / 2, (nx + 1) / 2]
    wcs.wcs.crval = [line_data["wl0"].to(u.cm).value, 0, 0]
    wcs.wcs.cdelt = [
        np.diff(line_data["wl_grid"].to(u.cm).value)[0],
        voxel_dy.to(u.Mm).value,
        voxel_dx.to(u.Mm).value
    ]

    return NDCube(
        cube_data,
        wcs=wcs,
        unit=intensity_unit,
        meta={
            "line_name": line_name,
            "rest_wav": line_data["wl0"],
            "atom": line_data["atom"],
            "ion": line_data["ion"]
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
    parser.add_argument("--vz-file", type=str, default="vz/result_prim_2.0270000",
                       help="Velocity file relative to data-dir")
    
    # Grid parameters
    parser.add_argument("--cube-shape", nargs=3, type=int, default=[512, 768, 256],
                       help="Cube dimensions (nx ny nz)")
    parser.add_argument("--voxel-dx", type=float, default=0.192,
                       help="Voxel size in x (Mm)")
    parser.add_argument("--voxel-dy", type=float, default=0.192,
                       help="Voxel size in y (Mm)")
    parser.add_argument("--voxel-dz", type=float, default=0.064,
                       help="Voxel size in z (Mm)")
    
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
        "vz": args.vz_file,
    }
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
    print(f"  Velocity grid: ±{vel_lim:.1f} at {vel_res:.1f} resolution")
    print(f"  Precision: {precision}")
    if downsample:
        print(f"  Downsampling: {downsample}x")
    if limit_lines:
        print(f"  Limited to lines: {limit_lines}")
    print()

    # ---------------- Build grids -----------------
    # Velocity grid (symmetric about zero, inclusive)
    vel_grid = np.arange(
        -vel_lim.to(u.cm / u.s).value,
        vel_lim.to(u.cm / u.s).value + vel_res.to(u.cm / u.s).value,
        vel_res.to(u.cm / u.s).value
    ) * (u.cm / u.s)
    
    dv_cm_s = vel_grid[1].cgs.value - vel_grid[0].cgs.value
    v_edges = np.concatenate([
        vel_grid.value - 0.5 * dv_cm_s,
        [vel_grid.value[-1] + 0.5 * dv_cm_s]
    ])

    # ---------------- Load simulation data -----------------
    print(f"Loading cubes ({print_mem()})")
    temp_cube = load_cube(paths["T"], shape=tuple(args.cube_shape), unit=u.K, 
                         downsample=downsample, precision=precision)
    rho_cube = load_cube(paths["rho"], shape=tuple(args.cube_shape), unit=u.g/u.cm**3, 
                        downsample=downsample, precision=precision)
    vz_cube = load_cube(paths["vz"], shape=tuple(args.cube_shape), unit=u.cm/u.s, 
                       downsample=downsample, precision=precision)

    # Convert to log10 temperature and density
    ne_arr = (rho_cube / (mean_mol_wt * const.u.cgs.to(u.g))).to(1/u.cm**3)
    logN_cube = np.log10(ne_arr.value, where=ne_arr.value > 0.0, 
                        out=np.zeros_like(ne_arr.value)).astype(precision)
    logT_cube = np.log10(temp_cube.value, where=temp_cube.value > 0.0, 
                        out=np.zeros_like(temp_cube.value)).astype(precision)
    del rho_cube, temp_cube, ne_arr

    # ---------------- Load contribution functions -----------------
    print(f"Loading contribution functions ({print_mem()})")
    goft, logT_goft, logN_grid = read_goft(goft_path, limit_lines, precision)

    # Use the GOFT temperature grid as our DEM temperature grid
    logT_grid = logT_goft
    dh_cm = voxel_dz.to(u.cm).value

    # ---------------- Calculate DEM -----------------
    print(f"Calculating DEM and average density per bin ({print_mem()})")
    dem_map, avg_ne_map = compute_dem(logT_cube, logN_cube, dh_cm, logT_grid)

    print(f"Interpolating contribution function on the DEM ({print_mem()})")
    interpolate_g_on_dem(goft, avg_ne_map, logT_grid, logN_grid, logT_goft, precision)

    # ---------------- Build EM(T,v) cube -----------------
    ne_sq_dh = (10.0 ** logN_cube.astype(np.float64)) ** 2 * dh_cm
    print(f"Calculating emission measure cube in (T,v) space ({print_mem()})")
    em_tv = build_em_tv(logT_cube, vz_cube, logT_grid, v_edges, ne_sq_dh)

    # ---------------- Synthesize spectra -----------------
    print(f"Synthesising spectra ({print_mem()})")
    synthesise_spectra(goft, em_tv, v_edges, logT_grid, vel_grid)

    # ---------------- Create output cubes -----------------
    print(f"Creating output cubes ({print_mem()})")
    line_cubes = {}
    for name, info in goft.items():
        line_cubes[name] = create_line_cube(name, info, voxel_dx, voxel_dy, intensity_unit)
    
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
        "v_edges": v_edges,
        "vel_grid": vel_grid,
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
        }
    }
    
    with open(output_file, "wb") as f:
        dill.dump(results_data, f)
    
    print(f"Saved results to {output_file} ({os.path.getsize(output_file) / 1e6:.2f} MB)")
    print("Synthesis complete!")

if __name__ == "__main__":
    main()