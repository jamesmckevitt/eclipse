import os
import pickle
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from scipy.io import readsav
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import dask.array as da
from dask.diagnostics import ProgressBar
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle
from mendeleev import element
import dill
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from matplotlib.legend_handler import HandlerTuple
from matplotlib.legend_handler import HandlerBase
from ndcube import NDCube
from astropy.wcs import WCS
from astropy.visualization import LogStretch, ImageNormalize
from types import MethodType
from astropy.coordinates import SkyCoord
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
    unit: u.Unit | None = None,
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
    file_path
        Path to the binary file.
    shape
        Tuple (nx, ny, nz) describing the *full* cube dimensions.
    unit
        Astropy unit to attach (e.g. u.K or u.g/u.cm**3). If None, returns
        a plain ndarray.
    downsample
        Integer factor; if non-False, keep every *downsample*-th cell along
        each axis (simple stride).
    precision
        np.float32 or np.float64 for returned dtype.

    Returns
    -------
    ndarray or Quantity with shape (nx', ny', nz').
    """
    data = np.fromfile(file_path, dtype=np.float32).reshape(shape, order="F")
    data = data.transpose(0, 2, 1)                        # (x,y,z)

    if downsample:
        data = data[::downsample, ::downsample, ::downsample]

    data = data.astype(precision, copy=False)
    return data * unit if unit is not None else data


def read_goft(
    sav_file: str | Path,
    limit_lines: list[str] | bool = False,
    precision: type = np.float64,
) -> Tuple[Dict[str, dict], np.ndarray, np.ndarray]:
    """
    Read a CHIANTI G(T,N) .sav file produced by IDL.

    Returns
    -------
    goft_dict
        Dictionary keyed by line name, each entry holding:
            'wl0'      - rest wavelength (Quantity, cm)
            'g_tn'     - 2-D array G(logT, logN)  [erg cm^3 s^-1]
            'wl_grid'  - placeholder for later wavelength grid
            'background' - filled later (True for background lines)
    logT_grid
        1-D array of log10(T/K) values.
    logN_grid
        1-D array of log10(N_e/cm^3) values.
    """
    raw = readsav(sav_file)
    goft_dict: Dict[str, dict] = {}

    logT_grid = raw["logTarr"].astype(precision)
    logN_grid = raw["logNarr"].astype(precision)

    for entry in raw["goftarr"]:
        line_name = entry[0].decode()          # e.g. "Fe12_195.1190"
        if limit_lines and line_name not in limit_lines:
            continue

        rest_wl = float(line_name.split("_")[1]) * u.AA      # A -> Quantity
        goft_dict[line_name] = {
            "wl0": rest_wl.to(u.cm),
            "g_tn": entry[4].astype(precision),              # erg cm^3 / s
            "atom": entry[1],                                 # e.g. 26 (for Fe)
            "ion": entry[2],                                 # e.g. 12 (for Fe XII)
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
    logT_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the differential emission measure DEM(T) and the emission-measure
    weighted mean electron density <n_e>(T).

    Returns
    -------
    dem_map  : (nx, ny, nT) - DEM  [cm^-5 per dex]
    avg_ne   : (nx, ny, nT) - mean n_e  [cm^-3]
    """
    nx, ny, _ = logT_cube.shape
    nT = len(logT_edges) - 1
    dlogT = logT_edges[1] - logT_edges[0]

    ne = 10.0 ** logN_cube.astype(np.float64)
    w2 = ne**2                       # weights for EM
    w3 = ne**3                       # weights for EM*n_e

    dem = np.zeros((nx, ny, nT))
    avg_ne = np.zeros_like(dem)

    for idx in tqdm(range(nT), desc="DEM bins", unit="bin", leave=False):
        lo, hi = logT_edges[idx], logT_edges[idx + 1]
        mask = (logT_cube >= lo) & (logT_cube < hi)          # (nx,ny,nz)

        em   = np.sum(w2 * mask, axis=2) * voxel_dz_cm       # cm^-5
        em_n = np.sum(w3 * mask, axis=2) * voxel_dz_cm       # cm^-5 * n_e

        dem[..., idx] = em / dlogT
        avg_ne[..., idx] = np.divide(em_n, em, where=em > 0.0)

    return dem, avg_ne


def interpolate_g_on_dem(
    goft: Dict[str, dict],
    avg_ne: np.ndarray,
    logT_centres: np.ndarray,
    logN_grid: np.ndarray,
    logT_grid: np.ndarray,
    precision: type = np.float32,
) -> None:
    """
    For every spectral line insert a cube g(x,y,T) evaluated at DEM
    bin centres and at the emission-measure weighted density <n_e>(T).
    """
    nT, nx, ny = len(logT_centres), *avg_ne.shape[:2]

    # build list of query points (logN, logT) for every (T-bin, pixel)
    logNe_flat = np.log10(avg_ne, where=avg_ne > 0.0, out=np.zeros_like(avg_ne)).transpose(2, 0, 1).ravel()
    logT_flat = np.broadcast_to(logT_centres[:, None, None],
                                (nT, nx, ny)).ravel()
    query_pts = np.column_stack((logNe_flat, logT_flat))

    for name, info in tqdm(goft.items(), desc="interpolating G", unit="line", leave=False):
        rgi = RegularGridInterpolator(
            (logN_grid, logT_grid), info["g_tn"],
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
    logT_edges: np.ndarray,
    v_edges: np.ndarray,
    ne_sq_dh: np.ndarray,
) -> np.ndarray:
    """
    Construct 4-D emission-measure cube EM(x,y,T,v) [cm^-5].
    """
    print("  Building 4-D emission-measure cube...")
    mask_T = (logT_cube[..., None] >= logT_edges[:-1]) & \
             (logT_cube[..., None] <  logT_edges[1:])
    mask_V = (vz_cube.value[..., None] >= v_edges[:-1]) & \
             (vz_cube.value[..., None] <  v_edges[1:])

    # Build the 4-D emission-measure cube EM(x,y,T,v) by summing over the z-axis:
    #   ne_sq_dh has shape (nx,ny,nz)  ->  n_e^2 * dh for each voxel
    #   mask_T   has shape (nx,ny,nz,nT) -> True where logT falls in each T-bin
    #   mask_V   has shape (nx,ny,nz,nv) -> True where vz falls in each v-bin
    # einsum string "ijk,ijkl,ijkm->ijlm" multiplies these three arrays
    # element-wise along z and sums over that axis, producing em_tv of shape (nx,ny,nT,nv).
    # This is further parallelised by Dask.

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
    logT_centres: np.ndarray,
    dv_cm_s: float,
) -> None:
    """
    Convolve EM(T,v) with thermal Gaussians plus Doppler shift to obtain the
    specific intensity cube I(x,y,lambda) for every line. The result is
    stored in goft[line]["si"].
    """
    kb = const.k_B.cgs.value
    c_cm_s = const.c.cgs.value
    v_centres = 0.5 * (v_edges[:-1] + v_edges[1:])      # (nv,)

    for line, data in tqdm(goft.items(), desc="spectra", unit="line", leave=False):

        wl0 = data["wl0"].cgs.value                     # cm
        wl_grid = data["wl_grid"].cgs.value                 # (n_lambda,)

        atom = element(int(data["atom"]))
        atom_weight_g = (atom.atomic_weight * u.u).cgs.value

        # thermal width per T-bin: sigma_T (nT,)
        sigma_T = wl0 * np.sqrt(2 * kb * (10 ** logT_centres) / atom_weight_g) / c_cm_s

        # Doppler-shifted center for each v-bin: (nv,)
        lam_cent = wl0 * (1 + v_centres / c_cm_s)

        # build phi(T,v,lambda) as (nT,nv,n_lambda)
        delta = wl_grid[None, None, :] - lam_cent[None, :, None]
        phi = np.exp(-0.5 * (delta / sigma_T[:, None, None]) ** 2)
        phi /= sigma_T[:, None, None] * np.sqrt(2 * np.pi)

        # EM(x,y,T,v) * G(T)  ->  (nx,ny,nT,nv)
        weighted = em_tv * data["g"][..., None]

        # collapse T and v: dot ((nT,nv) , (nT,nv)) -> (nx,ny,n_lambda)
        spec_map = np.tensordot(weighted, phi, axes=([2, 3], [0, 1]))

        data["si"] = spec_map / (4 * np.pi)


##############################################################################
# ---------------------------------------------------------------------------
#  Spectrum combiner
# ---------------------------------------------------------------------------
##############################################################################

def combine_lines(goft: dict, main_line: str):
    """
    Sum primary + background spectra on the wavelength grid of *main_line*.

    Returns
    -------
    total_si      : (nx,ny,nl)  -> primary + background
    background_si : (nx,ny,nl)  -> background only
    """
    wl_ref = goft[main_line]["wl_grid"].value
    nx, ny, nl = goft[main_line]["si"].shape

    total_si      = np.zeros((nx, ny, nl))
    background_si = np.zeros_like(total_si)

    for name, entry in goft.items():
        wl_src = entry["wl_grid"].value              # 1-D, length nl_src
        cube   = entry["si"]                         # (nx,ny,nl_src)

        # loop over all LOS pixels and interpolate their 1-D spectrum
        for i in range(nx):
            for j in range(ny):
                spec_1d = cube[i, j, :]              # 1-D array
                interp  = np.interp(wl_ref, wl_src, spec_1d,
                                    left=0.0, right=0.0)
                if entry["background"]:
                    background_si[i, j, :] += interp
                else:
                    total_si[i, j, :]      += interp

    total_si += background_si
    return total_si, background_si



##############################################################################
# ---------------------------------------------------------------------------
#                 M A I N   W O R K F L O W
# ---------------------------------------------------------------------------
##############################################################################

def main() -> None:
    # ---------------- user-tunable parameters -----------------
    precision      = np.float64       # global float dtype
    downsample     = False                # factor or False
    primary_lines  = ["Fe12_195.1190", "Fe12_195.1790"]
    main_line      = "Fe12_195.1190"
    limit_lines    = False            # e.g. ['Fe12_195.1190'] to speed up
    vel_res        = 5 * u.km / u.s
    vel_lim        = 300 * u.km / u.s
    voxel_dz       = 0.064 * u.Mm
    voxel_dx, voxel_dy = 0.192 * u.Mm, 0.192 * u.Mm
    if downsample:
        voxel_dz *= downsample
        voxel_dx *= downsample
        voxel_dy *= downsample
    mean_mol_wt    = 1.29                    # solar [doi:10.1051/0004-6361:20041507]
    # ----------------------------------------------------------

    # build velocity grid (symmetric about zero, inclusive)
    vel_grid = np.arange(-vel_lim.to(u.cm / u.s).value,
                          vel_lim.to(u.cm / u.s).value + vel_res.to(u.cm / u.s).value,
                          vel_res.to(u.cm / u.s).value) * (u.cm / u.s)
    dv_cm_s = vel_grid[1].cgs.value - vel_grid[0].cgs.value
    v_edges = np.concatenate([vel_grid.value - 0.5 * dv_cm_s,
                              [vel_grid.value[-1] + 0.5 * dv_cm_s]])

    # file paths
    base_dir = Path("data/atmosphere")
    files = dict(
        T   = "temp/eosT.0270000",
        rho = "rho/result_prim_0.0270000",
        vz  = "vz/result_prim_2.0270000",
    )
    paths = {k: base_dir / fname for k, fname in files.items()}

    # ----------------------------------------------------------
    # load simulation cubes
    # ----------------------------------------------------------
    print_mem = lambda: f"{psutil.virtual_memory().used/1e9:.2f}/" \
                        f"{psutil.virtual_memory().total/1e9:.2f} GB"

    print(f"Loading cubes ({print_mem()})")
    temp_cube = load_cube(paths["T"],   unit=u.K,         downsample=downsample, precision=precision)
    rho_cube  = load_cube(paths["rho"], unit=u.g/u.cm**3, downsample=downsample, precision=precision)
    vz_cube   = load_cube(paths["vz"],  unit=u.cm/u.s,    downsample=downsample, precision=precision)

    nx, ny, nz = temp_cube.shape

    # convert to log10 temperature and density
    ne_arr = (rho_cube / (mean_mol_wt * const.u.cgs.to(u.g))).to(1/u.cm**3)
    logN_cube = np.log10(ne_arr.value, where=ne_arr.value > 0.0, out=np.zeros_like(ne_arr.value)).astype(precision)
    logT_cube = np.log10(temp_cube.value, where=temp_cube.value > 0.0, out=np.zeros_like(temp_cube.value)).astype(precision)
    del rho_cube, temp_cube, ne_arr

    # ----------------------------------------------------------
    # read contribution functions
    # ----------------------------------------------------------
    print(f"Loading contribution functions ({print_mem()})")
    goft, logT_grid, logN_grid = read_goft("./data/gofnt.sav", limit_lines, precision)

    # attach wavelength grid and mark background lines
    for name, info in goft.items():
        info["wl_grid"] = (vel_grid * info["wl0"] / const.c + info["wl0"]).cgs
        info["background"] = name not in primary_lines

    # ----------------------------------------------------------
    # DEM and G interpolation
    # ----------------------------------------------------------
    logT_edges = np.linspace(logT_grid.min(), logT_grid.max(), len(logT_grid) + 1)
    logT_centres = 0.5 * (logT_edges[:-1] + logT_edges[1:])
    dh_cm = voxel_dz.to(u.cm).value

    print(f"Calculating DEM and average density per bin ({print_mem()})")
    dem_map, avg_ne_map = compute_dem(logT_cube, logN_cube, dh_cm, logT_edges)

    print(f"Interpolating contribution function on the DEM ({print_mem()})")
    interpolate_g_on_dem(goft, avg_ne_map, logT_centres,
                         logN_grid, logT_grid, precision)

    # ----------------------------------------------------------
    # EM(T,v) cube
    # ----------------------------------------------------------
    ne_sq_dh = (10.0 ** logN_cube.astype(np.float64)) ** 2 * dh_cm
    print(f"Calculating an emissivity cube in temperature and velocity space ({print_mem()})")
    em_tv = build_em_tv(logT_cube, vz_cube, logT_edges, v_edges, ne_sq_dh)

    # ----------------------------------------------------------
    # synthesis of spectra
    # ----------------------------------------------------------
    print(f"Synthesising spectra ({print_mem()})")
    synthesise_spectra(goft, em_tv, v_edges, logT_centres,
                       dv_cm_s)

    # ----------------------------------------------------------
    # combine lines
    # ----------------------------------------------------------
    print(f"Combining lines into a single spectrum cube ({print_mem()})")
    total_si, back_si = combine_lines(goft, main_line)
    total_si *= u.erg/u.s/u.cm**2/u.sr/u.cm

    # ----------------------------------------------------------
    # calculate integrated intensities
    # ----------------------------------------------------------
    print(f"Calculating the integrated intensities ({print_mem()})")
    dwl = np.diff(goft[main_line]["wl_grid"].to(u.cm))
    total_ii = np.sum(total_si[:, :, :-1] * dwl, axis=2)

    # ----------------------------------------------------------
    # make heliocentric ndcubes
    # ----------------------------------------------------------

    nx, ny, nl = total_si.shape
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['WAVE', 'SOLY', 'SOLX']
    wcs.wcs.cunit = ['cm', 'Mm', 'Mm']
    wcs.wcs.crpix = [(nl + 1) / 2, (ny + 1) / 2, (nx + 1) / 2]
    wcs.wcs.crval = [goft[main_line]["wl0"].to(u.cm).value,
                     0,
                     0]
    wcs.wcs.cdelt = [np.diff(goft[main_line]["wl_grid"].to(u.cm).value)[0],
                     voxel_dy.to(u.Mm).value,
                     voxel_dx.to(u.Mm).value
                     ]
    sim_si = NDCube(
        total_si,
        wcs=wcs,
        unit=total_si.unit,
        meta={
            "rest_wav": goft[main_line]["wl0"]
        }
    )

    nx, ny = total_ii.shape
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['SOLY', 'SOLX']
    wcs.wcs.cunit = ['Mm', 'Mm']
    wcs.wcs.crpix = [(ny + 1) / 2, (nx + 1) / 2]
    wcs.wcs.cdelt = [voxel_dy.to(u.Mm).value,
                     voxel_dx.to(u.Mm).value]
    sim_ii = NDCube(
        data=total_ii,
        wcs=wcs,
        unit=total_ii.unit,
        meta={
            "rest_wav": goft[main_line]["wl0"]
        }
    )

    # ----------------------------------------------------------
    # build per-line specific-intensity cubes
    # ----------------------------------------------------------
    line_cubes = {}
    for name, info in goft.items():
        cube_line = info["si"]                               # (nx,ny,nλ_line)

        nx, ny, nl = cube_line.shape

        wcs_line = WCS(naxis=3)
        wcs_line.wcs.ctype = ['WAVE', 'SOLY', 'SOLX']
        wcs_line.wcs.cunit = ['cm',   'Mm',   'Mm']
        wcs_line.wcs.crpix = [(nl + 1) / 2, (ny + 1) / 2, (nx + 1) / 2]
        wcs_line.wcs.crval = [info["wl0"].to(u.cm).value, 0, 0]
        wcs_line.wcs.cdelt = [np.diff(info["wl_grid"].to(u.cm).value)[0],
                              voxel_dy.to(u.Mm).value,
                              voxel_dx.to(u.Mm).value]

        line_cubes[name] = NDCube(
            cube_line,
            wcs=wcs_line,
            unit=total_si.unit,
            meta={"rest_wav": info["wl0"]}
        )
    print(f"Built per-line cubes: {len(line_cubes)} lines")

    # ----------------------------------------------------------------------
    # save the results
    # ----------------------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    output_file = scratch_dir / f"synthesised_spectra_{timestamp}.pkl"
    with open(output_file, "wb") as f:
        dill.dump({
            "sim_si": sim_si,
            "sim_ii": sim_ii,
            "line_cubes": line_cubes,
            "dem_map": dem_map,
            "em_tv": em_tv,
            "logT_centres": logT_centres,
            "v_edges": v_edges,
            "goft": goft,
            "logT_grid": logT_grid,
            "logN_grid": logN_grid,
        }, f)
    print(f"Saved the key information to {output_file} ({os.path.getsize(output_file) / 1e6:.2f} MB)")

    dest_path = Path("./run/input/synthesised_spectra.pkl")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_file, dest_path)
    print(f"Copied {output_file} to {dest_path}")

    output_file = scratch_dir / f"synthesised_spectra_session_{timestamp}.pkl"
    globals().update(locals())
    dill.dump_session(output_file)
    print(f"Saved the session to {output_file} ({os.path.getsize(output_file) / 1e9:.2f} GB)")

if __name__ == "__main__":
    main()