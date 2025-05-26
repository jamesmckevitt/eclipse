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
        Astropy unit to attach (e.g. u.K or u.g/u.cm**3).  If None, returns
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

        rest_wl = float(line_name.split("_")[1]) * u.AA      # Å -> Quantity
        goft_dict[line_name] = {
            "wl0": rest_wl.to(u.cm),
            "g_tn": entry[4].astype(precision),              # erg cm^3 / s
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
        with np.errstate(divide="ignore", invalid="ignore"):
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
    logNe_flat = np.log10(avg_ne).transpose(2, 0, 1).ravel()
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
#  Build EM(T,v) and synthesize spectra
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
    mask_T = (logT_cube[..., None] >= logT_edges[:-1]) & \
             (logT_cube[..., None] <  logT_edges[1:])
    mask_V = (vz_cube.value[..., None] >= v_edges[:-1]) & \
             (vz_cube.value[..., None] <  v_edges[1:])

    em_tv = np.einsum("ijk,ijkl,ijkm->ijlm",
                      ne_sq_dh, mask_T, mask_V, optimize=True)
    return em_tv


def synthesize_spectra(
    goft: Dict[str, dict],
    em_tv: np.ndarray,
    v_edges: np.ndarray,
    logT_centres: np.ndarray,
    dv_cm_s: float,
    ion_mass_g: float,
) -> None:
    """
    Convolve EM(T,v) with thermal Gaussians plus Doppler shift to obtain the
    specific intensity cube I(x,y,lambda) for every line.  The result is
    stored in `goft[line]["si"]`.
    """
    kb = const.k_B.cgs.value
    c_cm_s = const.c.cgs.value
    v_centres = 0.5 * (v_edges[:-1] + v_edges[1:])      # (nv,)

    for line, data in tqdm(goft.items(), desc="spectra", unit="line", leave=False):
        wl0 = data["wl0"].cgs.value                     # cm
        wl_grid = data["wl_grid"].cgs.value                 # (n_lambda,)

        # thermal width per T-bin: sigma_T (nT,)
        sigma_T = wl0 * np.sqrt(2 * kb * (10 ** logT_centres) / ion_mass_g) / c_cm_s

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
        data["si"] = spec_map * (dv_cm_s / (4 * np.pi))   # erg s^-1 cm^-2 sr^-1


##############################################################################
# ---------------------------------------------------------------------------
#  Simple spectrum combiner and viewer (optional)
# ---------------------------------------------------------------------------
##############################################################################

def combine_lines(goft: dict, main_line: str):
    """
    Sum primary + background spectra on the wavelength grid of *main_line*.

    Returns
    -------
    total_si      : (nx,ny,nλ)  -> primary + background
    background_si : (nx,ny,nλ)  -> background only
    """
    wl_ref = goft[main_line]["wl_grid"].value
    nx, ny, nλ = goft[main_line]["si"].shape

    total_si      = np.zeros((nx, ny, nλ))
    background_si = np.zeros_like(total_si)

    for name, entry in goft.items():
        wl_src = entry["wl_grid"].value              # 1-D, length nλ_src
        cube   = entry["si"]                         # (nx,ny,nλ_src)

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
    downsample     = 8                # factor or False
    primary_lines  = ["Fe12_195.1190", "Fe12_195.1790"]
    main_line      = "Fe12_195.1190"
    limit_lines    = False            # e.g. ['Fe12_195.1190'] to speed up
    vel_res        = 1 * u.km / u.s
    vel_lim        = 300 * u.km / u.s
    voxel_dz       = 0.064 * u.Mm
    ion_mass       = 55.845 * u.g / u.mol    # Fe atomic weight
    mean_mol_wt    = 1.29                    # solar [doi:10.1051/0004-6361:20041507]
    # ----------------------------------------------------------

    # build velocity grid (symmetric about zero, inclusive)
    vel_grid = np.arange(-vel_lim.to(u.cm / u.s).value,
                          vel_lim.to(u.cm / u.s).value + vel_res.to(u.cm / u.s).value,
                          vel_res.to(u.cm / u.s).value) * (u.cm / u.s)
    dv_cm_s = vel_grid[1].cgs.value - vel_grid[0].cgs.value
    v_edges = np.concatenate([vel_grid.value - 0.5 * dv_cm_s,
                              [vel_grid.value[-1] + 0.5 * dv_cm_s]])

    # # ---------------- coarse v-bins for the DEM -----------------
    # dv_dem = 25 * u.km/u.s                     # coarse LOS bin
    # v_edges_dem = np.arange(-vel_lim, vel_lim + dv_dem, dv_dem).to(u.cm/u.s).value
    # v_centres_dem = 0.5*(v_edges_dem[:-1] + v_edges_dem[1:])
    # nv_dem = len(v_centres_dem)

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

    print(f"Loading cubes       ({print_mem()})")
    temp_cube = load_cube(paths["T"],   unit=u.K,         downsample=downsample, precision=precision)
    rho_cube  = load_cube(paths["rho"], unit=u.g/u.cm**3, downsample=downsample, precision=precision)
    vz_cube   = load_cube(paths["vz"],  unit=u.cm/u.s,    downsample=downsample, precision=precision)

    nx, ny, nz = temp_cube.shape

    # convert to log10 temperature and density
    logT_cube = np.log10(temp_cube.value).astype(precision)
    ne_arr = (rho_cube / (mean_mol_wt * const.u.cgs.to(u.g))).to(1/u.cm**3)
    logN_cube = np.log10(ne_arr.value).astype(precision)
    del rho_cube, temp_cube, ne_arr

    # ----------------------------------------------------------
    # read contribution functions
    # ----------------------------------------------------------
    print(f"Loading G(T,N)      ({print_mem()})")
    goft, logT_grid, logN_grid = read_goft("gofnt.sav", limit_lines, precision)

    # attach wavelength grid and mark background lines
    for name, info in goft.items():
        info["wl_grid"] = (vel_grid * info["wl0"] / const.c + info["wl0"]).cgs
        info["background"] = name not in primary_lines

    # ----------------------------------------------------------
    # DEM and G interpolation
    # ----------------------------------------------------------
    nT_bins = 50
    logT_edges = np.linspace(logT_grid.min(), logT_grid.max(), nT_bins + 1)
    logT_centres = 0.5 * (logT_edges[:-1] + logT_edges[1:])
    dh_cm = voxel_dz.to(u.cm).value

    print(f"DEM / <n_e>          ({print_mem()})")
    dem_map, avg_ne_map = compute_dem(logT_cube, logN_cube, dh_cm, logT_edges)

    # globals().update(locals());raise ValueError("Kicking back to ipython")

    print(f"Interpolate G(T)    ({print_mem()})")
    interpolate_g_on_dem(goft, avg_ne_map, logT_centres,
                         logN_grid, logT_grid, precision)

    # ----------------------------------------------------------
    # EM(T,v) cube
    # ----------------------------------------------------------
    ne_sq_dh = (10.0 ** logN_cube.astype(np.float64)) ** 2 * dh_cm
    print(f"EM(T,v) cube        ({print_mem()})")
    em_tv = build_em_tv(logT_cube, vz_cube, logT_edges, v_edges, ne_sq_dh)

    # ----------------------------------------------------------
    # synthesis of spectra
    # ----------------------------------------------------------
    print(f"Synthesise spectra  ({print_mem()})")
    synthesize_spectra(goft, em_tv, v_edges, logT_centres,
                       dv_cm_s, ion_mass.cgs.value * const.u.cgs.value)

    # ----------------------------------------------------------
    # combine lines and quicklook image
    # ----------------------------------------------------------
    total_si, back_si = combine_lines(goft, main_line)
    wl_grid_main = goft[main_line]["wl_grid"].to(u.AA).value
    wl_res = wl_grid_main[1] - wl_grid_main[0]

    # ----------------------------------------------------------------------
    # interactive quick-look: click any pixel -> spectra + DEM windows
    # ----------------------------------------------------------------------
    def launch_viewer(
            total_si: np.ndarray,
            goft: dict,
            dem_map: np.ndarray,
            em_tv: np.ndarray,
            wl_ref: np.ndarray,
            v_edges: np.ndarray,
            logT_centres: np.ndarray,
            main_line: str,
    ):
        """
        Left-click on the intensity map to pop up
            1) the full spectrum for that LOS pixel,
            2) its 1-D DEM(T),
            3) its 2-D DEM(T,v).
        """

        # helper: wavelength ↔ velocity converters (km/s) for the main line
        wl0_A   = goft[main_line]["wl0"].to(u.AA).value
        c_km_s  = const.c.to(u.km / u.s).value
        wl_to_vel = lambda wl:  (wl - wl0_A) / wl0_A * c_km_s
        vel_to_wl = lambda vel: (vel / c_km_s) * wl0_A + wl0_A

        nx, ny, _ = total_si.shape
        v_centres = 0.5 * (v_edges[:-1] + v_edges[1:])
        nT, nv    = len(logT_centres), len(v_centres)

        # show the log-integrated intensity
        wl_res = wl_ref[1] - wl_ref[0]
        fig_map, ax_map = plt.subplots()
        im = ax_map.imshow(
            np.log10(total_si.sum(axis=2).T) / wl_res,
            origin="lower", cmap="inferno", aspect="equal"
        )
        ax_map.set_title("log10 ∫I(λ) dλ  (click a pixel)")
        plt.colorbar(im, ax=ax_map, label="log10(I)")

        # on-click callback --------------------------------------------------
        def on_click(event):
            if event.inaxes is not ax_map or event.xdata is None or event.ydata is None:
                return
            x_pix, y_pix = int(round(event.xdata)), int(round(event.ydata))
            if not (0 <= x_pix < nx and 0 <= y_pix < ny):
                return

            # 1. spectrum window --------------------------------------------
            fig_spec, ax_spec = plt.subplots(figsize=(6, 4))
            ax_spec.plot(wl_ref, total_si[x_pix, y_pix, :], color="k", lw=1)
            ax_spec.set_xlabel("Wavelength (Å)")
            ax_spec.set_ylabel("Specific intensity")
            ax_spec.set_title(f"Spectrum at pixel ({x_pix}, {y_pix})")
            ax_spec.grid(ls=":")

            # secondary x-axis in velocity
            ax_top = ax_spec.secondary_xaxis("top", functions=(wl_to_vel, vel_to_wl))
            ax_top.set_xlabel("Velocity (km/s)")

            # 2. 1-D DEM window ---------------------------------------------
            fig_dem, ax_dem = plt.subplots(figsize=(4, 4))
            ax_dem.step(logT_centres, dem_map[x_pix, y_pix, :],
                        where="mid", lw=1.5)
            ax_dem.set_xlabel("log10 T (K)")
            ax_dem.set_ylabel("DEM  [cm^-5 per dex]")
            ax_dem.set_title(f"DEM(T) at ({x_pix},{y_pix})")
            ax_dem.grid(ls=":")

            # 3. 2-D DEM(T,v) window ----------------------------------------
            fig_dem2d, ax_dem2d = plt.subplots(figsize=(5, 4))
            dem2d = em_tv[x_pix, y_pix, :, :]   # (nT,nv)
            im2 = ax_dem2d.imshow(
                np.log10(np.clip(dem2d, 1e15, None)),   # avoid log(0)
                origin="lower", aspect="auto",
                extent=[v_centres[0], v_centres[-1],
                        logT_centres[0], logT_centres[-1]],
                cmap="viridis"
            )
            ax_dem2d.set_xlabel("Velocity (cm/s)")
            ax_dem2d.set_ylabel("log10 T (K)")
            ax_dem2d.set_title(f"2-D DEM(T,v) at ({x_pix},{y_pix})")
            plt.colorbar(im2, ax=ax_dem2d, label="log10 EM  [cm^-5]")

            plt.tight_layout()
            plt.show()

        # connect the callback
        fig_map.canvas.mpl_connect("button_press_event", on_click)
        plt.tight_layout()
        plt.show()


    # ----------------------------------------------------------------------
    # call viewer after all processing is complete
    # ----------------------------------------------------------------------
    launch_viewer(
        total_si     = total_si,
        goft         = goft,
        dem_map      = dem_map,
        em_tv        = em_tv,
        wl_ref       = wl_grid_main,
        v_edges      = v_edges,
        logT_centres = logT_centres,
        main_line    = main_line,
    )



if __name__ == "__main__":
    main()