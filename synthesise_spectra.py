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
#  Interactive quick-look viewer
# ---------------------------------------------------------------------------
##############################################################################

def launch_viewer(
    *,                                 # force keyword use
    total_si     : np.ndarray,
    goft         : dict,
    dem_map      : np.ndarray,
    wl_ref       : np.ndarray,
    v_edges      : np.ndarray,
    logT_centres : np.ndarray,
    main_line    : str,
) -> None:
    """
    Opens one overview window (intensity + Doppler map).  Clicking on a pixel
    launches two further windows:

        - DEM(T)    - y-axis in log-scale
        - Spectra   - upper panel linear-y, lower panel log-y
                      - one curve per emission line
                      - thick black curve = summed spectrum
    """

    # ----------------- helper lambdas -----------------
    wl0_A   = goft[main_line]["wl0"].to(u.AA).value
    c_km_s  = const.c.to(u.km/u.s).value
    wl2v    = lambda wl:  (wl - wl0_A) / wl0_A * c_km_s         # A -> km s-1
    v2wl    = lambda vel: (vel / c_km_s)    * wl0_A + wl0_A     # km s-1 -> A

    # velocity grid centres that correspond to wl_ref
    v_centres_km = (0.5 * (v_edges[:-1] + v_edges[1:]) * u.cm/u.s)\
                    .to(u.km/u.s).value

    nx, ny, _ = total_si.shape
    wl_res     = wl_ref[1] - wl_ref[0]

    # ----------------- overview figure -----------------
    fig_ov, (ax_I, ax_V) = plt.subplots(2, 1, figsize=(7, 10))

    si = total_si.sum(axis=2) * wl_res  # integrate over wavelength
    log_si = np.log10(si, where=si > 0.0, out=np.zeros_like(si))

    imI = ax_I.imshow(
        log_si.T,  # log10 of integrated intensity
        origin="lower", aspect="equal", cmap="inferno"
    )
    ax_I.set_title(r"$\log_{10}\!\int I(\lambda)\,\mathrm{d}\lambda$")
    fig_ov.colorbar(imI, ax=ax_I, label=r"$\log_{10} I$  [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ dex$^{-1}$]")

    peak_idx = total_si.argmax(axis=2)             # index of lambda of max signal
    v_map    = v_centres_km[peak_idx]              # Doppler map
    imV = ax_V.imshow(
        v_map.T, origin="lower", aspect="equal",
        cmap="RdBu_r"
    )
    ax_V.set_title("Doppler velocity of peak intensity  [km s-1]")
    imV.set_clim(-30, 30)  # set color limits for velocity map
    fig_ov.colorbar(imV, ax=ax_V, label="v  (km s-1)")

    plt.tight_layout()
    plt.show(block=False)          # keep UI responsive

    # ----------------- click callback -----------------
    def _on_click(event):
      if event.inaxes not in (ax_I, ax_V):        # click somewhere else
        return
      if event.xdata is None or event.ydata is None:
        return

      # pixel indices (round - the images are pixel-aligned)
      x, y = map(int, map(round, (event.xdata, event.ydata)))
      if not (0 <= x < nx and 0 <= y < ny):
        return

      # -------- DEM window --------
      fig_dem, ax_dem = plt.subplots(figsize=(5, 4))
      dem_1d = dem_map[x, y, :]
      log_dem_1d = np.log10(dem_1d, where=dem_1d > 0.0, out=np.zeros_like(dem_1d))
      ax_dem.plot(logT_centres, log_dem_1d, where='mid', lw=1.8)
      ax_dem.set_xlabel(r"\log_{10} T  [K]")
      ax_dem.set_ylabel(r"\log_{10} DEM  [cm$^{-5}$ dex$^{-1}$]")
      ax_dem.set_title(f"DEM(T)  -  pixel ({x},{y})")
      ax_dem.grid(ls=":")
      fig_dem.tight_layout()
      plt.show(block=False)

      # -------- spectra window --------
      fig_sp, (ax_lin, ax_log) = plt.subplots(2, 1, sharex=True,
                          figsize=(7, 8))
      line_names   = list(goft.keys())
      n_lines      = len(line_names)
      cmap         = get_cmap("tab10", n_lines)

      # summed spectrum (already on wl_ref grid)
      summed = total_si[x, y, :]

      # plot each emission line - thin coloured curves
      for i, name in enumerate(line_names):
        spec_px   = goft[name]["si"][x, y, :]
        wl_src    = goft[name]["wl_grid"].to(u.AA).value
        spec_int  = np.interp(wl_ref, wl_src, spec_px, left=0.0, right=0.0)
        lbl       = f"{name}" + ("  (bg)" if goft[name]["background"] else "")
        ax_lin.plot(wl_ref, spec_int, color=cmap(i), lw=1.0, label=lbl)
        ax_log.plot(wl_ref, spec_int, color=cmap(i), lw=1.0)

      # summed spectrum - thick black curve on top
      ax_lin.plot(wl_ref, summed, color="k", lw=2.0, label="total")
      ax_log.plot(wl_ref, summed, color="k", lw=2.0)

      # axis cosmetics for linear panel
      ax_lin.set_ylabel("I  (linear)")
      ax_lin.set_title(f"Spectrum - pixel ({x},{y})")
      ax_lin.grid(ls=":")

      # axis cosmetics for log panel
      ax_log.set_yscale("log")
      ax_log.set_xlabel("Wavelength  (A)")
      ax_log.set_ylabel("I  (log)")
      # ax_log.grid(ls:")

      # add secondary x-axis (velocity) to both panels
      for ax in (ax_lin, ax_log):
        sec = ax.secondary_xaxis("top", functions=(wl2v, v2wl))
        sec.set_xlabel("Velocity  (km s-1)")

      # adjust log y-axis limits: bottom 10 orders below top
      y_top = ax_log.get_ylim()[1]
      ax_log.set_ylim(y_top / 1e10, y_top)

      ax_lin.legend(fontsize="small", ncol=2)

      fig_sp.tight_layout()
      plt.show(block=False)

    # connect callback
    fig_ov.canvas.mpl_connect("button_press_event", _on_click)

##############################################################################
# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
##############################################################################

def _find_mean_sigma_pixel(ndcube, margin_frac=0.20, sigma_factor=1.0):
    """
    Return (mean_coord, plus_sigma_coord, minus_sigma_coord)
    corresponding to the pixel intensities nearest to the mean and
    mean +- (sigma_factor)*std, as SkyCoord coordinates using pixel_to_world.
    """

    # Account for margin and crop to inner region
    nx, ny = ndcube.data.shape
    margin = int(margin_frac * min(nx, ny))
    inner = ndcube.data[margin:nx - margin, margin:ny - margin]

    # Calculate mean and standard deviation
    mean_val = np.mean(inner)
    std_val = np.std(inner)
    plus_sigma_val = mean_val + sigma_factor * std_val
    minus_sigma_val = mean_val - sigma_factor * std_val

    # Find indices of the closest pixels to mean, mean + sigma, and mean - sigma
    mean_idx = np.unravel_index(np.argmin(np.abs(inner - mean_val)), inner.shape)
    plus_sigma_idx = np.unravel_index(np.argmin(np.abs(inner - plus_sigma_val)), inner.shape)
    minus_sigma_idx = np.unravel_index(np.argmin(np.abs(inner - minus_sigma_val)), inner.shape)

    # Convert to global pixel indices
    mean_idx_global = (mean_idx[0] + margin, mean_idx[1] + margin)
    plus_sigma_idx_global = (plus_sigma_idx[0] + margin, plus_sigma_idx[1] + margin)
    minus_sigma_idx_global = (minus_sigma_idx[0] + margin, minus_sigma_idx[1] + margin)

    # Use pixel_to_world to get SkyCoord coordinates
    mean_coord = ndcube.wcs.pixel_to_world(mean_idx_global[1], mean_idx_global[0])
    plus_sigma_coord = ndcube.wcs.pixel_to_world(plus_sigma_idx_global[1], plus_sigma_idx_global[0])
    minus_sigma_coord = ndcube.wcs.pixel_to_world(minus_sigma_idx_global[1], minus_sigma_idx_global[0])

    return mean_coord, plus_sigma_coord, minus_sigma_coord


def calculate_doppler_map(total_si, v_edges):
    # Helper: Gaussian function
    def gaussian(x, amp, mu, sigma, offset):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

    v_cent_km = 0.5 * (v_edges[:-1] + v_edges[1:]) * (u.cm / u.s)
    v_cent_km = v_cent_km.to(u.km / u.s).value
    nx, ny, nl = total_si.shape
    v_map = np.full((nx, ny), np.nan)

    def fit_pixel(i, j):
        spectrum = total_si[i, j, :]
        try:
            amp0 = spectrum.max()
            mu0 = v_cent_km[np.argmax(spectrum)]
            sigma0 = 10  # km/s, rough guess
            offset0 = 0
            popt, _ = curve_fit(
                gaussian, v_cent_km, spectrum,
                p0=[amp0, mu0, sigma0, offset0],
                maxfev=5000
            )
            return (i, j, popt[1])  # centroid (mu)
        except Exception:
            return (i, j, np.nan)

    pixel_indices = [(i, j) for i in range(nx) for j in range(ny)]

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(fit_pixel)(i, j) for i, j in tqdm(pixel_indices, desc="Doppler map", unit="pixel", leave=False)
    )

    for i, j, mu in results:
        v_map[i, j] = mu

    return v_map


def plot_maps(
  total_si, v_map, voxel_dx, voxel_dy, downsample, margin, wl_grid_main, save,
  mean_idx=None, plus_sigma_idx=None, minus_sigma_idx=None, sigma_factor=1.0,
  key_pixel_colors=None
):
  """
  Intensity + Doppler maps (side-by-side), with mean and +-(sigma_factor) pixels marked
  on both panels. One shared legend on the velocity panel.
  key_pixel_colors: list of 3 colors for plus_sigma, mean, minus_sigma pixels.
  """
  ds = downsample if isinstance(downsample, int) and downsample > 1 else 1
  dx_pix = voxel_dx.to(u.Mm).value * ds
  dy_pix = voxel_dy.to(u.Mm).value * ds
  nx, ny = total_si.shape[:2]
  extent = (0, nx * dx_pix, 0, ny * dy_pix)

  wl_res = wl_grid_main[1] - wl_grid_main[0]

  # Remove the Gaussian fitting logic from here, as it's now precomputed
  fig = plt.figure(figsize=(11, 5))
  gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.0)
  axI = fig.add_subplot(gs[0, 0])
  axV = fig.add_subplot(gs[0, 1], sharey=axI)

  si = total_si.sum(axis=2) * wl_res  # integrate over wavelength
  log_si = np.log10(si, where=si > 0.0, out=np.zeros_like(si))

  # Intensity panel with lower colorbar limit set to 0
  imI = axI.imshow(
    log_si.T,  # log10 of integrated intensity
    origin="lower", aspect="equal", cmap="afmhot",
    extent=extent, vmin=0.0
  )
  # margin of shortest side
  this_margin = int(margin * min(nx, ny))
  rect = Rectangle(
    (this_margin * dx_pix, this_margin * dy_pix),
    (nx - 2 * this_margin) * dx_pix, (ny - 2 * this_margin) * dy_pix,
    fill=False, edgecolor="black", linewidth=1, linestyle="--"
  )
  axI.add_patch(rect)
  axI.set_xlabel("X (Mm)")
  axI.set_ylabel("Y (Mm)")
  cbarI = fig.colorbar(imI, ax=axI, orientation="horizontal", extend="neither", aspect=35, shrink=0.95, pad=0.115)
  cbarI.set_label(
    r"$\log_{10}\!\left(\int I(\lambda)\,\mathrm{d}\lambda\mathrm{ }\left[\mathrm{erg/s/cm}^2\mathrm{/sr}\right]\right)$"
  )
  axI.tick_params(direction="in", top=True, bottom=True, left=True, right=True)

  # Doppler panel
  imV = axV.imshow(
    v_map.T, origin="lower", aspect="equal", cmap="RdBu_r",
    extent=extent, vmin=-15, vmax=15
  )
  rect = Rectangle(
    (this_margin * dx_pix, this_margin * dy_pix),
    (nx - 2 * this_margin) * dx_pix, (ny - 2 * this_margin) * dy_pix,
    fill=False, edgecolor="black", linewidth=1, linestyle="--"
  )
  axV.add_patch(rect)
  axV.tick_params(labelleft=False, direction="in", top=True, bottom=True, right=True, left=True)
  axV.set_xlabel("X (Mm)")

  # Doppler panel colorbar (thin, below the axis)
  cbarV = fig.colorbar(imV, ax=axV, orientation="horizontal", extend="both", aspect=35, shrink=0.95, pad=0.115)
  cbarV.set_label(r"$v$ [km/s]")

  # Markers on both panels
  if mean_idx and plus_sigma_idx and minus_sigma_idx:
    # idxs = [minus_sigma_idx, mean_idx, plus_sigma_idx]
    # base_labels = [
    #   rf"$\mu - {sigma_factor:.0f}\sigma$",
    #   r"$\mu$",
    #   rf"$\mu + {sigma_factor:.0f}\sigma$"
    # ]
    # markers = ["1", "3", "2"]
    idxs = [plus_sigma_idx, mean_idx, minus_sigma_idx]
    base_labels = [
      rf"$\mu + {sigma_factor:.0f}\sigma$",
      r"$\mu$",
      rf"$\mu - {sigma_factor:.0f}\sigma$"
    ]
    # # Add intensity and velocity to each label
    # for i, idx in enumerate(idxs):
    #   I_1d = total_si[idx[0], idx[1], :]
    #   I_val = I_1d.sum() * wl_res
    #   logI_val = np.log10(I_val) if I_val > 0 else -np.inf
    #   v_val = v_map[idx[0], idx[1]]
    #   base_labels[i] += f"I={I_val:.2e}, v={v_val:.1f} km/s"
    markers = ["2", "3", "1"]
    if key_pixel_colors is None:
      colors = ["tab:blue", "tab:green", "tab:orange"]
    else:
      # reverse the order to match idxs
      colors = key_pixel_colors[::-1]
    for base_label, idx, marker, color in zip(base_labels, idxs, markers, colors):
      I_1d = total_si[idx[0], idx[1], :]
      I_val = I_1d.sum() * wl_res
      logI_val = np.log10(I_val) if I_val > 0 else -np.inf
      v_val = v_map[idx[0], idx[1]]
      label = f"{base_label}"

      # Mark intensity panel
      axI.scatter(
        idx[0] * dx_pix + dx_pix / 2,
        idx[1] * dy_pix + dy_pix / 2,
        color=color, s=250, marker=marker, linewidth=2,
      )
      # Mark velocity panel + legend
      axV.scatter(
        idx[0] * dx_pix + dx_pix / 2,
        idx[1] * dy_pix + dy_pix / 2,
        color=color, s=250, marker=marker, linewidth=2,
        label=label
      )
    axV.legend(loc="upper right", fontsize="small")

  plt.tight_layout()
  plt.savefig(save, dpi=600, bbox_inches="tight")
  plt.close(fig)


def plot_dems(
  dem_map,
  em_tv,
  logT_centres,
  v_edges,
  plus_idx,
  mean_idx,
  minus_idx,
  sigma_factor,
  xlim=(5.5, 7.0),
  ylim_dem=(25, 29),
  ylim_2d_dem=None,
  save="dem_and_2d_dem.png",
  goft=None,
  main_line="Fe12_195.1190",
  key_pixel_colors=None,
  logT_grid=None,
  logN_grid=None,
  figsize=(15, 7),
  cbar_offset=1.25,
  inset_axis_offset=0.8,
):
  """
  Plot DEM(T) (top row) and DEM(T,v) maps (bottom row) for three pixels.
  A slim, transparent inset axis overlays each map on the left, showing
  the line profile (intensity vs. velocity) without resizing the map.
  The right y-axis (wavelength) is now exactly aligned with the left y-axis (velocity).
  key_pixel_colors: list of 3 colors for plus_sigma, mean, minus_sigma pixels.
  """

  idxs   = [minus_idx, mean_idx, plus_idx]
  # titles = [r"$\mu-\sigma$", r"$\mu$", r"$\mu+\sigma$"]
  titles = [
    rf"$\mu - {sigma_factor:.0f}\sigma$",
    r"$\mu$",
    rf"$\mu + {sigma_factor:.0f}\sigma$"
  ]

  v_centres = 0.5 * (v_edges[:-1] + v_edges[1:]) * u.cm/u.s
  v_centres_kms = v_centres.to(u.km/u.s).value
  extent = (logT_centres[0], logT_centres[-1],
        v_centres_kms[0],    v_centres_kms[-1])

  # define primary->secondary and secondary->primary for wavelength axis
  if goft and main_line in goft:
    wl0   = goft[main_line]["wl0"].to(u.angstrom).value
    c_kms = const.c.to(u.km/u.s).value
    v2wl  = lambda v: wl0 * (1 + v / c_kms)
    wl2v  = lambda wl: (wl - wl0) / wl0 * c_kms

  fig, axes = plt.subplots(
    2, 3, figsize=figsize,
    sharey="row", gridspec_kw=dict(wspace=0.0, hspace=0.0)
  )

  # -------------------------------------------------------- top row DEM(T)
  if key_pixel_colors is None:
      colours = ["tab:blue", "tab:green", "tab:orange"]
  else:
      colours = key_pixel_colors
  for ax, idx, title, c in zip(axes[0], idxs, titles, colours):
    log_dem = np.log10(dem_map[idx], where=dem_map[idx] > 0,
                out=np.zeros_like(dem_map[idx]))
    ax.step(logT_centres, log_dem, where="mid", color=c, lw=1.8)
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim_dem)
    ax.set_xlabel(r"$\log_{10} T$  [K]")
    ax.grid(ls=":")
  axes[0, 0].set_ylabel(
    r"$\log_{10}\left(\xi\:\mathrm{[1/cm}^{5}/\mathrm{dex}\mathrm{]}\right)$"
  )

  # ------------------------------------------- build 2-D DEM arrays + limits
  datas, vmin, vmax = [], None, None
  for idx in idxs:
    data = np.log10(em_tv[idx].T, where=em_tv[idx].T > 0,
              out=np.zeros_like(em_tv[idx].T))
    data[data == 0] = np.nan
    datas.append(data)

    mx = (logT_centres >= xlim[0]) & (logT_centres <= xlim[1])
    my = np.ones_like(v_centres_kms, dtype=bool)
    if ylim_2d_dem:
      my &= (v_centres_kms >= ylim_2d_dem[0]) & (v_centres_kms <= ylim_2d_dem[1])
    sub = data[np.ix_(my, mx)]
    vmin = np.nanmin(sub) if vmin is None else min(vmin, np.nanmin(sub))
    vmax = np.nanmax(sub) if vmax is None else max(vmax, np.nanmax(sub))

  # ----------------------------------------------------- bottom row maps
  ims = []
  for i, ax in enumerate(axes[1]):
    im = ax.imshow(
      datas[i], origin="lower", aspect="auto",
      cmap="Purples", extent=extent,
      vmin=vmin, vmax=vmax
    )
    ims.append(im)
    ax.set_xlim(*xlim)
    if ylim_2d_dem:
      ax.set_ylim(*ylim_2d_dem)
    ax.set_xlabel(r"$\log_{10}\left(T\:\mathrm{[K]}\right)$")
    ax.grid(ls=":")
    # inset profile
    if goft and main_line in goft:
      spec = goft[main_line]["si"][idxs[i]]

      stick = inset_axes(
        ax, width="50%", height="100%",
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=ax.transAxes,
        loc="lower left", borderpad=0,
      )
      stick.set_facecolor("none")

      stick.set_axisbelow(False)
      for spine in stick.spines.values():
        spine.set_zorder(3)

      stick.plot(spec, v_centres_kms, color="red", lw=1.3, zorder=1)

      stick.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
      stick.xaxis.set_ticks_position("top")
      stick.xaxis.set_label_position("top")
      stick.tick_params(axis="x", labelsize=7, direction="in")

      fig.canvas.draw()
      offset_text = stick.xaxis.get_offset_text().get_text().replace("1e", "")
      stick.xaxis.get_offset_text().set_visible(False)

      stick.set_xlabel(
        f"Fe XII 195.119 $\\AA$ intensity \n [$10^{{{offset_text}}}$ erg/s/cm$^{{2}}$/sr/cm]",
        fontsize=8, zorder=3
      )

      ticks = stick.get_xticks()
      ticks = ticks[ticks != 0]
      stick.set_xticks(ticks)

      stick.yaxis.set_ticks([])

      stick.spines[['right', 'bottom', 'left']].set_visible(False)
      stick.spines['top'].set_position(('axes', inset_axis_offset))

      stick.set_xlim(0, spec.max()*1.05)
      stick.set_ylim(ax.get_ylim())

  # ------------------------------------------------ right‐hand wavelength axis on bottom panels
  if goft and main_line in goft:
    for ax in axes[1]:
      v_ticks = ax.get_yticks()
      vmin_ax, vmax_ax = ax.get_ylim()
      v_ticks = v_ticks[(v_ticks >= vmin_ax) & (v_ticks <= vmax_ax)]
      wl_ticks = v2wl(v_ticks)
      wl_ticklabels = [f"{wl:.3f}" for wl in wl_ticks]
      ax_r = ax.secondary_yaxis("right")
      ax_r.set_yticks(v_ticks)
      ax_r.set_yticklabels(wl_ticklabels)
      ax_r.set_ylabel(r"Wavelength (Fe XII 195.119 $\AA$) [$\AA$]")
      ax.tick_params(axis='y', direction='in', which='both')
      ax_r.tick_params(axis='y', direction='in', which='both')

  # ------------------------------------------------ Tick parameters for all main panels
  for row in axes:
    for ax in row:
      ax.tick_params(direction="in", top=True, bottom=True)

      # add ticks on the right y axis
      ax.yaxis.set_ticks_position("both")
      ax.yaxis.set_tick_params(which="both", direction="in", right=True)

  axes[1, 0].set_ylabel("Velocity [km/s]")

  # ------------------------------------------------ shared colourbar (move to right of all panels)
  cax = inset_axes(
      axes[1, -1], width="3%", height="90%",
      loc="center left",
      bbox_to_anchor=(cbar_offset, 0., 1, 1),
      bbox_transform=axes[1, -1].transAxes,
      borderpad=0,
  )
  cbar = fig.colorbar(
    ims[0], cax=cax, orientation="vertical", extend="min"
  )
  cbar.set_label(
    r"$\log_{10}\,\left(\Xi\:\mathrm{[1/cm}^{5}/\mathrm{dex}/\Delta v] \right)$"
  )

  # --------------------------------------------------------
  # Overplot the contribution function
  # --------------------------------------------------------
  if goft and main_line in goft and logT_grid is not None and logN_grid is not None:
          g_tn = goft[main_line]["g_tn"]
          integrated_g_over_T = np.trapz(g_tn, logT_grid, axis=1)
          optimal_density_idx = np.argmax(integrated_g_over_T)
          optimal_density = logN_grid[optimal_density_idx]
          g_at_optimal_density = g_tn[optimal_density_idx, :]

          # Compute mean and standard deviation of logT weighted by G
          mean_logT = np.average(logT_grid, weights=g_at_optimal_density)
          std_logT = np.sqrt(np.average((logT_grid - mean_logT)**2, weights=g_at_optimal_density))

          # Positions of vertical lines at mean +- xsigma
          vlines = [mean_logT - 2*std_logT, mean_logT + 2*std_logT]

          # --------------------------------------------------------
          # Add vertical lines to all panels
          # --------------------------------------------------------
          for ax_row in axes:
                  for ax in ax_row:
                          for vline in vlines:
                                #   ax.axvline(vline, color='grey', linestyle='--', linewidth=1, alpha=1.0, zorder=0)
                                  ax.axvline(vline, color='grey', linestyle=(0,(5,10)), linewidth=1, alpha=1.0, zorder=0)

          # --------------------------------------------------------
          # Add annotation for vertical lines at top right
          # --------------------------------------------------------
          # Add a custom legend entry for the vertical lines at mean +- xsigma
          custom_line = Line2D([0], [0], color='grey', linestyle=(0,(5,5)), linewidth=1, alpha=1.0)
          handles, labels = axes[0, -1].get_legend_handles_labels()
          handles.append(custom_line)
          labels.append(fr"$G_{{\mathrm{{Fe\,XII\,195.119}}}}(T,\,N_{{e}}={optimal_density:.1f})\pm2\sigma_{{G}}$")
          axes[0, -1].legend(handles=handles, labels=labels, loc="upper right", fontsize="small")

  plt.tight_layout()
  plt.savefig(save, dpi=300, bbox_inches="tight")
  plt.close(fig)

def plot_spectrum(
    goft: dict,
    total_si: np.ndarray,
    wl_grid_main: np.ndarray,
    minus_idx: int,
    mean_idx: int,
    plus_idx: int,
    main_line: str = "Fe12_195.1190",
    secondary_line: str = "Fe12_195.1790",
    main_label: str | None = None,
    secondary_label: str | None = None,
    key_pixel_colors: tuple[str, str, str] | None = None,
    sigma_factor: float = 1.0,
    xlim_vel: tuple[float, float] | None = None,   # (v_min, v_max) km s-1
    yorders: float | None = None,                  # log-y span below ymax
    ylimits: tuple[float, float] | None = None,     # optional (ymin, ymax) for bottom row
    save: str = "spectrum.png",
    figsize: tuple[float, float] = (15, 8),
) -> None:
    """
    Plot three spectra (μ - sigma, μ, μ + sigma) in two rows (linear + log y).

    New parameters:
    main_label, secondary_label:
        If provided, these override the legend labels for main_line
        and secondary_line respectively.
    """

    # ---------------------------------------------------------------------
    # constants and helpers
    # ---------------------------------------------------------------------
    wl0_A = goft[main_line]["wl0"].to(u.AA).value
    c_kms = const.c.to(u.km / u.s).value

    wl2v = lambda wl: (wl - wl0_A) / wl0_A * c_kms          # A -> km s-1
    v2wl = lambda v: wl0_A * (1 + v / c_kms)                # km s-1 -> A

    # velocity limits -> wavelength limits
    wl_min, wl_max = (None, None)
    if xlim_vel is not None:
        wl_min, wl_max = v2wl(xlim_vel[0]), v2wl(xlim_vel[1])

    # default total-spectrum colours
    if key_pixel_colors is None:
        key_pixel_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:3]

    # determine legend labels
    label_main = main_label if main_label is not None else main_line
    label_sec  = secondary_label if secondary_label is not None else secondary_line

    # ---------------------------------------------------------------------
    # figure & axes grid
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(
        2, 3,
        figsize=figsize,
        sharex="col", sharey="row",
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
    )

    pixel_indices = (minus_idx, mean_idx, plus_idx)
    col_titles = (
        rf"$\mu - {sigma_factor:.0f}\sigma$",
        r"$\mu$",
        rf"$\mu + {sigma_factor:.0f}\sigma$",
    )

    # ---------------------------------------------------------------------
    # main plotting loop
    # ---------------------------------------------------------------------
    for col, (pix, title, tot_colour) in enumerate(
        zip(pixel_indices, col_titles, key_pixel_colors)
    ):
        ax_lin = axes[0, col]   # linear y-axis (upper row)
        ax_log = axes[1, col]   # log    y-axis (lower row)

        # plot every individual line
        for line_name, info in goft.items():
            wl_src = info["wl_grid"].to(u.AA).value
            spec_int = np.interp(wl_grid_main, wl_src, info["si"][pix],
                                 left=0.0, right=0.0)

            if line_name == main_line:
                colr, z, lw = "red", 3, 1.2
            elif line_name == secondary_line:
                colr, z, lw = "blue", 3, 1.2
            else:
                colr, z, lw = "darkgrey", 1, 0.8

            ax_lin.plot(wl_grid_main, spec_int, c=colr, lw=lw, zorder=z)
            ax_log.plot(wl_grid_main, spec_int, c=colr, lw=lw, zorder=z)

        # summed spectrum
        ax_lin.plot(wl_grid_main, total_si[pix], c=tot_colour, lw=2.0, zorder=4)
        ax_log.plot(wl_grid_main, total_si[pix], c=tot_colour, lw=2.0, zorder=4)

        # basic cosmetics
        ax_lin.set_title(title)
        ax_lin.grid(ls=":", alpha=0.5)
        ax_log.grid(ls=":", alpha=0.5)
        ax_lin.tick_params(direction="in", which="both", top=True, right=True)
        ax_log.tick_params(direction="in", which="both", top=True, right=True)

        ax_log.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
        ax_log.set_yscale("log")

        # enforce wavelength limits (velocity follows automatically)
        if wl_min is not None:
            ax_lin.set_xlim(wl_min, wl_max)

        # ================================================================
        # velocity & wavelength ticks
        # ================================================================
        if xlim_vel is not None:
            v_min, v_max = xlim_vel
        else:
            v_min, v_max = wl2v(ax_lin.get_xlim())

        v_locator = MaxNLocator(nbins=5, symmetric=True)
        v_ticks   = v_locator.tick_values(v_min, v_max)

        sec = ax_lin.secondary_xaxis("top", functions=(wl2v, v2wl))
        sec.set_xlabel(f"Velocity ({main_label}) [km/s]")
        sec.set_xticks(v_ticks)
        sec.set_xticklabels([f"{v:.0f}" for v in v_ticks])
        sec.tick_params(direction="in", which="both", top=True, right=True)

        wl_ticks = v2wl(v_ticks)
        wl_labels = [f"{wl:.3f}" for wl in wl_ticks]

        fixed_loc = FixedLocator(wl_ticks)
        fixed_fmt = FixedFormatter(wl_labels)
        ax_lin.xaxis.set_major_locator(fixed_loc)
        ax_lin.xaxis.set_major_formatter(fixed_fmt)
        ax_log.xaxis.set_major_locator(fixed_loc)
        ax_log.xaxis.set_major_formatter(fixed_fmt)

    # ---------------------------------------------------------------------
    # optional tweaks: log-y lower limit & y-axis sci-format
    # ---------------------------------------------------------------------
    for ax in axes[0]:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.set_ylim(bottom=0)

    if yorders is not None:
        for ax in axes[1]:
            _, ymax = ax.get_ylim()
            ax.set_ylim(ymax / 10 ** yorders, ymax)

    # override bottom-row y-limits if provided
    if ylimits is not None:
        ymin, ymax = ylimits
        for ax in axes[1]:
            ax.set_ylim(ymin, ymax)

    # ---------------------------------------------------------------------
    # set top and bottom y-axis labels
    # ---------------------------------------------------------------------
    fig.canvas.draw()
    offset_text = axes[0, 0].yaxis.get_offset_text().get_text().replace("1e", "")
    axes[0, 0].yaxis.get_offset_text().set_visible(False)
    axes[0, 0].set_ylabel(
        fr"$\mathrm{{Intensity\:[10^{{{offset_text}}}\:erg/s/cm^2/sr/cm]}}$"
    )
    axes[1, 0].set_ylabel(
        r"Intensity [erg/s/cm$^2$/sr/cm]"
    )

    # # ---------------------------------------------------------------------
    # # legend (upper-left panel)
    # # ---------------------------------------------------------------------
    # lines_for_total = [
    #   plt.Line2D([0], [0], c=key_pixel_colors[0], lw=2.0),
    #   plt.Line2D([0], [0], c=key_pixel_colors[1], lw=2.0),
    #   plt.Line2D([0], [0], c=key_pixel_colors[2], lw=2.0),
    # ]
    # handles = [
    #   plt.Line2D([0], [0], c="red",      lw=1.2, label=label_main),
    #   plt.Line2D([0], [0], c="blue",     lw=1.2, label=label_sec),
    #   plt.Line2D([0], [0], c="darkgrey", lw=0.8, label="Background"),
    #   tuple(lines_for_total),
    # ]
    # labels = [label_main, label_sec, "Background", "Total"]
    # axes[0, 0].legend(
    #   handles=handles,
    #   labels=labels,
    #   loc="upper left",
    #   fontsize="small",
    #   handler_map={tuple: HandlerTuple(ndivide=None)}
    # )

    # ------------------------------------------------------------------
    #  custom handler: three coloured strokes separated by "/"
    # ------------------------------------------------------------------
    class HandlerTripleWithSlash(HandlerBase):
        """
        Draw three lines separated by forward slashes, with a little
        horizontal padding around every slash.
        """
        def __init__(self, pad_frac=0.08, **kw):
            super().__init__(**kw)
            self.pad_frac = pad_frac    # fraction of the full handle-width

        def create_artists(self, legend, tup, x0, y0, width, height, fontsize, trans):
            # unpack the three Line2D that came in as a tuple
            l1, l2, l3 = tup

            # total free space taken up by two slashes  (each drawn as "/")
            slash_width = width * 0.08
            pad = width * self.pad_frac

            # compute segment widths: split remaining space equally in three
            w_line = (width - 2*slash_width - 4*pad) / 3.0

            # helpful y-coordinate (centre of legend handle)
            yc = y0 + height * 0.5

            artists = []

            # ------- first coloured stroke -------
            x_left = x0
            x_right = x_left + w_line
            artists.append(Line2D([x_left, x_right], [yc, yc],
                                color=l1.get_color(), lw=l1.get_linewidth(),
                                solid_capstyle='butt', transform=trans))

            # ------- slash -------
            x_left = x_right + pad
            x_right = x_left + slash_width
            artists.append(Line2D([x_left, x_right], [y0 + height*0.1,
                                                    y0 + height*0.9],
                                color="k", lw=1.0, transform=trans))

            # ------- second coloured stroke -------
            x_left = x_right + pad
            x_right = x_left + w_line
            artists.append(Line2D([x_left, x_right], [yc, yc],
                                color=l2.get_color(), lw=l2.get_linewidth(),
                                solid_capstyle='butt', transform=trans))

            # ------- second slash -------
            x_left = x_right + pad
            x_right = x_left + slash_width
            artists.append(Line2D([x_left, x_right], [y0 + height*0.1,
                                                    y0 + height*0.9],
                                color="k", lw=1.0, transform=trans))

            # ------- third coloured stroke -------
            x_left = x_right + pad
            x_right = x_left + w_line
            artists.append(Line2D([x_left, x_right], [yc, yc],
                                color=l3.get_color(), lw=l3.get_linewidth(),
                                solid_capstyle='butt', transform=trans))

            return artists

    lines_for_total = [plt.Line2D([0], [0], c=c, lw=2.0) for c in key_pixel_colors]

    handles = [
        Line2D([0], [0], c='red',  lw=1.2, label=label_main),
        Line2D([0], [0], c='blue', lw=1.2, label=label_sec),
        Line2D([0], [0], c='grey', lw=0.8, label='Background'),
        tuple(lines_for_total),
    ]

    labels = [label_main, label_sec, "Background", "Total"]

    axes[0, 0].legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            fontsize="small",
            handlelength=3,                 # make the handle box a bit wider
            handler_map={tuple: HandlerTripleWithSlash(pad_frac=0.1)})

    # ---------------------------------------------------------------------
    # finish up
    # ---------------------------------------------------------------------
    plt.tight_layout(pad=0.1)
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_g_function(
    goft,
    line_name,
    logT_grid,
    logN_grid,
    save="g_function.png",
    xlim=None,
    ylim=None,
    show_vlines=True,
    figsize=(6, 5)
):
    """
    Plot G(logT, logN) for the specified emission line as a 2D colored image.
    Optionally restrict x/y limits and plot vertical lines at mean+-2sigma for the
    optimal density (where G is maximized when integrated over T).
    """
    g_data = goft[line_name]["g_tn"]

    # Optionally crop logT_grid and g_data to xlim
    if xlim is not None:
        mask_T = (logT_grid >= xlim[0]) & (logT_grid <= xlim[1])
        g_data = g_data[:, mask_T]
        logT_grid_plot = logT_grid[mask_T]
    else:
        logT_grid_plot = logT_grid

    # Optionally crop logN_grid and g_data to ylim
    if ylim is not None:
        mask_N = (logN_grid >= ylim[0]) & (logN_grid <= ylim[1])
        g_data = g_data[mask_N, :]
        logN_grid_plot = logN_grid[mask_N]
    else:
        logN_grid_plot = logN_grid

    fig, ax = plt.subplots(figsize=figsize)
    # # use pcolormesh so the axes “know” the exact grid values
    # # X runs over logT, Y over logN, and g_data[i,j] corresponds to (logT_grid_plot[j], logN_grid_plot[i])
    # X, Y = np.meshgrid(logT_grid_plot, logN_grid_plot)
    # im = ax.pcolormesh(
    #   X, Y, g_data,
    #   shading='nearest',
    #   cmap='Greys'
    # )
    extent = (
        logT_grid_plot[0], logT_grid_plot[-1],
        logN_grid_plot[0], logN_grid_plot[-1]
    )
    im = ax.imshow(
        g_data,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="Greys",
    )
    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    # Move the order of magnitude from the colorbar to the label
    fig.canvas.draw()
    offset_text = cbar.ax.yaxis.get_offset_text().get_text().replace("1e", "")
    cbar.ax.yaxis.get_offset_text().set_visible(False)
    cbar.set_label(rf"$G_{{\mathrm{{Fe\,XII\,195.119\,\AA}}}}(T,N)$ [$10^{{{offset_text}}}$ erg cm$^3$/s]")
    ax.set_xlabel(r"$\log_{10}(T\:\mathrm{[K]})$")
    ax.set_ylabel(r"$\log_{10}(N_e\:\mathrm{[1/cm^{3}]})$")

    # Plot vertical lines at mean+-2sigma for the optimal density
    if show_vlines:
        g_tn = goft[line_name]["g_tn"]
        # Use possibly cropped logT_grid and logN_grid for vlines
        g_tn_plot = g_tn
        if ylim is not None:
            g_tn_plot = g_tn_plot[mask_N, :]
        if xlim is not None:
            g_tn_plot = g_tn_plot[:, mask_T]
        # Integrate G over T for each density
        integrated_g_over_T = np.trapz(g_tn_plot, logT_grid_plot, axis=1)
        optimal_density_idx = np.argmax(integrated_g_over_T)
        optimal_density = logN_grid_plot[optimal_density_idx]
        g_at_optimal_density = g_tn_plot[optimal_density_idx, :]

        # Compute mean and std of logT weighted by G
        mean_logT = np.average(logT_grid_plot, weights=g_at_optimal_density)
        std_logT = np.sqrt(np.average((logT_grid_plot - mean_logT) ** 2, weights=g_at_optimal_density))
        vlines = [mean_logT - 2 * std_logT, mean_logT + 2 * std_logT]
        for vline in vlines:
            ax.axvline(
                vline,
                color='grey',
                linestyle=(0, (5, 10)),
                linewidth=1,
                alpha=1.0,
                zorder=2,
            )
        # Add annotation for the vlines
        ax.legend(
            [plt.Line2D([0], [0], color='grey', linestyle=(0, (5, 5)), linewidth=1)],
            # [fr"$G(T,{{N_{{e}}={optimal_density:.1f}}})\pm 2\sigma_{{G}}$"],
            [fr"$G_{{\mathrm{{Fe\,XII\,195.119\,\AA}}}}(T,\,N_{{e}}={optimal_density:.1f})\pm2\sigma_{{G}}$"],
            loc="upper right",
            fontsize="small",
        )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)

##############################################################################
# ---------------------------------------------------------------------------
#                 M A I N   W O R K F L O W
# ---------------------------------------------------------------------------
##############################################################################

def main() -> None:
    # ---------------- user-tunable parameters -----------------
    precision      = np.float64       # global float dtype
    downsample     = 4                # factor or False
    primary_lines  = ["Fe12_195.1190", "Fe12_195.1790"]
    main_line      = "Fe12_195.1190"
    limit_lines    = ['Fe12_195.1190']            # e.g. ['Fe12_195.1190'] to speed up
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

    sigma_factor = 1.0
    margin = 0.2
    mean_coords, plus_coords, minus_coords = _find_mean_sigma_pixel(
        sim_ii, margin, sigma_factor=sigma_factor
    )

    # ----------------------------------------------------------------------
    # save the results
    # ----------------------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    output_file = scratch_dir / f"synthesised_spectra_{timestamp}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump({
            "sim_si": sim_si,
            "sim_ii": sim_ii,
            "plus_coords": plus_coords,
            "mean_coords": mean_coords,
            "minus_coords": minus_coords,
            "sigma_factor": sigma_factor,
            "margin": margin,
        }, f)
    print(f"Saved the key information to {output_file} ({os.path.getsize(output_file) / 1e6:.2f} MB)")

    dest_path = Path("./run/input/synthesised_spectra.pkl")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_file, dest_path)
    print(f"Copied {output_file} to {dest_path}")

    output_file = f"synthesised_spectra_session_{timestamp}.pkl"
    globals().update(locals())
    dill.dump_session(output_file)
    print(f"Saved the session to {output_file} ({os.path.getsize(output_file) / 1e9:.2f} GB)")

    # # ----------------------------------------------------------------------
    # # call viewer after all processing is complete
    # # ----------------------------------------------------------------------
    # launch_viewer(
    #     total_si     = total_si.value,
    #     goft         = goft,
    #     dem_map      = dem_map,
    #     wl_ref       = goft[main_line]["wl_grid"].to(u.AA).value,
    #     v_edges      = v_edges,
    #     logT_centres = logT_centres,
    #     main_line    = main_line,
    # )

    # # ----------------------------------------------------------------------
    # # plot output
    # # ----------------------------------------------------------------------

    # # Calculate Doppler velocity map before plotting
    # print(f"Calculating Doppler velocity map ({print_mem()})")
    # v_map = calculate_doppler_map(total_si.value, v_edges)

    # key_pixel_colors = ["deeppink", "black", "mediumseagreen"]  # in order of minus, mean, plus

    # plot_maps(
    #     total_si.value, v_map, voxel_dx, voxel_dy, downsample, margin,
    #     goft[main_line]["wl_grid"].cgs.value, "fig_synthetic_maps.png",
    #     mean_idx=mean_idx, plus_sigma_idx=plus_idx, minus_sigma_idx=minus_idx,
    #     sigma_factor=sigma_factor, key_pixel_colors=key_pixel_colors
    # )

    # plot_dems(
    #     dem_map, em_tv, logT_centres, v_edges,
    #     plus_idx, mean_idx, minus_idx, sigma_factor,
    #     xlim=(5.5, 6.9),
    #     ylim_dem=(26.5, 30),
    #     ylim_2d_dem=(-50, 50),
    #     save="fig_synthetic_dems.png",
    #     goft=goft, main_line=main_line,
    #     key_pixel_colors=key_pixel_colors,
    #     logT_grid=logT_grid,
    #     logN_grid=logN_grid,
    #     figsize=(12, 6),
    #     cbar_offset=1.325,
    #     inset_axis_offset=0.77,
    # )

    # plot_g_function(
    #     goft, main_line, logT_grid, logN_grid,
    #     save="fig_g_function.png",
    #     xlim=(5.5, 6.9),
    #     show_vlines=True,
    #     figsize=(6, 3)
    # )

    # plot_spectrum(
    #     goft, total_si.value, goft[main_line]["wl_grid"].to('AA').value,
    #     minus_idx, mean_idx, plus_idx,
    #     main_line=main_line, secondary_line="Fe12_195.1790",
    #     key_pixel_colors=key_pixel_colors,
    #     sigma_factor=sigma_factor,
    #     save="fig_synthetic_spectra.png",
    #     xlim_vel=(-250, 250),
    #     yorders=9,
    #     ylimits=(None, 8e13),
    #     main_label="Fe XII 195.119",
    #     secondary_label="Fe XII 195.179",
    #     figsize=(12, 6)
    # )

if __name__ == "__main__":
    main()