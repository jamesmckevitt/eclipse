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
    logNe_flat = np.log10(avg_ne, where=avg_ne > 0.0).transpose(2, 0, 1).ravel()
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

        atom = element(data["atom"])
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

    imI = ax_I.imshow(
        np.log10(total_si.sum(axis=2).T / wl_res),
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
      ax_dem.step(logT_centres, np.log10(dem_1d, where=dem_1d > 0.0), where='mid', lw=1.8)
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
      ax_log.grid(ls=":")

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

def _find_median_pixel(total_si, margin_frac=0.20):
    """Return (i,j) indices of the median-intensity pixel inside the
    central (1-2*margin_frac)² region of total_si."""
    nx, ny = total_si.shape[:2]
    margin = int(margin_frac * min(nx, ny))
    inner  = total_si[margin:nx - margin, margin:ny - margin]
    med    = np.argsort(inner.sum(axis=2).ravel())[len(inner.ravel()) // 2]
    i_in, j_in = np.unravel_index(med, inner.sum(axis=2).shape)
    return (i_in + margin, j_in + margin), margin


def plot_maps(total_si, v_edges, voxel_dx, voxel_dy, downsample,
              median_idx, margin, save="maps.png"):
    """Intensity + Doppler maps (side-by-side)."""
    ds       = downsample if isinstance(downsample, int) and downsample > 1 else 1
    dx_pix   = voxel_dx.to(u.Mm).value * ds
    dy_pix   = voxel_dy.to(u.Mm).value * ds
    nx, ny   = total_si.shape[:2]
    extent   = (0, nx*dx_pix, 0, ny*dy_pix)

    # Doppler map
    v_cent_km = 0.5*(v_edges[:-1] + v_edges[1:]) * u.cm/u.s
    v_cent_km = v_cent_km.to(u.km/u.s).value
    v_map     = v_cent_km[total_si.argmax(axis=2)]

    # wavelength resolution for intensity normalisation
    wl_res = wl_grid_main[1] - wl_grid_main[0]

    fig   = plt.figure(figsize=(11, 5))
    gs    = fig.add_gridspec(nrows=1, ncols=2, wspace=0.0)
    axI   = fig.add_subplot(gs[0, 0])
    axV   = fig.add_subplot(gs[0, 1], sharey=axI)

    # intensity panel
    imI = axI.imshow(np.log10(total_si.sum(axis=2).T / wl_res),
                     origin="lower", aspect="equal", cmap="afmhot", extent=extent)
    x_med = median_idx[0]*dx_pix + dx_pix/2
    y_med = median_idx[1]*dy_pix + dy_pix/2
    axI.scatter(x_med, y_med, color="cyan", s=100, edgecolor="k")
    rect = Rectangle((margin*dx_pix, margin*dy_pix),
                     (nx-2*margin)*dx_pix, (ny-2*margin)*dy_pix,
                     fill=False, edgecolor="cyan", linewidth=1, linestyle="--")
    axI.add_patch(rect)
    axI.set_xlabel("X (Mm)")
    axI.set_ylabel("Y (Mm)")
    axI.set_title("Intensity")
    fig.colorbar(imI, ax=axI, orientation="horizontal", extend="both", shrink=0.9,
                 label=r"$\log_{10}\!\left(\int I(\lambda)\,\mathrm{d}\lambda\right)$ "
                       r"[erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]")

    # velocity panel
    imV = axV.imshow(v_map.T, origin="lower", aspect="equal", cmap="RdBu_r",
                     extent=extent, vmin=-15, vmax=15)
    axV.set_xlabel("X (Mm)")
    axV.set_title("Doppler velocity")
    fig.colorbar(imV, ax=axV, orientation="horizontal", extend="both", shrink=0.9,
                 label=r"$v$  [km s$^{-1}$]")

    for ax in (axI, axV):
        ax.tick_params(direction="in", top=True, bottom=True, left=True, right=True)

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum(goft, total_si, wl_grid_main, median_idx,
                  main_line, save="spectrum.png"):
    """Linear+log spectrum for a single pixel."""
    wl0_A  = goft[main_line]["wl0"].to(u.AA).value
    c_kms  = const.c.to(u.km/u.s).value
    wl2v   = lambda wl: (wl - wl0_A)/wl0_A * c_kms
    v2wl   = lambda v: (v / c_kms) * wl0_A + wl0_A

    fig    = plt.figure(figsize=(4, 5))
    gs     = fig.add_gridspec(nrows=2, ncols=1, hspace=0.0)
    ax_lin = fig.add_subplot(gs[0, 0])
    ax_log = fig.add_subplot(gs[1, 0], sharex=ax_lin)

    cmap = get_cmap("tab10", len(goft))
    for i, (name, info) in enumerate(goft.items()):
        spec_px  = info["si"][median_idx]
        wl_src   = info["wl_grid"].to(u.AA).value
        spec_int = np.interp(wl_grid_main, wl_src, spec_px, left=0.0, right=0.0)
        ax_lin.plot(wl_grid_main, spec_int, color=cmap(i), lw=1.0)
        ax_log.plot(wl_grid_main, spec_int, color=cmap(i), lw=1.0)

    summed = total_si[median_idx]
    ax_lin.plot(wl_grid_main, summed, color="k", lw=2.0)
    ax_log.plot(wl_grid_main, summed, color="k", lw=2.0)

    for ax in (ax_lin, ax_log):
        ax.tick_params(direction="in", top=True, bottom=True, left=True, right=True)

    ax_lin.set_ylabel(r"$I$ (linear)")
    ax_lin.tick_params(axis="x", labelbottom=False)
    ax_log.set_yscale("log")
    ax_log.set_xlabel(r"Wavelength  [$\mathrm{\AA}$]")
    ax_log.set_ylabel(r"$I$ (log)")
    ax_log.set_ylim(ax_log.get_ylim()[1]/1e7, ax_log.get_ylim()[1])

    sec = ax_lin.secondary_xaxis("top", functions=(wl2v, v2wl))
    sec.set_xlabel(r"Velocity  [km s$^{-1}$]")
    sec.tick_params(direction="in")

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dem(dem_map, logT_centres, median_idx, ylim=(25, 29),
             xlim=(5.5, 7.0), save="dem.png"):
    """DEM(T) for a single pixel."""
    dem_1d = dem_map[median_idx]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.step(logT_centres, np.log10(dem_1d, where=dem_1d>0), where="mid", lw=1.8)
    ax.set_xlabel(r"$\log_{10} T$  [K]")
    ax.set_ylabel(r"$\log_{10} \xi$  [cm$^{-5}$ dex$^{-1}$]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(ls=":")
    plt.tight_layout()
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
    downsample     = False                # factor or False
    primary_lines  = ["Fe12_195.1190", "Fe12_195.1790"]
    main_line      = "Fe12_195.1190"
    limit_lines    = False            # e.g. ['Fe12_195.1190'] to speed up
    vel_res        = 1 * u.km / u.s
    vel_lim        = 300 * u.km / u.s
    voxel_dz       = 0.064 * u.Mm
    voxel_dx, voxel_dy = 0.192 * u.Mm, 0.192 * u.Mm
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
    logN_cube = np.log10(ne_arr.value).astype(precision)
    logT_cube = np.log10(temp_cube.value).astype(precision)
    del rho_cube, temp_cube, ne_arr

    # ----------------------------------------------------------
    # read contribution functions
    # ----------------------------------------------------------
    print(f"Loading contribution functions ({print_mem()})")
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

    # # ----------------------------------------------------------------------
    # # call viewer after all processing is complete
    # # ----------------------------------------------------------------------
    # launch_viewer(
    #     total_si     = total_si,
    #     goft         = goft,
    #     dem_map      = dem_map,
    #     wl_ref       = goft[main_line]["wl_grid"].to(u.AA).value,
    #     v_edges      = v_edges,
    #     logT_centres = logT_centres,
    #     main_line    = main_line,
    # )

    # ----------------------------------------------------------------------
    # save the results
    # ----------------------------------------------------------------------
    output_file = "synthesised_spectra.pkl"
    print(f"Saving results to {output_file} ({print_mem()})")
    with open(output_file, "wb") as f:
        pickle.dump({
            "goft": goft,
            "dem_map": dem_map,
            "total_si": total_si,
            "background_si": back_si,
            "v_edges": v_edges,
            "logT_centres": logT_centres,
        }, f)
    filesize = os.path.getsize(output_file) / 1e9
    print(f"Saved {output_file} ({filesize:.2f} GB)")

    globals().update(locals());raise ValueError("Kicking back to ipython")

    # ----------------------------------------------------------------------
    # paper output with physical axes in Mm
    # ----------------------------------------------------------------------
    median_idx, margin = _find_median_pixel(total_si)
    plot_maps(total_si, v_edges, voxel_dx, voxel_dy, downsample,
              median_idx, margin, save="maps.png")
    plot_spectrum(goft, total_si, wl_grid_main, median_idx,
                  main_line, save="spectrum_median_pixel.png")
    plot_dem(dem_map, logT_centres, median_idx,
            save="dem_median_pixel.png")


    globals().update(locals());raise ValueError("Kicking back to ipython")
if __name__ == "__main__":
    main()