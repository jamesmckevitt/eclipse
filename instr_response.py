from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple
import math

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.nddata import NDData
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
from scipy.ndimage import zoom
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import convolve2d
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
from typing import Tuple
import contextlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import matplotlib
import dill
import os

# -----------------------------------------------------------------------------
# Configuration objects
# -----------------------------------------------------------------------------
@dataclass
class Detector:
    qe_vis: float = 1.0
    qe_euv: float = 0.76
    e_per_ph_euv: u.Quantity = 18.0 * u.electron / u.photon
    e_per_ph_vis: u.Quantity = 2.0 * u.electron / u.photon
    read_noise_rms: u.Quantity = 10.0 * u.electron / u.pixel
    dark_current: u.Quantity = 1.0 * u.electron / (u.pixel * u.s)
    gain_e_per_dn: u.Quantity = 2.0 * u.electron / u.DN
    max_dn: u.Quantity = 65535 * u.DN / u.pixel
    pix_size: u.Quantity = (13.5 * u.um).cgs / u.pixel
    wvl_res: u.Quantity = (16.9 * u.mAA).cgs / u.pixel  # EUVST
    # wvl_res: u.Quantity = (22.3 * u.mAA).cgs / u.pixel  # EIS
    plate_scale_angle: u.Quantity = 0.159 * u.arcsec / u.pixel  # EUVST
    # plate_scale_angle: u.Quantity = 1 * u.arcsec / u.pixel  # EIS
    si_fano: float = 0.115

    @property
    def plate_scale_length(self) -> u.Quantity:
        return angle_to_distance(self.plate_scale_angle * 1*u.pix) / u.pixel


@dataclass
class Telescope:
    D_ap: u.Quantity = 0.28 * u.m
    pm_eff: float = 0.161  # EUVST
    grat_eff: float = 0.0623
    filt_eff: float = 0.507
    # pm_eff: float = 1  # EIS
    # grat_eff: float = 1
    # filt_eff: float = 1
    psf_focus_res: u.Quantity = 0.5 * u.um / u.pixel
    psf_mesh_res: u.Quantity = 6.12e-4 * u.mm / u.pixel
    psf_focus_file: Path = Path("data/swc/psf_euvst_v20230909_195119_focus.txt")
    psf_mesh_file: Path = Path("data/swc/psf_euvst_v20230909_derived_195119_mesh.txt")
    psf: np.ndarray | None = field(default=None, init=False)

    @property
    def collecting_area(self) -> u.Quantity:
        return 0.5 * np.pi * (self.D_ap / 2) ** 2  # EUVST
        # return (0.23/0.76) * (u.cm)**2  # EIS


@dataclass
class Simulation:
    expos: u.Quantity = u.Quantity([20], u.s)
    n_iter: int = 2
    vis_sl: u.Quantity = 1 * u.photon / (u.s * u.pixel)
    slit_width: u.Quantity = 0.2 * u.arcsec  # EUVST 0.2" slit
    slit_scan_step: u.Quantity = 0.2 * u.arcsec
    # slit_width: u.Quantity = 1 * u.arcsec  # EIS 1" slit
    # slit_scan_step: u.Quantity = 1 * u.arcsec
    ncpu: int = -1

    def __post_init__(self):
        if self.slit_width != self.slit_scan_step:
            raise NotImplementedError(
                "Code cannot yet properly handle different values for slit width and scan step size. "
            )

# Global configuration used by plotting helpers --------------------------------
DET = Detector()
TEL = Telescope()
SIM = Simulation()


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def wl_to_vel(wl: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    """Convert wavelength to line-of-sight velocity."""
    return (wl - wl0) / wl0 * const.c


def vel_to_wl(v: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    """Convert line-of-sight velocity to wavelength."""
    return wl0 * (1 + v / const.c)


def gaussian(wave, peak, centre, sigma, back):
    return peak * np.exp(-0.5 * ((wave - centre) / sigma) ** 2) + back


def fano_noise(E: float, fano: float) -> int:
    sigma = np.sqrt(fano * E)
    n = np.random.normal(loc=E, scale=sigma)
    return int(max(round(n), 0))


def angle_to_distance(angle: u.Quantity) -> u.Quantity:
    if angle.unit.physical_type != "angle":
        raise ValueError("Input must be an angle")
    return 2 * const.au * np.tan(angle.to(u.rad) / 2)


def save_maps(path: str | Path, log_intensity: np.ndarray, v_map: u.Quantity,
              x_pix_size: float, y_pix_size: float) -> None:
    """Save intensity and velocity maps for later comparison."""
    np.savez(
        path,
        log_si=log_intensity,
        v_map=v_map.to(u.km / u.s).value,
        x_pix_size=x_pix_size,
        y_pix_size=y_pix_size,
    )


def load_maps(path: str | Path) -> dict:
    """Load previously saved intensity and velocity maps."""
    dat = np.load(path)
    return dict(
        log_si=dat["log_si"],
        v_map=dat["v_map"],
        x_pix_size=float(dat["x_pix_size"]),
        y_pix_size=float(dat["y_pix_size"]),
    )


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager that patches joblib so it uses the supplied tqdm
    instance to report progress.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):  # type: ignore[attr-defined]
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack  # type: ignore[attr-defined]
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# -----------------------------------------------------------------------------
# PSF handling
# -----------------------------------------------------------------------------

def _load_psf_ascii(fname: Path, skip: int) -> np.ndarray:
    return np.loadtxt(fname, skiprows=skip, encoding="utf-16 LE")


def _resample_psf(psf: np.ndarray, res_in: u.Quantity, res_out: u.Quantity) -> np.ndarray:
    factor = (res_in / res_out).decompose().value
    return zoom(psf, factor, order=1)


def _combine_psfs(psf_focus: np.ndarray, psf_mesh: np.ndarray, crop: float = 0.99, size: int | None = None) -> np.ndarray:
    """Convolve focus and mesh PSF and crop to given energy or size."""
    psf = convolve2d(psf_focus, psf_mesh, mode="same")
    if size is not None:
        if size % 2 == 0:
            size += 1
        r0, c0 = np.array(psf.shape) // 2
        half = size // 2
        psf = psf[r0 - half : r0 + half + 1, c0 - half : c0 + half + 1]
    else:
        flat = psf.ravel()
        idx = flat.argsort()[::-1]
        csum = flat[idx].cumsum()
        thr = flat[idx[np.searchsorted(csum, flat.sum() * crop)]]
        flat[flat < thr] = 0
        rows, cols = np.where(flat.reshape(psf.shape))
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        side = max(r1 - r0, c1 - c0) + 1
        r0 = (r0 + r1) // 2 - side // 2
        c0 = (c0 + c1) // 2 - side // 2
        psf = psf[r0 : r0 + side, c0 : c0 + side]
    return psf / psf.sum()


# -----------------------------------------------------------------------------
# Atmosphere I/O & resampling
# -----------------------------------------------------------------------------


def load_atmosphere(npz_file: str) -> Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, dict]:
    dat = np.load(npz_file)
    cube = dat["I_cube"] * (u.erg / u.s / u.cm**2 / u.sr / u.cm)
    wl_grid = dat["wl_grid"] * u.cm
    spt_res = dat["spt_res_x"] * (u.cm/u.pix)
    wl0 = dat["wl0"] * u.cm
    plotting = {
        "mean_idx": tuple(dat["mean_idx"]),
        "minus_idx": tuple(dat["minus_idx"]),
        "plus_idx": tuple(dat["plus_idx"]),
        "sigma_factor": float(dat["sigma_factor"]),
        "margin": int(dat["margin"]),
    }
    return cube, wl_grid, spt_res, wl0, plotting


def rebin_atmosphere(
    cube_sim: u.Quantity,
    wl_sim: u.Quantity,
    spt_sim: u.Quantity,
    det: Detector,
    sim: Simulation,
    plotting: dict,
) -> Tuple[u.Quantity, u.Quantity, dict]:
    y_pitch_out = det.plate_scale_length
    slit_width_as = sim.slit_width
    scan_step_as = sim.slit_scan_step
    print("  Spectral rebinning to instrument resolution (nx,ny,*nl*)...")

    wl_det = np.arange(
        wl_sim[0].to(u.cm).value,
        wl_sim[-1].to(u.cm).value + det.wvl_res.to(u.cm/u.pix).value,
        det.wvl_res.to(u.cm/u.pix).value,
    ) * wl_sim.unit
    resampler = FluxConservingResampler(extrapolation_treatment="zero_fill")

    def _reb_spec_block(block):
        block_nx, ny, nl = block.shape
        out = np.zeros((block_nx, ny, len(wl_det)))
        for i in range(block_nx):
            for j in range(ny):
                spec = Spectrum1D(block[i, j, :] * cube_sim.unit, spectral_axis=wl_sim)
                out[i, j, :] = resampler(spec, wl_det).flux.value
        return out

    # Parallelise over chunks along the first axis for better scaling
    n_chunks = min(16, cube_sim.shape[0])
    chunk_size = cube_sim.shape[0] // n_chunks
    blocks = [
        cube_sim.value[i * chunk_size : (i + 1) * chunk_size if i < n_chunks - 1 else cube_sim.shape[0]]
        for i in range(n_chunks)
    ]

    # total number of blocks corresponds to the number of Parallel tasks
    with tqdm_joblib(tqdm(total=len(blocks), desc="  > spectral rebin", leave=False)):
        results = Parallel(n_jobs=sim.ncpu if sim.ncpu > 0 else -1)(
            delayed(_reb_spec_block)(block) for block in blocks
        )

    cube_spec = np.concatenate(results, axis=0) * cube_sim.unit

    # ------------------------------------------------------------------
    # Helper: flux-conserving resample of a 3-D cube along one axis
    # ------------------------------------------------------------------
    def _resample_axis(cube, old_step, new_step, axis, n_jobs):
        """
        Parameters
        ----------
        cube : Quantity[(Nx, Ny, Nl)]
            The data cube.
        old_step : Quantity
            Physical size of one native pixel (same units for X & Y).
        new_step : Quantity
            Desired size of one output pixel (same units as old_step).
        axis : {0, 1}
            0 → rebin X (first) axis, 1 → rebin Y (second) axis.
        n_jobs : int
            Passed to joblib.Parallel.

        Returns
        -------
        reb : Quantity
            Cube rebinned along `axis`, flux conserved.
        """
        # ---------- coordinate grids ------------------------------------------------
        if axis == 0:         # X axis
            Nx, Ny, Nl = cube.shape
            Ny_range   = range(Ny)
            N_in       = Nx
        else:                 # Y axis
            Nx, Ny, Nl = cube.shape
            Ny_range   = range(Nx)          # we will iterate over X instead
            N_in       = Ny

        x_in   = np.arange(N_in) * old_step               # pixel centres
        tot_len = N_in * old_step
        N_out = int(np.floor((tot_len / new_step).decompose().value))
        x_out = (np.arange(N_out) + 0.5) * new_step       # centre of each new pixel

        # ---------- single 1-D resampler we re-use everywhere -----------------------
        fcr = FluxConservingResampler(extrapolation_treatment="zero_fill")

        # ---------- loop over “rows" in parallel ------------------------------------
        def _one_row(j):
            """
            Resample one (λ-stacked) 1-D row: either (Nx, Nl) at fixed Y
            or (Ny, Nl) at fixed X, depending on `axis`.
            Returns a 2-D array (N_out, Nl) in plain numpy dtype.
            """
            if axis == 0:                          # rebin along X for fixed Y=j
                row = cube[:, j, :]                # shape (Nx, Nl)
            else:                                  # rebin along Y for fixed X=j
                row = cube[j, :, :]                # shape (Ny, Nl)

            out = np.empty((N_out, Nl), dtype=cube.dtype)
            for k in range(Nl):
                spec = Spectrum1D(row[:, k] * cube.unit, spectral_axis=x_in)
                out[:, k] = fcr(spec, x_out).flux.value
            return out

        with tqdm_joblib(
            tqdm(total=len(Ny_range), desc="  > spatial rebin", leave=False)
        ):
            stacked = Parallel(n_jobs=n_jobs)(
                delayed(_one_row)(j) for j in Ny_range
            )

        # ---------- assemble cube back in correct orientation -----------------------
        reb_val = np.stack(stacked, axis=1 if axis == 0 else 0)  # (N_out, Ny, Nl) or (Nx, N_out, Nl)
        return reb_val * cube.unit
    # ------------------------------------------------------------------


    print("  Scanning slit across observation (*nx*,ny,nl)...")
    dx_len   = (spt_sim * u.pix).to(u.cm) if spt_sim.unit.is_equivalent(u.cm/u.pix) else spt_sim.to(u.cm)
    slit_w   = angle_to_distance(slit_width_as).to(u.cm)
    cube_scan = _resample_axis(
        cube_spec,
        old_step = dx_len,
        new_step = slit_w,
        axis     = 0,                # rebin X
        n_jobs   = sim.ncpu if sim.ncpu > 0 else -1,
    )

    print("  Rebinning each slit scan to detector plate scale (nx,*ny*,nl)...")
    y_pitch_cm = (y_pitch_out * u.pix).to(u.cm) if y_pitch_out.unit.is_equivalent(u.cm/u.pix) else y_pitch_out.to(u.cm)
    cube_det = _resample_axis(
        cube_scan,
        old_step = dx_len,           # same native step in Y
        new_step = y_pitch_cm,
        axis     = 1,                # rebin Y
        n_jobs   = sim.ncpu if sim.ncpu > 0 else -1,
    )

    # --- Calculate new iloc for plotting indices ---
    def map_idx(idx):
        if idx is None:
            return None
        x_factor = (spt_sim / angle_to_distance(scan_step_as)).decompose().value
        y_factor = (spt_sim / y_pitch_out).decompose().value
        x_new = int(round(idx[0] * x_factor))
        y_new = int(round(idx[1] * y_factor))
        return (x_new, y_new)

    plotting_new = plotting.copy()
    for key in ["mean_idx", "minus_idx", "plus_idx"]:
        plotting_new[key] = map_idx(plotting.get(key))

    return cube_det, wl_det, plotting_new


# -----------------------------------------------------------------------------
# Radiometric pipeline
# -----------------------------------------------------------------------------

def intensity_to_photons(I: u.Quantity, wl_axis: u.Quantity) -> u.Quantity:
    E_ph = (const.h * const.c / wl_axis).to("erg") * (1 / u.photon)
    return (I / E_ph).to(u.photon / u.s / u.cm**2 / u.sr / u.cm)


def add_effective_area(ph_cm2_sr_cm_s: u.Quantity, tel: Telescope) -> u.Quantity:
    A_eff = tel.collecting_area.cgs * tel.pm_eff * tel.grat_eff * tel.filt_eff
    return ph_cm2_sr_cm_s * A_eff


def photons_to_pixel_rate(ph_sr_cm_s: u.Quantity, wl_pitch: u.Quantity, plate_scale: u.Quantity, slit_width: u.Quantity) -> u.Quantity:
    pixel_solid_angle = ((plate_scale * u.pixel * slit_width).cgs / const.au.cgs ** 2) * u.sr
    return ph_sr_cm_s * pixel_solid_angle * wl_pitch


def apply_psf(signal: u.Quantity, psf: np.ndarray) -> u.Quantity:
    n_scan, n_slit, _ = signal.shape
    blurred = np.empty_like(signal.value)
    for i in range(n_scan):
        blurred[i] = convolve2d(signal.value[i], psf, mode="same")
    return blurred * signal.unit


def to_electrons(photon_rate: u.Quantity, t_exp: u.Quantity, det: Detector) -> u.Quantity:
    e_per_ph = fano_noise(det.e_per_ph_euv.value, det.si_fano) * u.electron / u.photon
    e = photon_rate * t_exp * det.qe_euv * e_per_ph
    e += det.dark_current * t_exp
    e += np.random.normal(0, det.read_noise_rms.value, photon_rate.shape) * (u.electron / u.pixel)
    e[e < 0] = 0
    return e


def to_dn(electrons: u.Quantity, det: Detector) -> u.Quantity:
    dn = electrons / det.gain_e_per_dn
    dn = dn.to(det.max_dn.unit)
    dn[dn > det.max_dn] = det.max_dn
    return dn


# -----------------------------------------------------------------------------
# Noise & stray-light models
# -----------------------------------------------------------------------------

def add_poisson(data: u.Quantity) -> u.Quantity:
    unit = data.unit
    return np.random.poisson(data.value) * unit


def add_stray_light(electrons: u.Quantity, t_exp: u.Quantity, det: Detector, sim: Simulation) -> u.Quantity:
    n_vis_ph = np.random.poisson((sim.vis_sl * t_exp).value, size=electrons.shape) * (u.photon / u.pixel)
    e_per_ph = fano_noise(det.e_per_ph_vis.value, det.si_fano) * (u.electron / u.photon)
    return electrons + n_vis_ph * e_per_ph * det.qe_vis


# -----------------------------------------------------------------------------
# Spectral fitting (per-pixel Gaussian)
# -----------------------------------------------------------------------------

def _guess_params(wv: np.ndarray, prof: np.ndarray) -> list:
    back = prof.min()
    prof_c = prof - back
    prof_c[prof_c < 0] = 0
    peak = prof_c.max()
    centre = wv[np.nanargmax(prof_c)]
    if peak == 0:
        sigma = 1.0
    else:
        sigma = np.trapezoid(prof_c, wv) / (peak * np.sqrt(2 * np.pi))
    return [peak, centre, sigma, back]


def _fit_one(wv: np.ndarray, prof: np.ndarray) -> np.ndarray:
    p0 = _guess_params(wv, prof)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt, _ = curve_fit(gaussian, wv, prof, p0=p0, maxfev=5000)
            return popt
        except RuntimeError:
            return np.array([-1, -1, -1, -1])


def fit_cube_gauss(signal_cube: u.Quantity, wv: u.Quantity, n_jobs: int = Simulation.ncpu) -> u.Quantity:
    n_scan, n_slit, _ = signal_cube.shape

    def _fit_block(block):
        block_nx, n_slit, n_wl = block.shape
        out = np.zeros((block_nx, n_slit, 4))
        for i in range(block_nx):
            for j in range(n_slit):
                out[i, j, :] = _fit_one(wv.value, block[i, j, :])
        return out

    n_chunks = min(16, signal_cube.shape[0])
    chunk_size = signal_cube.shape[0] // n_chunks
    blocks = [
        signal_cube.value[i * chunk_size : (i + 1) * chunk_size if i < n_chunks - 1 else signal_cube.shape[0]]
        for i in range(n_chunks)
    ]
    # use context manager so tqdm works with joblib
    with tqdm_joblib(tqdm(total=len(blocks), desc="Fit chunks", leave=False)):
        results = Parallel(n_jobs=n_jobs if n_jobs > 0 else -1)(
            delayed(_fit_block)(block) for block in blocks
        )
    arr = np.concatenate(results, axis=0)
    return arr * np.array([signal_cube.unit, wv.unit, wv.unit, signal_cube.unit])


# -----------------------------------------------------------------------------
# Monte-Carlo wrapper
# -----------------------------------------------------------------------------

def simulate_once(I_cube: u.Quantity, wl_axis: u.Quantity, t_exp: u.Quantity, det: Detector, tel: Telescope, sim: Simulation) -> Tuple[u.Quantity, ...]:

    signal0 = add_poisson(I_cube)
    signal1 = intensity_to_photons(signal0, wl_axis)
    signal2 = add_effective_area(signal1, tel)
    signal3 = photons_to_pixel_rate(signal2, det.wvl_res, det.plate_scale_length, angle_to_distance(sim.slit_width))
    signal4 = apply_psf(signal3, tel.psf)
    signal5 = to_electrons(signal4, t_exp, det)
    signal6 = add_stray_light(signal5, t_exp, det, sim)
    signal7 = to_dn(signal6, det)

    return (signal0, signal1, signal2, signal3, signal4, signal5, signal6, signal7)

def monte_carlo(I_cube: u.Quantity, wl_axis: u.Quantity, t_exp: u.Quantity, det: Detector, tel: Telescope, sim: Simulation, n_iter: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    signals, fits = [], []
    for _ in tqdm(range(n_iter), desc="Monte-Carlo", unit="iter", leave=False):
        signals.append(simulate_once(I_cube, wl_axis, t_exp, det, tel, sim))
        fits.append(fit_cube_gauss(signals[-1][-1], wl_axis))
    return np.array(signals), np.array(fits)


# -----------------------------------------------------------------------------
# Analysis metrics
# -----------------------------------------------------------------------------

def velocity_from_fit(
    fit_arr: u.Quantity | np.ndarray,
    wl0: u.Quantity,
    chunk_size: int = 128,
) -> u.Quantity:
    """
    Convert fitted line centres to LOS velocity.
    Works with either a Quantity array or an object-dtype array whose
    elements are Quantities. A tqdm bar (leave=False) shows progress.
    """
    centres_raw = fit_arr[..., 1]                       # (n_scan, n_slit)
    # Ensure we have a pure Quantity array
    if isinstance(centres_raw, u.Quantity):
        centres = centres_raw.to(wl0.unit)
    else:  # object array of Quantity scalars
        get_val = np.vectorize(lambda q: q.to_value(wl0.unit))
        centres = u.Quantity(get_val(centres_raw), wl0.unit)

    n_scan = centres.shape[0]
    v_val = np.empty_like(centres.value)                # float buffer
    mask_bad = np.all(fit_arr == -1, axis=-1)

    for i0 in tqdm(range(0, n_scan, chunk_size),
                   desc="Velocity", leave=False):
        i1 = min(i0 + chunk_size, n_scan)
        v_chunk = ((centres[i0:i1] - wl0) / wl0 * const.c).to(u.cm/u.s)
        v_val[i0:i1] = v_chunk.value

    v = v_val * (u.cm / u.s)
    v = np.where(mask_bad, -1 * u.cm / u.s, v)
    return v


def analyse(fits_all: u.Quantity | np.ndarray,
            v_true: u.Quantity,
            wl0: u.Quantity) -> dict:
    """
    Monte-Carlo velocity statistics given pre-computed ground truth.
    """
    v_all = velocity_from_fit(fits_all, wl0)
    return {
        "v_mean": v_all.mean(axis=0),
        "v_std":  v_all.std(axis=0),
        "v_err":  v_true - v_all.mean(axis=0),
        "v_samples": v_all,
        "v_true":    v_true,
    }


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def plot_radiometric_pipeline(
    signals: Tuple[u.Quantity, ...],
    wl_axis: u.Quantity,
    idx_sim_minus: Tuple[int, int] | None,
    idx_sim_mean: Tuple[int, int] | None,
    idx_sim_plus: Tuple[int, int] | None,
    spt_pitch_sim: u.Quantity,          # kept for API-compatibility (unused)
    spt_pitch_instr: u.Quantity,        # kept for API-compatibility (unused)
    save: str = "fig_radiometric_pipeline.png",
    row_labels: Iterable[str] = (r"$\mu-\sigma$", r"$\mu$", r"$\mu+\sigma$"),
    key_pixel_colors: Iterable[str] = ("mediumseagreen", "black", "deeppink"),
) -> plt.Figure:
    # indices are already expressed in detector pixels – no further scaling
    idxs_reb = (idx_sim_minus, idx_sim_mean, idx_sim_plus)

    wl_A = wl_axis.to(u.angstrom).value
    def spectrum(stage_idx: int, row_idx: int) -> np.ndarray:
        return signals[stage_idx][idxs_reb[row_idx] + (slice(None),)]

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(10, 6),
        sharex="row",
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
    )
    fig.subplots_adjust(right=0.86)

    for row in range(3):
        colour = key_pixel_colors[row]
        lab_ax = axes[row, 0].inset_axes([-0.42, 0, 0.1, 1], frameon=False)
        lab_ax.set_axis_off()
        lab_ax.text(0, 0.5, row_labels[row], va="center", ha="left", rotation=90, fontsize=9)

        ax0 = axes[row, 0]
        sp1 = spectrum(1, row)
        ax0.step(wl_A, sp1, where="mid", color=colour, lw=1)
        if row == 0:
            ax0.set_title("signal1/2/3", fontsize=8)
        ax0.set_ylabel(r"ph s$^{-1}$ cm$^{-2}$ sr$^{-1}$ cm$^{-1}$", color=colour, fontsize=7)
        ax0.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

        ax_r1 = ax0.twinx()
        sp2 = spectrum(2, row)
        ax_r1.step(wl_A, sp2, where="mid", color="tab:orange", lw=1)
        ax_r1.set_ylim(sp2.min(), sp2.max())
        ax_r1.set_ylabel(r"ph s$^{-1}$ sr$^{-1}$ cm$^{-1}$", color="tab:orange", fontsize=7)
        ax_r1.yaxis.labelpad = 8
        ax_r1.tick_params(direction="in", colors="tab:orange", which="both", top=True, bottom=True, right=True)
        ax_r1.patch.set_visible(False)

        ax_r2 = ax0.twinx()
        sp3 = spectrum(3, row)
        ax_r2.step(wl_A, sp3, where="mid", color="tab:blue", lw=1)
        ax_r2.set_ylim(sp3.min(), sp3.max())
        ax_r2.spines.right.set_position(("axes", 1.15))
        ax_r2.set_ylabel(r"ph s$^{-1}$ pix$^{-1}$", color="tab:blue", fontsize=7)
        ax_r2.yaxis.labelpad = 24
        ax_r2.tick_params(direction="in", colors="tab:blue", which="both", top=True, bottom=True, right=True)
        ax_r2.patch.set_visible(False)

        ax1 = axes[row, 1]
        sp4 = spectrum(4, row)
        ax1.step(wl_A, sp4, where="mid", color=colour, lw=1)
        if row == 0:
            ax1.set_title("signal4", fontsize=8)
        ax1.set_ylabel(r"ph s$^{-1}$ pix$^{-1}$ (PSF)", color=colour, fontsize=7)
        ax1.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

        ax2 = axes[row, 2]
        sp5 = spectrum(5, row)
        ax2.step(wl_A, sp5, where="mid", color=colour, lw=1)
        if row == 0:
            ax2.set_title("signal5", fontsize=8)
        ax2.set_ylabel(r"e$^-$ pix$^{-1}$", color=colour, fontsize=7)
        ax2.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

        ax3 = axes[row, 3]
        sp6 = spectrum(6, row)
        ax3.step(wl_A, sp6, where="mid", color=colour, lw=1)
        if row == 0:
            ax3.set_title("signal6/7", fontsize=8)
        ax3.set_ylabel(r"e$^-$ pix$^{-1}$ (stray)", color=colour, fontsize=7)
        ax3.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

        ax_r3 = ax3.twinx()
        sp7 = spectrum(7, row)
        ax_r3.step(wl_A, sp7, where="mid", color="tab:red", lw=1)
        ax_r3.spines.right.set_position(("axes", 1.12))
        ax_r3.set_ylim(sp7.min(), sp7.max())
        ax_r3.set_ylabel(r"DN pix$^{-1}$", color="tab:red", fontsize=7)
        ax_r3.yaxis.labelpad = 16
        ax_r3.tick_params(direction="in", colors="tab:red", which="both", top=True, bottom=True, right=True)
        ax_r3.patch.set_visible(False)

        if row == 2:
            for col in range(4):
                axes[row, col].set_xlabel("Wavelength [Å]")
        for col in range(4):
            axes[row, col].tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

    fig.savefig(save, dpi=300)
    plt.close(fig)
    return fig


def plot_maps(
    signal_cube: u.Quantity,
    fit_cube: u.Quantity,
    wl_axis: u.Quantity,
    wl0: u.Quantity,
    idx_sim_minus: Tuple[int, int] | None,
    idx_sim_mean: Tuple[int, int] | None,
    idx_sim_plus: Tuple[int, int] | None,
    photon_cube: u.Quantity | np.ndarray,   # signal6  (photons / pix / λ)
    save: str,
    *,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    key_pixel_colors: Iterable[str] = ("deeppink", "mediumseagreen", "black"),
    previous: dict | None = None,
    save_data_path: str | None = None,
) -> None:
    # --------------------------------------------------------------
    # photons in each CCD row:  Σ_λ  photon_cube(x,y,λ)
    # --------------------------------------------------------------
    ph_int = photon_cube.sum(axis=2) if isinstance(photon_cube, np.ndarray) \
             else photon_cube.sum(axis=2).value          # photons  pix⁻¹

    # ---------------- log10 intensity ----------------------------

    n_scan, n_slit = ph_int.shape
    x_pix_size = SIM.slit_scan_step.to(u.arcsec).value
    y_pix_size = DET.plate_scale_angle.to(u.arcsec / u.pix).value

    x = (np.arange(n_scan) - n_scan // 2) * x_pix_size
    y = (np.arange(n_slit) - n_slit // 2) * y_pix_size
    extent = [
        x[0] - x_pix_size / 2,
        x[-1] + x_pix_size / 2,
        y[0] - y_pix_size / 2,
        y[-1] + y_pix_size / 2,
    ]

    nrows = 2 if previous else 1
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(11, 5),
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
        sharey="row",       # y-axis shared within each row (axV shares y with axI)
    )
    if nrows == 1:
        axes = axes.reshape(1, 2)

    axI, axV = axes[0]
    # imI = axI.imshow(ph_int.T, origin="lower", aspect="auto",
    imI = axI.imshow(np.log10(ph_int, out=np.zeros_like(ph_int), where=ph_int > 0).T,
                     origin="lower", aspect="auto",  #  vmin=0,
                     cmap="afmhot", extent=extent)
    v_map = velocity_from_fit(fit_cube, wl0).to(u.km / u.s)
    imV = axV.imshow(v_map.T.value, origin="lower", aspect="auto",
                     cmap="RdBu_r", vmin=-15, vmax=15, extent=extent)

    # ------------------------------------------------------------------
    # photon colour-bar
    # ------------------------------------------------------------------
    cbarI = fig.colorbar(imI, ax=axI, orientation="horizontal",
                         pad=0.14, shrink=0.95, aspect=35)
    # cbarI.set_label(r"ph/row ")
    cbarI.set_label(
        r"$\log_{10}\!\left(\sum_{\lambda\,\mathrm{pix}}I(\lambda)\:\mathrm{ }\left[\mathrm{ph/s/pix}\right]\right)$"
    )

    # ------------------------------------------------------------------
    # velocity colour-bar (unchanged)
    # ------------------------------------------------------------------
    cbarV = fig.colorbar(imV, ax=axV, orientation="horizontal",
                         pad=0.14, extend="both", shrink=0.95, aspect=35)
    cbarV.set_label(r"$v$ [km/s]")

    # ---------------- spatial zoom if requested ----------------
    if xlim:
        axI.set_xlim(*xlim)
        axV.set_xlim(*xlim)
    if ylim:
        axI.set_ylim(*ylim)
        axV.set_ylim(*ylim)

    # ---------------- formatting, markers, save ----------------
    def _format(ax):
        ax.set_xlabel("X [arcsec]")
        if ax is axI:
            ax.set_ylabel("Y [arcsec]")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        interval = 15.0
        max_x = max(abs(extent[0]), abs(extent[1]))
        max_y = max(abs(extent[2]), abs(extent[3]))
        xticks = np.arange(-np.ceil(max_x / interval) * interval, np.ceil(max_x / interval) * interval + interval / 2, interval)
        yticks = np.arange(-np.ceil(max_y / interval) * interval, np.ceil(max_y / interval) * interval + interval / 2, interval)
        xticks = xticks[(xticks >= extent[0]) & (xticks <= extent[1])]
        yticks = yticks[(yticks >= extent[2]) & (yticks <= extent[3])]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        # pixels are already expressed in arcsec; enforce equal scaling
        ax.set_aspect(1.0)
        ax.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

    _format(axI)
    _format(axV)

    markers = ["2", "3", "1"]
    labels = [r"$\mu-1\sigma$", r"$\mu$", r"$\mu+1\sigma$"]

    # for idx, color, marker in zip(
    for idx, color, marker, label in zip(
        # [idx_sim_minus, idx_sim_mean, idx_sim_plus],
        list(reversed([idx_sim_minus, idx_sim_mean, idx_sim_plus])),
        key_pixel_colors,
        markers,
        # labels
        # list(reversed(key_pixel_colors)),  # reverse this order (for some reason...)
        # list(reversed(markers)),
        list(reversed(labels))
    ):
        if idx is not None:
            x_pos = (idx[0] - n_scan // 2) * x_pix_size
            y_pos = (idx[1] - n_slit // 2) * y_pix_size
            for ax in (axI, axV):
                ax.scatter(x_pos, y_pos, marker=marker, color=color, s=250, linewidth=2, label=label)

    axV.legend(
        loc="upper right",
        fontsize="small")

    plt.tight_layout()
    plt.savefig(save, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_exposure_time_map(
    analysis_per_exp: dict[float, dict],
    precision_requirement: float,
    x_pix_size: float,
    y_pix_size: float,
    save: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Plot a map showing the minimum exposure time required to achieve the given Doppler velocity precision.

    Parameters
    ----------
    analysis_per_exp : dict[float, dict]
        Dictionary mapping exposure times (seconds) to analysis results (containing 'v_std' maps).
    precision_requirement : float
        Required velocity precision in km/s.
    x_pix_size : float
        Pixel size in X-direction (arcsec).
    y_pix_size : float
        Pixel size in Y-direction (arcsec).
    save : str
        Filename to save the plot.
    cmap : str, optional
        Colormap for the plot.
    vmin, vmax : float, optional
        Min/max values for colorbar scale.
    """
    exp_times_sorted = sorted(analysis_per_exp.keys())
    shape = next(iter(analysis_per_exp.values()))["v_std"].shape
    exp_time_map = np.full(shape, np.nan)

    for exp_time in exp_times_sorted:
        v_std_map = analysis_per_exp[exp_time]["v_std"].to(u.km / u.s).value
        mask = (v_std_map <= precision_requirement) & np.isnan(exp_time_map)
        exp_time_map[mask] = exp_time

    n_scan, n_slit = exp_time_map.shape
    x = (np.arange(n_scan) - n_scan // 2) * x_pix_size
    y = (np.arange(n_slit) - n_slit // 2) * y_pix_size
    extent = [x[0] - x_pix_size / 2, x[-1] + x_pix_size / 2, y[0] - y_pix_size / 2, y[-1] + y_pix_size / 2]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(exp_time_map.T, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, shrink=0.95, aspect=35)
    cbar.set_label("Exposure time [s] to reach {:.1f} km/s precision".format(precision_requirement))

    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    ax.set_aspect(1.0)
    ax.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_maps(
    v_map: u.Quantity,
    v_std_map: u.Quantity,
    save: str,
    x_pix_size: float,
    y_pix_size: float,
    key_pixel_colors: Iterable[str] = ("mediumseagreen", "black", "deeppink"),
    idx_minus: Tuple[int, int] | None = None,
    idx_mean: Tuple[int, int] | None = None,
    idx_plus: Tuple[int, int] | None = None,
    vmin: float = -15,
    vmax: float = 15,
    std_vmax: float = 5,
) -> None:
    n_scan, n_slit = v_map.shape
    x = (np.arange(n_scan) - n_scan // 2) * x_pix_size
    y = (np.arange(n_slit) - n_slit // 2) * y_pix_size
    extent = [x[0] - x_pix_size / 2, x[-1] + x_pix_size / 2, y[0] - y_pix_size / 2, y[-1] + y_pix_size / 2]

    fig, (axV, axStd) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw=dict(wspace=0.0, hspace=0.0), sharey=True)

    # Doppler velocity map (left panel)
    imV = axV.imshow(v_map.T.to(u.km / u.s).value, origin="lower", aspect="auto",
                     cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent)
    cbarV = fig.colorbar(imV, ax=axV, orientation="horizontal", pad=0.14, extend="both", shrink=0.95, aspect=35)
    cbarV.set_label(r"$v$ [km/s]")

    # Velocity standard deviation map (right panel)
    imStd = axStd.imshow(v_std_map.T.to(u.km / u.s).value, origin="lower", aspect="auto",
                         cmap="magma", vmin=0, vmax=std_vmax, extent=extent)
    cbarStd = fig.colorbar(imStd, ax=axStd, orientation="horizontal", pad=0.14, shrink=0.95, aspect=35, extend="max")
    cbarStd.set_label(r"$\sigma_v$ [km/s]")

    # Formatting
    for ax in [axV, axStd]:
        ax.set_xlabel("X [arcsec]")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect(1.0)
        ax.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

        interval = 15.0
        max_x = max(abs(extent[0]), abs(extent[1]))
        max_y = max(abs(extent[2]), abs(extent[3]))
        xticks = np.arange(-np.ceil(max_x / interval) * interval, np.ceil(max_x / interval) * interval + interval / 2, interval)
        yticks = np.arange(-np.ceil(max_y / interval) * interval, np.ceil(max_y / interval) * interval + interval / 2, interval)
        xticks = xticks[(xticks >= extent[0]) & (xticks <= extent[1])]
        yticks = yticks[(yticks >= extent[2]) & (yticks <= extent[3])]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

    axV.set_ylabel("Y [arcsec]")

    # # Markers for key pixels
    # markers = ["2", "3", "1"]
    # labels = [r"$\mu-1\sigma$", r"$\mu$", r"$\mu+1\sigma$"]
    # for idx, color, marker, label in zip(
    #     [idx_minus, idx_mean, idx_plus],
    #     key_pixel_colors,
    #     markers,
    #     labels
    # ):
    #     if idx is not None:
    #         x_pos = (idx[0] - n_scan // 2) * x_pix_size
    #         y_pos = (idx[1] - n_slit // 2) * y_pix_size
    #         for ax in (axV, axStd):
    #             ax.scatter(x_pos, y_pos, marker=marker, color=color, s=250, linewidth=2, label=label)

    # axStd.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.savefig(save, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_multi_maps(
    npz_files: list[str | Path],
    map_type: str = "intensity",                    # "intensity" | "velocity"
    *,
    key_pixel_colors: Iterable[str] = ("mediumseagreen", "black", "deeppink"),
    markers: list[str] = ("2", "3", "1"),
    labels: list[str]  = (r"$\mu-1\sigma$", r"$\mu$", r"$\mu+1\sigma$"),
    save: str = "fig_multi_maps.png",
    figsize: tuple = (10, 6.75),
):
    """
    Stack several intensity / velocity maps vertically.

    If `map_type == "intensity"` the quantity shown is the *integrated photon
    count per CCD row* (Σ_λ ph pix⁻¹ s⁻¹) exactly like in `plot_maps`, not the
    specific intensity.
    """
    # ------------------------------------------------------------------
    # ---- load maps + ancillary info ----------------------------------
    maps, dxs, dys, idxs_list = [], [], [], []
    for npz_file in npz_files:
        dat = np.load(npz_file, allow_pickle=True)

        # ---- map -----------------------------------------------------
        if map_type == "intensity":
            # Prefer integrated photon map derived from first_signals[4]
            if "first_signals" in dat.files:
                photon_cube = dat["first_signals"][4]          # signal4
                if isinstance(photon_cube, np.ndarray):
                    m = photon_cube.sum(axis=2)                       # ph  pix⁻¹ s⁻¹
                else:                                                 # Quantity
                    m = photon_cube.sum(axis=2).value
            elif "si_map" in dat:
                m = dat["si_map"]                                     # fallback
            elif "log_si" in dat:
                m = 10 ** dat["log_si"]
            else:
                raise ValueError(f"no suitable intensity map in {npz_file}")
        elif map_type == "velocity":
            if "analysis_res" in dat:
                m = dat["analysis_res"].item()["v_mean"].to_value(u.km / u.s)
            elif "v_map" in dat:
                m = dat["v_map"]
            else:
                raise ValueError(f"no velocity map in {npz_file}")
        else:
            raise ValueError("map_type must be 'intensity' or 'velocity'")
        maps.append(m)

        # ---- pixel sizes --------------------------------------------
        if {"x_pix_size", "y_pix_size"} <= set(dat.files):
            dxs.append(float(dat["x_pix_size"]))
            dys.append(float(dat["y_pix_size"]))
        else:
            sim = dat["SIM"].item()
            det = dat["DET"].item()
            dxs.append(sim.slit_scan_step.to_value(u.arcsec))
            dys.append(det.plate_scale_angle.to_value(u.arcsec / u.pix))

        # ---- key-pixel indices --------------------------------------
        idxs: list[tuple[int, int] | None] = [None, None, None]
        if "plotting" in dat.files:
            plotting = dat["plotting"].item()
            for n, k in enumerate(("minus_idx", "mean_idx", "plus_idx")):
                if plotting.get(k) is not None:
                    idxs[n] = tuple(plotting[k])
        idxs_list.append(idxs)

    # ------------------------------------------------------------------
    # ---- figure + adaptive GridSpec ----------------------------------
    nrows   = len(maps)
    heights = [m.shape[1] * dy for m, dy in zip(maps, dys)]
    fig     = plt.figure(figsize=figsize)
    gs      = fig.add_gridspec(nrows, 1, height_ratios=heights, hspace=0.0)
    axes    = [fig.add_subplot(gs[i, 0]) for i in range(nrows)]

    # ------------------------------------------------------------------
    # ---- global X-extent (shared across rows) ------------------------
    xmins, xmaxs = [], []
    for m, dx in zip(maps, dxs):
        n_scan = m.shape[0]
        x      = (np.arange(n_scan) - n_scan // 2) * dx
        xmins.append(x[0] - dx / 2)
        xmaxs.append(x[-1] + dx / 2)
    xlim = (min(xmins), max(xmaxs))

    # ------------------------------------------------------------------
    # ---- plot each map ----------------------------------------------
    for ax, m, dx, dy, idxs in zip(axes, maps, dxs, dys, idxs_list):
        n_scan, n_slit = m.shape
        x = (np.arange(n_scan) - n_scan // 2) * dx
        y = (np.arange(n_slit) - n_slit // 2) * dy
        extent = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]

        # ---- data + colour scaling ----------------------------------
        if map_type == "intensity":
            data = np.log10(m, out=np.zeros_like(m), where=m > 0)
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            cmap       = "afmhot"
            cbar_label = r"$\log_{10}\!\left(\sum_{\lambda\,\mathrm{pix}}I(\lambda)\:\mathrm{ }\left[\mathrm{ph/s/pix}\right]\right)$"
        else:  # velocity
            data = m
            vmax = np.nanmax(np.abs(data))
            vmin = -vmax
            cmap = "RdBu_r"
            cbar_label = r"$v$ [km/s]"

        im = ax.imshow(
            data.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )

        # ---- ticks / labels ----------------------------------------
        ax.set_xlim(*xlim)
        ax.set_aspect(1)
        ax.tick_params(direction="in", which="both", top=True, right=True)

        # ---- markers -----------------------------------------------
        for idx, color, marker, lab in zip(
            reversed(idxs),
            key_pixel_colors,
            markers,
            reversed(labels),
        ):
            if idx is None:
                continue
            # ax.scatter(
            #     (idx[0] - n_scan // 2) * dx,
            #     (idx[1] - n_slit // 2) * dy,
            #     marker=marker,
            #     color=color,
            #     s=230,
            #     lw=2,
            #     label=lab,
            # )
        # place legend only on the bottom row, lower-right corner
        # if ax is axes[-1]:
        #     ax.legend(loc="lower right", fontsize="small")

        # ---- individual colour-bar ---------------------------------
        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            pad=0.04,
            shrink=0.92,
            aspect=20,
        )
        cbar.set_label(cbar_label)

    axes[-1].set_xlabel("X [arcsec]")
    for ax in axes:
        ax.set_ylabel("Y [arcsec]")

    plt.savefig(save, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_std_map(
    v_std_map: u.Quantity,
    save: str,
    x_pix_size: float,
    y_pix_size: float,
    key_pixel_colors: Iterable[str] = ("mediumseagreen", "black", "deeppink"),
    idx_minus: Tuple[int, int] | None = None,
    idx_mean: Tuple[int, int] | None = None,
    idx_plus: Tuple[int, int] | None = None,
) -> None:
    n_scan, n_slit = v_std_map.shape
    x = (np.arange(n_scan) - n_scan // 2) * x_pix_size
    y = (np.arange(n_slit) - n_slit // 2) * y_pix_size
    extent = [x[0] - x_pix_size / 2, x[-1] + x_pix_size / 2, y[0] - y_pix_size / 2, y[-1] + y_pix_size / 2]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(v_std_map.T.to(u.km / u.s).value, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=0, vmax=5)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.1)
    cbar.set_label(r"$\sigma_v$ [km/s]")

    interval = 15.0
    max_x = max(abs(extent[0]), abs(extent[1]))
    max_y = max(abs(extent[2]), abs(extent[3]))
    xticks = np.arange(-np.ceil(max_x / interval) * interval, np.ceil(max_x / interval) * interval + interval / 2, interval)
    yticks = np.arange(-np.ceil(max_y / interval) * interval, np.ceil(max_y / interval) * interval + interval / 2, interval)
    xticks = xticks[(xticks >= extent[0]) & (xticks <= extent[1])]
    yticks = yticks[(yticks >= extent[2]) & (yticks <= extent[3])]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    ax.set_aspect(1.0)
    ax.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

    for idx, color in zip([idx_minus, idx_mean, idx_plus], key_pixel_colors):
        if idx is not None:
            ax.plot(
                (idx[0] - n_scan // 2) * x_pix_size,
                (idx[1] - n_slit // 2) * y_pix_size,
                marker="o",
                color=color,
                markersize=8,
                fillstyle="none",
                lw=2,
            )

    plt.tight_layout(pad=0.1)
    plt.savefig(save, dpi=300)
    plt.close(fig)


def plot_intensity_vs_vstd(
    intensity: np.ndarray,
    v_std: u.Quantity,
    save: str,
    *,                           # keep new kw-only args after here
    fit_intensity_min: float | None = None,
    vstd_max: float | None = None,                 # NEW: upper σ_v cut-off (km/s)
) -> None:
    """
    Scatter of per-pixel intensity versus 1-σ velocity uncertainty plus two
    log–log linear fits.  A pop-up window lets the user click two points;
    a power-law (straight in log-log space) passing through them is added
    to the main plot.

    Parameters
    ----------
    intensity : ndarray
        Σ_λ photon count per CCD pixel [ph s⁻¹ pix⁻¹].
    v_std : Quantity
        Velocity standard-deviation map [km s⁻¹].
    save : str
        Output filename.
    fit_intensity_min : float, optional
        Ignore pixels with intensity < this value when fitting.
    vstd_max : float, optional
        Ignore pixels with σ_v > this value (km s⁻¹) in both scatter and fits.
    """
    inten = intensity.ravel()
    vstd  = v_std.to(u.km / u.s).value.ravel()

    # ------------------------------------------------------------------
    # ---- basic masks --------------------------------------------------
    valid_scatter = (inten > 0) & (vstd > 0)
    if vstd_max is not None:                          #  ← NEW
        valid_scatter &= (vstd <= vstd_max)

    if fit_intensity_min is not None:
        valid_fit = valid_scatter & (inten >= fit_intensity_min)
    else:
        valid_fit = valid_scatter

    # ------------------------------------------------------------------
    # ---- log-space arrays --------------------------------------------
    log_i_all = np.log10(inten[valid_scatter])
    log_v_all = np.log10(vstd [valid_scatter])

    log_i_fit = np.log10(inten[valid_fit])
    log_v_fit = np.log10(vstd [valid_fit])

    # ------------------------------------------------------------------
    # ---- global linear fit (log–log) ---------------------------------
    coeff = np.polyfit(log_i_fit, log_v_fit, 1)
    fit_x = np.linspace(log_i_fit.min(), log_i_fit.max(), 100)
    fit_y = coeff[0] * fit_x + coeff[1]

    # ------------------------------------------------------------------
    # ---- ridge (upper envelope) fit ----------------------------------
    nbins = 25
    bins  = np.linspace(log_i_fit.min(), log_i_fit.max(), nbins + 1)
    bin_cent  = 0.5 * (bins[:-1] + bins[1:])
    max_log_v = np.full(nbins, np.nan)
    for k in range(nbins):
        m = (log_i_fit >= bins[k]) & (log_i_fit < bins[k + 1])
        if np.any(m):
            max_log_v[k] = log_v_fit[m].max()
    valid_bin   = ~np.isnan(max_log_v)
    ridge_coeff = np.polyfit(bin_cent[valid_bin], max_log_v[valid_bin], 1)
    ridge_y     = ridge_coeff[0] * fit_x + ridge_coeff[1]

    # ------------------------------------------------------------------
    # ---- main figure --------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(inten[valid_scatter], vstd[valid_scatter],
               s=1, color="black", alpha=1)
    # ax.plot(10 ** fit_x, 10 ** fit_y,
    #         color="red", label="global fit")
    # ax.plot(10 ** fit_x, 10 ** ridge_y,
    #         color="limegreen", ls="--", lw=1.5, label="ridge fit")

    # ------------------------------------------------------------------
    # # ---- optional vertical / horizontal threshold lines --------------
    # if fit_intensity_min is not None:
    #     ax.axvline(fit_intensity_min, color="grey", ls=":", lw=1)
    # if vstd_max is not None:                          #  ← NEW
    #     ax.axhline(vstd_max, color="grey", ls=":", lw=1)

    # ------------------------------------------------------------------
    # ---- pop-up for manual two-point power-law -----------------------
    try:
        if matplotlib.get_backend().lower() not in {
            "agg", "module://matplotlib_inline.backend_inline"
        }:
            pop_fig, pop_ax = plt.subplots()
            pop_ax.scatter(inten[valid_scatter], vstd[valid_scatter],
                           s=4, color="tab:blue", alpha=0.6)
            pop_ax.set_xscale("log")
            pop_ax.set_yscale("log")
            pop_ax.set_xlabel("Intensity")
            pop_ax.set_ylabel(r"$\sigma_v$ [km/s]")
            pop_ax.set_title("Click two points to define a power-law")
            pop_ax.tick_params(direction="in", which="both",
                               top=True, right=True)

            plt.show(block=False)
            pts = np.array(pop_fig.ginput(2, timeout=-1))
            plt.close(pop_fig)

            if pts.shape == (2, 2):
                (x1, y1), (x2, y2) = pts
                if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
                    m = (np.log10(y2) - np.log10(y1)) / (np.log10(x2) - np.log10(x1))
                    c = np.log10(y1) - m * np.log10(x1)

                    x_line = np.logspace(np.log10(inten[valid_scatter].min()),
                                         np.log10(inten[valid_scatter].max()), 200)
                    y_line = 10 ** (m * np.log10(x_line) + c)

                    label = rf"Ridge fit: $y = {10**c:.2g}\,x^{{{m:.2g}}}$"
                    ax.plot(x_line, y_line, color="purple", lw=1.4, label=label)
    except Exception as exc:
        warnings.warn(f"Interactive line selection skipped ({exc})")

    # ------------------------------------------------------------------
    # ---- final cosmetics & save --------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlabel("Intensity")
    ax.set_xlabel(r"Intensity [erg/s/cm$^2$/sr]")
    ax.set_ylabel(r"$\sigma_v$ [km/s]")
    ax.legend(fontsize="small")
    ax.tick_params(direction="in", which="both", top=True, bottom=True,
                   left=True, right=True)

    plt.tight_layout(pad=0.1)
    plt.savefig(save, dpi=300)
    plt.close(fig)

def plot_spectra(
    dn_cube: u.Quantity,
    wl_axis: u.Quantity,
    idx_sim_minus: Tuple[int, int] | None,
    idx_sim_mean: Tuple[int, int] | None,
    idx_sim_plus: Tuple[int, int] | None,
    wl0: u.Quantity,
    *,                                       # keep following args keyword-only
    fit_cube: u.Quantity | np.ndarray,
    sigma_factor: float = 1.0,
    key_pixel_colors: Tuple[str, str, str] = ("deeppink", "black", "mediumseagreen"),
    save: str = "fig_spectra_dn.png",
    figsize: Tuple[float, float] = (6, 9.5),
) -> None:
    """
    Vertical layout (+1 σ at top, mean centre, −1 σ bottom) with a shared
    wavelength axis (bottom) and a shared velocity axis (top).  Each panel has
    an independent y-axis.  No vertical gaps between panels.
    """
    # ------------------------------------------------------------------
    # ---- handy converters --------------------------------------------
    wl_A  = wl_axis.to(u.angstrom).value
    wl0_A = wl0.to(u.angstrom).value
    c_kms = const.c.to_value(u.km / u.s)
    wl2v  = lambda wl: (wl - wl0_A) / wl0_A * c_kms
    v2wl  = lambda v: wl0_A * (1 + v / c_kms)

    # ------------------------------------------------------------------
    # ---- ordering:  +σ,  μ,  −σ --------------------------------------
    idxs   = [idx_sim_plus, idx_sim_mean, idx_sim_minus]
    titles = [
        rf"$\mu + {sigma_factor:.0f}\sigma$",
        r"$\mu$",
        rf"$\mu - {sigma_factor:.0f}\sigma$",
    ]

    # ------------------------------------------------------------------
    # ---- create stacked axes via GridSpec ----------------------------
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(nrows=3, ncols=1, hspace=0.0)
    axes = gs.subplots(sharex=True)

    # ------------------------------------------------------------------
    # ---- x-axis limits & tick locations ------------------------------
    v_lim        = (-250, 250)                               # km s⁻¹
    wl_lim_A     = v2wl(np.array(v_lim))
    v_ticks      = np.arange(-200, 201, 100)                 # every 100 km s⁻¹
    wl_tick_A    = v2wl(v_ticks)

    # ------------------------------------------------------------------
    # ---- loop over the three panels ----------------------------------
    first_secax: plt.Axes | None = None
    for ax, idx, title, color in zip(axes, idxs, titles, key_pixel_colors):
        if idx is None:
            ax.set_visible(False)
            continue

        # -------- observed spectrum -----------------------------------
        sel   = dn_cube[idx + (slice(None),)]
        spec  = sel.value if isinstance(sel, u.Quantity) else sel
        ax.step(wl_A, spec, where="mid", color=color, lw=1.5, zorder=2)

        # -------- fitted Gaussian ------------------------------------
        p = fit_cube[idx + (slice(None),)]
        if np.all(p != -1):
            peak, centre, sigma, back = (p[k].value for k in range(4))
            wl_hi = np.linspace(wl_axis.cgs.value.min(),
                                wl_axis.cgs.value.max(),
                                wl_axis.size * 10)           # ×10 sampling
            gauss = gaussian(wl_hi, peak, centre, sigma, back)
            ax.plot((wl_hi * u.cm).to_value(u.angstrom),
                    gauss, ls="--", color=color, lw=1.0, zorder=1)

        # -------- set y-axis lower limit to 0 -------------------------
        ax.set_ylim(bottom=0)

        # -------- cosmetics ------------------------------------------
        ax.set_ylabel("DN/pix")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(ls=":", alpha=0.5)

        # ticks on all four sides, pointing inwards
        ax.tick_params(direction="in", which="both", top=True, right=True)

        # -------------- velocity axis (top) --------------------------
        secax = ax.secondary_xaxis("top", functions=(wl2v, v2wl))
        secax.set_xlim(*v_lim)
        secax.set_xticks(v_ticks)
        secax.tick_params(direction="in", which="both", top=True)

        if first_secax is None:
            # first visible panel → keep labels
            # secax.set_xlabel("Velocity [km/s]")
            secax.set_xlabel("Velocity (Fe XII 195.119 Å) [km/s]")
            first_secax = secax
        else:
            # other panels: keep ticks but drop labels
            secax.set_xlabel("")
            secax.set_xticklabels([])

        # ------------------------------------------------------------------
        # Right-side “σ-label” styled like a y-axis label
        # ------------------------------------------------------------------
        ax.annotate(
            title,
            xy=(1.02, 0.5), xycoords=("axes fraction", "axes fraction"),
            rotation=90,
            ha="left", va="center",
            fontsize=ax.yaxis.label.get_size(),
        )

    # ------------------------------------------------------------------
    # ---- move scientific-notation offset to the y-label --------------
    fig.canvas.draw()  # populate offset texts
    for ax in axes:
        if not ax.get_visible():
            continue
        off_txt = ax.yaxis.get_offset_text().get_text()
        if off_txt:                                      # e.g. '1e3'
            exponent = off_txt.replace("1e", "")
            ax.yaxis.get_offset_text().set_visible(False)
            ax.set_ylabel(fr"Intensity [$10^{{{exponent}}}$ DN/pix]")

    # ------------------------------------------------------------------
    # ---- shared X-axis (bottom wavelength axis) ----------------------
    axes[-1].set_xlabel("Wavelength [Å]")
    axes[-1].set_xlim(*wl_lim_A)
    axes[-1].set_xticks(wl_tick_A)
    axes[-1].set_xticklabels([f"{w:.3f}" for w in wl_tick_A])
    axes[-1].tick_params(direction="in", which="both", top=True, right=True)

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main() -> None:
    global DET, TEL, SIM

    prev_maps = None
    save_maps_path = None
    args = sys.argv[1:]
    if "--prev" in args:
        prev_maps = load_maps(args[args.index("--prev") + 1])
    if "--save" in args:
        save_maps_path = args[args.index("--save") + 1]

    # Load PSF
    print("Loading PSF files...")
    psf_focus = _load_psf_ascii(TEL.psf_focus_file, skip=21)
    psf_mesh = _load_psf_ascii(TEL.psf_mesh_file, skip=16)
    psf_focus = _resample_psf(psf_focus, TEL.psf_focus_res, DET.pix_size)
    psf_mesh = _resample_psf(psf_mesh, TEL.psf_mesh_res, DET.pix_size)
    TEL.psf = _combine_psfs(psf_focus, psf_mesh, size=5)

    # Load synthetic atmosphere cube
    print("Loading atmosphere...")
    cube_sim, wl_sim, spt_sim, wl0, plotting = load_atmosphere("synthesised_spectra.npz")

    print("Rebinning atmosphere cube to instrument resolution for each slit position...")
    cube_reb, wl_axis, plotting = rebin_atmosphere(cube_sim, wl_sim, spt_sim, DET, SIM, plotting)

    print("Fitting ground truth cube...")
    fit_truth = fit_cube_gauss(cube_reb.cgs, wl_axis.cgs)
    v_true = velocity_from_fit(fit_truth, wl0)

    # --- Efficient in-memory buffers for post-loop plotting -----------------
    first_signal_per_exp: dict[float, Tuple[u.Quantity, ...]] = {}
    first_fit_per_exp:    dict[float, u.Quantity] = {}
    analysis_per_exp:     dict[float, dict] = {}
    si_map_per_exp:       dict[float, np.ndarray] = {}

    # Monte-Carlo simulation for each exposure time
    for t_exp in tqdm(SIM.expos, desc="Exposure time", unit="exposure"):
        signals, fits = monte_carlo(
            cube_reb, wl_axis, t_exp, DET, TEL, SIM, n_iter=SIM.n_iter
        )
        sec = t_exp.to_value(u.s)
        first_signal_per_exp[sec] = signals[0]          # tuple of 8 stages
        first_fit_per_exp[sec]    = fits[0]
        analysis_per_exp[sec]     = analyse(fits, v_true, wl0)
        del signals, fits

    output_file = "instr_modelling.pkl"
    globals().update(locals())
    dill.dump_session(output_file)
    print(f"Saved the session to {output_file} ({os.path.getsize(output_file) / 1e9:.2f} GB)")

    # ---------------------------  Post-processing plots  ---------------------------
    # print("Post-processing results...")
    # for t_exp in SIM.expos:
    #     sec = t_exp.to_value(u.s)
    #     first_signals = first_signal_per_exp[sec]
    #     first_fits    = first_fit_per_exp[sec]
    #     analysis_res  = analysis_per_exp[sec]

    #     ts   = datetime.now().strftime("%y%m%d_%H%M")
    #     path = f"{ts}_exp{sec:.1f}.npz"
    #     np.savez(
    #         path,
    #         first_signals=first_signals,
    #         first_fits=first_fits,
    #         analysis_res=analysis_res,
    #         TEL=TEL,
    #         DET=DET,
    #         SIM=SIM,
    #         plotting=plotting,
    #     )
    #     print(f"Saved data for exposure {sec} s to {path}")

    #     globals().update(locals());raise ValueError("Kicking back to ipython")

    #     plot_maps(
    #         cube_reb,
    #         first_fits,
    #         wl_axis,
    #         wl0,
    #         plotting["minus_idx"],
    #         plotting["mean_idx"],
    #         plotting["plus_idx"],
    #         photon_cube=first_signals[4],
    #         save=f"fig_maps_{sec}.png",
    #         previous=prev_maps,
    #         save_data_path=save_maps_path,
    #     )

    #     plot_radiometric_pipeline(
    #         signals=first_signals,
    #         wl_axis=wl_axis,
    #         idx_sim_minus=plotting["minus_idx"],
    #         idx_sim_mean=plotting["mean_idx"],
    #         idx_sim_plus=plotting["plus_idx"],
    #         spt_pitch_sim=spt_sim,
    #         spt_pitch_instr=DET.plate_scale_length,
    #         save=f"fig_radiometric_pipeline_{sec}.png",
    #     )

    #     plot_velocity_std_map(
    #         v_std_map=analysis_res["v_std"],
    #         save=f"fig_vstd_{sec}.png",
    #         x_pix_size=SIM.slit_scan_step.to(u.arcsec).value,
    #         y_pix_size=DET.plate_scale_angle.to(u.arcsec / u.pix).value,
    #         idx_minus=plotting["minus_idx"],
    #         idx_mean=plotting["mean_idx"],
    #         idx_plus=plotting["plus_idx"],
    #     )

    #     plot_intensity_vs_vstd(
    #         intensity=first_signals[0].sum(axis=2) * (wl_axis[1] - wl_axis[0]).cgs.value,
    #         v_std=analysis_res["v_std"],
    #         save=f"fig_int_vs_vstd_{sec}.png",
    #         vstd_max=1e9,
    #     )

    #     plot_spectra(
    #         dn_cube=first_signals[7],
    #         wl_axis=wl_axis,
    #         idx_sim_minus=plotting["minus_idx"],
    #         idx_sim_mean=plotting["mean_idx"],
    #         idx_sim_plus=plotting["plus_idx"],
    #         wl0=wl0,
    #         fit_cube=first_fits,                 # <-- new argument
    #         sigma_factor=plotting["sigma_factor"],
    #         key_pixel_colors=("mediumseagreen", "black", "deeppink"),
    #         save=f"fig_spectra_dn_{sec}.png",
    #     )

    #     plot_velocity_maps(
    #         analysis_res["v_mean"],
    #         analysis_res["v_std"],
    #         save=f"fig_velocity_maps_{sec}.png",
    #         x_pix_size=SIM.slit_scan_step.to_value(u.arcsec),
    #         y_pix_size=DET.plate_scale_angle.to_value(u.arcsec / u.pix),
    #         idx_minus=plotting["minus_idx"],
    #         idx_mean=plotting["mean_idx"],
    #         idx_plus=plotting["plus_idx"],
    #     )

    plot_exposure_time_map(
        analysis_per_exp=analysis_per_exp,
        precision_requirement=2.0,  # km/s
        x_pix_size=SIM.slit_scan_step.to_value(u.arcsec),
        y_pix_size=DET.plate_scale_angle.to_value(u.arcsec / u.pix),
        save="fig_exposure_time_map.png",
        cmap="viridis",
    )

    # npz_files = [
    #     "/gpfs/data/fs70652/jamesm/solar/solc/solc_euvst_sw_response/eis_250618_2009_exp20.0.npz",
    #     "/gpfs/data/fs70652/jamesm/solar/solc/solc_euvst_sw_response/euvst_250618_2015_exp20.0.npz"
    # ]
    # plot_multi_maps(
    #     [str(f) for f in npz_files],
    #     map_type="intensity",
    #     save="fig_multi_intensity.png",
    #     figsize=(8, 5.5),
    # )

if __name__ == "__main__":
    main()