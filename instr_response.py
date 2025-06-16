from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple

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
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import sys
from typing import Tuple

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
    wvl_res: u.Quantity = (16.9 * u.mAA).cgs / u.pixel
    plate_scale_angle: u.Quantity = 0.159 * u.arcsec / u.pixel
    si_fano: float = 0.115

    @property
    def plate_scale_length(self) -> u.Quantity:
        return angle_to_distance(self.plate_scale_angle * 1*u.pix) / u.pixel


@dataclass
class Telescope:
    D_ap: u.Quantity = 0.28 * u.m
    pm_eff: float = 0.161
    grat_eff: float = 0.0623
    filt_eff: float = 0.507
    psf_focus_res: u.Quantity = 0.5 * u.um / u.pixel
    psf_mesh_res: u.Quantity = 6.12e-4 * u.mm / u.pixel
    psf_focus_file: Path = Path("data/swc/psf_euvst_v20230909_195119_focus.txt")
    psf_mesh_file: Path = Path("data/swc/psf_euvst_v20230909_derived_195119_mesh.txt")
    psf: np.ndarray | None = field(default=None, init=False)

    @property
    def collecting_area(self) -> u.Quantity:
        return 0.5 * np.pi * (self.D_ap / 2) ** 2


@dataclass
class Simulation:
    expos: u.Quantity = u.Quantity([1, 2], u.s)
    n_iter: int = 2
    vis_sl: u.Quantity = 1 * u.photon / (u.s * u.pixel)
    slit_width: u.Quantity = 0.2 * u.arcsec
    slit_scan_step: u.Quantity = 0.2 * u.arcsec
    ncpu: int = -1


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

def slit_sample_cube(
    cube_hr: u.Quantity,
    dx: u.Quantity,
    slit_w: u.Quantity,
    scan_step: u.Quantity,
) -> u.Quantity:
    """Slide slit along X and integrate cube into mean specific intensity."""
    Nx, Ny, Nl = cube_hr.shape

    left_edges = np.arange(
        0,
        Nx * dx.to(u.cm/u.pix).value - slit_w.to(u.cm).value,
        scan_step.to(u.cm).value
    ) * u.cm

    right_edge_of_sim = (cube_hr.shape[0] * u.pix) * dx
    right_edge_of_slit_scans = left_edges[-1] + slit_w.to(u.cm)

    print(f"  Trimming {((right_edge_of_sim - right_edge_of_slit_scans).to(u.Mm)):.2f} from the right edge due to slit discretisation (slit is {(slit_w.to(u.Mm)):.2f} wide)...")

    cube_out = np.empty((len(left_edges), Ny, Nl), dtype=cube_hr.dtype)

    # make a running total (cummulative sum) along the scan axis, so
    #  that the intensity between two positions can simply be found by
    #  subtracting the two positions in the prefix array
    prefix = np.concatenate(
        [np.zeros((1, Ny, Nl), dtype=cube_hr.dtype), np.cumsum(cube_hr, axis=0) * dx],
        axis=0,
    )

    # define a function to interpolate the prefix array at a given position
    #  this is used to find the intensity at the left and right edge of the slit
    def interp_prefix(pos: u.Quantity) -> u.Quantity:
        pos_idx = (pos / dx).decompose().value
        i0 = np.floor(pos_idx).astype(int)
        alpha = pos_idx - i0
        i0 = np.clip(i0, 0, Nx)
        i1 = np.clip(i0 + 1, 0, Nx)
        return (1 - alpha) * prefix[i0] + alpha * prefix[i1]

    # loop over all slit positions and calculate the mean intensity
    #  by integrating the prefix array between the left and right edge of the slit
    for s, L in enumerate(left_edges):
        R = L + slit_w
        integral = interp_prefix(R) - interp_prefix(L)
        cube_out[s] = (integral / slit_w).value

    return cube_out * cube_hr.unit


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

  def _reb_spec(i: int) -> NDData:
    row = np.zeros((cube_sim.shape[1], len(wl_det))) * cube_sim.unit
    for j in range(cube_sim.shape[1]):
      spec = Spectrum1D(cube_sim[i, j, :], spectral_axis=wl_sim)
      row[j, :] = resampler(spec, wl_det).flux
    return row

  rows = Parallel(n_jobs=sim.ncpu)(
    delayed(_reb_spec)(i) for i in tqdm(range(cube_sim.shape[0]), desc="Spectral rebin", unit="slice", leave=False)
  )
  cube_spec = np.stack(rows)

  print("  Scanning slit across observation (*nx*,ny,nl)...")
  cube_scan = slit_sample_cube(
    cube_spec,
    spt_sim,
    angle_to_distance(slit_width_as),
    angle_to_distance(scan_step_as),
  )

  print("  Rebinning slit scan to detector plate scale (nx,*ny*,nl)...")
  factor_y = (spt_sim / y_pitch_out).decompose().value
  cube_det = zoom(cube_scan, (1, factor_y, 1), order=1) * cube_scan.unit

  # --- Calculate new iloc for plotting indices ---
  def map_idx(idx):
    if idx is None:
      return None
    # idx = (x, y) in original cube
    # x: scan axis, y: slit axis
    # For scan axis: step size changes from spt_sim to scan_step_as
    # For slit axis: step size changes from spt_sim to y_pitch_out
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


def photons_to_pixel_rate(ph_sr_cm_s: u.Quantity, wl_pitch: u.Quantity, spt_pitch: u.Quantity) -> u.Quantity:
    pixel_solid_angle = ((spt_pitch * u.pixel) ** 2 / const.au ** 2) * u.sr
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
    def _scan(i: int) -> np.ndarray:
        return np.array([_fit_one(wv.value, signal_cube.value[i, j]) for j in range(n_slit)])

    rows = Parallel(n_jobs=n_jobs)(delayed(_scan)(i) for i in tqdm(range(n_scan), desc="Gaussian fits", unit="scan", leave=False))
    arr = np.array(rows)
    return arr * np.array([signal_cube.unit, wv.unit, wv.unit, signal_cube.unit])


# -----------------------------------------------------------------------------
# Monte-Carlo wrapper
# -----------------------------------------------------------------------------

def simulate_once(I_cube: u.Quantity, wl_axis: u.Quantity, t_exp: u.Quantity, det: Detector, tel: Telescope, sim: Simulation) -> Tuple[u.Quantity, ...]:

    signal0 = add_poisson(I_cube)
    signal1 = intensity_to_photons(signal0, wl_axis)
    signal2 = add_effective_area(signal1, tel)
    signal3 = photons_to_pixel_rate(signal2, det.wvl_res, det.plate_scale_length)
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

def velocity_from_fit(fit_arr: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    arr = fit_arr[..., 1]
    unit = arr.flat[0].unit
    values = np.vectorize(lambda q: q.value)(arr)
    centres = values * unit
    mask_bad = np.all(fit_arr == -1, axis=-1)
    v = (centres - wl0) / wl0 * const.c
    v = np.where(mask_bad, -1 * u.cm / u.s, v)
    return v


def analyse(fits_all: u.Quantity, fits_true: u.Quantity, wl0: u.Quantity) -> dict:
    v_true = velocity_from_fit(fits_true, wl0)
    v_all = velocity_from_fit(fits_all, wl0)
    return {
        "v_mean": v_all.mean(axis=0),
        "v_std": v_all.std(axis=0),
        "v_err": v_true - v_all.mean(axis=0),
        "v_samples": v_all,
        "v_true": v_true,
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
    spt_pitch_sim: u.Quantity,
    spt_pitch_instr: u.Quantity,
    save: str = "fig_radiometric_pipeline.png",
    row_labels: Iterable[str] = (r"$\mu-\sigma$", r"$\mu$", r"$\mu+\sigma$"),
    key_pixel_colors: Iterable[str] = ("mediumseagreen", "black", "deeppink"),
) -> plt.Figure:
    factor = (spt_pitch_sim / spt_pitch_instr).decompose().value
    def _map(idx_sim: Tuple[int, int]) -> Tuple[int, int]:
        return tuple(int(round(i * factor)) for i in idx_sim)
    idxs_reb = tuple(_map(x) for x in (idx_sim_minus, idx_sim_mean, idx_sim_plus))

    wl_A = wl_axis.to(u.angstrom).value
    def spectrum(stage_idx: int, row_idx: int) -> np.ndarray:
        return signals[stage_idx][idxs_reb[row_idx] + (slice(None),)]

    fig, axes = plt.subplots(3, 4, figsize=(10, 6), sharex="row", constrained_layout=True)
    fig.subplots_adjust(right=0.86, wspace=0.18, hspace=0.06)

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
        ax0.tick_params(direction="in", which="both", top=True, right=True)

        ax_r1 = ax0.twinx()
        sp2 = spectrum(2, row)
        ax_r1.step(wl_A, sp2, where="mid", color="tab:orange", lw=1)
        ax_r1.set_ylim(sp2.min(), sp2.max())
        ax_r1.set_ylabel(r"ph s$^{-1}$ sr$^{-1}$ cm$^{-1}$", color="tab:orange", fontsize=7)
        ax_r1.yaxis.labelpad = 8
        ax_r1.tick_params(direction="in", colors="tab:orange", which="both", right=True)
        ax_r1.patch.set_visible(False)

        ax_r2 = ax0.twinx()
        sp3 = spectrum(3, row)
        ax_r2.step(wl_A, sp3, where="mid", color="tab:blue", lw=1)
        ax_r2.set_ylim(sp3.min(), sp3.max())
        ax_r2.spines.right.set_position(("axes", 1.15))
        ax_r2.set_ylabel(r"ph s$^{-1}$ pix$^{-1}$", color="tab:blue", fontsize=7)
        ax_r2.yaxis.labelpad = 24
        ax_r2.tick_params(direction="in", colors="tab:blue", which="both", right=True)
        ax_r2.patch.set_visible(False)

        ax1 = axes[row, 1]
        sp4 = spectrum(4, row)
        ax1.step(wl_A, sp4, where="mid", color=colour, lw=1)
        if row == 0:
            ax1.set_title("signal4", fontsize=8)
        ax1.set_ylabel(r"ph s$^{-1}$ pix$^{-1}$ (PSF)", color=colour, fontsize=7)
        ax1.tick_params(direction="in", which="both", top=True, right=True)

        ax2 = axes[row, 2]
        sp5 = spectrum(5, row)
        ax2.step(wl_A, sp5, where="mid", color=colour, lw=1)
        if row == 0:
            ax2.set_title("signal5", fontsize=8)
        ax2.set_ylabel(r"e$^-$ pix$^{-1}$", color=colour, fontsize=7)
        ax2.tick_params(direction="in", which="both", top=True, right=True)

        ax3 = axes[row, 3]
        sp6 = spectrum(6, row)
        ax3.step(wl_A, sp6, where="mid", color=colour, lw=1)
        if row == 0:
            ax3.set_title("signal6/7", fontsize=8)
        ax3.set_ylabel(r"e$^-$ pix$^{-1}$ (stray)", color=colour, fontsize=7)
        ax3.tick_params(direction="in", which="both", top=True, right=True)

        ax_r3 = ax3.twinx()
        sp7 = spectrum(7, row)
        ax_r3.step(wl_A, sp7, where="mid", color="tab:red", lw=1)
        ax_r3.spines.right.set_position(("axes", 1.12))
        ax_r3.set_ylim(sp7.min(), sp7.max())
        ax_r3.set_ylabel(r"DN pix$^{-1}$", color="tab:red", fontsize=7)
        ax_r3.yaxis.labelpad = 16
        ax_r3.tick_params(direction="in", colors="tab:red", which="both", right=True)
        ax_r3.patch.set_visible(False)

        if row == 2:
            for col in range(4):
                axes[row, col].set_xlabel("Wavelength [Å]")

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
  save: str,
  key_pixel_colors: Iterable[str] = ("mediumseagreen", "black", "deeppink"),
) -> None:
  si = signal_cube.sum(axis=2)
  log_si = np.log10(si, where=si > 0.0, out=np.zeros_like(si))
  v_map = velocity_from_fit(fit_cube, wl0).to(u.km / u.s)

  # --- Compute arcsec axes with correct pixel sizes ---
  n_scan, n_slit = si.shape
  # X: scan axis, pixel size = slit width
  x_pix_size = SIM.slit_scan_step.to(u.arcsec).value
  # Y: slit axis, pixel size = plate scale
  y_pix_size = DET.plate_scale_angle.to(u.arcsec/u.pix).value

  x = (np.arange(n_scan) - n_scan // 2) * x_pix_size
  y = (np.arange(n_slit) - n_slit // 2) * y_pix_size
  extent = [
    x[0] - x_pix_size/2, x[-1] + x_pix_size/2,
    y[0] - y_pix_size/2, y[-1] + y_pix_size/2
  ]

  fig, (axI, axV) = plt.subplots(1, 2, figsize=(10, 5))
  imI = axI.imshow(log_si.T, origin="lower", aspect="auto",
            cmap="afmhot", vmin=0, extent=extent)
  axI.set_title(r'$\log_{10} \int I(\lambda) d\lambda$')
  fig.colorbar(imI, ax=axI, orientation="horizontal",
          label="log10 Intensity")

  imV = axV.imshow(v_map.T.value, origin="lower", aspect="auto",
            cmap="RdBu_r", vmin=-15, vmax=15, extent=extent)
  axV.set_title("Doppler velocity of peak intensity [km/s]")
  fig.colorbar(imV, ax=axV, orientation="horizontal",
          label="v [km/s]")

  # Set axis labels and ticks every 15 arcsec (e.g. -30, -15, 0, 15, 30)
  interval = 15.0  # arcsec
  max_x = max(abs(extent[0]), abs(extent[1]))
  max_y = max(abs(extent[2]), abs(extent[3]))
  xticks = np.arange(
    -np.ceil(max_x/interval)*interval,
     np.ceil(max_x/interval)*interval + interval/2,
     interval
  )
  yticks = np.arange(
    -np.ceil(max_y/interval)*interval,
     np.ceil(max_y/interval)*interval + interval/2,
     interval
  )
  xticks = xticks[(xticks >= extent[0]) & (xticks <= extent[1])]
  yticks = yticks[(yticks >= extent[2]) & (yticks <= extent[3])]

  for ax in (axI, axV):
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # Set correct aspect ratio: ratio of y_pix_size to x_pix_size
    ax.set_aspect(y_pix_size / x_pix_size)

  # Overplot key pixels
  for idx, color in zip([idx_sim_minus, idx_sim_mean, idx_sim_plus], key_pixel_colors):
    if idx is not None:
      x_pos = (idx[0] - n_scan // 2) * x_pix_size
      y_pos = (idx[1] - n_slit // 2) * y_pix_size
      for ax in (axI, axV):
        ax.plot(x_pos, y_pos, marker="o", color=color,
            markersize=8, fillstyle="none", lw=2)

  plt.tight_layout()
  plt.savefig(save, dpi=300)
  plt.close(fig)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main() -> None:
    global DET, TEL, SIM

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
    # cube_reb, wl_axis = rebin_atmosphere(cube_sim, wl_sim, spt_sim, DET, SIM)
    cube_reb, wl_axis, plotting = rebin_atmosphere(cube_sim, wl_sim, spt_sim, DET, SIM, plotting)

    print("Fitting ground truth cube...")
    fit_truth = fit_cube_gauss(cube_reb.cgs, wl_axis.cgs)

    # Monte-Carlo simulation for each exposure time
    for t_exp in tqdm(SIM.expos, desc="Exposure time", unit="exposure"):
        signals, fits = monte_carlo(cube_reb, wl_axis, t_exp, DET, TEL, SIM, n_iter=SIM.n_iter)
        analysis_results = analyse(fits, fit_truth, wl0)
        plot_maps(
            signals[0][-1],    # final stage of first simulation
            fits[0],           # added fits[0] to use fitted centers
            wl_axis,
            wl0,
            plotting["minus_idx"],
            plotting["mean_idx"],
            plotting["plus_idx"],
            save=f"fig_maps_{t_exp.to_value(u.s)}.png",
        )
        plot_radiometric_pipeline(
            signals=signals[0],
            wl_axis=wl_axis,
            idx_sim_minus=plotting["minus_idx"],
            idx_sim_mean=plotting["mean_idx"],
            idx_sim_plus=plotting["plus_idx"],
            spt_pitch_sim=spt_sim,
            spt_pitch_instr=DET.plate_scale_length,
            save="fig_detector_pipeline.png",
        )
    print("Done.")


if __name__ == "__main__":
    main()
