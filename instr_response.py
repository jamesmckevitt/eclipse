from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple
import math
import numpy as np
import astropy.units as u
import astropy.constants as const
import astropy.wcs
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
import shutil
from matplotlib.colors import ListedColormap, BoundaryNorm
import yaml
import argparse
import os
import scipy.interpolate
from ndcube import NDCube
import pickle
from astropy.wcs import WCS
import reproject

# ------------------------------------------------------------------
#  Throughput helpers & AluminiumFilter
# ------------------------------------------------------------------
def _load_throughput_table(path: str | Path) -> tuple[u.Quantity, np.ndarray]:
    """Return (λ, T) arrays from a 2-col ASCII table (skip comments). λ is in nm."""
    arr = np.loadtxt(path, skiprows=2)
    wl = arr[:, 0] * u.nm
    tr = arr[:, 1]
    return wl, tr


def _interp_tr(wavelength_nm: float, wl_tab: np.ndarray, tr_tab: np.ndarray) -> float:
    """Linear interpolation."""
    f = scipy.interpolate.interp1d(wl_tab, tr_tab, bounds_error=False, fill_value=np.nan)
    return float(f(wavelength_nm))

@dataclass
class AluminiumFilter:
    """Multi-layer EUV filter (Al + Al₂O₃ + C) in front of SWC detector."""
    al_thickness: u.Quantity = 1500 * u.angstrom
    oxide_thickness: u.Quantity = 95 * u.angstrom
    c_thickness: u.Quantity = 0 * u.angstrom
    mesh_throughput: float = 0.8
    al_table: Path = Path("data/throughput/throughput_aluminium_1000_angstrom.dat")
    oxide_table: Path = Path("data/throughput/throughput_aluminium_oxide_1000_angstrom.dat")
    c_table: Path = Path("data/throughput/throughput_carbon_1000_angstrom.dat")
    table_thickness: u.Quantity = 1000 * u.angstrom

    def total_throughput(self, wl0: u.Quantity) -> float:
        """Calculate throughput at a given central wavelength (wl0, astropy Quantity)."""
        wl_nm = wl0.to_value(u.nm)
        wl_al, tr_al = _load_throughput_table(self.al_table)
        wl_ox, tr_ox = _load_throughput_table(self.oxide_table)
        wl_c,  tr_c  = _load_throughput_table(self.c_table)
        t_al = _interp_tr(wl_nm, wl_al, tr_al) ** (self.al_thickness.cgs / self.table_thickness.cgs)
        t_ox = _interp_tr(wl_nm, wl_ox, tr_ox) ** (self.oxide_thickness.cgs / self.table_thickness.cgs)
        t_c  = _interp_tr(wl_nm, wl_c,  tr_c)  ** (self.c_thickness.cgs / self.table_thickness.cgs)
        return t_al * t_ox * t_c * self.mesh_throughput
# ------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Configuration objects
# -----------------------------------------------------------------------------
@dataclass
class Detector_SWC:
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

class Detector_EIS:
    qe_euv: float = 1  # EIS telescope effective area already includes QE
    qe_vis: float = 1
    pix_size: u.Quantity = (13.5 * u.um).cgs / u.pixel
    wvl_res: u.Quantity = (22.3 * u.mAA).cgs / u.pixel
    plate_scale_angle: u.Quantity = 1 * u.arcsec / u.pixel

    @property
    def plate_scale_length(self) -> u.Quantity:
        return angle_to_distance(self.plate_scale_angle * 1*u.pix) / u.pixel
    
    @property
    def e_per_ph_euv(self) -> u.Quantity:
        return Detector_SWC.e_per_ph_euv
    @property
    def e_per_ph_vis(self) -> u.Quantity:
        return Detector_SWC.e_per_ph_vis
    @property
    def read_noise_rms(self) -> u.Quantity:
        return Detector_SWC.read_noise_rms
    @property
    def dark_current(self) -> u.Quantity:
        return Detector_SWC.dark_current
    @property
    def gain_e_per_dn(self) -> u.Quantity:
        return Detector_SWC.gain_e_per_dn
    @property
    def max_dn(self) -> u.Quantity:
        return Detector_SWC.max_dn
    @property
    def si_fano(self) -> float:
        return Detector_SWC.si_fano

@dataclass
class Telescope_EUVST:
    D_ap: u.Quantity = 0.28 * u.m
    pm_eff: float = 0.161
    grat_eff: float = 0.0623
    psf_focus_res: u.Quantity = 0.5 * u.um / u.pixel
    psf_mesh_res: u.Quantity = 6.12e-4 * u.mm / u.pixel
    psf_focus_file: Path = Path("data/psf/psf_euvst_v20230909_195119_focus.txt")
    psf_mesh_file: Path = Path("data/psf/psf_euvst_v20230909_derived_195119_mesh.txt")
    psf: np.ndarray | None = field(default=None, init=False)
    filter: AluminiumFilter = field(default_factory=AluminiumFilter)

    @property
    def collecting_area(self) -> u.Quantity:
        return 0.5 * np.pi * (self.D_ap / 2) ** 2

    def throughput(self, wl0: u.Quantity) -> float:
        return self.pm_eff * self.grat_eff * self.filter.total_throughput(wl0)

    def ea_and_throughput(self, wl0: u.Quantity) -> u.Quantity:
        return self.collecting_area * self.throughput(wl0)

class Telescope_EIS:
    def ea_and_throughput(self, wl0: u.Quantity) -> u.Quantity:
        return 0.23 * u.cm**2

@dataclass
class Simulation:
  expos: u.Quantity = u.Quantity([0.5, 1, 2, 5, 10, 20, 40, 80], u.s)
  n_iter: int = 25
  slit_width: u.Quantity = 0.2 * u.arcsec
  ncpu: int = -1
  instrument: str = "SWC"
  vis_sl: u.Quantity = 0 * u.photon / (u.s * u.pixel)

  @property
  def slit_scan_step(self) -> u.Quantity:
    return self.slit_width

  def __post_init__(self):
    allowed_slits = {
      "EIS": [1, 2, 4],
      "SWC": [0.2, 0.4, 1],
    }
    inst = self.instrument.upper()
    slit_val = self.slit_width.to_value(u.arcsec)
    if inst == "EIS":
      if slit_val not in allowed_slits["EIS"]:
        raise ValueError("For EIS, slit_width must be 1, 2, or 4 arcsec.")
    elif inst in ("SWC"):
      if slit_val not in allowed_slits["SWC"]:
        raise ValueError("For SWC, slit_width must be 0.2, 0.4, or 1 arcsec.")

# Global configuration used by plotting helpers --------------------------------
DET = Detector_SWC()
TEL = Telescope_EUVST()
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


def distance_to_angle(distance: u.Quantity) -> u.Quantity:
    if distance.unit.physical_type != "length":
        raise ValueError("Input must be a length")
    # return 2 * np.arctan(distance / (2 * const.au)).to(u.rad) * u.rad
    return (2 * np.arctan(distance / (2 * const.au))).to(u.arcsec)


def parse_yaml_input(val):
    if isinstance(val, str):
        return u.Quantity(val)
    else:
        raise ValueError("Thickness must be a string with units, e.g. '10 angstrom'.")


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


def resample_ndcube_spectral_axis(ndcube, spectral_axis, output_resolution):
    """
    Resample the spectral axis of an NDCube using FluxConservingResampler.

    Parameters
    ----------
    ndcube : NDCube
        The input NDCube.
    spectral_axis : int
        The index of the spectral axis (e.g., 0, 1, or 2).
    output_resolution : astropy.units.Quantity
        The desired output spectral resolution (e.g., 0.01 * u.nm).

    Returns
    -------
    NDCube
        A new NDCube with the spectral axis resampled.
    """
    # Get the world coordinates of the spectral axis
    spectral_world = ndcube.axis_world_coords(spectral_axis)[0]
    spectral_world = spectral_world.to(output_resolution.unit)

    # Define new spectral grid
    new_spec_grid = np.arange(
        spectral_world.min().value,
        spectral_world.max().value + output_resolution.value,
        output_resolution.value
    ) * output_resolution.unit

    n_spec = len(new_spec_grid)

    # Move spectral axis to last for easier iteration
    data = np.moveaxis(ndcube.data, spectral_axis, -1)
    shape = data.shape
    flat_data = data.reshape(-1, shape[-1])

    resampler = FluxConservingResampler(extrapolation_treatment="zero_fill")
    resampled = np.zeros((flat_data.shape[0], n_spec))

    def _resample_pixel(i):
      spec = Spectrum1D(flux=flat_data[i] * ndcube.unit, spectral_axis=spectral_world)
      res = resampler(spec, new_spec_grid)
      return res.flux.value

    with tqdm_joblib(tqdm(total=flat_data.shape[0], desc="Resampling spectral axis", unit="pixel", leave=False)):
      results = Parallel(n_jobs=-1)(
        delayed(_resample_pixel)(i) for i in range(flat_data.shape[0])
      )
    resampled = np.vstack(results)

    # Reshape back to original spatial shape, but with new spectral length
    new_shape = list(shape[:-1]) + [n_spec]
    resampled = resampled.reshape(new_shape)

    # Move spectral axis back to original position
    resampled = np.moveaxis(resampled, -1, spectral_axis)

    # Update WCS for new spectral axis
    new_wcs = ndcube.wcs.deepcopy()

    wcs_axis = new_wcs.wcs.naxis - 1 - spectral_axis  # Reverse axis order for WCS
    center_pixel = (n_spec + 1) / 2  # 1-based index (FITS convention)
    new_wcs.wcs.crpix[wcs_axis] = center_pixel
    new_wcs.wcs.crval[wcs_axis] = new_spec_grid[int(center_pixel - 1)].to_value(new_wcs.wcs.cunit[wcs_axis])
    new_wcs.wcs.cdelt[wcs_axis] = (new_spec_grid[1] - new_spec_grid[0]).to_value(new_wcs.wcs.cunit[wcs_axis])

    return NDCube(resampled, wcs=new_wcs, unit=ndcube.unit, meta=ndcube.meta)


def reproject_ndcube_heliocentric_to_helioprojective(
    new_cube_spec: NDCube,
    sim: Simulation,
    det: Detector_SWC,
) -> NDCube:
    """ Reproject an NDCube from heliocentric to helioprojective coordinates. """

    nx, ny, _ = new_cube_spec.shape
    wcs_hc = new_cube_spec.wcs

    dx = wcs_hc.wcs.cdelt[2] * wcs_hc.wcs.cunit[2]
    dy = wcs_hc.wcs.cdelt[1] * wcs_hc.wcs.cunit[1]
    x_angle = distance_to_angle(dx)
    y_angle = distance_to_angle(dy)

    wcs_hp = WCS(naxis=3)
    wcs_hp.wcs.ctype = [wcs_hc.wcs.ctype[0], 'HPLN-TAN', 'HPLT-TAN']
    wcs_hp.wcs.cunit = [wcs_hc.wcs.cunit[0], 'arcsec', 'arcsec']
    wcs_hp.wcs.crpix = [wcs_hc.wcs.crpix[0],
                        (ny + 1) / 2,
                        (nx + 1) / 2]
    wcs_hp.wcs.crval = [wcs_hc.wcs.crval[0], 0, 0]
    wcs_hp.wcs.cdelt = [wcs_hc.wcs.cdelt[0], y_angle.to_value(u.arcsec), x_angle.to_value(u.arcsec)]
    new_cube_spec_hp = NDCube(new_cube_spec.data, wcs=wcs_hp, unit=new_cube_spec.unit, meta=new_cube_spec.meta)

    nx_in, ny_in, nl_in = new_cube_spec_hp.shape
    fov_x = nx_in * x_angle
    fov_y = ny_in * y_angle
    pitch_x = sim.slit_width
    pitch_y = det.plate_scale_angle
    nx_out = int(np.floor((fov_x / pitch_x).decompose().value))
    ny_out = int(np.floor((fov_y / pitch_y).decompose().value))
    shape_out = [nx_out, ny_out, nl_in]

    crpix_spec = (shape_out[2] + 1) / 2
    crpix_y = (shape_out[1] + 1) / 2
    crpix_x = (shape_out[0] + 1) / 2

    wcs_tgt = WCS(naxis=3)
    wcs_tgt.wcs.ctype = [wcs_hc.wcs.ctype[0], 'HPLN-TAN', 'HPLT-TAN']
    wcs_tgt.wcs.cunit = [wcs_hc.wcs.cunit[0], 'arcsec', 'arcsec']
    wcs_tgt.wcs.crpix = [crpix_spec, crpix_y, crpix_x]
    wcs_tgt.wcs.crval = [wcs_hc.wcs.crval[0], 0, 0]
    wcs_tgt.wcs.cdelt = [wcs_hc.wcs.cdelt[0],
                        (det.plate_scale_angle * u.pix).to_value(u.arcsec),
                        (sim.slit_width).to_value(u.arcsec)]

    new_cube_spec_hp_spat = new_cube_spec_hp.reproject_to(
        wcs_tgt,
        shape_out=shape_out,
        algorithm='interpolation',
        parallel=True,
        order='bilinear',
    ) * new_cube_spec_hp.unit

    return new_cube_spec_hp_spat


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


def load_atmosphere(pkl_file: str) -> Tuple[u.Quantity, u.Quantity, dict]:
    with open(pkl_file, "rb") as f:
        tmp = pickle.load(f)
    cube = tmp["sim_si"]
    plotting = {
        "plus_coords": tuple(tmp["plus_coords"]),
        "mean_coords": tuple(tmp["mean_coords"]),
        "minus_coords": tuple(tmp["minus_coords"]),
        "sigma_factor": float(tmp["sigma_factor"]),
        "margin": int(tmp["margin"]),
    }
    return cube, plotting


def rebin_atmosphere(
    cube_sim: u.Quantity,
    det: Detector_SWC,
    sim: Simulation,
) -> Tuple[u.Quantity, u.Quantity, dict]:

    print("  Spectral rebinning to instrument resolution (nx,ny,*nl*)...")
    cube_spec = resample_ndcube_spectral_axis(cube_sim, spectral_axis=2, output_resolution=det.wvl_res*u.pix)

    print("  Spatially rebinning to plate scale (nx,*ny*,nl) and slit width (*nx*,ny,nl)...")

    # Reproject does not support non-celestial WCS (07/2025), so can't reproject directly.
    # The workaround: Turn existing NDCube into a new one with celestial WCS, then reproject to the appropriate resolution.
    # The next problem: NDCube does not (because astropy does not) support flux-conserving reprojection.
    # The interpolation sampling can be done with a new function here:
    # Reproject to helioprojective coordinates with new spatial resolution
    cube_det = reproject_ndcube_heliocentric_to_helioprojective(
        cube_spec,
        sim,
        det
    )

    # The original approach, not compatible with NDCube, was to hack the flux conserving spectral resampler to conserve flux, included here:
    # # ------------------------------------------------------------------
    # # Helper: flux-conserving resample of a 3-D cube along one axis
    # # ------------------------------------------------------------------
    # def _resample_axis(cube, old_step, new_step, axis, n_jobs):
    #     """
    #     Parameters
    #     ----------
    #     cube : Quantity[(Nx, Ny, Nl)]
    #         The data cube.
    #     old_step : Quantity
    #         Physical size of one native pixel (same units for X & Y).
    #     new_step : Quantity
    #         Desired size of one output pixel (same units as old_step).
    #     axis : {0, 1}
    #         0 → rebin X (first) axis, 1 → rebin Y (second) axis.
    #     n_jobs : int
    #         Passed to joblib.Parallel.

    #     Returns
    #     -------
    #     reb : Quantity
    #         Cube rebinned along `axis`, flux conserved.
    #     """
    #     # ---------- coordinate grids ------------------------------------------------
    #     if axis == 0:         # X axis
    #         Nx, Ny, Nl = cube.shape
    #         Ny_range   = range(Ny)
    #         N_in       = Nx
    #     else:                 # Y axis
    #         Nx, Ny, Nl = cube.shape
    #         Ny_range   = range(Nx)          # we will iterate over X instead
    #         N_in       = Ny

    #     x_in   = np.arange(N_in) * old_step               # pixel centres
    #     tot_len = N_in * old_step
    #     N_out = int(np.floor((tot_len / new_step).decompose().value))
    #     x_out = (np.arange(N_out) + 0.5) * new_step       # centre of each new pixel

    #     # ---------- single 1-D resampler we re-use everywhere -----------------------
    #     fcr = FluxConservingResampler(extrapolation_treatment="zero_fill")

    #     # ---------- loop over “rows" in parallel ------------------------------------
    #     def _one_row(j):
    #         """
    #         Resample one (λ-stacked) 1-D row: either (Nx, Nl) at fixed Y
    #         or (Ny, Nl) at fixed X, depending on `axis`.
    #         Returns a 2-D array (N_out, Nl) in plain numpy dtype.
    #         """
    #         if axis == 0:                          # rebin along X for fixed Y=j
    #             row = cube[:, j, :]                # shape (Nx, Nl)
    #         else:                                  # rebin along Y for fixed X=j
    #             row = cube[j, :, :]                # shape (Ny, Nl)

    #         out = np.empty((N_out, Nl), dtype=cube.dtype)
    #         for k in range(Nl):
    #             spec = Spectrum1D(row[:, k] * cube.unit, spectral_axis=x_in)
    #             out[:, k] = fcr(spec, x_out).flux.value
    #         return out

    #     with tqdm_joblib(
    #         tqdm(total=len(Ny_range), desc="  > spatial rebin", leave=False)
    #     ):
    #         stacked = Parallel(n_jobs=n_jobs)(
    #             delayed(_one_row)(j) for j in Ny_range
    #         )

    #     # ---------- assemble cube back in correct orientation -----------------------
    #     reb_val = np.stack(stacked, axis=1 if axis == 0 else 0)  # (N_out, Ny, Nl) or (Nx, N_out, Nl)
    #     return reb_val * cube.unit
    # # ------------------------------------------------------------------


    # print("  Scanning slit across observation (*nx*,ny,nl)...")
    # dx_len   = (spt_sim * u.pix).to(u.cm) if spt_sim.unit.is_equivalent(u.cm/u.pix) else spt_sim.to(u.cm)
    # slit_w   = angle_to_distance(slit_width_as).to(u.cm)
    # cube_scan = _resample_axis(
    #     cube_spec,
    #     old_step = dx_len,
    #     new_step = slit_w,
    #     axis     = 0,                # rebin X
    #     n_jobs   = sim.ncpu if sim.ncpu > 0 else -1,
    # )

    # print("  Rebinning each slit scan to detector plate scale (nx,*ny*,nl)...")
    # y_pitch_cm = (y_pitch_out * u.pix).to(u.cm) if y_pitch_out.unit.is_equivalent(u.cm/u.pix) else y_pitch_out.to(u.cm)
    # cube_det = _resample_axis(
    #     cube_scan,
    #     old_step = dx_len,           # same native step in Y
    #     new_step = y_pitch_cm,
    #     axis     = 1,                # rebin Y
    #     n_jobs   = sim.ncpu if sim.ncpu > 0 else -1,
    # )

    return cube_det


# -----------------------------------------------------------------------------
# Radiometric pipeline
# -----------------------------------------------------------------------------

def intensity_to_photons(I: u.Quantity) -> u.Quantity:
    wl_axis = I.axis_world_coords(2)[0]
    E_ph = (const.h * const.c / wl_axis).to("erg") * (1 / u.photon)
    return (I / E_ph).to(u.photon / u.s / u.cm**2 / u.sr / u.cm)


def add_effective_area(ph_cm2_sr_cm_s: u.Quantity, tel: Telescope_EUVST) -> u.Quantity:
    wl0 = ph_cm2_sr_cm_s.meta['rest_wav']
    wl_axis = ph_cm2_sr_cm_s.axis_world_coords(2)[0]
    A_eff = np.array([tel.ea_and_throughput(wl).cgs.value for wl in wl_axis]) * u.cm**2
    return ph_cm2_sr_cm_s * A_eff


def photons_to_pixel_rate(ph_sr_cm_s: u.Quantity, wl_pitch: u.Quantity, plate_scale: u.Quantity, slit_width: u.Quantity) -> u.Quantity:
    pixel_solid_angle = ((plate_scale * u.pixel * slit_width).cgs / const.au.cgs ** 2) * u.sr
    return ph_sr_cm_s * pixel_solid_angle * wl_pitch


def apply_psf(signal: NDCube, psf: np.ndarray) -> NDCube:
    """
    Convolve each detector row (first axis) of an NDCube with a 2-D PSF.

    Parameters
    ----------
    signal : NDCube
        Input cube with shape (n_scan, n_slit, n_lambda).
        The first axis is stepped by the raster scan.
    psf : np.ndarray
        2-D point-spread function sampled on the detector grid
        (dispersion x slit-height).

    Returns
    -------
    NDCube
        New cube with identical WCS / unit / meta but PSF-blurred data.
    """
    data_in = signal.data                       # ndarray view (no units)
    n_scan   = data_in.shape[0]

    blurred = np.empty_like(data_in)
    for i in range(n_scan):
        blurred[i] = convolve2d(data_in[i], psf, mode="same")

    return NDCube(
        data=blurred,
        wcs=signal.wcs.deepcopy(),
        unit=signal.unit,
        meta=signal.meta,
    )


def to_electrons(photon_rate: NDCube, t_exp: u.Quantity, det: Detector_SWC) -> NDCube:
    """
    Convert a photon-rate NDCube to an electron-count NDCube.

    Parameters
    ----------
    photon_rate : NDCube
        Cube of photon s⁻¹ pixel⁻¹.
    t_exp : Quantity
        Exposure time.
    det : Detector_SWC
        Detector description.

    Returns
    -------
    NDCube
        Electron counts per pixel for the given exposure.
    """
    rate_q = photon_rate.data * photon_rate.unit                      # Quantity array

    e_per_ph = fano_noise(det.e_per_ph_euv.value, det.si_fano) * (u.electron / u.photon)

    e = rate_q * t_exp * det.qe_euv * e_per_ph                        # signal
    e += det.dark_current * t_exp                                     # dark current
    e += np.random.normal(0, det.read_noise_rms.value,
                          photon_rate.data.shape) * (u.electron / u.pixel)  # read noise

    e = e.to(u.electron / u.pixel)
    e_val = e.value
    e_val[e_val < 0] = 0                                              # clip negatives

    return NDCube(
        data=e_val,
        wcs=photon_rate.wcs.deepcopy(),
        unit=e.unit,
        meta=photon_rate.meta,
    )


def to_dn(electrons: NDCube, det: Detector_SWC) -> NDCube:
    """
    Convert an electron-count NDCube to DN and clip at the detector's full-well.

    Parameters
    ----------
    electrons : NDCube
        Electron counts per pixel (u.electron / u.pixel).
    det : Detector_SWC
        Detector description containing the gain and max DN.

    Returns
    -------
    NDCube
        Same cube in DN / pixel, with values clipped to det.max_dn.
    """
    dn_q = (electrons.data * electrons.unit) / det.gain_e_per_dn          # Quantity
    dn_q = dn_q.to(det.max_dn.unit)

    dn_val = dn_q.value
    dn_val[dn_val > det.max_dn.value] = det.max_dn.value                  # clip

    return NDCube(
        data=dn_val,
        wcs=electrons.wcs.deepcopy(),
        unit=dn_q.unit,
        meta=electrons.meta,
    )


# -----------------------------------------------------------------------------
# Noise & stray-light models
# -----------------------------------------------------------------------------

def add_poisson(cube: NDCube) -> NDCube:
    """
    Apply Poisson noise to an input NDCube and return a new NDCube
    with the same WCS, unit, and metadata.

    Parameters
    ----------
    cube : NDCube
        Input data cube.

    Returns
    -------
    NDCube
        New cube containing Poisson-noised data.
    """
    noisy = np.random.poisson(cube.data) * cube.unit
    return NDCube(
        data=noisy.value,
        wcs=cube.wcs.deepcopy(),
        unit=noisy.unit,
        meta=cube.meta,
    )


def add_stray_light(
    electrons: NDCube,
    t_exp: u.Quantity,
    det: Detector_SWC,
    sim: Simulation
) -> NDCube:
    """
    Add visible-light stray-light to a cube of electron counts.

    Parameters
    ----------
    electrons : NDCube
        Electron counts per pixel (unit: u.electron / u.pixel).
    t_exp : astropy.units.Quantity
        Exposure time.
    det : Detector_SWC
        Detector description.
    sim : Simulation
        Simulation parameters (contains vis_sl - photon/s/pix).

    Returns
    -------
    NDCube
        New cube with stray-light signal added.
    """
    # Draw Poisson realisation of stray-light photons
    n_vis_ph = np.random.poisson(
        (sim.vis_sl * t_exp).to_value(u.photon / u.pixel),
        size=electrons.data.shape
    ) * (u.photon / u.pixel)

    # Convert photons to electrons
    e_per_ph = fano_noise(det.e_per_ph_vis.value, det.si_fano) * (u.electron / u.photon)
    stray_e  = n_vis_ph * e_per_ph * det.qe_vis               # u.electron / pixel

    # Add to original signal
    out_q = electrons.data * electrons.unit + stray_e
    out_q = out_q.to(electrons.unit)

    return NDCube(
        data=out_q.value,
        wcs=electrons.wcs.deepcopy(),
        unit=out_q.unit,
        meta=electrons.meta,
    )


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


# def fit_cube_gauss(signal_cube: u.Quantity, wv: u.Quantity, n_jobs: int = Simulation.ncpu) -> u.Quantity:
def fit_cube_gauss(signal_cube: NDCube, n_jobs: int = -1) -> u.Quantity:
    n_scan, n_slit, _ = signal_cube.shape
    wv = signal_cube.axis_world_coords(2)[0].cgs

    def _fit_block(block):
        block_nx, n_slit, n_wl = block.shape
        out = np.zeros((block_nx, n_slit, 4))
        for i in range(block_nx):
            for j in range(n_slit):
                out[i, j, :] = _fit_one(wv.value, block[i, j, :])
        return out
    
    with tqdm_joblib(tqdm(total=n_scan, desc="Fit chunks", leave=False)):
        arr = Parallel(n_jobs=n_jobs if n_jobs > 0 else -1)(
            delayed(_fit_block)(signal_cube.data[i:i+1]) for i in range(n_scan)
        )

    return arr * np.array([signal_cube.unit, wv.unit, wv.unit, signal_cube.unit])


# -----------------------------------------------------------------------------
# Monte-Carlo wrapper
# -----------------------------------------------------------------------------

def simulate_once(I_cube: NDCube, t_exp: u.Quantity, det: Detector_SWC, tel: Telescope_EUVST, sim: Simulation) -> Tuple[u.Quantity, ...]:

    signal0 = add_poisson(I_cube)
    signal1 = intensity_to_photons(signal0)
    signal2 = add_effective_area(signal1, tel)
    signal3 = photons_to_pixel_rate(signal2, det.wvl_res, det.plate_scale_length, angle_to_distance(sim.slit_width))
    signal4 = signal3  # signal4 = apply_psf(signal3, tel.psf)
    signal5 = to_electrons(signal4, t_exp, det)
    signal6 = add_stray_light(signal5, t_exp, det, sim)
    signal7 = to_dn(signal6, det)

    return (signal0, signal1, signal2, signal3, signal4, signal5, signal6, signal7)

def monte_carlo(I_cube: u.Quantity, t_exp: u.Quantity, det: Detector_SWC, tel: Telescope_EUVST, sim: Simulation, n_iter: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    signals, fits = [], []
    for _ in tqdm(range(n_iter), desc="Monte-Carlo", unit="iter", leave=False):
        signals.append(simulate_once(I_cube, t_exp, det, tel, sim))
        fits.append(fit_cube_gauss(signals[-1][-1]))
    return np.array(signals), np.array(fits)


# -----------------------------------------------------------------------------
# Analysis metrics
# -----------------------------------------------------------------------------

def velocity_from_fit(
    fit_arr: u.Quantity | np.ndarray,
    wl0: u.Quantity,
    n_jobs: int = -1,
) -> u.Quantity:
    """
    Convert fitted line centres to LOS velocity.
    Works with either a Quantity array or an object-dtype array whose
    elements are Quantities. Uses joblib.Parallel for speed.
    """
    centres_raw = fit_arr[..., 1]  # (n_scan, n_slit)
    # Ensure we have a pure Quantity array
    if isinstance(centres_raw, u.Quantity):
        centres = centres_raw.to(wl0.unit)
    else:  # object array of Quantity scalars
        get_val = np.vectorize(lambda q: q.to_value(wl0.unit))
        centres = u.Quantity(get_val(centres_raw), wl0.unit)

    n_scan = centres.shape[0]

    def _one_row(i):
        return ((centres[i] - wl0) / wl0 * const.c).to(u.cm / u.s).value

    with tqdm_joblib(tqdm(total=n_scan, desc="Velocity calc", leave=False)):
        v_val = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_one_row)(i) for i in range(n_scan)
            )
        )

    v = v_val * (u.cm / u.s)
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
# Main workflow
# -----------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="YAML config file", required=True)
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set up instrument, detector, telescope, simulation from config
    instrument = config.get("instrument", "SWC").upper()
    psf = config.get("psf", False)
    exposures = config.get("expos", [1])
    n_iter = config.get("n_iter", 25)
    ncpu = config.get("ncpu", -1)
    slit_width = config.get("slit_width", '0.2 arcsec')
    oxide_thickness = config.get("oxide_thickness", '0 nm')
    c_thickness = config.get("c_thickness", '0 nm')
    vis_sl = config.get("vis_sl", '0 photon / (s * pixel)')

    slit_width = parse_yaml_input(slit_width)
    oxide_thickness = parse_yaml_input(oxide_thickness)
    c_thickness = parse_yaml_input(c_thickness)
    vis_sl = parse_yaml_input(vis_sl)

    if instrument == "SWC":
        filter_obj = AluminiumFilter(
            oxide_thickness=oxide_thickness,
            c_thickness=c_thickness,
        )
        DET = Detector_SWC()
        TEL = Telescope_EUVST(filter=filter_obj)
    elif instrument == "EIS":
        DET = Detector_EIS()
        TEL = Telescope_EIS()
        if oxide_thickness.value != 0 or c_thickness.value != 0:
            raise ValueError("EIS does not support oxide or C thicknesses.")
    else:
        raise ValueError(f"Unknown instrument: {instrument}")
    
    if psf:
        if instrument != "SWC":
            raise ValueError("PSF loading is only supported for SWC/EUVST instrument.")
        if instrument == "SWC":
            raise NotImplementedError("PSF loading is not implemented yet, waiting for NAOJ measurements of mirror scattering.")

    SIM_kwargs = dict(
        expos=u.Quantity(exposures, u.s),
        n_iter=n_iter,
        slit_width=slit_width,
        ncpu=ncpu,
        instrument=instrument,
    )
    SIM_kwargs["vis_sl"] = vis_sl

    SIM = Simulation(**SIM_kwargs)

    # Load PSF (only for SWC/EUVST)
    if instrument == "SWC" and psf:
        print("Loading PSF files...")
        psf_focus = _load_psf_ascii(TEL.psf_focus_file, skip=21)
        psf_mesh = _load_psf_ascii(TEL.psf_mesh_file, skip=16)
        psf_focus = _resample_psf(psf_focus, TEL.psf_focus_res, DET.pix_size)
        psf_mesh = _resample_psf(psf_mesh, TEL.psf_mesh_res, DET.pix_size)
        TEL.psf = _combine_psfs(psf_focus, psf_mesh, size=5)

    # Load synthetic atmosphere cube
    print("Loading atmosphere...")
    cube_sim, plotting = load_atmosphere("./run/input/synthesised_spectra.pkl")

    print("Rebinning atmosphere cube to instrument resolution for each slit position...")
    cube_reb = rebin_atmosphere(cube_sim, DET, SIM)

    print("Fitting ground truth cube...")
    fit_truth = fit_cube_gauss(cube_reb)
    v_true = velocity_from_fit(fit_truth, cube_reb.meta['rest_wav'])

    # --- Efficient in-memory buffers for post-loop plotting -----------------
    results = {}

    # Loop over exposure time only (no contamination loop)
    first_signal_per_exp: dict[float, Tuple[u.Quantity, ...]] = {}
    first_fit_per_exp:    dict[float, u.Quantity] = {}
    analysis_per_exp:     dict[float, dict] = {}

    for t_exp in tqdm(SIM.expos, desc=f"Exposure time", unit="exposure"):
        signals, fits = monte_carlo(
            cube_reb, t_exp, DET, TEL, SIM, n_iter=SIM.n_iter
        )
        sec = t_exp.to_value(u.s)
        first_signal_per_exp[sec] = signals[0][-1]
        first_fit_per_exp[sec]    = fits[0]
        analysis_per_exp[sec]     = analyse(fits, v_true, cube_reb.meta['rest_wav'])
        del signals, fits

    # Save results for this run
    results = {
        "first_signal_per_exp": first_signal_per_exp,
        "first_fit_per_exp": first_fit_per_exp,
        "analysis_per_exp": analysis_per_exp,
    }

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    config_path = Path(args.config)
    config_base = config_path.stem
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    output_file = scratch_dir / f"instrument_response_{config_base}_{timestamp}.pkl"
    with open(output_file, "wb") as f:
        dill.dump({
            "results": results,
            "config": config,
            "instrument": instrument,
            "DET": DET,
            "TEL": TEL,
            "SIM": SIM,
            "plotting": plotting,
            "cube_sim": cube_sim,
            "cube_reb": cube_reb
        }, f)
    print(f"Saved results to {output_file} ({os.path.getsize(output_file) / 1e6:.1f} MB)")

    dest_path = Path(f"run/result/instrument_response_{config_base}.pkl")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_file, dest_path)
    print(f"Copied {output_file} to {dest_path})")

    # Save dill session to scratch directory
    session_file = scratch_dir / f"instrument_response_session_{config_base}_{timestamp}.pkl"
    globals().update(locals())
    dill.dump_session(session_file)
    print(f"Saved session to {session_file} ({os.path.getsize(session_file) / 1e6:.1f} MB)")

if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="target cannot be converted to ICRS",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="No observer defined on WCS, SpectralCoord will be converted without any velocity frame change",
            category=UserWarning,
        )

        main()