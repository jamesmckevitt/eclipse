"""
Radiometric pipeline functions for converting intensities to detector signals.
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
import astropy.constants as const
from ndcube import NDCube
from scipy.signal import convolve2d
from .utils import fano_noise


def intensity_to_photons(I: NDCube) -> NDCube:
    """Convert intensity to photon flux."""
    wl_axis = I.axis_world_coords(2)[0]
    E_ph = (const.h * const.c / wl_axis).to("erg") * (1 / u.photon)
    photon_data = (I.data * I.unit / E_ph).to(u.photon / u.s / u.cm**2 / u.sr / u.cm)
    
    return NDCube(
        data=photon_data.value,
        wcs=I.wcs.deepcopy(),
        unit=photon_data.unit,
        meta=I.meta,
    )


def add_effective_area(ph_cm2_sr_cm_s: NDCube, tel) -> NDCube:
    """Add telescope effective area to photon flux."""
    wl0 = ph_cm2_sr_cm_s.meta['rest_wav']
    wl_axis = ph_cm2_sr_cm_s.axis_world_coords(2)[0]
    A_eff = np.array([tel.ea_and_throughput(wl).cgs.value for wl in wl_axis]) * u.cm**2
    
    out_data = (ph_cm2_sr_cm_s.data * ph_cm2_sr_cm_s.unit * A_eff)
    
    return NDCube(
        data=out_data.value,
        wcs=ph_cm2_sr_cm_s.wcs.deepcopy(),
        unit=out_data.unit,
        meta=ph_cm2_sr_cm_s.meta,
    )


def photons_to_pixel_rate(ph_sr_cm_s: NDCube, wl_pitch: u.Quantity, plate_scale: u.Quantity, slit_width: u.Quantity) -> NDCube:
    """Convert photon flux to pixel count rate."""
    pixel_solid_angle = ((plate_scale * u.pixel * slit_width).cgs / const.au.cgs ** 2) * u.sr
    
    out_data = (ph_sr_cm_s.data * ph_sr_cm_s.unit * pixel_solid_angle * wl_pitch)
    
    return NDCube(
        data=out_data.value,
        wcs=ph_sr_cm_s.wcs.deepcopy(),
        unit=out_data.unit,
        meta=ph_sr_cm_s.meta,
    )


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


def to_electrons(photon_rate: NDCube, t_exp: u.Quantity, det) -> NDCube:
    """
    Convert a photon-rate NDCube to an electron-count NDCube.

    Parameters
    ----------
    photon_rate : NDCube
        Cube of photon s⁻¹ pixel⁻¹.
    t_exp : Quantity
        Exposure time.
    det : Detector_SWC or Detector_EIS
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


def to_dn(electrons: NDCube, det) -> NDCube:
    """
    Convert an electron-count NDCube to DN and clip at the detector's full-well.

    Parameters
    ----------
    electrons : NDCube
        Electron counts per pixel (u.electron / u.pixel).
    det : Detector_SWC or Detector_EIS
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


def add_stray_light(electrons: NDCube, t_exp: u.Quantity, det, sim) -> NDCube:
    """
    Add visible-light stray-light to a cube of electron counts.

    Parameters
    ----------
    electrons : NDCube
        Electron counts per pixel (unit: u.electron / u.pixel).
    t_exp : astropy.units.Quantity
        Exposure time.
    det : Detector_SWC or Detector_EIS
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
