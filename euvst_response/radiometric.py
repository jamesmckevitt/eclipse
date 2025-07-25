"""
Radiometric pipeline functions for converting intensities to detector signals.
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
import astropy.constants as const
from ndcube import NDCube
from scipy.signal import convolve2d
from .utils import wl_to_vel, vel_to_wl


def intensity_to_photons(I: NDCube) -> NDCube:
    """Convert intensity to photon flux."""
    wl_axis = I.axis_world_coords(2)[0]
    E_ph = (const.h * const.c / wl_axis).to("erg") * (1 / u.photon)
    
    photon_data = (I.data * I.unit / E_ph).to(u.photon / u.cm**2 / u.sr / u.cm)
    
    return NDCube(
        data=photon_data.value,
        wcs=I.wcs.deepcopy(),
        unit=photon_data.unit,
        meta=I.meta,
    )


def add_effective_area(ph_flux: NDCube, tel) -> NDCube:
    """Add telescope effective area to photon flux."""
    wl0 = ph_flux.meta['rest_wav']
    wl_axis = ph_flux.axis_world_coords(2)[0]
    A_eff = np.array([tel.ea_and_throughput(wl).cgs.value for wl in wl_axis]) * u.cm**2
    
    out_data = (ph_flux.data * ph_flux.unit * A_eff)
    
    return NDCube(
        data=out_data.value,
        wcs=ph_flux.wcs.deepcopy(),
        unit=out_data.unit,
        meta=ph_flux.meta,
    )


def photons_to_pixel_counts(ph_flux: NDCube, wl_pitch: u.Quantity, plate_scale: u.Quantity, slit_width: u.Quantity) -> NDCube:
    """Convert photon flux to pixel counts (total over exposure)."""
    pixel_solid_angle = ((plate_scale * u.pixel * slit_width).cgs / const.au.cgs ** 2) * u.sr
    
    out_data = (ph_flux.data * ph_flux.unit * pixel_solid_angle * wl_pitch)
    
    return NDCube(
        data=out_data.value,
        wcs=ph_flux.wcs.deepcopy(),
        unit=out_data.unit,
        meta=ph_flux.meta,
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


def to_electrons(photon_counts: NDCube, t_exp: u.Quantity, det) -> NDCube:
    """
    Convert a photon-count NDCube to an electron-count NDCube.

    Parameters
    ----------
    photon_counts : NDCube
        Cube of total photon counts per pixel (over exposure).
    t_exp : Quantity
        Exposure time (used for dark current and read noise).
    det : Detector_SWC or Detector_EIS
        Detector description.

    Returns
    -------
    NDCube
        Electron counts per pixel for the given exposure.
    """
    # Get rest wavelength from metadata (keep as Quantity with units)
    rest_wavelength = photon_counts.meta['rest_wav']  # Should be a Quantity
    
    # The photon_counts now already include the exposure time, so we just need QE
    photon_counts_with_qe = (photon_counts.data * photon_counts.unit * det.qe_euv).to_value(u.photon / u.pixel)
    
    # Apply proper Fano noise per pixel using the detector's method
    # Flatten arrays for processing
    flat_photons = photon_counts_with_qe.flatten()
    flat_electrons = np.zeros_like(flat_photons)
    
    # Apply Fano noise to each pixel using detector's method
    for i, n_photons in enumerate(flat_photons):
        flat_electrons[i] = det.fano_noise(n_photons, rest_wavelength)
    
    # Reshape back to original shape and convert to quantity
    electron_counts = flat_electrons.reshape(photon_counts_with_qe.shape)
    e = electron_counts * (u.electron / u.pixel)
    
    # Add dark current and read noise (these still depend on exposure time)
    e += det.dark_current * t_exp                                     # dark current
    e += np.random.normal(0, det.read_noise_rms.value,
                          photon_counts.data.shape) * (u.electron / u.pixel)  # read noise

    e = e.to(u.electron / u.pixel)
    e_val = e.value
    e_val[e_val < 0] = 0                                              # clip negatives

    return NDCube(
        data=e_val,
        wcs=photon_counts.wcs.deepcopy(),
        unit=e.unit,
        meta=photon_counts.meta,
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


def apply_exposure_and_poisson(I: NDCube, t_exp: u.Quantity) -> NDCube:
    """
    Apply exposure time to intensity and add Poisson noise.
    
    This converts intensity (per second) to total counts over the exposure
    and applies appropriate Poisson noise.

    Parameters
    ----------
    I : NDCube
        Input intensity cube (per second).
    t_exp : u.Quantity
        Exposure time.

    Returns
    -------
    NDCube
        New cube with exposure applied and Poisson noise added.
    """
    # Convert intensity rate to total intensity over exposure
    total_intensity = (I.data * I.unit * t_exp)
    
    # Apply Poisson noise
    noisy = np.random.poisson(total_intensity.value) * total_intensity.unit
    
    return NDCube(
        data=noisy.value,
        wcs=I.wcs.deepcopy(),
        unit=noisy.unit,
        meta=I.meta,
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

    # Assume visible stray light is ~600nm (typical visible wavelength)
    visible_wavelength = 600 * u.nm  # Keep as Quantity with units
    
    # Apply proper Fano noise to visible photons per pixel using detector's method
    flat_vis_photons = n_vis_ph.to_value(u.photon / u.pixel).flatten()
    flat_vis_electrons = np.zeros_like(flat_vis_photons)
    
    # Apply Fano noise to each pixel for visible light
    for i, n_photons in enumerate(flat_vis_photons):
        if n_photons > 0:
            # Apply quantum efficiency
            detected_photons = np.random.binomial(int(np.round(n_photons)), det.qe_vis)
            if detected_photons > 0:
                flat_vis_electrons[i] = det.fano_noise(detected_photons, visible_wavelength)
    
    # Reshape back and convert to quantity
    stray_electrons = flat_vis_electrons.reshape(n_vis_ph.shape) * (u.electron / u.pixel)

    # Add to original signal
    out_q = electrons.data * electrons.unit + stray_electrons
    out_q = out_q.to(electrons.unit)

    return NDCube(
        data=out_q.value,
        wcs=electrons.wcs.deepcopy(),
        unit=out_q.unit,
        meta=electrons.meta,
    )
