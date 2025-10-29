"""
Radiometric pipeline functions for converting intensities to detector signals.
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
import astropy.constants as const
from ndcube import NDCube
from scipy.signal import convolve2d
from .utils import wl_to_vel, vel_to_wl, debug_break
from scipy.special import voigt_profile


def _vectorized_fano_noise(photon_counts: np.ndarray, rest_wavelength: u.Quantity, det) -> np.ndarray:
    """
    Vectorized version of Fano noise calculation for improved performance.
    
    Parameters
    ----------
    photon_counts : np.ndarray
        Array of photon counts (unitless values)
    rest_wavelength : u.Quantity
        Rest wavelength with units
    det : Detector_SWC or Detector_EIS
        Detector object with fano noise parameters
        
    Returns
    -------
    np.ndarray
        Array of electron counts with Fano noise applied
    """
    # Handle zero or negative photon counts
    mask_positive = photon_counts > 0
    electron_counts = np.zeros_like(photon_counts)
    
    if not np.any(mask_positive):
        return electron_counts
    
    # Get CCD temperature - must be set via with_temperature()
    if not hasattr(det, '_ccd_temperature'):
        raise ValueError("CCD temperature not set. Use Detector_SWC.with_temperature() to create detector instance.")
    
    # Convert to Kelvin for the calculation
    temp_kelvin = det._ccd_temperature.to(u.K, equivalencies=u.temperature()).value
    
    # Convert wavelength to photon energy: E = hc/lambda
    photon_energy_ev = (const.h * const.c / (rest_wavelength.to(u.angstrom))).to(u.eV).value
    
    # Calculate temperature-dependent energy per electron-hole pair
    w_T = 3.71 - 0.0006 * (temp_kelvin - 300.0)  # eV per electron-hole pair
    
    # Mean number of electrons per photon
    mean_electrons_per_photon = photon_energy_ev / w_T
    
    # Fano noise variance per photon
    sigma_fano_per_photon = np.sqrt(det.si_fano * mean_electrons_per_photon)
    
    # Work only with positive photon counts
    positive_photons = photon_counts[mask_positive]
    
    # For efficiency, use a simpler approximation for most cases
    # The exact method is: for each photon, sample from Normal(mean_e, sigma_fano)
    # Approximation: for N photons, sample from Normal(N*mean_e, sqrt(N)*sigma_fano)
    # This is mathematically equivalent for large N and much faster
    
    mean_total_electrons = positive_photons * mean_electrons_per_photon
    std_total_electrons = np.sqrt(positive_photons) * sigma_fano_per_photon
    
    # Sample total electrons per pixel
    total_electrons = np.random.normal(
        loc=mean_total_electrons,
        scale=std_total_electrons
    )
    
    # Ensure non-negative
    total_electrons = np.maximum(total_electrons, 0)
    
    # Map back to full array
    electron_counts[mask_positive] = total_electrons
    
    return electron_counts


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


def add_telescope_throughput(ph_flux: NDCube, tel) -> NDCube:
    """Add telescope optical throughput (collecting area x optical efficiencies) to photon flux."""
    wl0 = ph_flux.meta['rest_wav']
    wl_axis = ph_flux.axis_world_coords(2)[0]
    throughput = np.array([tel.ea_and_throughput(wl).cgs.value for wl in wl_axis]) * u.cm**2
    
    out_data = (ph_flux.data * ph_flux.unit * throughput)
    
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


def apply_focusing_optics_psf(signal: NDCube, tel) -> NDCube:
    """
    Convolve each detector row (first axis) of an NDCube with a parameterized PSF
    from the focusing optics (primary mirror and diffraction grating).

    Parameters
    ----------
    signal : NDCube
        Input cube with shape (n_scan, n_slit, n_lambda).
        The first axis is stepped by the raster scan.
    tel : Telescope_EUVST or Telescope_EIS
        Telescope configuration containing PSF parameters.
        For Gaussian: psf_params = [width]
        For Voigt: psf_params = [width, gamma]

    Returns
    -------
    NDCube
        New cube with identical WCS / unit / meta but PSF-blurred data.
    """
    
    # Extract data and units
    data_in = signal.data  # ndarray view (no units)
    unit = signal.unit
    n_scan, n_slit, n_lambda = data_in.shape

    # Get PSF parameters from telescope
    psf_type = tel.psf_type.lower()
    psf_params = tel.psf_params
    
    # Extract width parameter (first parameter for both Gaussian and Voigt)
    width_pixels = psf_params[0].to(u.pixel).value
    
    # Create 2D PSF kernel
    # Make kernel size based on width (use 6*width to capture most of the profile)
    kernel_size = max(7, int(6 * width_pixels))
    if kernel_size % 2 == 0:  # Ensure odd size for symmetric kernel
        kernel_size += 1
    
    # Create coordinate grids centered at 0
    center = kernel_size // 2
    y, x = np.mgrid[:kernel_size, :kernel_size]
    y = y - center
    x = x - center
    
    # Create radial distance from center
    r = np.sqrt(x**2 + y**2)
    
    # Create PSF based on type
    if psf_type == "gaussian":
        sigma = width_pixels
        psf = np.exp(-0.5 * (r / sigma)**2)
        
    elif psf_type == "voigt":
        # For Voigt: need both width and gamma parameters
        if len(psf_params) < 2:
            raise ValueError("Voigt PSF requires two parameters: [width, gamma]")

        sigma_gauss = width_pixels
        # Get gamma parameter for Lorentzian component
        gamma_lorentz = psf_params[1].to(u.pixel).value
        
        # Create 2D Voigt PSF (approximate as radially symmetric)
        psf = voigt_profile(r, sigma_gauss, gamma_lorentz)
        
    else:
        raise ValueError(f"Unsupported PSF type: {psf_type}. Supported types: 'gaussian', 'voigt'")
    
    # Normalize PSF
    psf = psf / np.sum(psf)

    # Convolve each scan position
    blurred = np.empty_like(data_in)
    for i in range(n_scan):
        blurred[i] = convolve2d(data_in[i], psf, mode="same")

    return NDCube(
        data=blurred,
        wcs=signal.wcs.deepcopy(),
        unit=unit,
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

    # Apply quantum efficiency first using binomial distribution (proper physics)
    photons_detected = np.random.binomial(
        photon_counts.to(u.photon/u.pix).data.astype(int),  # Extract unitless data
        det.qe_euv
    )

    # Apply proper Fano noise per pixel using a vectorized approach
    electron_counts = _vectorized_fano_noise(photons_detected.astype(float), rest_wavelength, det)

    e = electron_counts * (u.electron / u.pixel)

    # Add dark current with Poisson noise
    dark_current_mean = (det.dark_current * t_exp).to(u.electron / u.pixel).value
    dark_current_poisson = np.random.poisson(dark_current_mean) * (u.electron / u.pixel)
    e += dark_current_poisson
    
    # Add read noise
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


def add_visible_stray_light(electrons: NDCube, t_exp: u.Quantity, det, sim, tel=None) -> NDCube:
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
        Simulation parameters (contains vis_sl - photon/s/cm2).
    tel : Telescope_EUVST or Telescope_EIS, optional
        Telescope configuration for filter throughput calculation.

    Returns
    -------
    NDCube
        New cube with stray-light signal added.
    """
    # Convert vis_sl from photon/s/cm2 to photon/s/pixel using detector pixel area
    pixel_area = ((det.pix_size*1*u.pix)**2)/u.pix  # cm/pix -> cm2/pixel
    vis_sl_per_pixel = (sim.vis_sl * pixel_area).to(u.photon / (u.s * u.pixel))
    
    # Apply filter throughput if telescope with filter is available
    if tel is not None and hasattr(tel, 'filter'):
        filter_throughput = tel.filter.visible_light_throughput()
        vis_sl_per_pixel *= filter_throughput
    
    # Draw Poisson realisation of stray-light photons
    n_vis_ph = np.random.poisson(
        (vis_sl_per_pixel * t_exp).to_value(u.photon / u.pixel),
        size=electrons.data.shape
    ) * (u.photon / u.pixel)

    # Assume visible stray light is ~600nm (typical visible wavelength)
    visible_wavelength = 600 * u.nm  # Keep as Quantity with units
    
    # Apply quantum efficiency first, then vectorized Fano noise
    vis_photons_detected = np.random.binomial(
        n_vis_ph.to_value(u.photon / u.pixel).astype(int),
        det.qe_vis
    )
    
    # Apply vectorized Fano noise to detected visible photons
    stray_electrons_values = _vectorized_fano_noise(vis_photons_detected.astype(float), visible_wavelength, det)
    stray_electrons = stray_electrons_values * (u.electron / u.pixel)

    # Add to original signal
    out_q = electrons.data * electrons.unit + stray_electrons
    out_q = out_q.to(electrons.unit)

    return NDCube(
        data=out_q.value,
        wcs=electrons.wcs.deepcopy(),
        unit=out_q.unit,
        meta=electrons.meta,
    )


def add_pinhole_visible_light(electrons: NDCube, t_exp: u.Quantity, det, sim, tel) -> NDCube:
    """
    Add visible light contributions from pinholes to electron counts.
    
    This function adds the visible light that bypasses the aluminum filter
    through pinholes and creates diffraction patterns on the detector.

    Parameters
    ----------
    electrons : NDCube
        Electron counts per pixel (unit: u.electron / u.pixel).
    t_exp : u.Quantity
        Exposure time.
    det : Detector_SWC
        Detector configuration (must be SWC for pinhole support).
    sim : Simulation
        Simulation parameters containing pinhole configuration.
    tel : Telescope_EUVST
        Telescope configuration with aluminum filter.

    Returns
    -------
    NDCube
        New cube with pinhole visible light contributions added.
    """
    if not (sim.enable_pinholes and len(sim.pinhole_sizes) > 0):
        return electrons  # No pinholes enabled
    
    # Import here to avoid circular imports
    from .pinhole_diffraction import calculate_pinhole_diffraction_pattern
    
    # Get detector and data properties
    data_shape = electrons.data.shape  # Should be (n_scan, n_slit, n_spectral)
    
    # Visible light wavelength (typical)
    visible_wavelength = 600 * u.nm
    
    # Initialize additional electron contributions
    additional_electrons = np.zeros_like(electrons.data)

    for pinhole_diameter, pinhole_position in zip(sim.pinhole_sizes, sim.pinhole_positions):
        # Calculate pinhole area
        pinhole_area = np.pi * (pinhole_diameter / 2)**2
        
        # === Visible Light Contribution Through Pinhole ===
        # Calculate total photons incident on the pinhole area (unfiltered)
        # sim.vis_sl is photon/s/cm^2, pinhole_area is in cm^2
        vis_photons_per_sec_through_pinhole = sim.vis_sl * pinhole_area
        vis_photons_total_through_pinhole = (vis_photons_per_sec_through_pinhole * t_exp).to(u.photon)
        
        # Calculate visible diffraction pattern - this shows how the pinhole photons spread
        n_scan, n_slit, n_spectral = data_shape
        vis_pattern = calculate_pinhole_diffraction_pattern(
            detector_shape=(n_slit, n_spectral),
            pixel_size=det.pix_size*u.pix,
            pinhole_diameter=pinhole_diameter,
            pinhole_position_slit=pinhole_position,
            slit_width=sim.slit_width,
            plate_scale=det.plate_scale_angle,
            distance=det.filter_distance,
            wavelength=visible_wavelength
        )
        
        # Distribute the total pinhole photons according to diffraction pattern
        # vis_pattern is normalized (peak = 1), so we need to ensure photon conservation
        # Normalize the pattern so the total integrated intensity equals 1.0
        pattern_total = np.sum(vis_pattern)
        if pattern_total > 0:
            vis_pattern_normalized = vis_pattern / pattern_total
        else:
            vis_pattern_normalized = vis_pattern

        vis_photons_distributed = vis_photons_total_through_pinhole.to(u.photon).value * vis_pattern_normalized
        
        # Sample Poisson photons for this pinhole contribution
        vis_photons_poisson = np.random.poisson(vis_photons_distributed)
        
        # Apply quantum efficiency
        vis_photons_detected = np.random.binomial(
            vis_photons_poisson.astype(int),
            det.qe_vis
        )

        # Apply Fano noise to detected visible photons
        vis_electrons_values = _vectorized_fano_noise(vis_photons_detected.astype(float), visible_wavelength, det)

        # Add to all scan positions (visible light affects all equally)
        for scan_idx in range(n_scan):
            additional_electrons[scan_idx] += vis_electrons_values

    # Add pinhole contributions to original signal
    additional_electrons_quantity = additional_electrons * (u.electron / u.pixel)
    out_q = electrons.data * electrons.unit + additional_electrons_quantity
    out_q = out_q.to(electrons.unit)

    return NDCube(
        data=out_q.value,
        wcs=electrons.wcs.deepcopy(),
        unit=out_q.unit,
        meta=electrons.meta,
    )
