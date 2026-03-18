"""
Radiometric pipeline functions for converting intensities to detector signals.

Pipeline order
--------------
The correct ordering of operations is critical for physical correctness.
The pipeline proceeds as:

1. :func:`apply_exposure` — multiply intensity by exposure time (deterministic).
2. :func:`intensity_to_photons` — convert erg intensity to photon flux.
3. :func:`add_telescope_throughput` — multiply by collecting area × optical
   efficiencies (mirror, grating, filter, microroughness).
4. :func:`photons_to_pixel_counts` — project onto the detector pixel solid
   angle and wavelength bin, producing **expected photon counts per pixel**.
5. :func:`apply_focusing_optics_psf` — convolve with the optical PSF (optional).
6. :func:`add_poisson` — **Poisson shot noise**, sampled here because the
   values are now actual photon counts per pixel.
7. :func:`to_electrons` — quantum efficiency (binomial), Fano noise, dark
   current (Poisson, per pixel), and read noise (Gaussian).
8. :func:`add_visible_stray_light` / :func:`add_pinhole_visible_light` —
   stray light added in electron space.
9. :func:`to_dn` — gain conversion to digital numbers, with saturation clip.

.. important::
   Poisson noise must be applied to **photon counts per pixel** (step 6),
   not to the erg-valued intensity (step 1).  The previous implementation
   applied ``np.random.poisson`` to the erg numerical values (~10¹⁴),
   which produced effectively zero noise (relative σ ~ 10⁻⁸) instead of
   the correct ~1/√N ~ 5–6 % for a typical ~300 photon/pixel signal.
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
import astropy.constants as const
from ndcube import NDCube
from scipy.signal import convolve2d
from .utils import wl_to_vel, vel_to_wl, debug_break


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
    
    # Get CCD temperature from the detector dataclass
    if not hasattr(det, 'ccd_temperature'):
        raise ValueError("CCD temperature not set. Pass ccd_temperature when constructing the Detector instance.")

    # Convert to Kelvin for the calculation
    temp_kelvin = det.ccd_temperature.to(u.K, equivalencies=u.temperature()).value
    
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


def _fwhm_to_sigma(fwhm: float) -> float:
    """Convert FWHM to Gaussian sigma: sigma = FWHM / (2 * sqrt(2 * ln2))."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def apply_focusing_optics_psf(signal: NDCube, tel) -> NDCube:
    """
    Convolve each detector frame (n_slit, n_lambda) of an NDCube with an
    anisotropic 2-D PSF from the focusing optics.

    The PSF is specified separately in the spatial (slit) and spectral
    (wavelength) directions via ``tel.psf_params``.

    Parameters
    ----------
    signal : NDCube
        Input cube with shape (n_scan, n_slit, n_lambda).
        The first axis is stepped by the raster scan.
    tel : Telescope_EUVST or Telescope_EIS
        Telescope configuration containing PSF parameters.
        psf_params = [spatial_fwhm, spectral_fwhm] in pixel units.

    Returns
    -------
    NDCube
        New cube with identical WCS / unit / meta but PSF-blurred data.
    """
    data_in = signal.data
    unit = signal.unit
    n_scan, n_slit, n_lambda = data_in.shape

    psf_type = tel.psf_type.lower()
    psf_params = tel.psf_params

    if len(psf_params) < 2:
        raise ValueError(
            "psf_params must contain two elements: "
            "[spatial_fwhm, spectral_fwhm] in pixel units."
        )

    # FWHM in pixels for each axis
    spatial_fwhm = psf_params[0].to(u.pixel).value   # along slit (axis 0 of each frame)
    spectral_fwhm = psf_params[1].to(u.pixel).value  # along lambda (axis 1 of each frame)

    # Convert FWHM -> sigma
    sigma_spatial = _fwhm_to_sigma(spatial_fwhm)
    sigma_spectral = _fwhm_to_sigma(spectral_fwhm)

    # Kernel size: 6*sigma rounded up to next odd integer, minimum 7
    ky = max(7, int(np.ceil(6 * sigma_spatial)))
    kx = max(7, int(np.ceil(6 * sigma_spectral)))
    if ky % 2 == 0:
        ky += 1
    if kx % 2 == 0:
        kx += 1

    # Coordinate grids centred at zero
    cy, cx = ky // 2, kx // 2
    y, x = np.mgrid[:ky, :kx]
    y = (y - cy).astype(float)
    x = (x - cx).astype(float)

    # Build PSF
    if psf_type == "gaussian":
        psf = np.exp(-0.5 * ((y / sigma_spatial) ** 2
                             + (x / sigma_spectral) ** 2))
    else:
        raise ValueError(
            f"Unsupported PSF type: {psf_type}. Supported: 'gaussian'."
        )

    # Normalise
    psf /= psf.sum()

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

    # Add dark current with Poisson noise (independent per pixel).
    # Each pixel gets its own Poisson realisation so that the noise is
    # spatially uncorrelated, as it is on a real CCD.  A previous version
    # drew a single scalar and broadcast it identically to every pixel,
    # which would bias spectral fits by correlating the noise across the
    # line profile (the noise wouldn't average down with more pixels).
    dark_current_mean = (det.dark_current * t_exp).to(u.electron / u.pixel).value
    dark_current_poisson = np.random.poisson(dark_current_mean, size=photon_counts.data.shape) * (u.electron / u.pixel)
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

    dn_val = np.round(dn_q.value)                                         # round to nearest whole number
    dn_val[dn_val > det.max_dn.value] = det.max_dn.value                  # clip

    return NDCube(
        data=dn_val,
        wcs=electrons.wcs.deepcopy(),
        unit=dn_q.unit,
        meta=electrons.meta,
    )


def add_poisson(cube: NDCube) -> NDCube:
    r"""
    Apply Poisson photon shot noise to photon counts per pixel.

    This function must be called **after** the signal has been converted
    to photon counts per detector pixel (i.e. after
    ``photons_to_pixel_counts`` and any PSF / pinhole effects).  Applying
    Poisson sampling earlier — e.g. to intensity values in erg units —
    would produce dramatically incorrect noise because ``np.random.poisson``
    treats its argument as the expected count, and erg-valued intensities
    are many orders of magnitude larger than the actual photon count
    (~1e14 vs ~1e2), making the relative shot noise negligibly small
    instead of the correct ~1/sqrt(N) level.

    Negative values (which can arise from floating-point arithmetic or
    PSF boundary effects) are clipped to zero before sampling.

    Parameters
    ----------
    cube : NDCube
        Input data cube whose values are expected photon counts per
        pixel (unit ``photon / pix``).

    Returns
    -------
    NDCube
        New cube containing Poisson-sampled integer photon counts.

    See Also
    --------
    apply_exposure : Multiplies intensity by exposure time (no noise).
    """
    data_clipped = np.maximum(cube.data, 0)
    noisy = np.random.poisson(data_clipped) * cube.unit
    return NDCube(
        data=noisy.value,
        wcs=cube.wcs.deepcopy(),
        unit=noisy.unit,
        meta=cube.meta,
    )


def apply_exposure(I: NDCube, t_exp: u.Quantity) -> NDCube:
    r"""
    Multiply a per-second intensity cube by the exposure time.

    Poisson (photon shot) noise is **not** applied here.  It is applied
    later in the pipeline by :func:`add_poisson`, after the intensity has
    been converted to photon counts per detector pixel — the only stage
    where the numerical values represent actual expected photon counts
    and Poisson sampling is physically meaningful.

    .. note::
       A previous version of this function (``apply_exposure_and_poisson``)
       applied ``np.random.poisson`` directly to the erg-valued intensity
       numbers (~1e14).  Because Poisson noise scales as sqrt(lambda), the
       relative noise was ~1e-8 — effectively zero — rather than the
       correct ~1/sqrt(N) = 5-6 % for the ~300 photon/pixel signal.  The
       corrected pipeline now produces realistic shot noise.

    Parameters
    ----------
    I : NDCube
        Input intensity cube (per second).
    t_exp : u.Quantity
        Exposure time.

    Returns
    -------
    NDCube
        New cube with exposure applied (deterministic, no stochastic noise).
    """
    total_intensity = (I.data * I.unit * t_exp)

    return NDCube(
        data=total_intensity.value,
        wcs=I.wcs.deepcopy(),
        unit=total_intensity.unit,
        meta=I.meta,
    )


# Keep old name as an alias for backward compatibility
apply_exposure_and_poisson = apply_exposure


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
