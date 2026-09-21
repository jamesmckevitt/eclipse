"""
Radiometric pipeline functions for converting intensities to detector signals.
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
import astropy.constants as const
from ndcube import NDCube
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d
from scipy.special import erf
from scipy.stats import poisson
from .utils import wl_to_vel, vel_to_wl, debug_break, _fwhm_to_sigma


def _poisson_inverse_transform(mean_counts, size=None) -> np.ndarray:
    """
    Draw Poisson counts by inverse-transform (quantile) sampling.

    One uniform draw per element is passed through the Poisson inverse-CDF.
    The marginal distribution is identical to ``np.random.poisson``, but this
    consumes a fixed number of RNG draws per element, where the rejection
    sampling in ``np.random.poisson`` consumes a variable number.  That keeps
    the random stream synchronised across runs whose only difference is the
    Poisson mean, which is what makes common-random-number variance reduction
    possible.

    Parameters
    ----------
    mean_counts : float or np.ndarray
        Poisson mean, either scalar or per element.
    size : tuple of int, optional
        Shape to draw.  Defaults to the shape of *mean_counts*.

    Returns
    -------
    np.ndarray
        Sampled counts as int64.
    """
    if size is None:
        size = np.shape(mean_counts)

    u_draw = np.random.random(size=size)
    # np.random.random() draws from [0, 1) and scipy's ppf returns -1 at
    # exactly 0, which would give a negative count.  Clamp to the smallest
    # positive double, which leaves the CRN property intact.
    np.maximum(u_draw, np.nextafter(0.0, 1.0), out=u_draw)

    return poisson.ppf(u_draw, mean_counts).astype(np.int64)


def _vectorized_fano_noise(photon_counts: np.ndarray, rest_wavelength: u.Quantity, det,
                           *, noise: bool = True) -> np.ndarray:
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
    noise : bool, optional
        When False, return the mean number of electrons each photon liberates
        rather than drawing around it.  The conversion gain is unchanged; only
        its spread is dropped.  Default True.

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
    
    # Fano noise standard deviation per photon
    sigma_fano_per_photon = np.sqrt(det.si_fano * mean_electrons_per_photon)
    
    # Work only with positive photon counts
    positive_photons = photon_counts[mask_positive]
    
    # For efficiency, use a simpler approximation for most cases
    # The exact method is: for each photon, sample from Normal(mean_e, sigma_fano)
    # Approximation: for N photons, sample from Normal(N*mean_e, sqrt(N)*sigma_fano)
    # This is mathematically equivalent for large N and much faster
    
    mean_total_electrons = positive_photons * mean_electrons_per_photon

    if noise:
        std_total_electrons = np.sqrt(positive_photons) * sigma_fano_per_photon

        # Sample total electrons per pixel
        total_electrons = np.random.normal(
            loc=mean_total_electrons,
            scale=std_total_electrons
        )

        # Ensure non-negative
        total_electrons = np.maximum(total_electrons, 0)
    else:
        total_electrons = mean_total_electrons
    
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
    
    out_data = (ph_flux.data * ph_flux.unit * pixel_solid_angle * wl_pitch.to(u.cm/u.pix))
    
    return NDCube(
        data=out_data.value,
        wcs=ph_flux.wcs.deepcopy(),
        unit=out_data.unit,
        meta=ph_flux.meta,
    )


def slit_image_width(slit_width: u.Quantity, det) -> float:
    """
    The width of the slit's image across the dispersion, in detector pixels.

    RSC-2022021C measures widths along the dispersion in the same arcsec as
    along the slit, converting both with the spatial scale (0.0118 arcsec per
    micron for SW), and adds the slit to the optics in those units. A slit
    *slit_width* wide therefore covers *slit_width* over the plate scale in
    spectral pixels, as it would along the slit.
    """
    return (slit_width / (det.plate_scale_angle * u.pixel)).to_value(u.dimensionless_unscaled)


def spectral_optics_fwhm(tel, det) -> float:
    """
    FWHM of the optics alone along the dispersion, in detector pixels.

    ``tel.psf_params[1]`` is the spectral FWHM with a slit
    ``tel.psf_slit_width`` wide, the optics FWHM and the slit width added in
    quadrature, so the optics part is what is left when the slit is taken
    out the same way. For the default SWC values that is 2.207 pixels, or
    0.351 arcsec, where RSC-2022021C gives 0.352 arcsec; the two agree to the
    rounding of 43.00 mA to 2.54 pixels and of the plate scale to 0.159
    arcsec per pixel.
    """
    reference = getattr(tel, "psf_slit_width", None)
    if reference is None:
        raise ValueError(
            f"{type(tel).__name__} has no psf_slit_width, so its spectral PSF "
            f"is not separated into the optics and the slit and cannot be "
            f"convolved with a slit. Give psf_slit_width, the slit width "
            f"psf_params was measured with, or use spectral_psf: quadrature.")
    measured = tel.psf_params[1].to_value(u.pixel)
    slit = slit_image_width(reference, det)
    optics_squared = measured**2 - slit**2
    if optics_squared <= 0:
        raise ValueError(
            f"The spectral FWHM of {measured} pixels is no wider than the "
            f"{slit:.3f}-pixel image of the {reference} slit it is quoted for, "
            f"so it leaves nothing for the optics.")
    return float(np.sqrt(optics_squared))


def spectral_psf_fwhm(tel, det, slit_width: u.Quantity) -> float:
    """
    FWHM of the spectral PSF with a slit *slit_width* wide, in detector pixels.

    The optics FWHM and the slit width add in quadrature, as RSC-2022021C
    adds them for the spectral resolution it quotes:
    ``FWHM(w)**2 = FWHM(w0)**2 + s(w)**2 - s(w0)**2``, where ``w0`` is
    ``tel.psf_slit_width``, ``FWHM(w0)`` is ``tel.psf_params[1]`` and ``s``
    is :func:`slit_image_width`. With the reference slit this is
    ``psf_params[1]`` exactly. For the default SWC values it gives 2.54,
    3.35, 5.49 and 10.30 pixels for the 0.2, 0.4, 0.8 and 1.6 arcsec slits.

    A telescope without ``psf_slit_width`` has one spectral FWHM whatever
    the slit.
    """
    measured = tel.psf_params[1].to_value(u.pixel)
    reference = getattr(tel, "psf_slit_width", None)
    if reference is None:
        return measured
    # Refuses a FWHM that leaves nothing for the optics.
    spectral_optics_fwhm(tel, det)
    # The difference of squares comes first so that it is exactly zero for
    # the reference slit, which then gets psf_params[1] to the last bit.
    difference = slit_image_width(slit_width, det)**2 - slit_image_width(reference, det)**2
    return float(np.sqrt(measured**2 + difference))


def spectral_line_spread(tel, det, slit_width: u.Quantity) -> np.ndarray:
    """
    The spectral PSF as the optics convolved with the slit, sampled at whole pixels.

    RSC-2022021C defines the line spread function as the PSF of the optics
    after the slit convolved with a rectangle the width of the slit. This
    is that convolution, of a Gaussian of :func:`spectral_optics_fwhm` with
    a rectangle :func:`slit_image_width` wide, evaluated at the centre of
    each pixel as the Gaussian kernel is, and normalised to sum to one. It
    reaches three optics sigma beyond the rectangle on each side, the reach
    the Gaussian kernel has, and is at least seven pixels long.

    A wide slit gives a flat-topped profile. For the 0.2 arcsec slit the
    profile is narrower than the Gaussian of :func:`spectral_psf_fwhm`,
    because a rectangle adds less to a width than a Gaussian of the same
    FWHM does.
    """
    sigma = _fwhm_to_sigma(spectral_optics_fwhm(tel, det))
    half = 0.5 * slit_image_width(slit_width, det)
    reach = int(np.ceil(half + 3 * sigma))
    n = max(7, 2 * reach + 1)
    x = np.arange(n) - n // 2
    scale = np.sqrt(2.0) * sigma
    kernel = 0.5 * (erf((x + half) / scale) - erf((x - half) / scale))
    return kernel / kernel.sum()


def apply_focusing_optics_psf(
    signal: NDCube,
    tel,
    det,
    sim,
    *,
    convolve_spatial: bool = True,
    boundary: str = "replicate",
) -> NDCube:
    """
    Convolve each detector frame (n_slit, n_lambda) of an NDCube with an
    anisotropic 2-D PSF from the focusing optics.

    The PSF is specified separately in the spatial (slit) and spectral
    (wavelength) directions via ``tel.psf_params``. Along the dispersion it
    also depends on the slit, whose image is part of the line profile: with
    ``sim.spectral_psf`` ``"quadrature"`` the PSF is a Gaussian of
    :func:`spectral_psf_fwhm`, and with ``"convolution"`` it is
    :func:`spectral_line_spread`.

    Parameters
    ----------
    signal : NDCube
        Input cube with shape (n_slit, n_scan, n_lambda).
        The middle axis is stepped by the raster scan.
    tel : Telescope_EUVST or Telescope_EIS
        Telescope configuration containing PSF parameters.
        psf_params = [spatial_fwhm, spectral_fwhm] in pixel units.
    det : Detector_SWC or Detector_EIS
        The detector, whose plate scale sets how many spectral pixels the
        slit's image covers.
    sim : Simulation
        The slit width and ``spectral_psf``, how the slit enters the PSF.
    convolve_spatial : bool, optional
        When False, convolve the spectral axis only and leave the slit axis
        alone.  This is for a field that is uniform along the slit, where
        convolving a constant with a normalised kernel returns the same
        constant and the spatial pass is an identity operation.  Doing it
        anyway would not be harmless: the convolution treats everything
        outside the array as dark, so a uniform field would lose flux off the
        ends of the slit that it really does have.  Default True.
    boundary : {"replicate", "zero"}, optional
        What lies beyond the ends of the slit.  ``"replicate"`` continues the
        edge rows outward; ``"zero"`` treats everything outside the field as
        dark.  Zero fill is wrong for a raster, because the Sun carries on
        past the field of view and the rows just inside the edge really do
        receive PSF contributions from it.  With the default SWC spatial PSF
        the kernel is seven rows wide and zero fill costs the edge row about
        a third of the kernel weight, the next row 8 per cent, and the one
        after 1 per cent.  Default ``"replicate"``.

    Returns
    -------
    NDCube
        New cube with identical WCS / unit / meta but PSF-blurred data.
    """
    if boundary not in ("replicate", "zero"):
        raise ValueError(
            f"boundary must be 'replicate' or 'zero', got {boundary!r}."
        )
    data_in = signal.data
    unit = signal.unit
    n_slit, n_scan, n_lambda = data_in.shape

    psf_type = tel.psf_type.lower()
    psf_params = tel.psf_params

    if psf_type != "gaussian":
        raise ValueError(
            f"Unsupported PSF type: {psf_type}. Supported: 'gaussian'."
        )

    if len(psf_params) < 2:
        raise ValueError(
            "psf_params must contain two elements: "
            "[spatial_fwhm, spectral_fwhm] in pixel units."
        )

    if sim.spectral_psf not in ("quadrature", "convolution"):
        raise ValueError(
            f"spectral_psf must be 'quadrature' or 'convolution', got "
            f"{sim.spectral_psf!r}."
        )
    quadrature = sim.spectral_psf == "quadrature"

    # FWHM in pixels along the slit (axis 0 of each frame)
    spatial_fwhm = psf_params[0].to(u.pixel).value
    sigma_spatial = _fwhm_to_sigma(spatial_fwhm)

    # Kernel size: 6*sigma rounded up to next odd integer, minimum 7
    ky = max(7, int(np.ceil(6 * sigma_spatial)))
    if ky % 2 == 0:
        ky += 1

    # Along lambda (axis 1 of each frame) the profile depends on the slit.
    if quadrature:
        sigma_spectral = _fwhm_to_sigma(spectral_psf_fwhm(tel, det, sim.slit_width))
        kx = max(7, int(np.ceil(6 * sigma_spectral)))
        if kx % 2 == 0:
            kx += 1
        x_1d = np.arange(kx) - kx // 2
        spectral_1d = np.exp(-0.5 * (x_1d / sigma_spectral) ** 2)
    else:
        spectral_1d = spectral_line_spread(tel, det, sim.slit_width)
        kx = spectral_1d.size

    if not convolve_spatial:
        # Spectral axis only.  Zero padding is correct here: the wavelength
        # grid extends several sigma past the line, so there is no flux at its
        # edges to lose.
        psf_1d = spectral_1d / spectral_1d.sum()
        return NDCube(
            data=convolve1d(data_in, psf_1d, axis=2, mode="constant", cval=0.0),
            wcs=signal.wcs.deepcopy(),
            unit=unit,
            meta=signal.meta,
        )

    if quadrature:
        # Coordinate grids centred at zero
        cy, cx = ky // 2, kx // 2
        y, x = np.mgrid[:ky, :kx]
        y = (y - cy).astype(float)
        x = (x - cx).astype(float)

        # Build PSF
        psf = np.exp(-0.5 * ((y / sigma_spatial) ** 2
                             + (x / sigma_spectral) ** 2))
    else:
        # The optics blur along the slit does not depend on the slit, so the
        # PSF is the spatial Gaussian times the spectral profile.
        y_1d = np.arange(ky) - ky // 2
        psf = np.outer(np.exp(-0.5 * (y_1d / sigma_spatial) ** 2), spectral_1d)

    # Normalise
    psf /= psf.sum()

    # Convolve each scan position's detector frame (n_slit, n_lambda). Padding
    # the slit axis by half a kernel is exactly enough for every real row to
    # see replicated rows rather than the zeros convolve2d assumes outside the
    # array; the spectral axis is left to zero-fill, which is correct there.
    pad = (ky // 2) if boundary == "replicate" else 0
    blurred = np.empty_like(data_in)
    for i in range(n_scan):
        frame = data_in[:, i, :]
        if pad:
            frame = np.pad(frame, ((pad, pad), (0, 0)), mode="edge")
        convolved = convolve2d(frame, psf, mode="same")
        blurred[:, i, :] = convolved[pad:pad + n_slit, :] if pad else convolved

    return NDCube(
        data=blurred,
        wcs=signal.wcs.deepcopy(),
        unit=unit,
        meta=signal.meta,
    )


def to_electrons(
    photon_counts: NDCube,
    t_exp: u.Quantity,
    det,
    *,
    dark_current_inverse_transform: bool = False,
    noise: bool = True,
) -> NDCube:
    """
    Convert a photon-count NDCube to an electron-count NDCube.

    Parameters
    ----------
    photon_counts : NDCube
        Photon counts per pixel, non-negative.  With *noise* True these must
        be whole numbers, as from :func:`sample_photon_arrivals`; with it
        False they may be the fractional expected counts.
    t_exp : Quantity
        Exposure time (used for dark current and read noise).
    det : Detector_SWC or Detector_EIS
        Detector description.
    dark_current_inverse_transform : bool, optional
        When True, draw dark-current shot noise with
        :func:`_poisson_inverse_transform` rather than ``np.random.poisson``.
        The distribution is unchanged, but the random stream stays synchronised
        across runs that differ only in dark-current level, which is what
        common-random-number variance reduction needs.  Default False.
    noise : bool, optional
        When False, every random draw here is replaced by its mean: quantum
        efficiency becomes a straight multiplication, the Fano spread and the
        read noise are dropped, and the dark current contributes its expected
        number of electrons.  Default True.

    Returns
    -------
    NDCube
        Electron counts per pixel for the given exposure.
    """
    # Get rest wavelength from metadata (keep as Quantity with units)
    rest_wavelength = photon_counts.meta['rest_wav']  # Should be a Quantity

    # Apply quantum efficiency.  With noise on this is a binomial draw over
    # whole photons; with it off the same expectation, qe * N, without the
    # cast to integers that a draw would need.
    incident = photon_counts.to(u.photon / u.pix).data
    if noise:
        photons_detected = np.random.binomial(incident.astype(int), det.qe_euv)
    else:
        photons_detected = incident * det.qe_euv

    # Apply proper Fano noise per pixel using a vectorized approach
    electron_counts = _vectorized_fano_noise(photons_detected.astype(float),
                                             rest_wavelength, det, noise=noise)

    e = electron_counts * (u.electron / u.pixel)

    # Add dark current with Poisson shot noise (per pixel)
    dark_current_mean = (det.dark_current * t_exp).to(u.electron / u.pixel).value
    if not noise:
        dark_current_counts = np.full(photon_counts.data.shape, dark_current_mean)
    elif dark_current_inverse_transform:
        dark_current_counts = _poisson_inverse_transform(
            dark_current_mean, size=photon_counts.data.shape
        )
    else:
        dark_current_counts = np.random.poisson(dark_current_mean, size=photon_counts.data.shape)
    dark_current_signal = dark_current_counts * (u.electron / u.pixel)
    e += dark_current_signal

    # Add read noise.  It is zero-mean, so with noise off there is nothing to
    # add rather than something to average.
    if noise:
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


def sample_photon_arrivals(
    photon_counts: NDCube,
    *,
    photon_shot_inverse_transform: bool = False,
    noise: bool = True,
) -> NDCube:
    """
    Sample a discrete Poisson realisation of photon arrivals per pixel.

    This represents the fundamental quantum nature of light: even for a
    perfectly stable source, the number of photons arriving at any pixel
    in a finite time is a Poisson-distributed integer.  This step is
    purely physical (a property of the photon field) and is independent
    of the detector technology.

    Parameters
    ----------
    photon_counts : NDCube
        Expected (mean) photon counts per pixel.
    photon_shot_inverse_transform : bool, optional
        When True, draw photon shot noise with
        :func:`_poisson_inverse_transform` rather than ``np.random.poisson``.
        The distribution is unchanged, but the random stream stays synchronised
        across runs that differ only in photon flux, which is what
        common-random-number variance reduction needs.  Default False.
    noise : bool, optional
        When False, return the expected counts rather than a draw around them.
        The result is left as floating point: rounding it to whole photons
        would put quantisation back in where the point was to remove the
        randomness.  Default True.

    Returns
    -------
    NDCube
        Poisson-sampled integer photon counts per pixel, or the expected
        counts as floats when *noise* is False.
    """
    q = photon_counts.data * photon_counts.unit

    canonical_units = u.photon / u.pix

    mean_counts = q.to(canonical_units).value
    mean_counts = np.maximum(mean_counts, 0)

    if not noise:
        return NDCube(
            data=mean_counts,
            wcs=photon_counts.wcs.deepcopy(),
            unit=canonical_units,
            meta=photon_counts.meta,
        )

    if photon_shot_inverse_transform:
        sampled = _poisson_inverse_transform(mean_counts)
    else:
        sampled = np.random.poisson(mean_counts)

    return NDCube(
        data=sampled.astype(np.int64),
        wcs=photon_counts.wcs.deepcopy(),
        unit=canonical_units,
        meta=photon_counts.meta,
    )


def apply_exposure(I: NDCube, t_exp: u.Quantity) -> NDCube:
    """
    Apply exposure time to intensity.

    Multiplies the intensity rate by the exposure time to give the total
    accumulated intensity.

    Parameters
    ----------
    I : NDCube
        Input intensity cube (per second).
    t_exp : u.Quantity
        Exposure time.

    Returns
    -------
    NDCube
        New cube with exposure time applied.
    """
    # Convert intensity rate to total intensity over exposure
    total_intensity = (I.data * I.unit * t_exp)

    return NDCube(
        data=total_intensity.value,
        wcs=I.wcs.deepcopy(),
        unit=total_intensity.unit,
        meta=I.meta,
    )


def add_visible_stray_light(electrons: NDCube, t_exp: u.Quantity, det, sim, tel=None,
                            *, noise: bool = True) -> NDCube:
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
    vis_mean = (vis_sl_per_pixel * t_exp).to_value(u.photon / u.pixel)
    if noise:
        n_vis_ph = np.random.poisson(
            vis_mean, size=electrons.data.shape) * (u.photon / u.pixel)
    else:
        n_vis_ph = np.full(electrons.data.shape, vis_mean) * (u.photon / u.pixel)

    # Assume visible stray light is ~600nm (typical visible wavelength)
    visible_wavelength = 600 * u.nm  # Keep as Quantity with units

    # Apply quantum efficiency first, then vectorized Fano noise
    vis_incident = n_vis_ph.to_value(u.photon / u.pixel)
    if noise:
        vis_photons_detected = np.random.binomial(
            vis_incident.astype(int), det.qe_vis)
    else:
        vis_photons_detected = vis_incident * det.qe_vis

    # Apply vectorized Fano noise to detected visible photons
    stray_electrons_values = _vectorized_fano_noise(
        vis_photons_detected.astype(float), visible_wavelength, det, noise=noise)
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


def add_pinhole_visible_light(electrons: NDCube, t_exp: u.Quantity, det, sim, tel,
                              *, noise: bool = True) -> NDCube:
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
    from .pinhole_diffraction import (
        airy_peak_fraction_per_pixel, calculate_pinhole_diffraction_pattern)
    
    # Get detector and data properties
    data_shape = electrons.data.shape  # Should be (n_slit, n_scan, n_spectral)
    
    # Visible light wavelength (typical)
    visible_wavelength = 600 * u.nm
    
    # Initialize additional electron contributions
    additional_electrons = np.zeros_like(electrons.data)

    # Spectral positions are optional; without them every pinhole projects to
    # the centre of the spectral window, as it always did.
    spectral_positions = (list(sim.pinhole_positions_spectral)
                          or [None] * len(sim.pinhole_sizes))

    for pinhole_diameter, pinhole_position, pinhole_spectral in zip(
            sim.pinhole_sizes, sim.pinhole_positions, spectral_positions):
        # Calculate pinhole area
        pinhole_area = np.pi * (pinhole_diameter / 2)**2

        # === Visible Light Contribution Through Pinhole ===
        # Calculate total photons incident on the pinhole area (unfiltered)
        # sim.vis_sl is photon/s/cm^2, pinhole_area is in cm^2
        vis_photons_per_sec_through_pinhole = sim.vis_sl * pinhole_area
        vis_photons_total_through_pinhole = (vis_photons_per_sec_through_pinhole * t_exp).to(u.photon)
        
        # Calculate visible diffraction pattern - this shows how the pinhole photons spread
        n_slit, n_scan, n_spectral = data_shape
        vis_pattern = calculate_pinhole_diffraction_pattern(
            detector_shape=(n_slit, n_spectral),
            pixel_size=det.pix_size*u.pix,
            pinhole_diameter=pinhole_diameter,
            pinhole_position_slit=pinhole_position,
            slit_width=sim.slit_width,
            plate_scale=det.plate_scale_angle,
            distance=det.filter_distance,
            wavelength=visible_wavelength,
            pinhole_position_spectral=pinhole_spectral,
        )

        # Scale the pattern by its ABSOLUTE normalisation rather than by its
        # sum over the detector array.  Dividing by the array sum forces every
        # transmitted photon onto the detector, which is badly wrong for small
        # pinholes: a 1 micron hole at 250 mm puts its first Airy minimum
        # 183 mm out, so nearly all of its light misses a detector tens of mm
        # across and must be lost, not redistributed.  vis_pattern peaks at
        # 1.0, so multiplying by the peak per-pixel fraction gives the correct
        # per-pixel fraction everywhere, and the array simply sums to less
        # than 1 when light falls off the detector.
        peak_fraction = airy_peak_fraction_per_pixel(
            pinhole_diameter=pinhole_diameter,
            distance=det.filter_distance,
            wavelength=visible_wavelength,
            pixel_size=det.pix_size * u.pix,
        )
        vis_pattern_normalized = vis_pattern * peak_fraction

        vis_photons_distributed = vis_photons_total_through_pinhole.to(u.photon).value * vis_pattern_normalized

        # Sample Poisson photons for this pinhole contribution
        if noise:
            vis_photons_poisson = np.random.poisson(vis_photons_distributed)
        else:
            vis_photons_poisson = vis_photons_distributed

        # Apply quantum efficiency
        if noise:
            vis_photons_detected = np.random.binomial(
                vis_photons_poisson.astype(int),
                det.qe_vis
            )
        else:
            vis_photons_detected = vis_photons_poisson * det.qe_vis

        # Apply Fano noise to detected visible photons
        vis_electrons_values = _vectorized_fano_noise(
            vis_photons_detected.astype(float), visible_wavelength, det,
            noise=noise)

        # Add to all scan positions (visible light affects all equally)
        for scan_idx in range(n_scan):
            additional_electrons[:, scan_idx, :] += vis_electrons_values

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
