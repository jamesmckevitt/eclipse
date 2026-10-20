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
from scipy.stats import binom, poisson
from . import resolution_budget
from .sampling import _carry_photon_wavelengths
from .utils import wl_to_vel, vel_to_wl, debug_break, _fwhm_to_sigma


def _poisson_inverse_transform(mean_counts, size=None) -> np.ndarray:
    """
    Draw Poisson counts by inverse-transform (quantile) sampling.

    Each element takes one uniform draw through the Poisson inverse CDF. The
    distribution is the same as `np.random.poisson`, but each element uses a
    fixed number of random draws, where its rejection sampling uses a
    variable number. This keeps the random stream synchronised across runs
    that differ only in the Poisson mean, which common-random-number
    variance reduction needs.

    Parameters
    ----------
    mean_counts : float or np.ndarray
        Poisson mean, scalar or per element.
    size : tuple of int, optional
        Shape to draw. Defaults to the shape of `mean_counts`.

    Returns
    -------
    np.ndarray
        Sampled counts as int64.
    """
    if size is None:
        size = np.shape(mean_counts)

    u_draw = np.random.random(size=size)
    # np.random.random() draws from [0, 1), and scipy's ppf gives -1 at
    # exactly 0, a negative count. Clamping to the smallest positive double
    # keeps the CRN property.
    np.maximum(u_draw, np.nextafter(0.0, 1.0), out=u_draw)

    return poisson.ppf(u_draw, mean_counts).astype(np.int64)


def _binomial_inverse_transform(trials, probability: float) -> np.ndarray:
    """
    Draw binomial counts by inverse-transform sampling, one uniform per element.

    As in :func:`_poisson_inverse_transform`, the distribution is that of
    `np.random.binomial`, but the number of random draws does not depend on
    the number of trials. Runs that differ only in photon flux then use the
    same random numbers for the quantum efficiency.
    """
    u_draw = np.random.random(size=np.shape(trials))
    np.maximum(u_draw, np.nextafter(0.0, 1.0), out=u_draw)
    return binom.ppf(u_draw, trials, probability).astype(np.int64)


def electrons_per_photon(rest_wavelength: u.Quantity, det) -> np.ndarray:
    """
    Mean number of electrons a detected photon frees in the detector silicon.

    This is the photon energy over the energy per electron-hole pair, which
    depends on the CCD temperature.

    Parameters
    ----------
    rest_wavelength : u.Quantity
        Photon wavelength, one value or an array.
    det : Detector_SWC or Detector_EIS
        Detector, with its `ccd_temperature`.

    Returns
    -------
    np.ndarray
        Electrons per photon, same shape as `rest_wavelength`.
    """
    if not hasattr(det, 'ccd_temperature'):
        raise ValueError("CCD temperature not set. Pass ccd_temperature when constructing the Detector instance.")
    temp_kelvin = det.ccd_temperature.to(u.K, equivalencies=u.temperature()).value
    photon_energy_ev = (const.h * const.c / (rest_wavelength.to(u.angstrom))).to(u.eV).value
    w_T = 3.71 - 0.0006 * (temp_kelvin - 300.0)  # eV per electron-hole pair
    return np.asarray(photon_energy_ev / w_T, dtype=float)


def dn_variance(dn: np.ndarray, wavelength: u.Quantity, t_exp: u.Quantity, det,
                visible_electrons: np.ndarray | float = 0.0, n_binned: int = 1, *,
                rms_wavelength: u.Quantity | None = None) -> np.ndarray:
    """
    Variance of each pixel's DN estimated from the DN.

    The dark current and visible light are known, and the rest of the
    signal is taken as EUV. The variance is the sum of:

    - EUV: (electrons per photon + Fano factor) times the EUV electrons.
      For photons of different energies, the electrons per photon is the
      mean of its square over its mean.
    - Mean dark current and visible light (Poisson).
    - Read noise: `read_noise_rms` squared.
    - Rounding to whole DN: 1/12 DN^2.

    A signal at or below the dark current and visible light is taken to
    hold no EUV. Each of the `n_binned` pixels summed after read-out adds
    its own dark current, read noise and rounding.

    Parameters
    ----------
    dn : np.ndarray
        DN in each pixel, or summed over `n_binned` pixels along the slit.
    wavelength : u.Quantity
        Wavelength of the mean photon energy in each pixel, same shape as
        `dn`, or one value for all, such as the line's rest wavelength. For
        summed pixels, that of all their photons, from
        `binned_photon_wavelengths`. Sets the electrons per photon.
    t_exp : u.Quantity
        Exposure time, for the dark current.
    det : Detector_SWC or Detector_EIS
        Detector.
    visible_electrons : np.ndarray or float, optional
        Mean electrons from visible light in each pixel, as added by
        `add_visible_stray_light` and `add_pinhole_visible_light` with
        `noise=False`, summed over the same pixels as `dn`. Default 0.
    n_binned : int, optional
        Number of pixels summed in each value of `dn`. Default 1.
    rms_wavelength : u.Quantity, optional
        Wavelength of the root mean square energy of the same photons, in
        the same form as `wavelength`. Sets how far the electrons of photons
        of different energies spread. Default `wavelength` (photons all of
        one energy).

    Returns
    -------
    np.ndarray
        Variance in DN^2, same shape as `dn`.
    """
    gain = det.gain_e_per_dn.to_value(u.electron / u.DN)
    dark = n_binned * (det.dark_current * t_exp).to_value(u.electron / u.pixel)
    read = n_binned * det.read_noise_rms.to_value(u.electron / u.pixel) ** 2
    visible = np.asarray(visible_electrons, dtype=float)
    electrons = np.asarray(dn, dtype=float) * gain
    euv = np.maximum(electrons - dark - visible, 0.0)
    mean = electrons_per_photon(wavelength, det)
    square = mean**2 if rms_wavelength is None else electrons_per_photon(rms_wavelength, det)**2
    per_electron = square / mean + det.si_fano
    variance_electrons = per_electron * euv + dark + visible + read
    return variance_electrons / gain ** 2 + n_binned / 12.0


def _vectorized_fano_noise(photon_counts: np.ndarray, rest_wavelength: u.Quantity, det,
                           *, noise: bool = True, every_pixel: bool = False,
                           rms_wavelength: u.Quantity | None = None) -> np.ndarray:
    """
    Electrons from photon counts with Fano noise, vectorized for speed.

    Parameters
    ----------
    photon_counts : np.ndarray
        Photon counts (unitless).
    rest_wavelength : u.Quantity
        Wavelength of the mean energy of each pixel's photons, one value or
        an array that broadcasts against `photon_counts`.
    det : Detector_SWC or Detector_EIS
        Detector, with the Fano noise parameters.
    noise : bool, optional
        If False, each photon frees its mean number of electrons rather
        than a draw around it. The conversion gain is unchanged; only its
        spread is dropped. Default True.
    every_pixel : bool, optional
        Draw one standard normal for every pixel, with or without photons,
        rather than one per pixel with photons. The random numbers then do
        not depend on how many pixels have photons, so runs that differ only
        in photon flux use the same ones. Same distribution. Default False.
    rms_wavelength : u.Quantity, optional
        Wavelength of the root mean square energy of each pixel's photons,
        in the same form as `rest_wavelength`. Photons of different energies
        free different numbers of electrons, which widens a pixel's spread.
        Default `rest_wavelength` (photons all of one energy).

    Returns
    -------
    np.ndarray
        Electron counts with Fano noise applied.
    """
    # Handle zero or negative photon counts
    mask_positive = photon_counts > 0
    electron_counts = np.zeros_like(photon_counts)
    standard = (np.random.standard_normal(photon_counts.shape) if noise and every_pixel
                else None)

    if not np.any(mask_positive):
        return electron_counts
    
    # Mean electrons per photon for each pixel. The wavelength may be one
    # value or an array that broadcasts against the counts, such as one per
    # row for a frame that spans the band.
    mean_electrons_per_photon = np.broadcast_to(
        electrons_per_photon(rest_wavelength, det), photon_counts.shape)[mask_positive]

    # Variance per photon: the Fano variance plus, for photons of different
    # energies, the variance of their mean electrons. That is the mean
    # square, from the root mean square energy, less the squared mean; it is
    # zero for one energy, but rounding can take it slightly below.
    variance_per_photon = det.si_fano * mean_electrons_per_photon
    if rms_wavelength is not None:
        rms_electrons_per_photon = np.broadcast_to(
            electrons_per_photon(rms_wavelength, det), photon_counts.shape)[mask_positive]
        variance_per_photon = variance_per_photon + np.maximum(
            rms_electrons_per_photon**2 - mean_electrons_per_photon**2, 0.0)
    sigma_per_photon = np.sqrt(variance_per_photon)

    # Work only with positive photon counts
    positive_photons = photon_counts[mask_positive]

    # Draw the total for N photons from Normal(N*mean_e, sqrt(N)*sigma_per_photon)
    # rather than each photon's electrons from its own energy. This has the
    # same mean and variance, is equivalent for large N and is much faster.

    mean_total_electrons = positive_photons * mean_electrons_per_photon

    if noise:
        std_total_electrons = np.sqrt(positive_photons) * sigma_per_photon

        # Sample total electrons per pixel
        if standard is None:
            total_electrons = np.random.normal(
                loc=mean_total_electrons,
                scale=std_total_electrons
            )
        else:
            total_electrons = mean_total_electrons + std_total_electrons * standard[mask_positive]

        # Ensure non-negative
        total_electrons = np.maximum(total_electrons, 0)
    else:
        total_electrons = mean_total_electrons
    
    # Map back to full array
    electron_counts[mask_positive] = total_electrons
    
    return electron_counts


def photons_per_energy(tel, wavelength: u.Quantity) -> u.Quantity:
    """
    Photons the telescope collects per erg of incident light, at each wavelength.

    This is the collecting area times the throughput, over the photon
    energy. ECLIPSE takes it at each photon's own wavelength, before the
    spectrograph blur, since the blur moves where a photon lands but not
    its wavelength.

    Parameters
    ----------
    tel : Telescope_EUVST or Telescope_EIS
        Telescope, queried for all wavelengths at once.
    wavelength : u.Quantity
        Wavelengths.

    Returns
    -------
    u.Quantity
        In cm2 / erg, one value per wavelength. Beyond the throughput tables
        EUVST's telescope gives NaN and EIS's raises a ValueError.
    """
    wavelength = np.atleast_1d(u.Quantity(wavelength))
    area = np.broadcast_to(u.Quantity(tel.ea_and_throughput(wavelength)).to(u.cm**2),
                           wavelength.shape, subok=True)
    return (area * wavelength / (const.h * const.c)).to(u.cm**2 / u.erg)


def intensity_to_photons(I: NDCube) -> NDCube:
    """
    Convert energy to photons by dividing by the photon energy at each wavelength.

    Parameters
    ----------
    I : NDCube
        Spectral radiance over the exposure, from `apply_exposure`, with
        wavelength on the last axis.

    Returns
    -------
    NDCube
        Photons, in photon / (cm2 sr cm).
    """
    wl_axis = I.axis_world_coords_values(2)[0]
    E_ph = (const.h * const.c / wl_axis).to("erg") * (1 / u.photon)
    
    photon_data = (I.data * I.unit / E_ph).to(u.photon / u.cm**2 / u.sr / u.cm)
    
    return NDCube(
        data=photon_data.value,
        wcs=I.wcs.deepcopy(),
        unit=photon_data.unit,
        meta=I.meta,
    )


def add_telescope_throughput(ph_flux: NDCube, tel) -> NDCube:
    """
    Multiply by the telescope's effective area at each wavelength, before the detector.

    Parameters
    ----------
    ph_flux : NDCube
        Photons, from `intensity_to_photons`.
    tel : Telescope_EUVST or Telescope_EIS
        Telescope; its `ea_and_throughput` is used.

    Returns
    -------
    NDCube
        Photons collected, in photon / (sr cm).

    Raises
    ------
    ValueError
        If the cube's wavelengths reach beyond the telescope's throughput
        tables.
    """
    wl0 = ph_flux.meta['rest_wav']
    wl_axis = ph_flux.axis_world_coords_values(2)[0]
    throughput = np.array([tel.ea_and_throughput(wl).cgs.value for wl in wl_axis]) * u.cm**2
    # Beyond the throughput tables there is no detector, so a window pixel
    # there is one the instrument does not have. Filled with zero it would
    # pull the fit; left as NaN it spreads through the blur, or makes the fit
    # wrong with no warning.
    outside = ~np.isfinite(throughput.value)
    if outside.any():
        missing = wl_axis[outside].to_value(u.AA)
        raise ValueError(
            f"The window of the line at {u.Quantity(wl0).to(u.AA):.4f} reaches "
            f"{missing.min():.4f} to {missing.max():.4f} Angstrom, where the telescope's "
            f"throughput tables, and so its band, end. Observe a line further inside the band.")

    out_data = (ph_flux.data * ph_flux.unit * throughput)
    
    return NDCube(
        data=out_data.value,
        wcs=ph_flux.wcs.deepcopy(),
        unit=out_data.unit,
        meta=ph_flux.meta,
    )


def photons_to_pixel_counts(ph_flux: NDCube, wl_pitch: u.Quantity, plate_scale: u.Quantity, slit_width: u.Quantity) -> NDCube:
    """
    Convert the photons collected to photons per detector pixel.

    Each pixel sees a patch of Sun one pixel along the slit and one slit
    width across, over one pixel in wavelength.

    Parameters
    ----------
    ph_flux : NDCube
        Photons collected, from `add_telescope_throughput`.
    wl_pitch : u.Quantity
        Wavelength step per pixel, `det.wvl_res`.
    plate_scale : u.Quantity
        Length on the Sun of one pixel along the slit,
        `det.plate_scale_length`.
    slit_width : u.Quantity
        Slit width as a length on the Sun, such as
        `angle_to_distance(sim.slit_width)`.

    Returns
    -------
    NDCube
        Photons per pixel.
    """
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
    Width of the slit's image along the dispersion, in detector pixels.

    RSC-2022021C gives widths along the dispersion in the same arcsec as
    along the slit, converting both with the spatial scale (0.0118 arcsec
    per micron for SW), and adds the slit to the optics in those units. A
    slit `slit_width` wide therefore covers `slit_width` over the plate
    scale in spectral pixels, as it would along the slit.
    """
    return (slit_width / (det.plate_scale_angle * u.pixel)).to_value(u.dimensionless_unscaled)


def _budget(tel):
    """`resolution_budget` and `pointing_jitter` in arcsec, for the resolution_budget functions."""
    jitter = getattr(tel, "pointing_jitter", None)
    return (getattr(tel, "resolution_budget", "none"),
            None if jitter is None else jitter.to_value(u.arcsec))


def _budget_pixels(tel, det, part) -> float:
    """FWHM in pixels of the resolution budget's blur, `part` one of its functions."""
    return part(*_budget(tel)) / (det.plate_scale_angle * u.pixel).to_value(u.arcsec)


def spatial_psf_fwhm(tel, det) -> float:
    """
    FWHM of the PSF along the slit, in detector pixels.

    `tel.psf_params[0]`, the optical design, and the resolution budget's
    other terms (:func:`~euvst_response.resolution_budget.along_slit_fwhm`)
    added in quadrature.
    """
    optics = tel.psf_params[0].to_value(u.pixel)
    return float(np.hypot(optics, _budget_pixels(tel, det, resolution_budget.along_slit_fwhm)))


def across_slit_fwhm(tel) -> u.Quantity | None:
    """
    FWHM of the telescope's blur across the slit, or None for none.

    `tel.psf_across_slit` and the resolution budget's terms before the slit
    (:func:`~euvst_response.resolution_budget.across_slit_fwhm`) added in
    quadrature.
    """
    optics = getattr(tel, "psf_across_slit", None)
    optics = 0.0 if optics is None else optics.to_value(u.arcsec)
    fwhm = np.hypot(optics, resolution_budget.across_slit_fwhm(*_budget(tel)))
    return None if fwhm == 0 else float(fwhm) * u.arcsec


def spectral_blur_fwhm(tel, det) -> float:
    """
    FWHM of the spectral PSF without the slit, in detector pixels.

    The optics (:func:`spectral_optics_fwhm`) and the resolution budget's
    other terms (:func:`~euvst_response.resolution_budget.spectral_fwhm`)
    added in quadrature. This is the Gaussian convolved with the slit's
    image for `spectral_psf="convolution"`.
    """
    return float(np.hypot(spectral_optics_fwhm(tel, det),
                          _budget_pixels(tel, det, resolution_budget.spectral_fwhm)))


def spectral_optics_fwhm(tel, det) -> float:
    """
    FWHM of the optics alone along the dispersion, in detector pixels.

    `tel.psf_params[1]` is the spectral FWHM with a slit `tel.psf_slit_width`
    wide: the optics FWHM and the slit width added in quadrature. The optics
    FWHM is what remains when the slit is removed in quadrature. For the
    default SWC values it is 2.207 pixels, or 0.351 arcsec, against 0.352
    arcsec in RSC-2022021C; the two agree to the rounding of 43.00 mA to
    2.54 pixels and of the plate scale to 0.159 arcsec per pixel.
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
    FWHM of the spectral PSF for a given slit, in detector pixels.

    The slit's image is part of the spectral PSF, so the PSF widens with the
    slit. The optics FWHM and the slit image width add in quadrature, and
    `tel.psf_params[1]` is the FWHM for the slit `tel.psf_slit_width`. The
    resolution budget's other terms
    (:func:`~euvst_response.resolution_budget.spectral_fwhm`) are added in
    quadrature too. For EUVST-SW's optics alone this gives 2.54, 3.35, 5.49
    and 10.30 pixels for the 0.2, 0.4, 0.8 and 1.6 arcsec slits. A
    telescope without `psf_slit_width` has the same optics FWHM for every
    slit.

    Parameters
    ----------
    tel : Telescope_EUVST or Telescope_EIS
        Telescope.
    det : Detector_SWC or Detector_EIS
        Detector, whose plate scale sets the width of the slit's image.
    slit_width : u.Quantity
        Slit width.

    Returns
    -------
    float
        FWHM in pixels.
    """
    # In quadrature, FWHM(w)**2 = FWHM(w0)**2 + s(w)**2 - s(w0)**2, with w0
    # the reference slit and s the slit image width (slit_image_width), as
    # RSC-2022021C combines them for its quoted spectral resolution.
    measured = tel.psf_params[1].to_value(u.pixel)
    budget = _budget_pixels(tel, det, resolution_budget.spectral_fwhm)
    reference = getattr(tel, "psf_slit_width", None)
    if reference is None:
        return float(np.hypot(measured, budget))
    # Raises if the FWHM leaves nothing for the optics.
    spectral_optics_fwhm(tel, det)
    # Take the difference of squares first so it is exactly zero for the
    # reference slit, which then gets psf_params[1] to the last bit.
    difference = slit_image_width(slit_width, det)**2 - slit_image_width(reference, det)**2
    return float(np.sqrt(measured**2 + difference + budget**2))


def _reach(fwhm: float) -> int:
    """Pixels a kernel six sigma wide reaches either side of its centre, at least three."""
    n = max(7, int(np.ceil(6 * _fwhm_to_sigma(fwhm))))
    if n % 2 == 0:
        n += 1
    return n // 2


def spectral_psf_reach(tel, det, slit_width: u.Quantity) -> int:
    """
    Pixels the spectral PSF kernel reaches either side of its centre.

    For either `spectral_psf`, the kernel spans six sigma of the Gaussian of
    :func:`spectral_psf_fwhm`, rounded up to an odd number of pixels and at
    least seven. The slit convolved with the optics is no wider than that
    Gaussian, and the pixels left out hold at most a few parts in ten
    thousand of it, for the narrowest slit.
    """
    return _reach(spectral_psf_fwhm(tel, det, slit_width))


def spectral_psf_margin(tel, det, slit_width: u.Quantity) -> int:
    """
    Pixels by which the spectral PSF reaches beyond that of the optics alone with the reference slit.

    The wavelength window of a synthesis has some margin around its lines,
    enough for the optics' PSF, `tel.psf_params[1]`. A wider slit, the
    resolution budget and the pointing jitter spread the lines further, and
    light spread past the ends of the window is lost. The window is
    therefore widened by this many pixels at each end. Zero when the PSF is
    the optics' alone with the reference slit.
    """
    optics = _reach(tel.psf_params[1].to_value(u.pixel))
    return max(0, spectral_psf_reach(tel, det, slit_width) - optics)


def _spectral_response(tel, det, slit_width: u.Quantity, spectral_psf: str) -> tuple:
    """
    Spectral response as a Gaussian convolved with a rectangle, in pixels.

    Returns the Gaussian's standard deviation and the rectangle's width. For
    `spectral_psf` `"quadrature"` the slit is in the Gaussian, of FWHM
    :func:`spectral_psf_fwhm`, and the width is 0. For `"convolution"` the
    Gaussian is the optics, of FWHM :func:`spectral_blur_fwhm`, and the
    rectangle is the slit's image.
    """
    if spectral_psf == "quadrature":
        return _fwhm_to_sigma(spectral_psf_fwhm(tel, det, slit_width)), 0.0
    if spectral_psf == "convolution":
        return _fwhm_to_sigma(spectral_blur_fwhm(tel, det)), slit_image_width(slit_width, det)
    raise ValueError(
        f"spectral_psf must be 'quadrature' or 'convolution', got {spectral_psf!r}."
    )


def _spectral_kernel(tel, det, slit_width: u.Quantity, spectral_psf: str) -> np.ndarray:
    """
    Spectral response at whole pixels out to :func:`spectral_psf_reach`.

    For `"quadrature"` a Gaussian, not normalised; for `"convolution"`
    :func:`spectral_line_spread`, which sums to one. Raises ValueError for
    any other `spectral_psf`.
    """
    if spectral_psf == "convolution":
        return spectral_line_spread(tel, det, slit_width)
    sigma, _ = _spectral_response(tel, det, slit_width, spectral_psf)
    reach = spectral_psf_reach(tel, det, slit_width)
    offsets = np.arange(-reach, reach + 1)
    return np.exp(-0.5 * (offsets / sigma) ** 2)


def spectral_line_spread(tel, det, slit_width: u.Quantity) -> np.ndarray:
    """
    Spectral PSF as a Gaussian convolved with the slit's image, per pixel.

    The Gaussian's FWHM is :func:`spectral_blur_fwhm`. This is the PSF
    used with `spectral_psf="convolution"`. A wide slit
    gives it a flat top. For the 0.2 arcsec slit it is narrower than the
    Gaussian of `spectral_psf_fwhm`, because a rectangle adds less to a
    width than a Gaussian of the same FWHM.

    Parameters
    ----------
    tel : Telescope_EUVST or Telescope_EIS
        Telescope.
    det : Detector_SWC or Detector_EIS
        Detector.
    slit_width : u.Quantity
        Slit width.

    Returns
    -------
    np.ndarray
        PSF at whole pixels either side of its centre, summing to one.
    """
    # RSC-2022021C defines the line spread function as the optics PSF after
    # the slit, convolved with a rectangle the width of the slit. Like the
    # Gaussian kernel, it is taken at pixel centres over the same pixels
    # (spectral_psf_reach).
    sigma, width = _spectral_response(tel, det, slit_width, "convolution")
    half = 0.5 * width
    reach = spectral_psf_reach(tel, det, slit_width)
    x = np.arange(-reach, reach + 1)
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
    Blur a cube on the detector pixels with the PSF, along the slit and in wavelength.

    For a synthesis file, `eclipse` blurs the spectra before laying them
    onto the pixels, so a line narrower than a pixel keeps its place within
    it. This function instead blurs a cube already on the pixels.
    `simulate_once` uses it for a cube not yet blurred, such as a single
    line of known intensity.

    Along the slit the PSF is a Gaussian of FWHM `spatial_psf_fwhm`. In
    wavelength it depends on `sim.spectral_psf`: a Gaussian of
    `spectral_psf_fwhm` for `"quadrature"`, or `spectral_line_spread` for
    `"convolution"`.

    Parameters
    ----------
    signal : NDCube
        Cube shaped (along the slit, slit positions, wavelength).
    tel : Telescope_EUVST or Telescope_EIS
        Telescope, with its PSF.
    det : Detector_SWC or Detector_EIS
        Detector, whose plate scale sets the width of the slit's image.
    sim : Simulation
        Gives the slit width and `spectral_psf`.
    convolve_spatial : bool, optional
        If False, blur in wavelength only, for a scene uniform along the
        slit, where blurring along it would change nothing. Default True.
    boundary : {"replicate", "zero"}, optional
        What lies beyond the ends of the slit. `"replicate"` extends the end
        rows outward. `"zero"` takes it as dark, so rows within a PSF width
        of an end come out darker. Default `"replicate"`.

    Returns
    -------
    NDCube
        Blurred cube, with the same coordinates, unit and metadata.
    """
    # A uniform scene is blurred in wavelength only: along the slit,
    # "replicate" changes nothing and "zero" removes light from the ends that
    # the scene does have. Zero fill is wrong for a raster too, as the Sun
    # continues past the field of view. With the default SWC PSF, nine rows
    # wide, it costs the end row about a third of the kernel's weight, the
    # next row 13 per cent and the one after 3 per cent.
    if boundary not in ("replicate", "zero"):
        raise ValueError(
            f"boundary must be 'replicate' or 'zero', got {boundary!r}."
        )
    n_slit, n_scan, n_lambda = signal.data.shape

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

    # Along lambda (axis 1 of each frame) the profile depends on the slit.
    spectral_1d = _spectral_kernel(tel, det, sim.slit_width, sim.spectral_psf)
    kx = spectral_1d.size
    quadrature = sim.spectral_psf == "quadrature"

    # FWHM in pixels along the slit (axis 0 of each frame)
    spatial_fwhm = spatial_psf_fwhm(tel, det)
    sigma_spatial = _fwhm_to_sigma(spatial_fwhm)
    ky = 2 * _reach(spatial_fwhm) + 1

    if not convolve_spatial:
        # Spectral axis only. Zero padding is correct here: the wavelength
        # grid extends several sigma past the line, so no flux is lost at its
        # edges.
        psf_1d = spectral_1d / spectral_1d.sum()

        def blur(cube):
            return convolve1d(cube, psf_1d, axis=2, mode="constant", cval=0.0)

        return _blurred_cube(signal, blur)

    if quadrature:
        sigma_spectral, _ = _spectral_response(tel, det, sim.slit_width, "quadrature")
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

    # Convolve each scan position's frame (n_slit, n_lambda). Padding the slit
    # axis by half a kernel is just enough for every real row to see
    # replicated rows rather than the zeros convolve2d assumes outside the
    # array. The spectral axis is zero-filled, which is correct there.
    pad = (ky // 2) if boundary == "replicate" else 0

    def blur(cube):
        blurred = np.empty(cube.shape)
        for i in range(n_scan):
            frame = cube[:, i, :]
            if pad:
                frame = np.pad(frame, ((pad, pad), (0, 0)), mode="edge")
            convolved = convolve2d(frame, psf, mode="same")
            blurred[:, i, :] = convolved[pad:pad + n_slit, :] if pad else convolved
        return blurred

    return _blurred_cube(signal, blur)


def _blurred_cube(signal: NDCube, blur) -> NDCube:
    """
    `signal`, photons on the pixels, blurred by `blur`, with the per-pixel
    photon values in its `meta` carried along (:func:`_carry_photon_meta`).
    """
    blurred, meta = _carry_photon_meta(blur, signal)
    return NDCube(data=blurred, wcs=signal.wcs.deepcopy(), unit=signal.unit, meta=meta)


def _carry_photon_meta(carry, cube: NDCube) -> tuple:
    """
    Photons of `cube` moved by `carry`, and its `meta` with the per-pixel
    photon values moved with them.

    `carry` is a linear map that keeps the wavelength axis. The values are
    the wavelengths of the mean and root mean square photon energies,
    `photon_wavelength` and `photon_rms_wavelength`. For the pinholes they
    also include the fraction of the light the filter passed,
    `photon_filter_transmission`, and the wavelengths of the mean and root
    mean square energies of the light it blocked,
    `blocked_photon_wavelength` and `blocked_photon_rms_wavelength`.

    The photons and the blocked light are each moved as photons, so these
    values are exact wherever they land. A pixel no photons reach keeps its
    own wavelength and blocks nothing.

    Returns `(moved, meta)`.
    """
    photons = np.asarray(cube.data, dtype=float)
    meta = cube.meta
    if "photon_wavelength" not in (meta or {}):
        return carry(photons), meta
    mean = u.Quantity(meta["photon_wavelength"])
    unit = mean.unit
    rms = u.Quantity(meta.get("photon_rms_wavelength", mean)).to_value(unit)
    own = cube.axis_world_coords_values(2)[0].to_value(unit)
    moved, moved_mean, moved_rms = _carry_photon_wavelengths(carry, photons, mean.value, rms,
                                                             own)
    meta = {**meta, "photon_wavelength": moved_mean * unit,
            "photon_rms_wavelength": moved_rms * unit}
    if "photon_filter_transmission" in meta:
        passed = np.asarray(meta["photon_filter_transmission"], dtype=float)
        blocked = photons * np.divide(1.0 - passed, passed, out=np.zeros(photons.shape),
                                      where=passed > 0)
        moved_blocked, blocked_mean, blocked_rms = _carry_photon_wavelengths(
            carry, blocked, u.Quantity(meta["blocked_photon_wavelength"]).to_value(unit),
            u.Quantity(meta["blocked_photon_rms_wavelength"]).to_value(unit), own)
        before = moved + moved_blocked
        meta.update({
            "photon_filter_transmission": np.divide(moved, before, out=np.ones(moved.shape),
                                                    where=before > 0),
            "blocked_photon_wavelength": blocked_mean * unit,
            "blocked_photon_rms_wavelength": blocked_rms * unit})
    return moved, meta


def to_electrons(
    photon_counts: NDCube,
    t_exp: u.Quantity,
    det,
    *,
    dark_current_inverse_transform: bool = False,
    noise: bool = True,
    photon_shot_inverse_transform: bool = False,
) -> NDCube:
    """
    Convert the photons arriving in each pixel to the electrons read out.

    Each photon is detected with probability `det.qe_euv` and frees a number
    of electrons that depends on its wavelength, with a spread set by the
    Fano factor. Dark current and read noise are then added. Pixels can go
    below zero, as read noise takes a dark pixel below the bias level as
    often as above it.

    Parameters
    ----------
    photon_counts : NDCube
        Photons in each pixel, from `sample_photon_arrivals`: whole numbers,
        or with `noise=False` their expected, fractional values. Its
        `meta["photon_wavelength"]`, the wavelength of the mean energy of
        each pixel's photons, sets the electrons per photon. Its
        `meta["photon_rms_wavelength"]`, that of their root mean square
        energy, sets how far the electrons of photons of different energies
        spread. Without them, `meta["rest_wav"]` sets both.
    t_exp : u.Quantity
        Exposure time, for the dark current.
    det : Detector_SWC or Detector_EIS
        Detector.
    dark_current_inverse_transform : bool, optional
        Draw the dark current by inverse-transform sampling, for common
        random numbers between runs that differ in dark current. Default
        False.
    noise : bool, optional
        If False, every random draw is replaced by its mean. Default True.
    photon_shot_inverse_transform : bool, optional
        Draw the detected photons by inverse-transform sampling, and the Fano
        spread for every pixel, for common random numbers between runs that
        differ in photon flux. Default False.

    Returns
    -------
    NDCube
        Electrons in each pixel.
    """
    # Each photon frees electrons by its own energy. The cube gives the
    # wavelengths of the mean and root mean square photon energies in each
    # pixel, as kept by `simulate_once`; otherwise the line's rest wavelength
    # is used for all.
    meta = photon_counts.meta or {}
    if "photon_wavelength" in meta:
        rest_wavelength = meta["photon_wavelength"]
        rms_wavelength = meta.get("photon_rms_wavelength", rest_wavelength)
    else:
        rest_wavelength = rms_wavelength = meta["rest_wav"]

    # Apply quantum efficiency: a binomial draw over whole photons, or without
    # noise its expectation qe * N, with no cast to integers.
    incident = photon_counts.to(u.photon / u.pix).data
    if not noise:
        photons_detected = incident * det.qe_euv
    elif photon_shot_inverse_transform:
        photons_detected = _binomial_inverse_transform(incident.astype(int), det.qe_euv)
    else:
        photons_detected = np.random.binomial(incident.astype(int), det.qe_euv)

    # Apply Fano noise per pixel, vectorized
    electron_counts = _vectorized_fano_noise(photons_detected.astype(float),
                                             rest_wavelength, det, noise=noise,
                                             every_pixel=photon_shot_inverse_transform,
                                             rms_wavelength=rms_wavelength)

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

    # Add read noise. It has zero mean, so without noise nothing is added.
    if noise:
        e += np.random.normal(0, det.read_noise_rms.value,
                              photon_counts.data.shape) * (u.electron / u.pixel)  # read noise

    # Not clipped at zero: a CCD reads out above a bias level, so read noise
    # takes a dark pixel below it as often as above, and ECLIPSE's
    # bias-subtracted signal goes negative. Clipping would raise the mean of
    # faint pixels, so noise on would no longer average to noise off.
    e = e.to(u.electron / u.pixel)
    e_val = e.value

    return NDCube(
        data=e_val,
        wcs=photon_counts.wcs.deepcopy(),
        unit=e.unit,
        meta=photon_counts.meta,
    )


def to_dn(electrons: NDCube, det) -> NDCube:
    """
    Convert electrons to DN, rounded to the nearest DN and clipped at the digitiser maximum.

    Nothing is clipped at the CCD full well, `Detector_SWC.full_well`,
    though it is lower than the digitiser maximum.

    Parameters
    ----------
    electrons : NDCube
        Electrons in each pixel, from `to_electrons`.
    det : Detector_SWC or Detector_EIS
        Detector, with its gain and `max_dn`.

    Returns
    -------
    NDCube
        DN in each pixel.
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
    Replace each value of a cube with a Poisson draw with that mean.

    Parameters
    ----------
    cube : NDCube
        Mean in each pixel.

    Returns
    -------
    NDCube
        Draws, with the same coordinates, unit and metadata.
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
    Poisson draw of the photons arriving in each pixel, around the expected number.

    Parameters
    ----------
    photon_counts : NDCube
        Expected photons in each pixel.
    photon_shot_inverse_transform : bool, optional
        Draw by inverse-transform sampling, for common random numbers between
        runs that differ in photon flux. Default False.
    noise : bool, optional
        If False, return the expected numbers instead, not rounded to whole
        photons. Default True.

    Returns
    -------
    NDCube
        Photons in each pixel, in photon per pixel.
    """
    q = photon_counts.data * photon_counts.unit

    canonical_units = u.photon / u.pix

    mean_counts = q.to(canonical_units).value
    mean_counts = np.maximum(mean_counts, 0)

    # Without noise, keep floats: rounding to whole photons would add back
    # quantisation where the aim is to remove randomness.
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
    Multiply a cube of spectral radiance by the exposure time.

    Parameters
    ----------
    I : NDCube
        Spectral radiance, per second.
    t_exp : u.Quantity
        Exposure time.

    Returns
    -------
    NDCube
        Spectral radiance over the exposure.
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
    Add the electrons freed by visible stray light, the same in every pixel.

    `sim.vis_sl` is in photons per second per cm2. If the telescope has a
    filter, as EUVST does, this is the light at the filter, which passes
    `AluminiumFilter.visible_light_throughput` of it; otherwise it is the
    light at the CCD. Each photon reaching the detector is detected with
    probability `det.qe_vis` and frees one electron.

    Parameters
    ----------
    electrons : NDCube
        Electrons in each pixel, from `to_electrons`.
    t_exp : u.Quantity
        Exposure time.
    det : Detector_SWC or Detector_EIS
        Detector.
    sim : Simulation
        Simulation, with `vis_sl`.
    tel : Telescope_EUVST or Telescope_EIS, optional
        Telescope. Its filter is applied if it has one. EIS's is not, so for
        EIS `vis_sl` is the light reaching the CCD.
    noise : bool, optional
        If False, add the expected electrons rather than a draw. Default
        True.

    Returns
    -------
    NDCube
        Electrons in each pixel with the stray light added, in electron per
        pixel.
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

    # Apply the quantum efficiency. A visible photon absorbed in silicon frees
    # exactly one electron: its couple of eV is below the 3.7 eV per
    # electron-hole pair, so the EUV model of many electrons with a Fano
    # spread does not apply (it would give 0.55).
    vis_incident = n_vis_ph.to_value(u.photon / u.pixel)
    if noise:
        vis_photons_detected = np.random.binomial(
            vis_incident.astype(int), det.qe_vis)
    else:
        vis_photons_detected = vis_incident * det.qe_vis
    stray_electrons = vis_photons_detected.astype(float) * (u.electron / u.pixel)

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
    Add the electrons freed by visible light through pinholes in the filter.

    The light through each pinhole spreads over the detector in the
    pinhole's diffraction pattern. Does nothing unless `sim.enable_pinholes`
    is set and there are pinholes.

    Parameters
    ----------
    electrons : NDCube
        Electrons in each pixel.
    t_exp : u.Quantity
        Exposure time.
    det : Detector_SWC
        EUVST-SW detector, whose `filter_distance` sets the diffraction.
    sim : Simulation
        Simulation, with `vis_sl` and the pinholes.
    tel : Telescope_EUVST
        Telescope.
    noise : bool, optional
        If False, add the expected electrons rather than a draw. Default
        True.

    Returns
    -------
    NDCube
        Electrons in each pixel with the pinholes' light added, in electron
        per pixel.
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
    # the centre of the spectral window.
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
        
        # Visible diffraction pattern: how the pinhole photons spread
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

        # Scale the pattern by its absolute normalisation, not its sum over the
        # detector. Dividing by the sum would put every transmitted photon on
        # the detector, but a 1 micron hole at 250 mm has its first Airy
        # minimum 183 mm out, so nearly all its light misses a detector tens
        # of mm across and must be lost, not redistributed. vis_pattern peaks
        # at 1.0, so multiplying by the peak per-pixel fraction gives the
        # right fraction everywhere, and the array sums to less than 1 when
        # light falls off the detector.
        peak_fraction = airy_peak_fraction_per_pixel(
            pinhole_diameter=pinhole_diameter,
            distance=det.filter_distance,
            wavelength=visible_wavelength,
            pixel_size=det.pix_size * u.pix,
        )
        vis_pattern_normalized = vis_pattern * peak_fraction

        vis_photons_distributed = vis_photons_total_through_pinhole.to(u.photon).value * vis_pattern_normalized

        # The pinhole does not move, so the light is the same at every scan
        # position, but each exposure gets its own draw.
        expected = np.broadcast_to(vis_photons_distributed[:, np.newaxis, :], data_shape)
        if noise:
            vis_photons_detected = np.random.binomial(np.random.poisson(expected), det.qe_vis)
        else:
            vis_photons_detected = expected * det.qe_vis

        # One electron per visible photon; see add_visible_stray_light.
        additional_electrons += vis_photons_detected

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
