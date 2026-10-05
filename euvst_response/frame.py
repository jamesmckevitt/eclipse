"""
Laying a spectrum onto the SW detectors.

:mod:`euvst_response.readout` says which wavelength each CCD row records and
what a frame collects while it is clocked out.  This module fills in the step
before: how much light each row receives in the first place, given a spectrum
of the Sun and the telescope in front of the detectors.

The arithmetic is the standard radiometric equation, the same one
:mod:`euvst_response.radiometric` applies to a synthesis cube::

    photons per second per pixel = I * Omega_pix * A_eff / E_photon

with the spectral radiance ``I`` integrated over the wavelengths the row
covers, rather than multiplied by a fixed pixel bandwidth.  That matters here
because the rows are not evenly spaced in wavelength and a frame spans the
whole band, where the effective area changes by a factor of several.

A line is spread over the rows by integrating a Gaussian between the row
boundaries, so its flux is conserved whatever the sampling.  That Gaussian is
the line as the Sun emits it.  Given *spectral_psf*, the instrument's own
spectral response is applied to it as it is laid onto the rows, so that a
line narrower than a row keeps its place within the row.

At the other end, :func:`detect` and :func:`digitise` take the photons a frame
recorded through the detector stages of :mod:`euvst_response.radiometric`,
with the two things a full-band frame needs that a cube does not: a photon
energy for each row, and a dark current time for each row.
"""

from __future__ import annotations

import warnings
from typing import Optional

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.special import erf

from .radiometric import (
    _vectorized_fano_noise,
    photons_per_energy,
    slit_image_width,
    spectral_line_spread,
    spectral_optics_fwhm,
    spectral_psf_fwhm,
    spectral_psf_reach,
)
from .readout import FocalPlane_SWC, expose
from .sampling import _blurred_antiderivative, light_onto_pixels
from .utils import angle_to_distance, _fwhm_to_sigma


def pixel_solid_angle(focal_plane: FocalPlane_SWC, slit_width: u.Quantity) -> u.Quantity:
    """
    The patch of Sun one pixel sees, in steradian.

    A pixel covers the plate scale along the slit and the slit width across it,
    which is what the raster steps by.
    """
    along_slit = angle_to_distance(focal_plane.plate_scale)
    across = angle_to_distance(u.Quantity(slit_width))
    return ((along_slit * across).cgs / const.au.cgs ** 2).value * u.sr


def _row_bounds(focal_plane: FocalPlane_SWC, ccd: str, column=None, margin: int = 0) -> np.ndarray:
    """
    Row boundaries in Angstrom, sorted so that the first is the shorter, for
    the rows of the CCD and *margin* more past each end.
    """
    if margin:
        rows = np.arange(-margin, focal_plane.n_rows + margin + 1) - 0.5
        edges = focal_plane.wavelength(rows, ccd, column).to_value(u.Angstrom)
    else:
        edges = focal_plane.row_edges(ccd, column).to_value(u.Angstrom)
    return np.column_stack([np.minimum(edges[:-1], edges[1:]),
                            np.maximum(edges[:-1], edges[1:])])


def _response(focal_plane: FocalPlane_SWC, ccd: str, column, telescope, det,
              slit_width: u.Quantity, spectral_psf: str, wavelength: np.ndarray):
    """
    The instrument's spectral response at each of *wavelength* (Angstrom),
    as the standard deviation of its Gaussian and the width of the slit's
    image the Gaussian is convolved with, both in Angstrom, the second 0
    with *spectral_psf* ``"quadrature"``, where the slit is in the Gaussian.

    The response is a width in rows, the same that
    :func:`~euvst_response.radiometric.apply_focusing_optics_psf` gives a
    synthesis, taken here at the width of the row the wavelength falls in,
    which changes by less than a part in 1e4 across the response.
    """
    if spectral_psf == "quadrature":
        sigma_rows = _fwhm_to_sigma(spectral_psf_fwhm(telescope, det, slit_width))
        box_rows = 0.0
    elif spectral_psf == "convolution":
        sigma_rows = _fwhm_to_sigma(spectral_optics_fwhm(telescope, det))
        box_rows = slit_image_width(slit_width, det)
    else:
        raise ValueError(
            f"spectral_psf must be 'quadrature' or 'convolution', got {spectral_psf!r}."
        )
    bounds = _row_bounds(focal_plane, ccd, column)
    centres = bounds.mean(axis=1)
    order = np.argsort(centres)
    row_width = np.interp(wavelength, centres[order], (bounds[:, 1] - bounds[:, 0])[order])
    return sigma_rows * row_width, box_rows * row_width


def _intervals_onto_rows(bounds: np.ndarray, low: np.ndarray, high: np.ndarray,
                         light: np.ndarray, sigma: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    The light in each row between *bounds* of intervals from *low* to *high*,
    each holding *light* spread evenly over it and blurred by a Gaussian of
    *sigma* convolved with a rectangle *box* wide, by the exact integrals.
    Only the rows the blur reaches from an interval are worked out.
    """
    order = np.argsort(bounds[:, 0])
    row_low, row_high = bounds[order, 0], bounds[order, 1]
    if not np.allclose(row_high[:-1], row_low[1:], rtol=1e-12, atol=0.0):
        raise ValueError("The rows must meet, each starting where the one before ends.")
    share = light_onto_pixels(low, high, np.append(row_low, row_high[-1]), sigma, box)
    rows = np.zeros(len(bounds))
    rows[order] = share @ np.asarray(light, dtype=float)
    return rows


def _check_margin(margin: int, lit_only: bool) -> None:
    if margin < 0:
        raise ValueError(f"margin is a number of rows, and cannot be negative, got {margin}.")
    if margin and lit_only:
        raise ValueError(
            "A margin is for the spectral blur, and the baffle comes after it: "
            "lay the spectrum with lit_only=False, blur it, and empty the dark "
            "rows then."
        )


def _photons_per_energy(telescope, wavelength: u.Quantity) -> u.Quantity:
    """
    :func:`~euvst_response.radiometric.photons_per_energy` at each wavelength.

    Outside its throughput tables the telescope has no effective area, and a
    single such wavelength would turn every row of the frame to NaN through
    the spectral blur, so that is an error here rather than a silent result.
    """
    wavelength = np.atleast_1d(u.Quantity(wavelength))
    collected = photons_per_energy(telescope, wavelength)
    if not np.all(np.isfinite(collected)):
        outside = wavelength[~np.isfinite(collected)].to_value(u.Angstrom)
        raise ValueError(
            f"The telescope has no effective area at {outside.min():.4f} to "
            f"{outside.max():.4f} Angstrom ({outside.size} wavelengths); the "
            f"spectrum must stay within its throughput tables."
        )
    return collected


def photons_from_lines(focal_plane: FocalPlane_SWC, ccd: str, telescope,
                       slit_width: u.Quantity, wavelengths: u.Quantity,
                       intensities: u.Quantity, widths: u.Quantity,
                       column=None, lit_only: bool = True, margin: int = 0, *,
                       det=None, spectral_psf: Optional[str] = None) -> u.Quantity:
    """
    Photons per second in each row of one CCD from a list of emission lines.

    With *spectral_psf*, each line is blurred by the instrument's spectral
    response as it is laid onto the rows, which keeps a line narrower than a
    row where it is; blurring the rows after, with
    :func:`apply_spectral_psf`, moved it toward the middle of its row.

    Parameters
    ----------
    focal_plane : FocalPlane_SWC
        Which wavelength each row records.
    ccd : str
        ``'left'`` or ``'right'``.
    telescope : Telescope_EUVST
        Supplies the effective area, asked for an array of wavelengths at once.
    slit_width : u.Quantity
        The slit the light came through, as an angle.
    wavelengths : u.Quantity
        Rest wavelengths of the lines.
    intensities : u.Quantity
        Spectrally integrated radiance of each line, in erg / (s cm2 sr).
    widths : u.Quantity
        Gaussian 1-sigma width of each line, as a wavelength.  This is the line
        as emitted: thermal plus any non-thermal broadening, without the
        instrument's spectral response.
    column : int, optional
        Which column along the slit, for a focal plane whose lines drift along
        it.  Ignored otherwise.
    lit_only : bool
        Zero the rows the baffle keeps dark.  They still pass charge, so the
        smear model needs them present but empty.  The baffle comes after the
        grating, so for rows that are to be blurred with
        :func:`apply_spectral_psf`, leave this off and empty those rows after.
    margin : int
        Rows to add past each end of the CCD, for :func:`apply_spectral_psf`:
        light just off the chip is blurred onto its edge rows.  Needs
        ``lit_only=False``.  Not needed with *spectral_psf*, where the light
        of a line off the chip is blurred onto the edge rows as it is laid.
    det : Detector_SWC, optional
        The detector, for the spectral response in rows.  Needed with
        *spectral_psf*.
    spectral_psf : str, optional
        ``"quadrature"`` or ``"convolution"``, as in the configuration: blur
        the lines with the spectral response a synthesis through the same
        slit gets from
        :func:`~euvst_response.radiometric.apply_focusing_optics_psf`.
        None, the default, lays the lines as emitted.

    Returns
    -------
    u.Quantity
        Photons per second per pixel, one value per row, and then *margin*
        more past each end.
    """
    _check_margin(margin, lit_only)
    if spectral_psf is not None and det is None:
        raise ValueError("The spectral response is in detector rows: give det with "
                         "spectral_psf.")
    wavelengths = np.atleast_1d(u.Quantity(wavelengths).to(u.Angstrom))
    intensities = np.atleast_1d(u.Quantity(intensities).to(u.erg / (u.s * u.cm**2 * u.sr)))
    widths = np.atleast_1d(u.Quantity(widths).to(u.Angstrom))
    if not (len(wavelengths) == len(intensities) == len(widths)):
        raise ValueError(
            f"Each line needs a wavelength, an intensity and a width, got "
            f"{len(wavelengths)}, {len(intensities)} and {len(widths)}."
        )
    if np.any(widths.value <= 0):
        raise ValueError("Line widths must be positive.")

    solid_angle = pixel_solid_angle(focal_plane, slit_width)
    # Photons per second per pixel the line would give if all of it landed in
    # one row; the Gaussian below shares that out between the rows it covers.
    total = (intensities * solid_angle * _photons_per_energy(telescope, wavelengths)).to(1 / u.s)

    bounds = _row_bounds(focal_plane, ccd, column, margin)
    rows = np.zeros(len(bounds))
    scale = np.sqrt(2.0) * widths.to_value(u.Angstrom)
    centre = wavelengths.to_value(u.Angstrom)
    if spectral_psf is None:
        for i, weight in enumerate(total.value):
            if weight == 0:
                continue
            # Fraction of the line between each pair of row boundaries.
            low = erf((bounds[:, 0] - centre[i]) / scale[i])
            high = erf((bounds[:, 1] - centre[i]) / scale[i])
            rows += weight * 0.5 * (high - low)
    else:
        # The line, a Gaussian, blurred by the response: a Gaussian of the two
        # widths in quadrature, convolved with the slit's image for
        # "convolution", and shared between the rows by its exact integrals.
        psf_sigma, box = _response(focal_plane, ccd, column, telescope, det, slit_width,
                                   spectral_psf, centre)
        sigma = np.hypot(widths.to_value(u.Angstrom), psf_sigma)
        for i, weight in enumerate(total.value):
            if weight == 0:
                continue
            share = (_blurred_antiderivative(bounds[:, 1] - centre[i], sigma[i], box[i], 1)
                     - _blurred_antiderivative(bounds[:, 0] - centre[i], sigma[i], box[i], 1))
            rows += weight * np.maximum(share, 0.0)

    if lit_only:
        first, last = focal_plane.lit_rows(ccd, column)
        rows[:first] = 0.0
        rows[last + 1:] = 0.0
    return rows / u.s


def photons_from_spectrum(focal_plane: FocalPlane_SWC, ccd: str, telescope,
                          slit_width: u.Quantity, wavelength: u.Quantity,
                          radiance: u.Quantity, column=None,
                          lit_only: bool = True, margin: int = 0, *,
                          det=None, spectral_psf: Optional[str] = None) -> u.Quantity:
    """
    Photons per second in each row of one CCD from a continuous spectrum.

    Use this for a continuum, or for anything already sampled on a wavelength
    grid.  The grid has to be finer than a row, since the radiance is
    integrated between the row boundaries by trapezium rule on it: each
    interval of the grid holds the light the rule gives it, spread evenly
    over it.  With *spectral_psf* that light is blurred by the instrument's
    spectral response as it is laid onto the rows, as
    :func:`photons_from_lines` blurs a line.

    Parameters
    ----------
    wavelength : u.Quantity
        Grid the spectrum is sampled on, increasing.
    radiance : u.Quantity
        Spectral radiance per unit wavelength, in erg / (s cm2 sr Angstrom).

    The other parameters are those of :func:`photons_from_lines`.

    Returns
    -------
    u.Quantity
        Photons per second per pixel, one value per row, and then *margin*
        more past each end.
    """
    _check_margin(margin, lit_only)
    if spectral_psf is not None and det is None:
        raise ValueError("The spectral response is in detector rows: give det with "
                         "spectral_psf.")
    wavelength = u.Quantity(wavelength).to(u.Angstrom)
    radiance = u.Quantity(radiance).to(u.erg / (u.s * u.cm**2 * u.sr * u.Angstrom))
    if wavelength.size != radiance.size:
        raise ValueError(
            f"The spectrum needs one radiance per wavelength, got "
            f"{radiance.size} for {wavelength.size}."
        )
    if np.any(np.diff(wavelength.value) <= 0):
        raise ValueError("The wavelength grid must increase.")

    solid_angle = pixel_solid_angle(focal_plane, slit_width)
    # Photons per second per pixel per Angstrom, on the input grid.
    density = (radiance * solid_angle * _photons_per_energy(telescope, wavelength)).to_value(
        1 / (u.s * u.Angstrom))

    grid = wavelength.to_value(u.Angstrom)
    bounds = _row_bounds(focal_plane, ccd, column, margin)
    low, high = grid[:-1], grid[1:]
    if spectral_psf is None:
        sigma = box = np.zeros(low.size)
    else:
        sigma, box = _response(focal_plane, ccd, column, telescope, det, slit_width,
                               spectral_psf, (low + high) / 2)
    rows = _intervals_onto_rows(bounds, low, high, np.diff(grid) * (density[1:] + density[:-1]) / 2,
                                sigma, box)

    if lit_only:
        first, last = focal_plane.lit_rows(ccd, column)
        rows[:first] = 0.0
        rows[last + 1:] = 0.0
    return rows / u.s


def thermal_width(wavelength: u.Quantity, temperature: u.Quantity,
                  atomic_weight: u.Quantity,
                  non_thermal: Optional[u.Quantity] = None) -> u.Quantity:
    """
    Gaussian 1-sigma width of a line, as a wavelength.

    Thermal motion at *temperature* for an ion of *atomic_weight*, with an
    optional non-thermal speed added in quadrature.
    """
    speed = np.sqrt(const.k_B * u.Quantity(temperature) / u.Quantity(atomic_weight).to(u.g))
    if non_thermal is not None:
        speed = np.sqrt(speed**2 + u.Quantity(non_thermal).to(u.cm / u.s)**2)
    return (u.Quantity(wavelength) * speed / const.c).to(u.Angstrom)


def apply_spectral_psf(rows: u.Quantity, telescope, det, slit_width: u.Quantity,
                       spectral_psf: str = "quadrature", margin: int = 0) -> u.Quantity:
    """
    Blur a frame's rows with the instrument's spectral response.

    Along the dispersion a pixel is a row, and the response is the one
    :func:`~euvst_response.radiometric.apply_focusing_optics_psf` gives a
    synthesis through the same slit: with *spectral_psf* ``"quadrature"`` a
    Gaussian of :func:`~euvst_response.radiometric.spectral_psf_fwhm`, and
    with ``"convolution"``
    :func:`~euvst_response.radiometric.spectral_line_spread`, on the same
    rows either way. The slit's image is part of the line profile, so a
    wider slit gives a wider response: 2.54 rows of FWHM for the 0.2 arcsec
    slit and 3.35 for the 0.4 arcsec one.

    Nothing is assumed beyond the rows given: light blurred past either end is
    lost, as off the edge of a detector, and none comes in.  The edge rows of
    a CCD do receive light from just off the chip, though, and at the butted
    edge the spectrum carries on across the gap.  To include it, lay the
    spectrum with a *margin* of rows past each end
    (:func:`photons_from_lines` and :func:`photons_from_spectrum` take one),
    at least :func:`~euvst_response.radiometric.spectral_psf_reach` of them,
    and pass the same margin here: the result is then the chip's own rows.
    The kernel is normalised, so any flux that stays within the rows is
    conserved.

    .. deprecated:: 0.12.0
        Blurring the rows once the light is on them moves a line narrower
        than a row toward the middle of its row, by up to a few km/s. Give
        *spectral_psf* and *det* to :func:`photons_from_lines` or
        :func:`photons_from_spectrum` instead, which blur the light as it is
        laid onto the rows.
    """
    warnings.warn(
        "apply_spectral_psf is deprecated and will be removed in a future release: blurring "
        "the rows once the light is on them moves a line narrower than a row toward the "
        "middle of its row. Give spectral_psf and det to photons_from_lines or "
        "photons_from_spectrum instead.", FutureWarning, stacklevel=2)
    margin = int(margin)
    if spectral_psf == "quadrature":
        sigma = _fwhm_to_sigma(spectral_psf_fwhm(telescope, det, slit_width))
        reach = spectral_psf_reach(telescope, det, slit_width)
        offsets = np.arange(-reach, reach + 1)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    elif spectral_psf == "convolution":
        kernel = spectral_line_spread(telescope, det, slit_width)
    else:
        raise ValueError(
            f"spectral_psf must be 'quadrature' or 'convolution', got {spectral_psf!r}."
        )
    kernel = kernel / kernel.sum()
    half = kernel.size // 2
    if margin and margin < half:
        raise ValueError(
            f"A margin of {margin} rows is less than the {half} the spectral "
            f"response reaches, so the edge rows would miss light."
        )
    unit = rows.unit if hasattr(rows, "unit") else 1
    values = np.asarray(rows.value if hasattr(rows, "value") else rows, dtype=float)
    if values.size <= 2 * margin:
        raise ValueError(
            f"{values.size} rows leave nothing inside a margin of {margin} at each end."
        )
    blurred = np.convolve(np.pad(values, half), kernel, mode="valid")
    return blurred[margin:values.size - margin] * unit


def expose_with_wavelength(rate: np.ndarray, wavelength: u.Quantity, exposure: u.Quantity,
                           sequence) -> tuple:
    """
    Photons in each pixel of a frame, as :func:`~euvst_response.readout.expose`
    gives them, and the wavelength that carries their mean energy.

    Without a shutter a pixel holds photons from every row its charge crossed,
    and :func:`detect` needs their mean energy to turn them into electrons.
    ``expose`` is linear in the rate, so exposing the energy-weighted rate
    gives the energy each pixel holds, and dividing by the photons gives the
    mean.  A pixel with no photons keeps its own row's wavelength, and a
    parallel overscan pixel with none the last image row's.

    Parameters
    ----------
    rate : np.ndarray
        Photons per second reaching each pixel of one CCD, ``(n_rows, n_columns)``.
    wavelength : u.Quantity
        The wavelength each row records, one per row.
    exposure, sequence
        As for ``expose``.

    Returns
    -------
    photons : np.ndarray
        Photons per pixel, ``(n_rows + parallel_overscan_rows, n_columns)``.
    wavelength : u.Quantity
        The wavelength of their mean energy, the same shape.
    """
    rate = np.asarray(rate, dtype=float)
    own = np.asarray(u.Quantity(wavelength).to_value(u.Angstrom), dtype=float)
    if rate.ndim != 2 or own.shape != (rate.shape[0],):
        raise ValueError(
            f"One wavelength per row: got {own.shape} for a rate of {rate.shape}."
        )
    hc = (const.h * const.c).to_value(u.erg * u.Angstrom)
    photons = expose(rate, exposure, sequence)
    energy = expose(rate * (hc / own)[:, np.newaxis], exposure, sequence)
    fallback = np.concatenate([own, np.full(photons.shape[0] - own.size, own[-1])])
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(photons > 0, hc * photons / energy, np.nan)
    return photons, np.where(np.isfinite(mean), mean, fallback[:, np.newaxis]) * u.Angstrom


def detect(photons: np.ndarray, wavelength: u.Quantity, dark_time: u.Quantity, det,
           *, noise: bool = True) -> np.ndarray:
    """
    Electrons per pixel from the photons a frame recorded.

    The stages are those of :func:`euvst_response.radiometric.to_electrons`:
    the quantum efficiency, the electrons each photon liberates with their
    Fano spread, the dark current, and the read noise.  A frame spans the
    band, so each row has its own photon energy, and without a shutter each
    row waits a different time for its read-out, so each has its own dark
    current.

    Parameters
    ----------
    photons : np.ndarray
        Photons recorded per pixel, shaped (rows, columns).  With *noise* on
        these must be whole numbers, as from a Poisson draw.
    wavelength : u.Quantity
        The wavelength of the photons in each row, one per row, or one per
        pixel as an array the shape of *photons*.  Without a shutter a pixel
        holds photons from every row its charge crossed, and the per-pixel
        form takes the wavelength that carries their mean energy, as
        :func:`expose_with_wavelength` gives it.
    dark_time : u.Quantity
        How long each row collects dark current, one per row or one for all.
        :func:`euvst_response.readout.dark_current_time` gives it.
    det : Detector_SWC
        The detector.
    noise : bool
        With it off every random draw is replaced by its mean.

    Returns
    -------
    np.ndarray
        Electrons per pixel, the same shape as *photons*.
    """
    photons = np.asarray(photons, dtype=float)
    if photons.ndim != 2:
        raise ValueError(f"A frame has two axes, rows and columns, not {photons.ndim}.")
    wavelength = np.asarray(u.Quantity(wavelength).to_value(u.Angstrom), dtype=float)
    if wavelength.ndim == 1 and wavelength.size == photons.shape[0]:
        wavelength = wavelength[:, np.newaxis]
    elif wavelength.shape != photons.shape:
        raise ValueError(
            f"One wavelength per row or per pixel: got {wavelength.shape} for a "
            f"frame of {photons.shape}."
        )
    if noise:
        if not np.all(np.mod(photons, 1) == 0):
            raise ValueError("With noise on the photons must be whole numbers, as from a Poisson draw.")
        detected = np.random.binomial(photons.astype(np.int64), det.qe_euv).astype(float)
    else:
        detected = photons * det.qe_euv
    electrons = _vectorized_fano_noise(detected, wavelength * u.Angstrom, det, noise=noise)

    dark = (det.dark_current * u.Quantity(dark_time)).to_value(u.electron / u.pixel)
    dark = np.broadcast_to(np.reshape(dark, (-1, 1)) if np.ndim(dark) else dark, photons.shape)
    if noise:
        electrons = (electrons + np.random.poisson(dark)
                     + np.random.normal(0.0, det.read_noise_rms.to_value(u.electron / u.pixel),
                                        photons.shape))
    else:
        electrons = electrons + dark
    # Not clipped at zero, as a CCD reads out above a bias level; see
    # radiometric.to_electrons.
    return electrons


def digitise(electrons: np.ndarray, det) -> np.ndarray:
    """
    DN per pixel: the electrons divided by the gain, rounded, and clipped at
    the detector's maximum, as :func:`euvst_response.radiometric.to_dn` does.
    """
    dn = np.asarray(electrons, dtype=float) / det.gain_e_per_dn.to_value(u.electron / u.DN)
    return np.minimum(np.round(dn), det.max_dn.to_value(u.DN / u.pixel))
