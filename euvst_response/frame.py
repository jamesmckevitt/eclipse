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
the line as the Sun emits it; the instrument's own spectral response is a
separate convolution, applied with :func:`apply_spectral_psf`.

At the other end, :func:`detect` and :func:`digitise` take the photons a frame
recorded through the detector stages of :mod:`euvst_response.radiometric`,
with the two things a full-band frame needs that a cube does not: a photon
energy for each row, and a dark current time for each row.
"""

from __future__ import annotations

from typing import Optional

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.special import erf

from .radiometric import (
    _vectorized_fano_noise,
    spectral_line_spread,
    spectral_psf_fwhm,
    spectral_psf_reach,
)
from .readout import FocalPlane_SWC
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


def _row_bounds(focal_plane: FocalPlane_SWC, ccd: str, column=None) -> np.ndarray:
    """Row boundaries in Angstrom, sorted so that the first is the shorter."""
    edges = focal_plane.row_edges(ccd, column).to_value(u.Angstrom)
    return np.column_stack([np.minimum(edges[:-1], edges[1:]),
                            np.maximum(edges[:-1], edges[1:])])


def _collecting(telescope, wavelength: u.Quantity) -> np.ndarray:
    """
    Effective area in cm^2 at each wavelength, from the telescope model.

    The telescope is asked for every wavelength at once, and may answer with
    one area for all of them.  Outside its throughput tables it has no
    effective area, and a single such wavelength would turn every row of the
    frame to NaN through the spectral blur, so that is an error here rather
    than a silent result.
    """
    wavelength = np.atleast_1d(wavelength)
    area = u.Quantity(telescope.ea_and_throughput(wavelength)).cgs.value
    area = np.array(np.broadcast_to(area, wavelength.shape), dtype=float)
    if not np.all(np.isfinite(area)):
        outside = wavelength[~np.isfinite(area)].to_value(u.Angstrom)
        raise ValueError(
            f"The telescope has no effective area at {outside.min():.4f} to "
            f"{outside.max():.4f} Angstrom ({outside.size} wavelengths); the "
            f"spectrum must stay within its throughput tables."
        )
    return area


def photons_from_lines(focal_plane: FocalPlane_SWC, ccd: str, telescope,
                       slit_width: u.Quantity, wavelengths: u.Quantity,
                       intensities: u.Quantity, widths: u.Quantity,
                       column=None, lit_only: bool = True) -> u.Quantity:
    """
    Photons per second in each row of one CCD from a list of emission lines.

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
        smear model needs them present but empty.

    Returns
    -------
    u.Quantity
        Photons per second per pixel, one value per row.
    """
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
    energy = (const.h * const.c / wavelengths).to(u.erg)
    area = _collecting(telescope, wavelengths) * u.cm**2
    # Photons per second per pixel the line would give if all of it landed in
    # one row; the Gaussian below shares that out between the rows it covers.
    total = (intensities * solid_angle * area / energy).to(1 / u.s)

    bounds = _row_bounds(focal_plane, ccd, column)
    rows = np.zeros(focal_plane.n_rows)
    scale = np.sqrt(2.0) * widths.to_value(u.Angstrom)
    centre = wavelengths.to_value(u.Angstrom)
    for i, weight in enumerate(total.value):
        if weight == 0:
            continue
        # Fraction of the line between each pair of row boundaries.
        low = erf((bounds[:, 0] - centre[i]) / scale[i])
        high = erf((bounds[:, 1] - centre[i]) / scale[i])
        rows += weight * 0.5 * (high - low)

    if lit_only:
        first, last = focal_plane.lit_rows(ccd, column)
        rows[:first] = 0.0
        rows[last + 1:] = 0.0
    return rows / u.s


def photons_from_spectrum(focal_plane: FocalPlane_SWC, ccd: str, telescope,
                          slit_width: u.Quantity, wavelength: u.Quantity,
                          radiance: u.Quantity, column=None,
                          lit_only: bool = True) -> u.Quantity:
    """
    Photons per second in each row of one CCD from a continuous spectrum.

    Use this for a continuum, or for anything already sampled on a wavelength
    grid.  The grid has to be finer than a row, since the radiance is
    integrated between the row boundaries by trapezium rule on it.

    Parameters
    ----------
    wavelength : u.Quantity
        Grid the spectrum is sampled on, increasing.
    radiance : u.Quantity
        Spectral radiance per unit wavelength, in erg / (s cm2 sr Angstrom).

    Returns
    -------
    u.Quantity
        Photons per second per pixel, one value per row.
    """
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
    energy = (const.h * const.c / wavelength).to(u.erg)
    area = _collecting(telescope, wavelength) * u.cm**2
    # Photons per second per pixel per Angstrom, on the input grid.
    density = (radiance * solid_angle * area / energy).to(1 / (u.s * u.Angstrom)).value

    grid = wavelength.to_value(u.Angstrom)
    cumulative = np.concatenate([[0.0], np.cumsum(np.diff(grid) * (density[1:] + density[:-1]) / 2)])
    bounds = _row_bounds(focal_plane, ccd, column)
    rows = (np.interp(bounds[:, 1], grid, cumulative, left=cumulative[0], right=cumulative[-1])
            - np.interp(bounds[:, 0], grid, cumulative, left=cumulative[0], right=cumulative[-1]))

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
                       spectral_psf: str = "quadrature") -> u.Quantity:
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

    Flux is conserved: the kernel is normalised and the ends are padded by
    repeating the edge value, which is right here because a frame covers the
    whole band rather than a window with empty edges.
    """
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
    unit = rows.unit if hasattr(rows, "unit") else 1
    values = np.asarray(rows.value if hasattr(rows, "value") else rows, dtype=float)
    padded = np.pad(values, half, mode="edge")
    return np.convolve(padded, kernel, mode="valid") * unit


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
        form takes the wavelength that carries their mean energy.
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
    return np.maximum(electrons, 0.0)


def digitise(electrons: np.ndarray, det) -> np.ndarray:
    """
    DN per pixel: the electrons divided by the gain, rounded, and clipped at
    the detector's maximum, as :func:`euvst_response.radiometric.to_dn` does.
    """
    dn = np.asarray(electrons, dtype=float) / det.gain_e_per_dn.to_value(u.electron / u.DN)
    return np.minimum(np.round(dn), det.max_dn.to_value(u.DN / u.pixel))
