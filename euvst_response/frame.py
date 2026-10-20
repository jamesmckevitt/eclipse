"""
Laying a spectrum onto the SW detectors.

The light each CCD row receives from a solar spectrum, by the radiometric
equation::

    photons per second per pixel = I * Omega_pix * A_eff / E_photon

with the spectral radiance `I` integrated over each row's wavelengths, as
the rows are not evenly spaced and the effective area changes across the
band. Lines are integrated between the row boundaries, so their flux is
kept for any sampling, and the spectral response, if given, is applied as
they are laid on the rows. :func:`detect` and :func:`digitise` take a
frame's photons to electrons and DN.
"""

from __future__ import annotations

from typing import Optional

import astropy.constants as const
import astropy.units as u
import numpy as np

from .radiometric import (
    _vectorized_fano_noise,
    photons_per_energy,
    slit_image_width,
    spectral_blur_fwhm,
    spectral_line_spread,  # noqa: F401
    spectral_psf_fwhm,
    spectral_psf_reach,  # noqa: F401
)
from .deprecated import apply_spectral_psf  # noqa: F401  (deprecated)
from .readout import FocalPlane_SWC, expose, expose_serial_overscan
from .sampling import (_REACH_IN_SIGMA, _carry_photon_wavelengths, light_onto_pixels,
                       points_onto_pixels)
from .utils import angle_to_distance, _fwhm_to_sigma


def pixel_solid_angle(focal_plane: FocalPlane_SWC, slit_width: u.Quantity) -> u.Quantity:
    """
    Solid angle of the Sun seen by one pixel, in steradian.

    A pixel covers the plate scale along the slit and the slit width, which
    is the raster step, across it.
    """
    along_slit = angle_to_distance(focal_plane.plate_scale)
    across = angle_to_distance(u.Quantity(slit_width))
    return ((along_slit * across).cgs / const.au.cgs ** 2).value * u.sr


def _row_bounds(focal_plane: FocalPlane_SWC, ccd: str, column=None, margin: int = 0) -> np.ndarray:
    """
    Row boundaries in Angstrom, shorter wavelength first, for the CCD's rows
    and `margin` more past each end.
    """
    if margin:
        rows = np.arange(-margin, focal_plane.n_rows + margin + 1) - 0.5
        edges = focal_plane.wavelength(rows, ccd, column).to_value(u.Angstrom)
    else:
        edges = focal_plane.row_edges(ccd, column).to_value(u.Angstrom)
    return np.column_stack([np.minimum(edges[:-1], edges[1:]),
                            np.maximum(edges[:-1], edges[1:])])


# Number of cells per row a line is laid onto, by exact integrals of its
# Gaussian, so the telescope throughput and photon energy are taken at each
# cell's wavelength rather than once for the whole line.
CELLS_PER_ROW = 16

# Lines are laid in blocks of this many, to bound the memory a whole band
# takes. The result does not depend on it.
_LINES_AT_ONCE = 10000


def _sorted_rows(focal_plane: FocalPlane_SWC, ccd: str, column=None, margin: int = 0):
    """
    Row edges in increasing wavelength, in Angstrom, and the order that
    sorts the rows that way.
    """
    bounds = _row_bounds(focal_plane, ccd, column, margin)
    order = np.argsort(bounds[:, 0])
    low, high = bounds[order, 0], bounds[order, 1]
    if not np.allclose(high[:-1], low[1:], rtol=1e-12, atol=0.0):
        raise ValueError("The rows must meet, each starting where the one before ends.")
    return np.append(low, high[-1]), order


def _cells(focal_plane: FocalPlane_SWC, ccd: str, column, row_edges: np.ndarray,
           reach: float) -> np.ndarray:
    """
    Cell edges, `CELLS_PER_ROW` cells to the CCD's narrowest row, covering
    the rows and `reach` Angstrom past each end. The edges are whole steps
    from 0 Angstrom, so extending the range does not move them.
    """
    step = np.abs(np.diff(focal_plane.row_edges(ccd, column).to_value(u.Angstrom))).min()
    step /= CELLS_PER_ROW
    first = np.floor((row_edges[0] - reach) / step)
    last = np.ceil((row_edges[-1] + reach) / step)
    return step * np.arange(first, last + 1)


def _response(focal_plane: FocalPlane_SWC, ccd: str, column, telescope, det,
              slit_width: u.Quantity, spectral_psf: str, wavelength: np.ndarray):
    """
    Spectral response of the instrument at each `wavelength` (Angstrom).

    Returns the standard deviation of the Gaussian and the width of the
    slit's image it is convolved with, both in Angstrom. With `spectral_psf`
    `"quadrature"` the slit is included in the Gaussian and the width is 0.

    The response is set in rows, as
    :func:`~euvst_response.radiometric.apply_focusing_optics_psf` sets it for
    a synthesis, and converted at the width of the row each wavelength falls
    in. The row width changes by less than a part in 1e4 across the
    response.
    """
    if spectral_psf == "quadrature":
        sigma_rows = _fwhm_to_sigma(spectral_psf_fwhm(telescope, det, slit_width))
        box_rows = 0.0
    elif spectral_psf == "convolution":
        sigma_rows = _fwhm_to_sigma(spectral_blur_fwhm(telescope, det))
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
    Light in each row of `bounds` from intervals `low` to `high`.

    Each interval holds `light` spread evenly over it, blurred by a Gaussian
    of standard deviation `sigma` convolved with a rectangle `box` wide, and
    integrated exactly over the rows. Only the rows the blur reaches from an
    interval are computed.
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

    Raises ValueError for a wavelength outside the telescope's throughput
    tables, where it has no effective area: a single one would turn every
    row of the frame to NaN through the spectral blur.
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

    Each line is split between cells, `CELLS_PER_ROW` to a row, by exact
    integrals of its Gaussian. Each cell's light is converted to photons at
    the cell's wavelength, with the telescope's effective area there. With
    `spectral_psf`, the light is blurred by the instrument's spectral
    response as it is laid onto the rows, so a line narrower than a row
    stays where it is. Blurring the rows afterwards with
    :func:`apply_spectral_psf` moves it toward the middle of its row.

    Parameters
    ----------
    focal_plane : FocalPlane_SWC
        Wavelength each row records.
    ccd : str
        `'left'` or `'right'`.
    telescope : Telescope_EUVST
        Gives the effective area, for an array of wavelengths at once.
    slit_width : u.Quantity
        Width of the slit the light came through, as an angle.
    wavelengths : u.Quantity
        Rest wavelengths of the lines.
    intensities : u.Quantity
        Spectrally integrated radiance of each line, in erg / (s cm2 sr).
    widths : u.Quantity
        Gaussian 1-sigma width of each line as emitted, as a wavelength:
        thermal plus any non-thermal broadening, without the instrument's
        spectral response.
    column : int, optional
        Column along the slit, for a focal plane whose lines drift along it.
        Ignored otherwise.
    lit_only : bool
        Zero the rows the baffle keeps dark. They still pass charge, so the
        smear model needs them present but empty. Only for the deprecated
        :func:`apply_spectral_psf`, leave this off, blur, and empty the dark
        rows afterwards.
    margin : int
        Rows to add past each end of the CCD, only for the deprecated
        :func:`apply_spectral_psf`. Needs `lit_only=False`. Not needed with
        `spectral_psf`, which blurs the light of a line off the chip onto
        the edge rows as it is laid.
    det : Detector_SWC, optional
        Detector, for the spectral response in rows. Needed with
        `spectral_psf`.
    spectral_psf : str, optional
        `"quadrature"` or `"convolution"`, as in the configuration: blur the
        lines with the spectral response of
        :func:`~euvst_response.radiometric.apply_focusing_optics_psf` for a
        synthesis through the same slit. None, the default, lays the lines
        as emitted.

    Returns
    -------
    u.Quantity
        Photons per second per pixel, one per row, plus `margin` more past
        each end.
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

    # A line outside the telescope's throughput tables is an error, as is a
    # spectrum reaching beyond them.
    at_centre = _photons_per_energy(telescope, wavelengths).to_value(u.cm**2 / u.erg)
    edges, order = _sorted_rows(focal_plane, ccd, column, margin)
    centre, width = wavelengths.to_value(u.Angstrom), widths.to_value(u.Angstrom)
    light = (intensities * pixel_solid_angle(focal_plane, slit_width)).to_value(
        u.erg / (u.s * u.cm**2))
    if spectral_psf is None:
        psf_sigma = box = np.zeros(centre.size)
    else:
        psf_sigma, box = _response(focal_plane, ccd, column, telescope, det, slit_width,
                                   spectral_psf, centre)
    reach = (_REACH_IN_SIGMA * psf_sigma.max() + box.max()
             if spectral_psf is not None and centre.size else 0.0)
    cells = _cells(focal_plane, ccd, column, edges, reach)
    sorted_rows = np.zeros(edges.size - 1)
    in_cells, at_centres = np.zeros(cells.size - 1), np.zeros(cells.size - 1)
    for start in range(0, centre.size, _LINES_AT_ONCE):
        block = slice(start, start + _LINES_AT_ONCE)
        collected = light[block] * at_centre[block]
        # Each line blurred by the response is a Gaussian of the two widths
        # in quadrature, convolved with the slit's image for "convolution".
        # Share it between the rows by exact integrals, in photons at the
        # line centre's wavelength.
        sorted_rows += points_onto_pixels(
            centre[block], edges, np.hypot(width[block], psf_sigma[block]), box[block]
        ) @ collected
        masses = points_onto_pixels(centre[block], cells, width[block])
        in_cells += masses @ light[block]
        at_centres += masses @ collected
    # Then correct, cell by cell across each line, for the throughput and
    # photon energy at each cell's wavelength rather than the line centre's,
    # with the same blur as the line.
    lit = np.flatnonzero(in_cells > 0)
    low, high = cells[lit], cells[lit + 1]
    middle = (low + high) / 2
    own = np.nan_to_num(photons_per_energy(telescope, middle * u.Angstrom).to_value(
        u.cm**2 / u.erg))
    change = in_cells[lit] * own - at_centres[lit]
    if spectral_psf is None:
        sigma = cell_box = 0.0
    else:
        sigma, cell_box = _response(focal_plane, ccd, column, telescope, det, slit_width,
                                    spectral_psf, middle)
    sorted_rows += light_onto_pixels(low, high, edges, sigma, cell_box) @ change
    rows = np.zeros(edges.size - 1)
    rows[order] = sorted_rows

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

    Use this for a continuum, or anything already sampled on a wavelength
    grid. The grid must be finer than a row, since the radiance is
    integrated between the row boundaries by the trapezium rule on it: each
    grid interval holds its trapezium-rule light, spread evenly over it.
    With `spectral_psf` that light is blurred by the instrument's spectral
    response as it is laid onto the rows, as in :func:`photons_from_lines`.

    Parameters
    ----------
    wavelength : u.Quantity
        Grid the spectrum is sampled on, increasing.
    radiance : u.Quantity
        Spectral radiance per unit wavelength, in erg / (s cm2 sr Angstrom).

    Other parameters as for :func:`photons_from_lines`.

    Returns
    -------
    u.Quantity
        Photons per second per pixel, one per row, plus `margin` more past
        each end.
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

    Thermal motion at `temperature` for an ion of `atomic_weight`, with the
    optional non-thermal speed `non_thermal` added in quadrature.
    """
    speed = np.sqrt(const.k_B * u.Quantity(temperature) / u.Quantity(atomic_weight).to(u.g))
    if non_thermal is not None:
        speed = np.sqrt(speed**2 + u.Quantity(non_thermal).to(u.cm / u.s)**2)
    return (u.Quantity(wavelength) * speed / const.c).to(u.Angstrom)


def expose_with_wavelength(rate: np.ndarray, wavelength: u.Quantity, exposure: u.Quantity,
                           sequence) -> tuple:
    """
    Photons in each pixel of a frame and the wavelength of their mean energy.

    The photons are as from :func:`~euvst_response.readout.expose`. This is
    :func:`expose_with_wavelengths` without the spread of photon energies,
    which :func:`detect` also takes.

    Parameters
    ----------
    rate : np.ndarray
        Photons per second reaching each pixel of one CCD, `(n_rows, n_columns)`.
    wavelength : u.Quantity
        Wavelength of the mean energy of the photons reaching each row (one
        per row) or each pixel (same shape as `rate`).
    exposure, sequence
        As for `expose`.

    Returns
    -------
    photons : np.ndarray
        Photons per pixel, `(n_rows + parallel_overscan_rows, n_columns)`.
    wavelength : u.Quantity
        Wavelength of their mean energy, same shape.
    """
    photons, mean, _ = expose_with_wavelengths(rate, wavelength, exposure, sequence)
    return photons, mean


def expose_with_wavelengths(rate: np.ndarray, wavelength: u.Quantity, exposure: u.Quantity,
                            sequence, rms_wavelength: Optional[u.Quantity] = None) -> tuple:
    """
    Photons in each pixel of a frame and the wavelengths of their mean and
    root mean square energies.

    The photons are as from :func:`~euvst_response.readout.expose`. Without
    a shutter a pixel holds photons from every row its charge crossed.
    :func:`detect` needs their mean energy to convert them to electrons, and
    their root mean square energy for the spread in electrons from photons
    of different energies. `expose` is linear in the rate, so exposing the
    rate weighted by the energy, and by its square, gives each pixel's total
    energy and squared energy; dividing by the photons gives the means. A
    pixel with no photons keeps its given wavelength, and a parallel
    overscan pixel with none takes the last image row's.

    Parameters
    ----------
    rate : np.ndarray
        Photons per second reaching each pixel of one CCD, `(n_rows, n_columns)`.
    wavelength : u.Quantity
        Wavelength of the mean energy of the photons reaching each row (one
        per row) or each pixel (same shape as `rate`).
    exposure, sequence
        As for `expose`.
    rms_wavelength : u.Quantity, optional
        Wavelength of the root mean square energy of the photons reaching
        each row or pixel, shaped as `wavelength`. Default `wavelength`, for
        photons of one energy.

    Returns
    -------
    photons : np.ndarray
        Photons per pixel, `(n_rows + parallel_overscan_rows, n_columns)`.
    wavelength, rms_wavelength : u.Quantity
        Wavelengths of their mean and root mean square energies, same shape.
    """
    rate = np.asarray(rate, dtype=float)
    if rate.ndim != 2:
        raise ValueError(f"A frame has two axes, rows and columns, not {rate.ndim}.")
    own = _per_pixel(wavelength, rate.shape)
    rms = own if rms_wavelength is None else _per_pixel(rms_wavelength, rate.shape)
    n_more = sequence.parallel_overscan_rows
    fallback = np.concatenate([own, np.repeat(own[-1:], n_more, axis=0)])
    photons, mean, rms = _carry_photon_wavelengths(
        lambda values: expose(values, exposure, sequence), rate, own, rms, fallback)
    return photons, mean * u.Angstrom, rms * u.Angstrom


def expose_serial_overscan_with_wavelengths(rate: np.ndarray, wavelength: u.Quantity,
                                            exposure: u.Quantity, sequence,
                                            rms_wavelength: Optional[u.Quantity] = None) -> tuple:
    """
    Photons the serial register carries into the overscan, with the
    wavelengths of their mean and root mean square energies.

    As :func:`expose_with_wavelengths`, for the samples of
    :func:`~euvst_response.readout.expose_serial_overscan`. A sample no
    photons reach takes the wavelength of the image pixel next to it.

    Returns
    -------
    photons : np.ndarray
        Photons per sample, `(n_rows + parallel_overscan_rows, 2 * serial_overscan)`.
    wavelength, rms_wavelength : u.Quantity
        Wavelengths of their mean and root mean square energies, same shape.
    """
    rate = np.asarray(rate, dtype=float)
    if rate.ndim != 2:
        raise ValueError(f"A frame has two axes, rows and columns, not {rate.ndim}.")
    own = _per_pixel(wavelength, rate.shape)
    rms = own if rms_wavelength is None else _per_pixel(rms_wavelength, rate.shape)
    n_more, overscan = sequence.parallel_overscan_rows, sequence.serial_overscan
    # The image columns next to each half's overscan. A CCD one column wide
    # has only the first half, whose column stands in for both.
    half = (rate.shape[1] + 1) // 2
    edges = np.concatenate([own, np.repeat(own[-1:], n_more, axis=0)])[
        :, [half - 1, min(half, rate.shape[1] - 1)]]
    fallback = np.repeat(edges, overscan, axis=1)
    photons, mean, rms = _carry_photon_wavelengths(
        lambda values: expose_serial_overscan(values, exposure, sequence), rate, own, rms,
        fallback)
    return photons, mean * u.Angstrom, rms * u.Angstrom


def _per_pixel(wavelength: u.Quantity, shape: tuple) -> np.ndarray:
    """Wavelengths per row or per pixel, broadcast to a frame of `shape`, in Angstrom."""
    values = np.asarray(u.Quantity(wavelength).to_value(u.Angstrom), dtype=float)
    if values.shape not in ((shape[0],), shape):
        raise ValueError(
            f"One wavelength per row or per pixel: got {values.shape} for a frame of "
            f"{shape}."
        )
    return np.broadcast_to(values[:, np.newaxis] if values.ndim == 1 else values, shape)


def detect(photons: np.ndarray, wavelength: u.Quantity, dark_time: u.Quantity, det,
           *, noise: bool = True, rms_wavelength: Optional[u.Quantity] = None) -> np.ndarray:
    """
    Electrons per pixel from the photons a frame recorded.

    The stages are those of :func:`euvst_response.radiometric.to_electrons`:
    quantum efficiency, electrons per photon with their Fano spread, dark
    current and read noise. A frame spans the band, so each row has its own
    photon energy. Without a shutter each row waits a different time to be
    read out, so each row has its own dark current.

    Parameters
    ----------
    photons : np.ndarray
        Photons recorded per pixel, shape (rows, columns). With `noise` on,
        these must be whole numbers, as from a Poisson draw.
    wavelength : u.Quantity
        Wavelength of the photons, one per row or one per pixel (same shape
        as `photons`). Without a shutter a pixel holds photons from every
        row its charge crossed; per pixel, give the wavelength of their mean
        energy, as from :func:`expose_with_wavelengths`.
    dark_time : u.Quantity
        Time each row collects dark current, one per row or one for all, as
        from :func:`euvst_response.readout.dark_current_time`, or one per
        pixel, same shape as `photons`.
    det : Detector_SWC
        Detector.
    noise : bool
        If False, every random draw is replaced by its mean.
    rms_wavelength : u.Quantity, optional
        Wavelength of the root mean square energy of the photons in each row
        or pixel, shaped as `wavelength`, as from
        :func:`expose_with_wavelengths`. Photons of different energies free
        different numbers of electrons, which widens the spread of a pixel's
        electrons. Default `wavelength`, for photons of one energy.

    Returns
    -------
    np.ndarray
        Electrons per pixel, same shape as `photons`.
    """
    photons = np.asarray(photons, dtype=float)
    if photons.ndim != 2:
        raise ValueError(f"A frame has two axes, rows and columns, not {photons.ndim}.")
    wavelength = _per_pixel(wavelength, photons.shape)
    rms = wavelength if rms_wavelength is None else _per_pixel(rms_wavelength, photons.shape)
    if noise:
        if not np.all(np.mod(photons, 1) == 0):
            raise ValueError("With noise on the photons must be whole numbers, as from a Poisson draw.")
        detected = np.random.binomial(photons.astype(np.int64), det.qe_euv).astype(float)
    else:
        detected = photons * det.qe_euv
    electrons = _vectorized_fano_noise(detected, wavelength * u.Angstrom, det, noise=noise,
                                       rms_wavelength=rms * u.Angstrom)

    dark = (det.dark_current * u.Quantity(dark_time)).to_value(u.electron / u.pixel)
    if np.ndim(dark) == 1:
        dark = np.reshape(dark, (-1, 1))
    dark = np.broadcast_to(dark, photons.shape)
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
    DN per pixel: electrons divided by the gain, rounded and clipped at the
    detector's maximum, as in :func:`euvst_response.radiometric.to_dn`.
    """
    dn = np.asarray(electrons, dtype=float) / det.gain_e_per_dn.to_value(u.electron / u.DN)
    return np.minimum(np.round(dn), det.max_dn.to_value(u.DN / u.pixel))
