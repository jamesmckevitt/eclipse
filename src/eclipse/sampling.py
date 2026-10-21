"""
Laying a scene onto detector pixels.

A pixel records the mean over its extent of the cells' radiance, each cell
uniform, blurred by the optics. The weight of cell b in pixel p is

    W[p, b] = (1 / |p|) int_p dx int_b dx' h(x - x')

for blur h, along each axis in turn. For a Gaussian blur, alone or
convolved with a rectangle such as a slit's image, these integrals have
closed forms, so the weights are exact for any cell, pixel and blur sizes.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import erfc

# A Gaussian's weight beyond this many sigma is below 1e-30 of its total.
_REACH_IN_SIGMA = 12.0


def _antiderivative(u, sigma, order: int) -> np.ndarray:
    """
    Antiderivative of order `order` of a unit Gaussian of standard deviation
    `sigma`, zero at minus infinity.

    `order` is 1 (the cumulative distribution), 2 or 3. Where `sigma` is 0
    the Gaussian is a unit spike. `sigma` is one value or one per element of
    `u`.
    """
    u = np.asarray(u, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), u.shape)
    spike = sigma == 0.0
    s = np.where(spike, 1.0, sigma)
    cdf = 0.5 * erfc(-u / (np.sqrt(2.0) * s))
    if order == 1:
        smooth = cdf
        sharp = np.where(u > 0, 1.0, np.where(u == 0, 0.5, 0.0))
    else:
        density = np.exp(-0.5 * (u / s) ** 2) / (np.sqrt(2.0 * np.pi) * s)
        positive = np.maximum(u, 0.0)
        if order == 2:
            smooth = u * cdf + s**2 * density
            sharp = positive
        else:
            smooth = (u**2 + s**2) / 2 * cdf + s**2 * u / 2 * density
            sharp = positive**2 / 2
    return np.where(spike, sharp, smooth) if spike.any() else smooth


def _blurred_antiderivative(u, sigma, width, order: int) -> np.ndarray:
    """
    :func:`_antiderivative` of the Gaussian convolved with a rectangle
    `width` wide, such as a slit's image, where `width` is above 0. `sigma`
    and `width` are each one value or one per element of `u`.
    """
    u = np.asarray(u, dtype=float)
    width = np.broadcast_to(np.asarray(width, dtype=float), u.shape)
    plain = _antiderivative(u, sigma, order)
    boxed = width > 0
    if not boxed.any():
        return plain
    w = np.where(boxed, width, 1.0)
    within = (_antiderivative(u + w / 2, sigma, order + 1)
              - _antiderivative(u - w / 2, sigma, order + 1)) / w
    return np.where(boxed, within, plain)


def light_onto_pixels(cell_low, cell_high, pixel_edges, sigma=0.0,
                      width=0.0) -> sparse.csr_matrix:
    """
    Share of each cell's blurred light recorded by each pixel.

    Each cell's light is spread evenly from `cell_low` to `cell_high` and
    blurred by a Gaussian of `sigma` convolved with a rectangle `width`
    wide. The shares are exact integrals of the blur over the cell and the
    pixel, computed only for the pixels the blur reaches from each cell.
    Cells may overlap and leave gaps; pixels are contiguous.

    Parameters
    ----------
    cell_low, cell_high : array
        Ends of each cell, in one unit.
    pixel_edges : array
        Pixel edges, increasing, in the same unit.
    sigma, width : float or array
        Blur, one value or one per cell, in the same unit; 0 for none.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape `(n_pixels, n_cells)`. The shares of a cell whose light all
        lands within the pixels add up to 1.
    """
    low = np.atleast_1d(np.asarray(cell_low, dtype=float))
    high = np.atleast_1d(np.asarray(cell_high, dtype=float))
    edges = np.asarray(pixel_edges, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), low.shape)
    width = np.broadcast_to(np.asarray(width, dtype=float), low.shape)
    reach = _REACH_IN_SIGMA * sigma + width
    # Pixels each cell's blurred light reaches: from the first pixel ending
    # above where it starts, to just before the first starting above where
    # it ends.
    first = np.searchsorted(edges[1:], low - reach, side="right")
    last = np.searchsorted(edges[:-1], high + reach, side="left")
    count = np.maximum(last - first, 0)
    cell = np.repeat(np.arange(low.size), count)
    pixel = first[cell] + np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    p_low, p_high = edges[pixel], edges[pixel + 1]
    b_low, b_high = low[cell], high[cell]
    s, w = sigma[cell], width[cell]

    def corner(p, b):
        return _blurred_antiderivative(p - b, s, w, 2)

    total = (corner(p_high, b_low) - corner(p_high, b_high)
             - corner(p_low, b_low) + corner(p_low, b_high))
    # The closed forms cancel to rounding error where the true share is 0.
    share = np.maximum(total, 0.0) / (b_high - b_low)
    return sparse.csr_matrix((share, (pixel, cell)), shape=(edges.size - 1, low.size))


def points_onto_pixels(position, pixel_edges, sigma, width=0.0) -> sparse.csr_matrix:
    """
    Share of each point's blurred light recorded by each pixel.

    This is :func:`light_onto_pixels` for light all at one place, such as a
    line at its centre blurred by its own width.

    Parameters
    ----------
    position : array
        Position of each point, in one unit.
    pixel_edges : array
        Pixel edges, increasing, in the same unit.
    sigma, width : float or array
        Blur, a Gaussian of `sigma` convolved with a rectangle `width` wide,
        one value or one per point, in the same unit. `sigma` is above 0
        where `width` is 0.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape `(n_pixels, n_points)`.
    """
    position = np.atleast_1d(np.asarray(position, dtype=float))
    edges = np.asarray(pixel_edges, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), position.shape)
    width = np.broadcast_to(np.asarray(width, dtype=float), position.shape)
    reach = _REACH_IN_SIGMA * sigma + width
    first = np.searchsorted(edges[1:], position - reach, side="right")
    last = np.searchsorted(edges[:-1], position + reach, side="left")
    count = np.maximum(last - first, 0)
    point = np.repeat(np.arange(position.size), count)
    pixel = first[point] + np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    x, s, w = position[point], sigma[point], width[point]
    share = (_blurred_antiderivative(edges[pixel + 1] - x, s, w, 1)
             - _blurred_antiderivative(edges[pixel] - x, s, w, 1))
    return sparse.csr_matrix((np.maximum(share, 0.0), (pixel, point)),
                             shape=(edges.size - 1, position.size))


def photons_onto_pixels(share: sparse.spmatrix, photons: np.ndarray,
                        wavelength: np.ndarray) -> tuple:
    """
    Photons laid onto pixels, with the wavelengths of their mean and root mean square energies.

    Parameters
    ----------
    share : scipy.sparse matrix
        Each cell's share in each pixel, from :func:`light_onto_pixels`.
    photons : np.ndarray
        Photons in each cell, any number of spectra with the cells on the
        last axis.
    wavelength : np.ndarray
        Wavelength of each cell's photons.

    Returns
    -------
    tuple
        `(on_pixels, photon_wavelength, rms_wavelength)`, all with the pixels
        on the last axis. A photon frees electrons in a detector by its
        energy, so `photon_wavelength` is the wavelength of the mean energy
        of the photons a pixel receives, and `rms_wavelength` that of their
        root mean square energy, which sets how far their electrons spread.
        Both are in the unit of `wavelength`, and NaN where no photons
        arrive, or only photons below the smallest normal float.
    """
    photons = np.asarray(photons, dtype=float)
    flat = photons.reshape(-1, photons.shape[-1])
    wavelength = np.asarray(wavelength, dtype=float)
    on, photon_wavelength, rms_wavelength = _carry_photon_wavelengths(
        lambda cells: np.asarray(share @ cells.T).T, flat, wavelength, wavelength, np.nan)
    shape = photons.shape[:-1] + (share.shape[0],)
    return on.reshape(shape), photon_wavelength.reshape(shape), rms_wavelength.reshape(shape)


def _carry_photon_wavelengths(carry, photons, mean, rms, fill) -> tuple:
    """
    Photons moved by `carry`, with the wavelengths of the mean and root mean
    square energies of the photons where they land.

    A photon's energy goes as one over its wavelength. Photons whose mean
    energy is that of wavelength `mean`, and whose root mean square energy
    is that of `rms`, hold an energy that goes as photons / mean and a
    squared energy that goes as photons / rms**2. Both add up like the
    photons, so any linear map `carry` moves them as it moves the photons.

    Parameters
    ----------
    carry : callable
        Linear map from an array shaped as `photons` to where they land.
    photons : np.ndarray
        Photons.
    mean, rms : np.ndarray
        Wavelengths of the mean and root mean square energies of `photons`,
        broadcasting against them.
    fill : float or np.ndarray
        Wavelength for both where no photons land, broadcasting against
        where they land. Also used where the photons that land are below
        the smallest normal float, as their ratios then have too few
        significant bits to mean anything.

    Returns
    -------
    tuple
        `(moved, mean, rms)`.
    """
    photons = np.asarray(photons, dtype=float)
    mean = np.broadcast_to(np.asarray(mean, dtype=float), photons.shape)
    rms = np.broadcast_to(np.asarray(rms, dtype=float), photons.shape)
    # The wavelengths are taken relative to the largest, so the energies
    # carried are about the size of the photons. Divided by wavelengths in
    # cm they would be up to 1e12 times larger, and could stay above zero
    # where the photons round to zero.
    scale = np.max(mean, initial=0.0, where=np.isfinite(mean))
    scale = scale if scale > 0 else 1.0
    moved = carry(photons)
    energy = carry(photons * np.divide(scale, mean, out=np.zeros(photons.shape), where=mean > 0))
    square = carry(photons * np.divide(scale, rms, out=np.zeros(photons.shape), where=rms > 0)**2)
    empty = np.broadcast_to(np.asarray(fill, dtype=float), moved.shape)
    tiny = np.finfo(float).tiny
    with_mean = (moved >= tiny) & (energy >= tiny)
    with_rms = (moved >= tiny) & (square >= tiny)
    ratio = np.divide(moved, energy, out=np.ones(moved.shape), where=with_mean)
    moved_mean = np.where(with_mean, scale * ratio, empty)
    ratio = np.divide(moved, square, out=np.ones(moved.shape), where=with_rms)
    moved_rms = np.where(with_rms, scale * np.sqrt(ratio), empty)
    return moved, moved_mean, moved_rms


def pixel_weights(cell_edges, pixel_edges, sigma: float = 0.0, width: float = 0.0,
                  extend: bool = False) -> np.ndarray:
    """
    Weight of each cell in each pixel: the mean over the pixel of the cell's
    light, uniform over the cell, after blurring.

    Parameters
    ----------
    cell_edges, pixel_edges : array
        Cell and pixel edges along one axis, increasing, in one unit.
    sigma : float
        Standard deviation of the Gaussian blur, in the same unit; 0 for
        none.
    width : float
        Width of a rectangle convolved with the Gaussian, such as a slit's
        image, in the same unit; 0 for none.
    extend : bool
        Extend the first and last cells outward as far as the blur reaches,
        for a scene that continues past its edges.

    Returns
    -------
    np.ndarray
        Shape `(n_pixels, n_cells)`. With no blur each row is the fraction
        of the pixel each cell covers.
    """
    cells = np.asarray(cell_edges, dtype=float)
    pixels = np.asarray(pixel_edges, dtype=float)
    reach = _REACH_IN_SIGMA * sigma + width
    if extend and reach > 0:
        beyond = reach + (pixels[-1] - pixels[0]) + (cells[-1] - cells[0])
        cells = cells.copy()
        cells[0] = min(cells[0], pixels[0]) - beyond
        cells[-1] = max(cells[-1], pixels[-1]) + beyond

    # Each cell's share of its light in a pixel, turned into the mean over
    # the pixel of the cell's light per unit length.
    share = light_onto_pixels(cells[:-1], cells[1:], pixels, sigma, width)
    weights = sparse.diags(1.0 / np.diff(pixels)) @ share @ sparse.diags(np.diff(cells))
    return weights.toarray()


def pixel_weights_through(cell_edges, pixel_edges, openings, sigma: float = 0.0,
                          extend: bool = False) -> np.ndarray:
    """
    Pixel weights for light passing only through `openings`.

    As :func:`pixel_weights`, for a slit with gaps in it. The blur comes before the openings,
    as the optics in front of a slit blur the image the slit then cuts. Each
    pixel still holds the mean over its whole extent, so a pixel half
    covered by a gap holds half the light, and a pixel inside a gap none.

    Parameters
    ----------
    cell_edges, pixel_edges, sigma, extend
        As for :func:`pixel_weights`.
    openings : array
        `(n, 2)` start and end of each opening, in the unit of the edges,
        increasing and not overlapping.

    Returns
    -------
    np.ndarray
        Weights, `(n_pixels, n_cells)`.
    """
    pixels = np.asarray(pixel_edges, dtype=float)
    openings = np.asarray(openings, dtype=float).reshape(-1, 2)
    if np.any(openings[:, 1] <= openings[:, 0]) or np.any(openings[1:, 0] < openings[:-1, 1]):
        raise ValueError("The openings must each end after they start, in increasing "
                         "order, without overlapping.")
    # Cut the pixels at every opening edge inside them, so each piece is wholly
    # open or closed, then average the open pieces' weights over their pixel.
    cuts = openings[(openings > pixels[0]) & (openings < pixels[-1])]
    pieces = np.union1d(pixels, cuts)
    middles = (pieces[:-1] + pieces[1:]) / 2
    is_open = np.any((middles[:, np.newaxis] > openings[:, 0])
                     & (middles[:, np.newaxis] < openings[:, 1]), axis=1)
    weights = pixel_weights(cell_edges, pieces, sigma, extend=extend)
    pixel = np.searchsorted(pixels, middles, side="right") - 1
    out = np.zeros((pixels.size - 1, weights.shape[1]))
    np.add.at(out, pixel, weights * (np.diff(pieces) * is_open)[:, np.newaxis])
    return out / np.diff(pixels)[:, np.newaxis]


def centred_edges(n: int, pitch: float, centre: float = 0.0) -> np.ndarray:
    """Edges of `n` pixels of width `pitch`, centred on `centre`."""
    return centre + (np.arange(n + 1) - n / 2) * pitch


def onto_pixels(data: np.ndarray, rows: np.ndarray, columns: np.ndarray,
                wavelengths: np.ndarray) -> np.ndarray:
    """
    `data`, shaped `(y, x, wavelength)`, laid onto pixels by the weights
    along each axis from :func:`pixel_weights`.
    """
    out = np.tensordot(data, wavelengths, axes=([2], [1]))       # (y, x, l)
    out = np.tensordot(out, columns, axes=([1], [1]))            # (y, l, q)
    out = np.tensordot(rows, out, axes=([1], [0]))               # (p, l, q)
    return np.ascontiguousarray(np.moveaxis(out, 2, 1))          # (p, q, l)
