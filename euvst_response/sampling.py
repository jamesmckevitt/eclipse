"""
Laying a scene onto detector pixels.

A synthesis gives the radiance of cells, each uniform over its extent in
space and in wavelength. A detector pixel records the mean over its own
extent of the scene the instrument forms, which is the scene blurred by the
optics. Both steps are linear, so a pixel's value is a weighted sum of the
cells', with weights

    W[p, b] = (1 / |p|) int_p dx int_b dx' h(x - x')

for pixel *p*, cell *b* and the optics' blur *h*, taken along each axis in
turn. For a Gaussian blur, and for one convolved with a rectangle such as a
slit's image, the integrals have closed forms in the Gaussian's repeated
antiderivatives, so the weights are exact whatever the sizes of the cells,
the pixels and the blur, and a narrow line keeps its place within a pixel.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import erfc

# A Gaussian's weight beyond this many sigma is below a part in 1e30 of it.
_REACH_IN_SIGMA = 12.0


def _antiderivative(u, sigma, order: int) -> np.ndarray:
    """
    The *order*-th antiderivative, 1 (the cumulative distribution), 2 or 3,
    of a unit Gaussian of *sigma*, vanishing at minus infinity; where *sigma*
    is 0, of a unit spike. *sigma* is one value, or one for each of *u*.
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
    *width* wide, such as a slit's image, where *width* is above 0. *sigma*
    and *width* are each one value, or one for each of *u*.
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
    The share of each cell's light that each pixel records, once blurred.

    Each cell's light is spread evenly from *cell_low* to *cell_high* and
    blurred by a Gaussian of *sigma* convolved with a rectangle *width*
    wide. The shares are the exact integrals of the blur over the cell and
    the pixel, and only the pixels the blur reaches from a cell are worked
    out. The cells may overlap and leave gaps; the pixels are contiguous.

    Parameters
    ----------
    cell_low, cell_high : array
        The ends of each cell, in one unit.
    pixel_edges : array
        The edges of the pixels, increasing, in that unit.
    sigma, width : float or array
        The blur, one value or one for each cell, in that unit; 0 for none.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(n_pixels, n_cells)``. A cell whose light all lands within the
        pixels has shares adding up to 1.
    """
    low = np.atleast_1d(np.asarray(cell_low, dtype=float))
    high = np.atleast_1d(np.asarray(cell_high, dtype=float))
    edges = np.asarray(pixel_edges, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), low.shape)
    width = np.broadcast_to(np.asarray(width, dtype=float), low.shape)
    reach = _REACH_IN_SIGMA * sigma + width
    # The first pixel that ends above where a cell's blurred light starts,
    # and the pixels before the first that starts above where it ends.
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


def photons_onto_pixels(share: sparse.spmatrix, photons: np.ndarray,
                        wavelength: np.ndarray) -> tuple:
    """
    Photons in cells laid onto pixels, and the wavelength of each pixel's mean photon energy.

    Parameters
    ----------
    share : scipy.sparse matrix
        Each cell's share in each pixel, from :func:`light_onto_pixels`.
    photons : np.ndarray
        The photons in each cell, any number of spectra with the cells on
        the last axis.
    wavelength : np.ndarray
        The wavelength of each cell's photons.

    Returns
    -------
    tuple
        ``(on_pixels, photon_wavelength)``, both with the pixels on the last
        axis. A photon frees electrons in a detector by its energy, so the
        second, in the unit of *wavelength*, is the wavelength whose energy
        is the mean of the photons a pixel receives, NaN where none arrive.
    """
    photons = np.asarray(photons, dtype=float)
    flat = photons.reshape(-1, photons.shape[-1])
    on = np.asarray(share @ flat.T).T
    per_wavelength = np.asarray(share @ (flat / np.asarray(wavelength, dtype=float)).T).T
    photon_wavelength = np.divide(on, per_wavelength, out=np.full(on.shape, np.nan),
                                  where=per_wavelength > 0)
    shape = photons.shape[:-1] + (share.shape[0],)
    return on.reshape(shape), photon_wavelength.reshape(shape)


def pixel_weights(cell_edges, pixel_edges, sigma: float = 0.0, width: float = 0.0,
                  extend: bool = False) -> np.ndarray:
    """
    The weight of each cell in each pixel: the mean over the pixel of the light
    of the cell, uniform over it, once blurred.

    Parameters
    ----------
    cell_edges, pixel_edges : array
        The edges of the cells and of the pixels along one axis, increasing,
        in one unit.
    sigma : float
        The standard deviation of the Gaussian blur, in that unit; 0 for none.
    width : float
        The width of a rectangle the Gaussian is convolved with, such as the
        image of a slit, in that unit; 0 for none.
    extend : bool
        Carry the first and the last cell on outward as far as the blur
        reaches, for a scene that goes on past its edges.

    Returns
    -------
    np.ndarray
        ``(n_pixels, n_cells)``. With no blur each row is the share of the
        pixel each cell covers.
    """
    cells = np.asarray(cell_edges, dtype=float)
    pixels = np.asarray(pixel_edges, dtype=float)
    reach = _REACH_IN_SIGMA * sigma + width
    if extend and reach > 0:
        beyond = reach + (pixels[-1] - pixels[0]) + (cells[-1] - cells[0])
        cells = cells.copy()
        cells[0] = min(cells[0], pixels[0]) - beyond
        cells[-1] = max(cells[-1], pixels[-1]) + beyond

    # Each cell's share of its light in a pixel, as the mean over the pixel
    # of the cell's light per unit length.
    share = light_onto_pixels(cells[:-1], cells[1:], pixels, sigma, width)
    weights = sparse.diags(1.0 / np.diff(pixels)) @ share @ sparse.diags(np.diff(cells))
    return weights.toarray()


def centred_edges(n: int, pitch: float, centre: float = 0.0) -> np.ndarray:
    """The edges of *n* pixels of *pitch*, centred on *centre*."""
    return centre + (np.arange(n + 1) - n / 2) * pitch


def onto_pixels(data: np.ndarray, rows: np.ndarray, columns: np.ndarray,
                wavelengths: np.ndarray) -> np.ndarray:
    """
    ``data`` ``(y, x, wavelength)`` laid onto pixels by the weights along each
    axis, from :func:`pixel_weights`.
    """
    out = np.tensordot(data, wavelengths, axes=([2], [1]))       # (y, x, l)
    out = np.tensordot(out, columns, axes=([1], [1]))            # (y, l, q)
    out = np.tensordot(rows, out, axes=([1], [0]))               # (p, l, q)
    return np.ascontiguousarray(np.moveaxis(out, 2, 1))          # (p, q, l)
