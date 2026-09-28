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
from scipy.special import erfc

# A Gaussian's weight beyond this many sigma is below a part in 1e30 of it.
_REACH_IN_SIGMA = 12.0


def _antiderivative(u: np.ndarray, sigma: float, order: int) -> np.ndarray:
    """
    The *order*-th antiderivative, 2 or 3, of a unit Gaussian of *sigma*,
    vanishing at minus infinity; with *sigma* 0, of a unit spike.
    """
    if sigma == 0.0:
        positive = np.maximum(u, 0.0)
        return positive if order == 2 else positive**2 / 2
    cdf = 0.5 * erfc(-u / (np.sqrt(2.0) * sigma))
    density = np.exp(-0.5 * (u / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
    if order == 2:
        return u * cdf + sigma**2 * density
    return (u**2 + sigma**2) / 2 * cdf + sigma**2 * u / 2 * density


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

    low, high = pixels[:-1, np.newaxis], pixels[1:, np.newaxis]
    first, last = cells[np.newaxis, :-1], cells[np.newaxis, 1:]
    # Only the pairs the blur can join; the rest are nothing, where the
    # closed forms would leave rounding error.
    near = (low - reach < last) & (first < high + reach)
    weights = np.zeros(near.shape)
    rows, columns = np.nonzero(near)
    p_low, p_high = pixels[:-1][rows], pixels[1:][rows]
    b_low, b_high = cells[:-1][columns], cells[1:][columns]

    if width == 0.0:
        def corners(p, b):
            return _antiderivative(p - b, sigma, 2)
        total = (corners(p_high, b_low) - corners(p_high, b_high)
                 - corners(p_low, b_low) + corners(p_low, b_high))
    else:
        half = width / 2

        def corners(p, b):
            return (_antiderivative(p - b + half, sigma, 3)
                    - _antiderivative(p - b - half, sigma, 3)) / width
        total = (corners(p_high, b_low) - corners(p_high, b_high)
                 - corners(p_low, b_low) + corners(p_low, b_high))
    # The closed forms cancel to rounding error where the true weight is 0.
    weights[rows, columns] = np.maximum(total, 0.0) / (p_high - p_low)
    return weights


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
