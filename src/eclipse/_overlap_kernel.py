"""Compiled loop for `eclipse.projection`: how much of a shape falls in each pixel."""

import numba
import numpy as np


@numba.njit(cache=True)
def _clip_side(xs, ys, count, out_x, out_y, axis, bound, keep_below):
    """
    Part of a convex polygon on one side of a line of constant u or v.

    The polygon's `count` vertices are in `xs` and `ys`, in order around it.
    The part at or below `bound` is kept if `keep_below`, else the part at
    or above it, and written to `out_x` and `out_y`. Returns its number of
    vertices.
    """
    kept = 0
    for i in range(count):
        j = i + 1 if i + 1 < count else 0
        here = xs[i] if axis == 0 else ys[i]
        there = xs[j] if axis == 0 else ys[j]
        beyond = here - bound if keep_below else bound - here
        beyond_next = there - bound if keep_below else bound - there
        if beyond <= 0.0:
            out_x[kept] = xs[i]
            out_y[kept] = ys[i]
            kept += 1
        if (beyond < 0.0 < beyond_next) or (beyond_next < 0.0 < beyond):
            along = beyond / (beyond - beyond_next)
            out_x[kept] = xs[i] + along * (xs[j] - xs[i])
            out_y[kept] = ys[i] + along * (ys[j] - ys[i])
            kept += 1
    return kept


@numba.njit(parallel=True, cache=True)
def _overlap_kernel(corners, weight, depth, middle, slope, reach, span, centres, first, shares):
    """Fill `first` and `shares` for `eclipse.projection._overlaps`, one shape at a time."""
    n_faces = corners.shape[0]
    low_corner = np.empty((n_faces, 2))
    high_corner = np.empty((n_faces, 2))
    for f in range(n_faces):
        for axis in range(2):
            low_corner[f, axis] = corners[f, :, axis].min()
            high_corner[f, axis] = corners[f, :, axis].max()
    for n in numba.prange(centres.shape[0]):
        xs, ys = np.empty(16), np.empty(16)
        other_x, other_y = np.empty(16), np.empty(16)
        first_u = np.floor(centres[n, 0] - reach[0])
        first_v = np.floor(centres[n, 1] - reach[1])
        first[n, 0] = np.int64(first_u)
        first[n, 1] = np.int64(first_v)
        for step_u in range(span[0]):
            low_u = first_u + step_u - centres[n, 0]
            for step_v in range(span[1]):
                low_v = first_v + step_v - centres[n, 1]
                total = 0.0
                for f in range(n_faces):
                    if (high_corner[f, 0] <= low_u or low_corner[f, 0] >= low_u + 1.0
                            or high_corner[f, 1] <= low_v or low_corner[f, 1] >= low_v + 1.0):
                        continue
                    for k in range(4):
                        xs[k] = corners[f, k, 0]
                        ys[k] = corners[f, k, 1]
                    count = _clip_side(xs, ys, 4, other_x, other_y, 0, low_u, False)
                    count = _clip_side(other_x, other_y, count, xs, ys, 0, low_u + 1.0, True)
                    count = _clip_side(xs, ys, count, other_x, other_y, 1, low_v, False)
                    count = _clip_side(other_x, other_y, count, xs, ys, 1, low_v + 1.0, True)
                    # Signed area and first moments, by the shoelace formula.
                    area, moment_u, moment_v = 0.0, 0.0, 0.0
                    for i in range(count):
                        j = i + 1 if i + 1 < count else 0
                        cross = xs[i] * ys[j] - xs[j] * ys[i]
                        area += cross
                        moment_u += (xs[i] + xs[j]) * cross
                        moment_v += (ys[i] + ys[j]) * cross
                    area *= 0.5
                    moment_u /= 6.0
                    moment_v /= 6.0
                    total += weight[f] * (depth[f] * area
                                          + slope[f, 0] * (moment_u - area * middle[f, 0])
                                          + slope[f, 1] * (moment_v - area * middle[f, 1]))
                # The near and far faces' parts cancel to rounding error where
                # the shadow only touches the pixel; that is not a negative share.
                if -1e-12 < total < 0.0:
                    total = 0.0
                shares[n, step_u, step_v] = total
