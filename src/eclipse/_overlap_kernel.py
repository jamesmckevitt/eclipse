"""Compiled loop for `eclipse.projection`: how much of a shape falls in each pixel."""

import numba
import numpy as np

# Relative rounding error of a sum of a few face integrals, many times that
# of one double precision number.
ROUNDING = 1e-12

# The face as a square of side one, in its own coordinates (s, t), corners in
# order around it.
_SQUARE_S = (-0.5, 0.5, 0.5, -0.5)
_SQUARE_T = (-0.5, -0.5, 0.5, 0.5)


@numba.njit(cache=True)
def _clip(xs, ys, count, out_x, out_y, normal_x, normal_y, bound):
    """
    Part of a convex polygon where `normal . (x, y) <= bound`.

    The polygon's `count` vertices are in `xs` and `ys`, in order around it.
    The part kept is written to `out_x` and `out_y`. Returns its number of
    vertices.
    """
    kept = 0
    for i in range(count):
        j = i + 1 if i + 1 < count else 0
        beyond = normal_x * xs[i] + normal_y * ys[i] - bound
        beyond_next = normal_x * xs[j] + normal_y * ys[j] - bound
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
def _overlap_kernel(middle, sides, scale, depth, reach, span, centres, first, shares):
    """
    Fill `first` and `shares` for `eclipse.projection._overlaps`, one shape at a time.

    Each face is seen as the parallelogram `middle + s a + t b`, with s and t
    from -1/2 to 1/2 and `a` and `b` its sides. The part of it in front of a
    pixel is cut out in (s, t), where the pixel's edges are straight lines,
    so that the face's seen area multiplies the integral rather than
    dividing it, and a face seen all but edge on adds no rounding error.
    """
    n_faces = middle.shape[0]
    low_corner = np.empty((n_faces, 2))
    high_corner = np.empty((n_faces, 2))
    for f in range(n_faces):
        for axis in range(2):
            half = 0.5 * (abs(sides[f, 0, axis]) + abs(sides[f, 1, axis]))
            low_corner[f, axis] = middle[f, axis] - half
            high_corner[f, axis] = middle[f, axis] + half
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
                total, size = 0.0, 0.0
                for f in range(n_faces):
                    if (high_corner[f, 0] <= low_u or low_corner[f, 0] >= low_u + 1.0
                            or high_corner[f, 1] <= low_v or low_corner[f, 1] >= low_v + 1.0):
                        continue
                    a_u, a_v = sides[f, 0, 0], sides[f, 0, 1]
                    b_u, b_v = sides[f, 1, 0], sides[f, 1, 1]
                    for k in range(4):
                        xs[k] = _SQUARE_S[k]
                        ys[k] = _SQUARE_T[k]
                    # u = middle_u + a_u s + b_u t from low_u to low_u + 1, and
                    # the same for v.
                    count = _clip(xs, ys, 4, other_x, other_y, -a_u, -b_u,
                                  middle[f, 0] - low_u)
                    count = _clip(other_x, other_y, count, xs, ys, a_u, b_u,
                                  low_u + 1.0 - middle[f, 0])
                    count = _clip(xs, ys, count, other_x, other_y, -a_v, -b_v,
                                  middle[f, 1] - low_v)
                    count = _clip(other_x, other_y, count, xs, ys, a_v, b_v,
                                  low_v + 1.0 - middle[f, 1])
                    # Area and first moments in (s, t), by the shoelace formula.
                    area, moment_s, moment_t = 0.0, 0.0, 0.0
                    for i in range(count):
                        j = i + 1 if i + 1 < count else 0
                        cross = xs[i] * ys[j] - xs[j] * ys[i]
                        area += cross
                        moment_s += (xs[i] + xs[j]) * cross
                        moment_t += (ys[i] + ys[j]) * cross
                    area *= 0.5
                    moment_s /= 6.0
                    moment_t /= 6.0
                    part = scale[f] * (depth[f, 0] * area + depth[f, 1] * moment_s
                                       + depth[f, 2] * moment_t)
                    total += part
                    size += abs(scale[f]) * (abs(depth[f, 0] * area) + abs(depth[f, 1] * moment_s)
                                             + abs(depth[f, 2] * moment_t))
                # The near and far faces' parts cancel where the shadow only
                # touches the pixel. What is left within rounding error of
                # their size is no share at all.
                if abs(total) <= ROUNDING * size:
                    total = 0.0
                shares[n, step_u, step_v] = total
