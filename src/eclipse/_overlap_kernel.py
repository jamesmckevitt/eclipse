"""Compiled loop for `eclipse.projection`: how much of a shape falls in each pixel."""

import numba
import numpy as np

# Relative rounding error of a sum of a few face integrals, many times that
# of one double precision number.
ROUNDING = 1e-12


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


@numba.njit(cache=True)
def _shape_shares(n_faces, corners, weight, depth, middle, slope, low_corner, high_corner,
                  centre_u, centre_v, reach_u, reach_v, span_u, span_v, buffers, first, shares):
    """
    Share of one shape in each pixel near it.

    The shape's faces are cut to each pixel, and the integral of their depth
    over the part inside summed. `first` gets the first pixel along u and v,
    and `shares` the share in each pixel from there.
    """
    xs, ys, other_x, other_y = buffers[0], buffers[1], buffers[2], buffers[3]
    first_u = np.floor(centre_u - reach_u)
    first_v = np.floor(centre_v - reach_v)
    first[0] = np.int64(first_u)
    first[1] = np.int64(first_v)
    for step_u in range(span_u):
        low_u = first_u + step_u - centre_u
        for step_v in range(span_v):
            low_v = first_v + step_v - centre_v
            total, size = 0.0, 0.0
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
                part = weight[f] * (depth[f] * area
                                    + slope[f, 0] * (moment_u - area * middle[f, 0])
                                    + slope[f, 1] * (moment_v - area * middle[f, 1]))
                total += part
                size += abs(weight[f]) * (
                    abs(depth[f] * area)
                    + abs(slope[f, 0]) * (abs(moment_u) + abs(area * middle[f, 0]))
                    + abs(slope[f, 1]) * (abs(moment_v) + abs(area * middle[f, 1])))
            # The near and far faces' parts cancel where the shadow only
            # touches the pixel. What is left within rounding error of their
            # size is no share at all.
            if abs(total) <= ROUNDING * size:
                total = 0.0
            shares[step_u, step_v] = total


@numba.njit(cache=True)
def _bounds(n_faces, corners, low_corner, high_corner):
    """Lowest and highest u and v of each face."""
    for f in range(n_faces):
        for axis in range(2):
            low_corner[f, axis] = corners[f, :, axis].min()
            high_corner[f, axis] = corners[f, :, axis].max()


@numba.njit(parallel=True, cache=True)
def _overlap_kernel(corners, weight, depth, middle, slope, reach, span, centres, first, shares):
    """Fill `first` and `shares` for `eclipse.projection._overlaps`, for shapes alike."""
    n_faces = corners.shape[0]
    low_corner = np.empty((n_faces, 2))
    high_corner = np.empty((n_faces, 2))
    _bounds(n_faces, corners, low_corner, high_corner)
    for n in numba.prange(centres.shape[0]):
        buffers = np.empty((4, 16))
        _shape_shares(n_faces, corners, weight, depth, middle, slope, low_corner, high_corner,
                      centres[n, 0], centres[n, 1], reach[0], reach[1], span[0], span[1],
                      buffers, first[n], shares[n])


# Corners of a parallelogram, in order around it, in halves of its two sides.
_ALONG_A = (-0.5, 0.5, 0.5, -0.5)
_ALONG_B = (-0.5, -0.5, 0.5, 0.5)


@numba.njit(cache=True)
def _cell_faces(linear, cell, corners, weight, depth, middle, slope):
    """
    Faces of a cell's shadow, as `eclipse.projection._cell_faces` gives them.

    Fills the arrays and returns the number of faces, with those seen edge
    on left out.
    """
    scaled = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            scaled[i, j] = linear[i, j] * cell[j]
    # Cofactors of the last row: the inverse's last column times the determinant.
    cofactor = np.empty(3)
    cofactor[0] = linear[0, 1] * linear[1, 2] - linear[0, 2] * linear[1, 1]
    cofactor[1] = linear[0, 2] * linear[1, 0] - linear[0, 0] * linear[1, 2]
    cofactor[2] = linear[0, 0] * linear[1, 1] - linear[0, 1] * linear[1, 0]
    determinant = (linear[2, 0] * cofactor[0] + linear[2, 1] * cofactor[1]
                   + linear[2, 2] * cofactor[2])
    volume = abs(determinant * cell[0] * cell[1] * cell[2])
    n_faces = 0
    for normal in range(3):
        a = 1 if normal == 0 else 0
        b = 1 if normal == 2 else 2
        seen_a_u, seen_a_v = scaled[0, a], scaled[1, a]
        seen_b_u, seen_b_v = scaled[0, b], scaled[1, b]
        area = seen_a_u * seen_b_v - seen_a_v * seen_b_u
        if abs(area) <= 1e-12 * np.hypot(seen_a_u, seen_a_v) * np.hypot(seen_b_u, seen_b_v):
            continue
        # The depth along a and b, against u and v.
        depth_a, depth_b = scaled[2, a], scaled[2, b]
        slope_u = (depth_a * seen_b_v - depth_b * seen_a_v) / area
        slope_v = (depth_b * seen_a_u - depth_a * seen_b_u) / area
        # The face whose outward normal points towards larger w is the near one.
        nearness = np.sign(cofactor[normal] * determinant)
        for outward in (1.0, -1.0):
            middle_u = 0.5 * outward * scaled[0, normal]
            middle_v = 0.5 * outward * scaled[1, normal]
            for k in range(4):
                corners[n_faces, k, 0] = middle_u + _ALONG_A[k] * seen_a_u + _ALONG_B[k] * seen_b_u
                corners[n_faces, k, 1] = middle_v + _ALONG_A[k] * seen_a_v + _ALONG_B[k] * seen_b_v
            weight[n_faces] = outward * nearness * np.sign(area) / volume
            depth[n_faces] = 0.5 * outward * scaled[2, normal]
            middle[n_faces, 0] = middle_u
            middle[n_faces, 1] = middle_v
            slope[n_faces, 0] = slope_u
            slope[n_faces, 1] = slope_v
            n_faces += 1
    return n_faces


@numba.njit(parallel=True, cache=True)
def _cell_overlap_kernel(linear, cell, centres, span, first, shares):
    """
    Like `_overlap_kernel`, for cells each seen its own way.

    `linear` is `(n, 3, 3)` and `cell` `(n, 3)`: for each cell, what
    `eclipse.projection._cell_faces` takes.
    """
    for n in numba.prange(centres.shape[0]):
        corners = np.empty((6, 4, 2))
        weight, depth = np.empty(6), np.empty(6)
        middle, slope = np.empty((6, 2)), np.empty((6, 2))
        n_faces = _cell_faces(linear[n], cell[n], corners, weight, depth, middle, slope)
        low_corner, high_corner = np.empty((6, 2)), np.empty((6, 2))
        _bounds(n_faces, corners, low_corner, high_corner)
        reach_u, reach_v = 0.0, 0.0
        for j in range(3):
            reach_u += 0.5 * abs(linear[n, 0, j] * cell[n, j])
            reach_v += 0.5 * abs(linear[n, 1, j] * cell[n, j])
        buffers = np.empty((4, 16))
        _shape_shares(n_faces, corners, weight, depth, middle, slope, low_corner, high_corner,
                      centres[n, 0], centres[n, 1], reach_u, reach_v, span[0], span[1],
                      buffers, first[n], shares[n])
