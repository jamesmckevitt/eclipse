"""Compiled loop for `eclipse.projection`: how much of a shape falls in each pixel."""

import numba
import numpy as np

# Cutting a face leaves rounding error in where the cut falls, about that of
# one double precision number (2e-16) in the face's own coordinates. A sum
# of face parts within this fraction of the faces' whole parts, a few hundred
# times that, is taken as zero: the shadow only touches the pixel.
ROUNDING = 1e-13

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


@numba.njit(cache=True)
def _bounds(n_faces, middle, sides, low_corner, high_corner):
    """Lowest and highest u and v of each face."""
    for f in range(n_faces):
        for axis in range(2):
            half = 0.5 * (abs(sides[f, 0, axis]) + abs(sides[f, 1, axis]))
            low_corner[f, axis] = middle[f, axis] - half
            high_corner[f, axis] = middle[f, axis] + half


@numba.njit(cache=True)
def _shape_shares(n_faces, middle, sides, scale, depth, low_corner, high_corner,
                  centre_u, centre_v, reach_u, reach_v, span_u, span_v, buffers, first, shares):
    """
    Share of one shape in each pixel near it.

    Each face is seen as the parallelogram `middle + s a + t b`, with s and t
    from -1/2 to 1/2 and `a` and `b` its sides. The part of it in front of a
    pixel is cut out in (s, t), where the pixel's edges are straight lines,
    so that the face's seen area is a factor of the integral, and a face
    seen all but edge on adds no rounding error. `first` gets the first
    pixel along u and v, and `shares` the share in each pixel from there.
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
                a_u, a_v = sides[f, 0, 0], sides[f, 0, 1]
                b_u, b_v = sides[f, 1, 0], sides[f, 1, 1]
                for k in range(4):
                    xs[k] = _SQUARE_S[k]
                    ys[k] = _SQUARE_T[k]
                # u = middle_u + a_u s + b_u t from low_u to low_u + 1, and the
                # same for v.
                count = _clip(xs, ys, 4, other_x, other_y, -a_u, -b_u, middle[f, 0] - low_u)
                count = _clip(other_x, other_y, count, xs, ys, a_u, b_u,
                              low_u + 1.0 - middle[f, 0])
                count = _clip(xs, ys, count, other_x, other_y, -a_v, -b_v, middle[f, 1] - low_v)
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
                total += scale[f] * (depth[f, 0] * area + depth[f, 1] * moment_s
                                     + depth[f, 2] * moment_t)
                # Cutting the face leaves rounding error in s and t, where the
                # face is a square of side one, whatever the size of the part.
                size += abs(scale[f]) * (abs(depth[f, 0])
                                         + 0.5 * (abs(depth[f, 1]) + abs(depth[f, 2])))
            # The near and far faces' parts cancel where the shadow only
            # touches the pixel.
            if abs(total) <= ROUNDING * size:
                total = 0.0
            shares[step_u, step_v] = total


@numba.njit(parallel=True, cache=True)
def _overlap_kernel(middle, sides, scale, depth, reach, span, centres, first, shares, n_runs):
    """Fill `first` and `shares` for `eclipse.projection._overlaps`: shapes seen the same way."""
    n_faces = middle.shape[0]
    low_corner = np.empty((n_faces, 2))
    high_corner = np.empty((n_faces, 2))
    _bounds(n_faces, middle, sides, low_corner, high_corner)
    n_cells = centres.shape[0]
    # Scratch space is made once for each of `n_runs` runs of cells, not for
    # each cell.
    for run in numba.prange(n_runs):
        buffers = np.empty((4, 16))
        for n in range(run * n_cells // n_runs, (run + 1) * n_cells // n_runs):
            _shape_shares(n_faces, middle, sides, scale, depth, low_corner, high_corner,
                          centres[n, 0], centres[n, 1], reach[0], reach[1], span[0], span[1],
                          buffers, first[n], shares[n])


@numba.njit(cache=True)
def _cell_faces(linear, cell, middle, sides, scale, depth):
    """
    Faces of a cell's shadow; see `eclipse.projection._cell_faces`.

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
        area = abs(scaled[0, a] * scaled[1, b] - scaled[1, a] * scaled[0, b])
        if area == 0.0:
            continue
        # The face whose outward normal points towards larger w is the near one.
        nearness = np.sign(cofactor[normal] * determinant)
        for outward in (1.0, -1.0):
            middle[n_faces, 0] = 0.5 * outward * scaled[0, normal]
            middle[n_faces, 1] = 0.5 * outward * scaled[1, normal]
            for axis in range(2):
                sides[n_faces, 0, axis] = scaled[axis, a]
                sides[n_faces, 1, axis] = scaled[axis, b]
            scale[n_faces] = outward * nearness * area / volume
            depth[n_faces, 0] = 0.5 * outward * scaled[2, normal]
            depth[n_faces, 1] = scaled[2, a]
            depth[n_faces, 2] = scaled[2, b]
            n_faces += 1
    return n_faces


@numba.njit(parallel=True, cache=True)
def _cell_overlap_kernel(linear, cell, centres, span, first, shares, n_runs):
    """
    Like `_overlap_kernel`, for cells each seen its own way.

    `linear` is `(n, 3, 3)` and `cell` `(n, 3)`: each cell's matrix and size,
    as for `eclipse.projection._cell_faces`.
    """
    n_cells = centres.shape[0]
    # Scratch space is made once for each of `n_runs` runs of cells, not for
    # each cell.
    for run in numba.prange(n_runs):
        middle, sides = np.empty((6, 2)), np.empty((6, 2, 2))
        scale, depth = np.empty(6), np.empty((6, 3))
        low_corner, high_corner = np.empty((6, 2)), np.empty((6, 2))
        buffers = np.empty((4, 16))
        for n in range(run * n_cells // n_runs, (run + 1) * n_cells // n_runs):
            n_faces = _cell_faces(linear[n], cell[n], middle, sides, scale, depth)
            _bounds(n_faces, middle, sides, low_corner, high_corner)
            reach_u, reach_v = 0.0, 0.0
            for j in range(3):
                reach_u += 0.5 * abs(linear[n, 0, j] * cell[n, j])
                reach_v += 0.5 * abs(linear[n, 1, j] * cell[n, j])
            _shape_shares(n_faces, middle, sides, scale, depth, low_corner, high_corner,
                          centres[n, 0], centres[n, 1], reach_u, reach_v, span[0], span[1],
                          buffers, first[n], shares[n])


def runs(n_cells: int) -> int:
    """How many runs of cells to share between the threads: a few for each."""
    return min(n_cells, 4 * numba.get_num_threads())
