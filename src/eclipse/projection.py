"""
A box seen from a position on the solar disk, along a slanted line of sight.

Each cell of the box is taken to be uniform. Its emission is shared between
the sky pixels its column of sight passes through, in proportion to the part
of the cell's volume in front of each pixel.
"""

import warnings
from typing import Iterator, List, Tuple

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from ndcube import NDCube
from scipy import sparse
from sunpy.coordinates import Heliocentric, HeliographicStonyhurst, Helioprojective
from sunpy.util.exceptions import SunpyUserWarning

#: Ways of sharing a box's cells between sky pixels; the first is the default.
PROJECTIONS = ("two-step", "exact")


def disk_frame(x: u.Quantity, y: u.Quantity,
               rotation: u.Quantity = 0 * u.deg) -> Tuple[np.ndarray, u.Quantity]:
    """
    Directions of a box at a position on the solar disk, as seen on the sky.

    The box's z points up, along the local vertical at the position. With no
    rotation its x points west and its y north. The sky's x points west, its
    y north and its z towards the observer, who is 1 AU from the Sun above
    its equator, with solar north up.

    Parameters
    ----------
    x, y : u.Quantity
        Helioprojective position of the box's centre, west and north of
        disk centre, as angles.
    rotation : u.Quantity, optional
        Angle from west to the box's x, turning towards north about the
        vertical, as seen from above. Default 0 deg.

    Returns
    -------
    frame : np.ndarray
        `(3, 3)` matrix taking a vector in the box's axes to the sky's.
    heliocentric_angle : u.Quantity
        Angle between the local vertical and the line of sight.

    Raises
    ------
    ValueError
        If the position is not on the disk.
    """
    observer = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, const.au)
    sky = Helioprojective(observer=observer)
    # Heliocentric coordinates: x west, y north and z towards the observer,
    # with the origin at the Sun's centre.
    heliocentric = Heliocentric(observer=observer)

    def position(coordinate):
        return SkyCoord(coordinate).transform_to(heliocentric).cartesian.xyz.to_value(u.Mm)

    def unit(vector):
        return vector / np.linalg.norm(vector)

    def across(vector, normal):
        """Unit vector along the part of `vector` perpendicular to `normal`."""
        return unit(vector - (vector @ normal) * normal)

    # Where the line of sight first meets the surface; NaN off the disk,
    # which is reported below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SunpyUserWarning)
        surface = SkyCoord(x, y, frame=sky).make_3d()
    if not np.isfinite(surface.distance):
        r = np.hypot(u.Quantity(x).to_value(u.arcsec), u.Quantity(y).to_value(u.arcsec)) * u.arcsec
        radius = np.arcsin((const.R_sun / const.au).decompose()).to(u.arcsec)
        raise ValueError(f"The position is {r:.1f} from disk centre, on or beyond the "
                         f"limb at {radius:.1f}; a box is seen from a position on the disk.")
    point = position(surface)
    up = unit(point)
    pole = unit(position(SkyCoord(0 * u.deg, 90 * u.deg, const.R_sun,
                                  frame=HeliographicStonyhurst())))
    north = across(pole, up)
    west = np.cross(north, up)
    # The sky's z is the line of sight from the position, which is parallel
    # to within the box's size to the line of sight from every other point
    # of it. The sky's y is the direction of increasing Ty there, towards
    # the pole of the helioprojective frame.
    sky_z = unit(position(observer) - point)
    sky_pole = position(SkyCoord(0 * u.deg, 90 * u.deg, const.au, frame=sky)) - position(observer)
    sky_y = across(sky_pole, sky_z)
    sky_x = np.cross(sky_y, sky_z)
    frame = np.vstack([sky_x, sky_y, sky_z]) @ np.column_stack([west, north, up])
    angle = u.Quantity(rotation).to_value(u.rad)
    turn = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                     [np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])
    theta = np.arctan2(np.linalg.norm(np.cross(up, sky_z)), up @ sky_z)
    return frame @ turn, (theta * u.rad).to(u.deg)


# ----------------------------------------------------------------------
# How much of a shape falls in each pixel
# ----------------------------------------------------------------------
# A face of a shape's shadow: its corners about the shape's centre, in
# pixels; the weight of the integral over it; and the depth, an affine
# function of (u, v): its value at `middle` and its slope.
_Face = Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]


def _parallelogram(middle: np.ndarray, side_a: np.ndarray, side_b: np.ndarray) -> np.ndarray:
    """Corners of a parallelogram in order around it."""
    return np.array([middle - 0.5 * side_a - 0.5 * side_b, middle + 0.5 * side_a - 0.5 * side_b,
                     middle + 0.5 * side_a + 0.5 * side_b, middle - 0.5 * side_a + 0.5 * side_b])


def _cell_faces(linear: np.ndarray, cell: np.ndarray) -> Tuple[List[_Face], np.ndarray]:
    """
    Faces of a cell's shadow, and its half extent along u and v.

    The volume of the cell in front of a region of the sky is the integral
    over the region of the depth of the cell, from its far faces to its near
    ones (the divergence theorem). Each face's part is an integral of an
    affine function over a parallelogram. Faces seen edge on have none.

    Parameters
    ----------
    linear : np.ndarray
        `(3, 3)` matrix from a vector in the box to (u, v, w): u and v in
        pixels, w along the line of sight. Points along a line of sight have
        the same u and v.
    cell : np.ndarray
        Cell's size along the box's x, y and z.
    """
    inverse = np.linalg.inv(linear)
    volume = abs(np.linalg.det(linear * cell))
    faces = []
    for normal in range(3):
        side_a, side_b = (np.eye(3)[axis] * cell[axis] for axis in range(3) if axis != normal)
        seen_a, seen_b = linear[:2] @ side_a, linear[:2] @ side_b
        area = seen_a[0] * seen_b[1] - seen_a[1] * seen_b[0]
        if abs(area) <= 1e-12 * np.linalg.norm(seen_a) * np.linalg.norm(seen_b):
            continue
        slope = np.array([linear[2] @ side_a, linear[2] @ side_b]) @ np.linalg.inv(
            np.column_stack([seen_a, seen_b]))
        # The face whose outward normal points towards larger w is the near one.
        nearness = np.sign(inverse[normal, 2])
        for outward in (1.0, -1.0):
            offset = 0.5 * outward * cell[normal] * np.eye(3)[normal]
            middle = linear[:2] @ offset
            faces.append((_parallelogram(middle, seen_a, seen_b),
                          outward * nearness * np.sign(area) / volume,
                          linear[2] @ offset, middle, slope))
    reach = 0.5 * np.abs(linear[:2] * cell).sum(axis=1)
    return faces, reach


def _patch_faces(side_a: np.ndarray, side_b: np.ndarray) -> Tuple[List[_Face], np.ndarray]:
    """The one face of a flat, uniform parallelogram, and its half extent."""
    area = side_a[0] * side_b[1] - side_a[1] * side_b[0]
    face = (_parallelogram(np.zeros(2), side_a, side_b), 1.0 / area, 1.0, np.zeros(2),
            np.zeros(2))
    return [face], 0.5 * (np.abs(side_a) + np.abs(side_b))


def _overlaps(faces: List[_Face], reach: np.ndarray,
              centres: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pixels a shape at each of `centres` falls on, and the share in each.

    Each face is cut to each pixel near it, and the integral of its depth
    over the part inside summed.

    Parameters
    ----------
    faces, reach
        From :func:`_cell_faces` or :func:`_patch_faces`.
    centres : np.ndarray
        `(n, 2)` centres of the shapes in pixels. Pixel `(p, q)` covers
        `p <= u < p + 1` and `q <= v < q + 1`.

    Yields
    ------
    p, q, share : np.ndarray
        For each pixel the shapes may reach, its indices and the share of
        each shape in it. The shares of a shape sum to one.
    """
    centres = np.ascontiguousarray(centres, dtype=float).reshape(-1, 2)
    # An interval of length 2 r touches at most floor(2 r) + 2 pixels.
    span = np.floor(2.0 * np.asarray(reach, dtype=float)).astype(np.int64) + 2
    first = np.empty((len(centres), 2), dtype=np.int64)
    shares = np.zeros((len(centres), *span))
    if faces and len(centres):
        # Compiled with numba, which is loaded only for a view from a position.
        from ._overlap_kernel import _overlap_kernel

        corners, weight, depth, middle, slope = (np.array(column, dtype=float)
                                                 for column in zip(*faces))
        _overlap_kernel(corners, weight, depth, middle, slope,
                        np.asarray(reach, dtype=float), span, centres, first, shares)
    for step_u in range(span[0]):
        for step_v in range(span[1]):
            yield first[:, 0] + step_u, first[:, 1] + step_v, shares[:, step_u, step_v]


# ----------------------------------------------------------------------
# The view
# ----------------------------------------------------------------------
class SkyView:
    """
    How each cell of a box is shared between sky pixels, from a position on the disk.

    Each cell is taken to be uniform, and its emission is shared between the
    pixels in proportion to the part of its volume in front of each. The sky
    pixels are as big as the box's cells across, so the emission measure of
    a pixel is the sum over cells of n_e^2 times their height times that
    share. A slanted line of sight crosses more cells, so the sum grows as
    one over the cosine of the heliocentric angle without any further factor.

    Two projections are offered:

    - `"two-step"` first shares each cell between the columns of the box,
      along the line of sight, to where it meets the box's base at z = 0.
      Every cell of a layer sits the same way on that grid, so this is
      computed once per layer, exactly. The columns are then shared between
      the sky pixels by area, each taken to be uniform across. That second
      sharing softens detail by up to a pixel.
    - `"exact"` shares each cell between the sky pixels directly. It is
      exact for uniform cells, and slower, since each cell sits differently
      on the sky's pixels.

    At disk centre with no rotation both give the view from straight above.

    Parameters
    ----------
    frame : np.ndarray
        `(3, 3)` matrix from :func:`disk_frame`.
    x_edges, y_edges, z_edges : u.Quantity
        Edges of the box's cells. x and y must be evenly spaced.
    wrap : bool
        Take the box to repeat sideways, as MURaM and Bifrost boxes do, so
        every line of sight crosses the box's full height. The image then
        covers where the box's base at z = 0 is seen. Otherwise lines of
        sight end at the box's sides, and the image covers every cell, with
        part columns along its edges.
    projection : str, optional
        `"two-step"` (default) or `"exact"`.

    Attributes
    ----------
    image_shape : tuple of int
        Pixels of the image, `(rows, columns)`: north, then west.
    x_edges, y_edges : u.Quantity
        Edges of the image's pixels: west and north on the sky from where
        the box's centre at z = 0 is seen, plus the box's centre in x and
        y, so that at disk centre they are the box's own.
    """

    def __init__(self, frame: np.ndarray, x_edges: u.Quantity, y_edges: u.Quantity,
                 z_edges: u.Quantity, wrap: bool, projection: str = PROJECTIONS[0]):
        if projection not in PROJECTIONS:
            raise ValueError(f"The projection is one of {', '.join(PROJECTIONS)}, not "
                             f"{projection!r}.")
        self.frame = np.asarray(frame, dtype=float)
        self.wrap = bool(wrap)
        self.projection = projection
        xe, ye, ze = (u.Quantity(edges).to_value(u.Mm) for edges in (x_edges, y_edges, z_edges))
        self._x_edges, self._y_edges, self._z_edges = xe, ye, ze
        self._shape = (ze.size - 1, ye.size - 1, xe.size - 1)
        self._pitch = np.array([xe[1] - xe[0], ye[1] - ye[0]])
        self._size = np.array([xe[-1] - xe[0], ye[-1] - ye[0]])
        self._centre = 0.5 * np.array([xe[0] + xe[-1], ye[0] + ye[-1]])
        # A point of the box is seen where its line of sight meets the base,
        # at z = 0: `lean` along x and y per unit height.
        towards = self.frame[2]
        self._lean = -towards[:2] / towards[2]

        # The pixels are centred on where the box's base is seen: their
        # edges are on whole multiples of the pitch from its centre when the
        # base spans an even number of pixels, halfway between for an odd
        # one. In a view from straight above they are the box's cells.
        half = 0.5 * self._size
        base = np.array([[sx * half[0], sy * half[1]] for sx in (-1, 1) for sy in (-1, 1)])
        spans = np.ceil(2.0 * np.abs(base @ self.frame[:2, :2].T).max(axis=0) / self._pitch
                        - 1e-9).astype(np.int64)
        self._edge_offset = 0.5 * (spans % 2)
        # The image covers the base of a repeating box. Otherwise it covers
        # every cell, or with two steps every column of the base they reach.
        if projection == "two-step":
            self._prepare_layers()
        if self.wrap:
            corners = np.column_stack([base, np.zeros(4)])
        elif projection == "two-step":
            low = (self._base_first * self._pitch + np.array([xe[0], ye[0]]) - self._centre)
            high = low + np.array(self.grid_shape[::-1]) * self._pitch
            corners = np.array([[x, y, 0.0] for x in (low[0], high[0]) for y in (low[1], high[1])])
        else:
            corners = np.array([[*corner, z] for corner in base for z in (ze[0], ze[-1])])
        seen = corners @ self.frame[:2].T / self._pitch - self._edge_offset
        # A rounding error does not add a pixel.
        self._first = np.floor(seen.min(axis=0) + 1e-9).astype(np.int64)
        last = np.ceil(seen.max(axis=0) - 1e-9).astype(np.int64)
        n_columns, n_rows = (last - self._first).tolist()
        self.image_shape = (n_rows, n_columns)
        self.x_edges, self.y_edges = (
            ((np.arange(self._first[i], last[i] + 1) + self._edge_offset[i]) * self._pitch[i]
             + self._centre[i]) * u.Mm for i in range(2))

        if projection == "two-step":
            self._prepare_resample()

    # -- Common geometry ------------------------------------------------
    def _on_sky(self, offset: np.ndarray) -> np.ndarray:
        """Image pixel coordinates of points of the base, `(n, 2)` x and y from its centre."""
        # Written out rather than as a matrix product, whose BLAS threads would
        # compete for the cores with the projection's own.
        seen = (offset[:, :1] * self.frame[:2, 0] + offset[:, 1:] * self.frame[:2, 1])
        return seen / self._pitch - self._edge_offset - self._first

    def _copies(self, reach: np.ndarray):
        """
        Shifts of the base, along x and y, at which a repeating box is seen in the image.

        `reach` is how far, in pixels, a shape reaches from where its centre is seen.
        """
        if not self.wrap:
            return [np.zeros(2)]
        # The image, widened by the reach, as seen on the base.
        n_rows, n_columns = self.image_shape
        low, high = -reach - 1.0, np.array([n_columns, n_rows]) + reach + 1.0
        corners = np.array([[u_, v_] for u_ in (low[0], high[0]) for v_ in (low[1], high[1])])
        sky = (corners + self._first + self._edge_offset) * self._pitch
        base = np.linalg.solve(self.frame[:2, :2], sky.T).T
        # Copy (m, n) covers -size / 2 to size / 2 about (m, n) times the size.
        half = 0.5 * self._size
        first = np.ceil((base.min(axis=0) - half) / self._size).astype(int)
        last = np.floor((base.max(axis=0) + half) / self._size).astype(int)
        return [np.array([m, n]) * self._size
                for m in range(first[0], last[0] + 1) for n in range(first[1], last[1] + 1)]

    def _image_parts(self, faces, reach, offsets):
        """Image pixels shapes on the base at `offsets` fall on, with copies when wrapping."""
        n_rows, n_columns = self.image_shape
        for shift in self._copies(reach):
            centres = self._on_sky(offsets + shift)
            near = np.flatnonzero((centres[:, 0] + reach[0] > 0)
                                  & (centres[:, 0] - reach[0] < n_columns)
                                  & (centres[:, 1] + reach[1] > 0)
                                  & (centres[:, 1] - reach[1] < n_rows))
            if near.size == 0:
                continue
            for p, q, share in _overlaps(faces, reach, centres[near]):
                seen = (share != 0.0) & (p >= 0) & (p < n_columns) & (q >= 0) & (q < n_rows)
                if seen.any():
                    yield near[seen], q[seen] * n_columns + p[seen], share[seen]

    # -- Two steps -------------------------------------------------------
    def _prepare_layers(self):
        """Step one: how each layer's cells are shared between the columns of the base."""
        nz, ny, nx = self._shape
        dz = np.diff(self._z_edges)
        heights = 0.5 * (self._z_edges[1:] + self._z_edges[:-1])
        # From a vector in the box to where it is seen on the base, in cells,
        # and its height.
        linear = np.array([[1.0 / self._pitch[0], 0.0, self._lean[0] / self._pitch[0]],
                           [0.0, 1.0 / self._pitch[1], self._lean[1] / self._pitch[1]],
                           [0.0, 0.0, 1.0]])
        # Every cell of a layer sits on the base's grid as its first does.
        self._layers = []
        for height, thickness in zip(heights, dz):
            faces, reach = _cell_faces(linear, np.array([*self._pitch, thickness]))
            first = np.array([[0.5, 0.5]]) + height * self._lean / self._pitch
            self._layers.append([(int(p[0]), int(q[0]), float(share[0]))
                                 for p, q, share in _overlaps(faces, reach, first)
                                 if share[0] != 0.0])
        if self.wrap:
            self._base_first = np.zeros(2, dtype=np.int64)
            self.grid_shape = (ny, nx)
        else:
            steps = np.array([(p, q) for layer in self._layers for p, q, _ in layer])
            self._base_first = steps.min(axis=0)
            last = steps.max(axis=0) + np.array([nx, ny])
            self.grid_shape = tuple(int(n) for n in (last - self._base_first)[::-1])

    def _prepare_resample(self):
        """Step two: each column of the base, a parallelogram on the sky, shared by area."""
        rows, columns = self.grid_shape
        q, p = np.divmod(np.arange(rows * columns), columns)
        offsets = ((np.column_stack([p, q]) + self._base_first + 0.5) * self._pitch
                   + np.array([self._x_edges[0], self._y_edges[0]]) - self._centre)
        sides = self.frame[:2, :2] * self._pitch / self._pitch[:, np.newaxis]
        faces, reach = _patch_faces(sides[:, 0], sides[:, 1])
        parts = list(self._image_parts(faces, reach, offsets))
        self._resample = sparse.csr_matrix(
            (np.concatenate([share for _, _, share in parts]),
             (np.concatenate([pixel for _, pixel, _ in parts]),
              np.concatenate([column for column, _, _ in parts]))),
            shape=(self.image_shape[0] * self.image_shape[1], rows * columns))

    # -- What the accumulators use ----------------------------------------
    @property
    def accumulation_shape(self) -> Tuple[int, int]:
        """Grid the cells are summed onto: the base's for two steps, else the image."""
        return self.grid_shape if self.projection == "two-step" else self.image_shape

    def parts(self) -> Iterator[Tuple[tuple, np.ndarray, np.ndarray]]:
        """
        Cells, the flat pixel of the accumulation grid each goes to, and its share.

        Yields `(cells, pixel, share)`, where `cells` indexes the box's
        `(nz, ny, nx)` arrays and `pixel` and `share` match what it picks.
        """
        nz, ny, nx = self._shape
        row, column = np.indices((ny, nx), dtype=np.int64)
        if self.projection == "two-step":
            rows, columns = self.grid_shape
            # Neighbouring layers mostly sit the same way on the base, so the
            # last layer's pixels are kept for the next.
            pixels = {}
            for layer, steps in enumerate(self._layers):
                kept = {}
                for p, q, share in steps:
                    if (p, q) in pixels:
                        kept[p, q] = pixels[p, q]
                    elif self.wrap:
                        kept[p, q] = ((row + q) % ny) * nx + (column + p) % nx
                    else:
                        kept[p, q] = ((row + q - self._base_first[1]) * columns
                                      + column + p - self._base_first[0])
                    yield layer, kept[p, q], share
                pixels = kept
            return
        dz = np.diff(self._z_edges)
        heights = 0.5 * (self._z_edges[1:] + self._z_edges[:-1])
        row, column = row.ravel(), column.ravel()
        x = self._x_edges[0] + (column + 0.5) * self._pitch[0] - self._centre[0]
        y = self._y_edges[0] + (row + 0.5) * self._pitch[1] - self._centre[1]
        linear = np.vstack([self.frame[:2] / self._pitch[:, np.newaxis], self.frame[2]])
        for layer, (height, thickness) in enumerate(zip(heights, dz)):
            faces, reach = _cell_faces(linear, np.array([*self._pitch, thickness]))
            # Each cell is seen where its line of sight meets the base.
            base = np.column_stack([x, y]) + height * self._lean
            if self.wrap:
                base = np.mod(base + 0.5 * self._size, self._size) - 0.5 * self._size
            for cells, pixel, share in self._image_parts(faces, reach, base):
                yield (layer, row[cells], column[cells]), pixel, share

    def to_image(self, maps: np.ndarray) -> np.ndarray:
        """
        Maps on the accumulation grid, `(n_pixels, ...)`, onto the image's pixels.
        """
        if self.projection != "two-step":
            return maps
        flat = maps.reshape(maps.shape[0], -1)
        return np.asarray(self._resample @ flat).reshape(
            self.image_shape[0] * self.image_shape[1], *maps.shape[1:])

    def reference_cube(self, z_size: u.Quantity, meta=None) -> NDCube:
        """Empty cube with the image's coordinates, for `create_line_cube`."""
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ["SOLX", "SOLY", "SOLZ"]
        wcs.wcs.cunit = ["Mm", "Mm", "Mm"]
        wcs.wcs.crpix = [1, 1, 1]
        wcs.wcs.crval = [0.5 * (self.x_edges[0] + self.x_edges[1]).to_value(u.Mm),
                         0.5 * (self.y_edges[0] + self.y_edges[1]).to_value(u.Mm), 0.0]
        wcs.wcs.cdelt = [(self.x_edges[1] - self.x_edges[0]).to_value(u.Mm),
                         (self.y_edges[1] - self.y_edges[0]).to_value(u.Mm),
                         u.Quantity(z_size).to_value(u.Mm)]
        return NDCube(np.zeros((1, *self.image_shape)), wcs=wcs, meta=meta)
