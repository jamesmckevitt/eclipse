"""A box laid on the curved solar surface, seen from a position on the disk."""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from eclipse.projection import (CurvedSkyView, SkyView, _cell_faces, _cell_overlaps,
                                _overlaps, disk_frame, disk_geometry)
from eclipse.synthesis import build_em_tv, compute_dem
from eclipse.utils import velocity_grid

LOGT = np.array([5.9, 6.0, 6.1, 6.2, 6.3])
GRID = velocity_grid(1 * u.km / u.s, 30 * u.km / u.s)


def _edges(n, size):
    return (np.arange(n + 1) - n / 2) * size * u.Mm


def _random_box(shape, seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(5.9, 6.3, shape), rng.uniform(8.5, 9.5, shape)


@pytest.mark.parametrize("seed", range(4))
def test_the_compiled_faces_match_the_shared_ones(seed):
    """Cells each seen their own way share themselves as the same cells seen alike."""
    rng = np.random.default_rng(seed)
    frame, _ = disk_frame(*rng.uniform(-600, 600, 2) * u.arcsec, rng.uniform(0, 360) * u.deg)
    linear = np.vstack([frame[:2] / 0.2, frame[2]]) @ np.diag(rng.uniform(0.8, 1.2, 3))
    cell = np.array([0.2, 0.2, rng.uniform(0.05, 2.0)])
    centres = rng.uniform(0, 20, (50, 2))
    faces, reach = _cell_faces(linear, cell)
    shared = {}
    for p, q, share in _overlaps(faces, reach, centres):
        for n in range(len(centres)):
            shared[n, p[n], q[n]] = shared.get((n, p[n], q[n]), 0.0) + share[n]
    own = {}
    for p, q, share in _cell_overlaps(np.broadcast_to(linear, (50, 3, 3)), cell, centres):
        for n in range(len(centres)):
            own[n, p[n], q[n]] = own.get((n, p[n], q[n]), 0.0) + share[n]
    for key in set(shared) | set(own):
        assert own.get(key, 0.0) == pytest.approx(shared.get(key, 0.0), abs=1e-12)


@pytest.mark.parametrize("wrap", [True, False])
@pytest.mark.parametrize("x, y, rotation", [(500, 0, 0), (450, 350, 20)])
def test_a_huge_sun_gives_the_flat_view(wrap, x, y, rotation):
    log_t, log_n = _random_box((5, 6, 7))
    edges = (_edges(7, 0.2), _edges(6, 0.2), np.arange(6) * 0.1 * u.Mm)
    frame, _ = disk_frame(x * u.arcsec, y * u.arcsec, rotation * u.deg)
    flat = SkyView(frame, *edges, wrap, "exact")
    curved = CurvedSkyView(disk_geometry(x * u.arcsec, y * u.arcsec, rotation * u.deg), *edges,
                           wrap, radius=1e11 * u.Mm)
    assert curved.image_shape == flat.image_shape
    assert u.allclose(curved.x_edges, flat.x_edges) and u.allclose(curved.y_edges, flat.y_edges)
    seen_flat, _ = compute_dem(log_t, log_n, 1e7, LOGT, "z", sky=flat)
    seen_curved, _ = compute_dem(log_t, log_n, 1e7, LOGT, "z", sky=curved)
    assert seen_curved == pytest.approx(seen_flat, rel=1e-6, abs=1e-6 * seen_flat.max())


@pytest.mark.parametrize("x, y, rotation", [(0, 0, 0), (500, 0, 0), (-450, 350, 20)])
def test_every_cell_of_a_box_that_does_not_repeat_is_seen_once(x, y, rotation):
    log_t, log_n = _random_box((5, 30, 40))
    sky = CurvedSkyView(disk_geometry(x * u.arcsec, y * u.arcsec, rotation * u.deg),
                        _edges(40, 2.0), _edges(30, 2.0), np.arange(6) * 0.5 * u.Mm, False)
    above, _ = compute_dem(log_t, log_n, 1e7, LOGT, "z")
    seen, _ = compute_dem(log_t, log_n, 1e7, LOGT, "z", sky=sky)
    assert seen.sum(axis=(0, 1)) == pytest.approx(above.sum(axis=(0, 1)), rel=1e-12)


@pytest.mark.parametrize("x, y, rotation", [(0, 0, 0), (500, 300, 20), (-700, -300, 60)])
def test_a_thin_repeating_layer_is_seen_through_one_over_mu(x, y, rotation):
    """Each pixel sees n_e^2 times the layer's thickness over the cosine at its own place."""
    n, size, thickness = 30, 2.0, 0.01
    log_t, log_n = np.full((1, n, n), 6.0), np.full((1, n, n), 9.0)
    geometry = disk_geometry(x * u.arcsec, y * u.arcsec, rotation * u.deg)
    sky = CurvedSkyView(geometry, _edges(n, size), _edges(n, size),
                        np.array([0.0, thickness]) * u.Mm, True)
    dem, _ = compute_dem(log_t, log_n, thickness * 1e8, LOGT, "z", sky=sky)
    em = dem.sum(axis=-1) * 0.1
    # Where each pixel is seen from the Sun's centre on the sky, and the
    # cosine there.
    radius = const.R_sun.to_value(u.Mm)
    sun = -radius * geometry.sky[:2] @ geometry.up
    columns = 0.5 * (sky.x_edges[1:] + sky.x_edges[:-1]).to_value(u.Mm)
    rows = 0.5 * (sky.y_edges[1:] + sky.y_edges[:-1]).to_value(u.Mm)
    off = np.hypot(columns[np.newaxis, :] - sun[0], rows[:, np.newaxis] - sun[1]) / radius
    mu = np.sqrt(1.0 - off ** 2)
    # The layer is laid on the sphere with lengths across the way from the
    # box's centre shortened by sin(a) / a, at the angle a round the Sun from
    # it, so the emission measure per area grows by a / sin(a).
    across = (columns[np.newaxis, :] - sun[0])[..., np.newaxis] * geometry.sky[0]
    up = (rows[:, np.newaxis] - sun[1])[..., np.newaxis] * geometry.sky[1]
    point = across + up + radius * mu[..., np.newaxis] * geometry.sky[2]
    angle = np.arccos(np.clip(point @ geometry.up / radius, -1.0, 1.0))
    stretch = np.where(angle > 0, angle / np.sin(np.where(angle > 0, angle, 1.0)), 1.0)
    # Each cell is laid flat, along the surface at its centre, so neighbouring
    # cells meet with slivers between them, which move about a tenth of the
    # cell's size over the radius between pixels, more where seen at a slant.
    assert em == pytest.approx(1e18 * thickness * 1e8 / mu * stretch, rel=0.2 * size / radius)


def test_a_flow_along_the_surface_is_seen_away_on_one_side_and_towards_on_the_other():
    """At disk centre, a westward flow tilts away from the observer west of centre."""
    n, size, speed = 41, 10.0, 20.0
    shape = (2, 1, n)
    log_t = np.full(shape, 6.0)
    velocity = np.zeros((3, *shape))
    velocity[0] = speed * 1e5
    sky = CurvedSkyView(disk_geometry(0 * u.arcsec, 0 * u.arcsec), _edges(n, size),
                        _edges(1, size), np.array([0.0, 0.1, 0.2]) * u.Mm, False)
    em_tv = build_em_tv(log_t, velocity, LOGT, GRID, np.ones(shape), "z", sky=sky)
    profile = em_tv.sum(axis=(0, 2))
    mean = (profile * GRID.to_value(u.km / u.s)).sum(axis=1) / profile.sum(axis=1)
    columns = 0.5 * (sky.x_edges[1:] + sky.x_edges[:-1]).to_value(u.Mm)
    # A column seen at x from the centre is tilted by asin(x / R), so the
    # flow along the line of sight there is the speed times x / R.
    expected = speed * columns / const.R_sun.to_value(u.Mm)
    # The pixels at the ends of the box see only part of a column.
    lit = np.flatnonzero(profile.sum(axis=1) > 0)[1:-1]
    assert mean[lit] == pytest.approx(expected[lit], abs=0.02)
    assert mean[lit][0] < 0 < mean[lit][-1]


def test_a_box_past_the_limb_is_refused():
    with pytest.raises(ValueError, match="far side"):
        CurvedSkyView(disk_geometry(950 * u.arcsec, 0 * u.arcsec), _edges(30, 10.0),
                      _edges(20, 2.0), np.arange(3) * 0.5 * u.Mm, False)


def test_a_repeating_box_whose_image_leaves_the_disk_is_refused():
    with pytest.raises(ValueError, match="off the disk"):
        CurvedSkyView(disk_geometry(900 * u.arcsec, 0 * u.arcsec), _edges(40, 10.0),
                      _edges(20, 2.0), np.arange(3) * 0.5 * u.Mm, True)


def _synthesise(tmp_path, monkeypatch, extra, flow=(1.0, 2.0, 3.0)):
    import sys

    from eclipse import synthesis
    from eclipse.atmosphere import Atmosphere, write_atmosphere

    shape = (4, 6, 8)
    velocities = {f"velocity_{axis}": np.full(shape, value) * u.km / u.s
                  for axis, value in zip("xyz", flow) if value is not None}
    atmosphere = Atmosphere(x_edges=_edges(shape[2], 0.5), y_edges=_edges(shape[1], 0.5),
                            z_edges=np.arange(shape[0] + 1) * 0.1 * u.Mm,
                            temperature=np.full(shape, 1.0e6) * u.K,
                            electron_density=np.full(shape, 1.0e9) / u.cm**3, **velocities)
    path = write_atmosphere(atmosphere, tmp_path / "box.h5")
    monkeypatch.setattr(sys, "argv", [
        "synthesise-spectra", "--atmosphere", str(path), "--lines", "Fe12_195.1190",
        "--output-dir", str(tmp_path), "--output-name", "seen.h5", *extra])

    def flat_goft(lines, **kwargs):
        logT_grid, logN_grid = np.linspace(5.0, 7.0, 21), np.linspace(8.0, 10.0, 21)
        return ({"Fe12_195.1190": {"wl0": (195.119 * u.Angstrom).to(u.cm),
                                   "g_tn": np.ones((logN_grid.size, logT_grid.size)),
                                   "atom": 26, "ion": 12, "hdf5_dbase_root": None}},
                logT_grid, logN_grid)

    monkeypatch.setattr(synthesis, "compute_goft_fiasco", flat_goft)
    synthesis.main()
    return tmp_path / "seen.h5"


def test_curvature_is_recorded(tmp_path, monkeypatch):
    from eclipse.synthesis_file import load_synthesis

    path = _synthesise(tmp_path, monkeypatch,
                       ("--position", "300 arcsec", "200 arcsec", "--curvature"))
    config = load_synthesis(path)["config"]
    assert config["curvature"] is True
    assert config["projection"] == "exact"


def test_curvature_needs_every_velocity(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="velocity_y"):
        _synthesise(tmp_path, monkeypatch,
                    ("--position", "300 arcsec", "0 arcsec", "--curvature"), (1.0, None, 3.0))


def test_curvature_takes_only_the_exact_projection(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="--projection"):
        _synthesise(tmp_path, monkeypatch, ("--position", "300 arcsec", "0 arcsec",
                                            "--curvature", "--projection", "two-step"))


def test_curvature_is_only_for_a_position(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="--curvature is for a box seen from a position"):
        _synthesise(tmp_path, monkeypatch, ("--curvature",))
