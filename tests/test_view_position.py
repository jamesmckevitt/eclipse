"""A box seen from a position on the solar disk, along a slanted line of sight."""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from eclipse.projection import PROJECTIONS, SkyView, _cell_faces, _overlaps, _patch_faces
from eclipse.synthesis import build_em_tv, compute_dem, disk_frame
from eclipse.utils import velocity_grid

LOGT = np.array([5.9, 6.0, 6.1, 6.2, 6.3])
GRID = velocity_grid(5 * u.km / u.s, 300 * u.km / u.s)


def _edges(n, size):
    return (np.arange(n + 1) - n / 2) * size * u.Mm


def _slab(nz=40, ny=24, nx=24, size=0.2, dz=0.1):
    """Uniform plasma at 1 MK filling a box of `nz` layers, `dz` Mm thick."""
    shape = (nz, ny, nx)
    z_edges = np.arange(nz + 1) * dz * u.Mm
    return (np.full(shape, 6.0), np.full(shape, 9.0), _edges(nx, size), _edges(ny, size),
            z_edges, dz * 1e8)


def _random_box(shape, seed=2):
    rng = np.random.default_rng(seed)
    log_t = rng.uniform(5.9, 6.3, shape)
    return log_t, rng.uniform(8.5, 9.5, shape)


def test_disk_centre_looks_straight_down():
    frame, angle = disk_frame(0 * u.arcsec, 0 * u.arcsec)
    assert frame == pytest.approx(np.eye(3), abs=1e-12)
    assert angle.to_value(u.deg) == pytest.approx(0.0, abs=1e-9)


def test_the_local_vertical_leans_away_from_disk_centre():
    """West of centre the vertical leans west."""
    frame, _ = disk_frame(600 * u.arcsec, 0 * u.arcsec)
    up = frame[:, 2]
    assert up[0] > 0 and up[1] == pytest.approx(0.0, abs=1e-12)
    assert frame.T @ frame == pytest.approx(np.eye(3), abs=1e-12)
    # North on the box is north on the sky on the equator.
    assert frame[:, 1] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


@pytest.mark.parametrize("x, y", [(600, 0), (-300, 450), (200, -700), (0, 950)])
def test_the_heliocentric_angle_follows_the_sine_rule(x, y):
    """sin(angle) = D sin(r) / R_sun, at angle r from disk centre seen from distance D."""
    _, angle = disk_frame(x * u.arcsec, y * u.arcsec)
    tx, ty = (x * u.arcsec).to_value(u.rad), (y * u.arcsec).to_value(u.rad)
    sin_r = np.sqrt(1.0 - (np.cos(tx) * np.cos(ty)) ** 2)
    expected = np.degrees(np.arcsin(sin_r * (const.au / const.R_sun).decompose().value))
    assert angle.to_value(u.deg) == pytest.approx(expected, rel=1e-9)


def test_a_position_off_the_disk_is_refused():
    with pytest.raises(ValueError, match="limb"):
        disk_frame(1000 * u.arcsec, 0 * u.arcsec)


def test_a_rotation_turns_the_box_about_the_vertical():
    """Turned by 90 deg, the box's x points north and its y east."""
    frame, _ = disk_frame(0 * u.arcsec, 0 * u.arcsec, 90 * u.deg)
    assert frame[:, 0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert frame[:, 1] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)
    turned, angle = disk_frame(300 * u.arcsec, -200 * u.arcsec, 30 * u.deg)
    plain, plain_angle = disk_frame(300 * u.arcsec, -200 * u.arcsec)
    assert angle == plain_angle
    # The vertical, and the line of sight in the box's base, turn with it.
    assert turned[:, 2] == pytest.approx(plain[:, 2], abs=1e-15)
    assert np.hypot(*turned[2, :2]) == pytest.approx(np.hypot(*plain[2, :2]), rel=1e-12)


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("wrap", [True, False])
def test_disk_centre_gives_the_view_from_above(projection, wrap):
    log_t, log_n = _random_box((6, 5, 7))
    frame, _ = disk_frame(0 * u.arcsec, 0 * u.arcsec)
    sky = SkyView(frame, _edges(7, 0.2), _edges(5, 0.2), np.arange(7) * 0.1 * u.Mm, wrap,
                  projection)
    above, _ = compute_dem(log_t, log_n, 1e7, LOGT, "z")
    seen, _ = compute_dem(log_t, log_n, 1e7, LOGT, "z", sky=sky)
    assert seen == pytest.approx(above, rel=1e-12, abs=1e-12 * above.max())
    assert u.allclose(sky.x_edges, _edges(7, 0.2)) and u.allclose(sky.y_edges, _edges(5, 0.2))


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_a_box_turned_at_disk_centre_is_the_view_from_above_turned(projection):
    """Turned by 90 deg the box's x runs north and its y east, so rows become columns."""
    log_t, log_n = _random_box((6, 4, 7))
    frame, _ = disk_frame(0 * u.arcsec, 0 * u.arcsec, 90 * u.deg)
    sky = SkyView(frame, _edges(7, 0.2), _edges(4, 0.2), np.arange(7) * 0.1 * u.Mm, True,
                  projection)
    above, _ = compute_dem(log_t, log_n, 1e7, LOGT, "z")
    seen, _ = compute_dem(log_t, log_n, 1e7, LOGT, "z", sky=sky)
    # Row j, column i from above is seen in row i (north), column ny - 1 - j (west).
    turned = np.transpose(above, (1, 0, 2))[:, ::-1]
    assert seen == pytest.approx(turned, rel=1e-12, abs=1e-12 * above.max())


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("x, y, rotation, nz",
                         [(500, 0, 0, 40), (0, -400, 0, 40), (450, 350, 0, 40),
                          (450, 350, 30, 40), (800, 300, 0, 40), (500, 0, 0, 1)])
def test_a_repeating_slab_is_uniform_and_seen_through_one_over_cos(projection, x, y, rotation,
                                                                    nz):
    """
    Every pixel of a repeating, uniform slab sees n_e^2 times its height over cos.

    One layer is the hardest case, a thin layer seen at a slant.
    """
    log_t, log_n, xe, ye, ze, dz_cm = _slab(nz=nz)
    frame, angle = disk_frame(x * u.arcsec, y * u.arcsec, rotation * u.deg)
    sky = SkyView(frame, xe, ye, ze, True, projection)
    dem, _ = compute_dem(log_t, log_n, dz_cm, LOGT, "z", sky=sky)
    em = dem.sum(axis=-1) * 0.1                       # cm^-5, LOGT is 0.1 apart
    column = 1e18 * dz_cm * nz                        # n_e^2 times the slab's height
    assert em == pytest.approx(column / np.cos(angle.to_value(u.rad)), rel=1e-9)


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_a_long_thin_repeating_strip_fills_its_image(projection):
    """A strip turned on the sky is seen again across its width many times over."""
    log_t, log_n, xe, ye, ze, dz_cm = _slab(nz=6, ny=2, nx=40)
    frame, angle = disk_frame(600 * u.arcsec, 0 * u.arcsec, 30 * u.deg)
    sky = SkyView(frame, xe, ye, ze, True, projection)
    dem, _ = compute_dem(log_t, log_n, dz_cm, LOGT, "z", sky=sky)
    em = dem.sum(axis=-1) * 0.1
    assert em == pytest.approx(1e18 * dz_cm * 6 / np.cos(angle.to_value(u.rad)), rel=1e-9)


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("x, y, rotation", [(500, 0, 0), (450, 350, 20), (-700, -500, 75)])
def test_every_cell_of_a_box_that_does_not_repeat_is_seen_once(projection, x, y, rotation):
    log_t, log_n = _random_box((7, 6, 8))
    frame, _ = disk_frame(x * u.arcsec, y * u.arcsec, rotation * u.deg)
    sky = SkyView(frame, _edges(8, 0.2), _edges(6, 0.2), np.arange(8) * 0.1 * u.Mm, False,
                  projection)
    above, _ = compute_dem(log_t, log_n, 1e7, LOGT, "z")
    seen, _ = compute_dem(log_t, log_n, 1e7, LOGT, "z", sky=sky)
    assert seen.sum(axis=(0, 1)) == pytest.approx(above.sum(axis=(0, 1)), rel=1e-12)


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_a_raised_feature_is_seen_outward_from_disk_centre(projection):
    """A cell 2 Mm up, west of centre, is seen 2 sin(angle) Mm further west."""
    nz, n = 21, 41
    log_t = np.full((nz, n, n), 4.0)
    log_t[20, n // 2, n // 2] = 6.0
    frame, angle = disk_frame(700 * u.arcsec, 0 * u.arcsec)
    sky = SkyView(frame, _edges(n, 0.2), _edges(n, 0.2), np.arange(nz + 1) * 0.1 * u.Mm,
                  False, projection)
    dem, _ = compute_dem(log_t, np.full(log_t.shape, 9.0), 1e7, LOGT, "z", sky=sky)
    image = dem.sum(axis=-1)
    columns = 0.5 * (sky.x_edges[1:] + sky.x_edges[:-1]).to_value(u.Mm)
    rows = 0.5 * (sky.y_edges[1:] + sky.y_edges[:-1]).to_value(u.Mm)
    height = 2.05                                      # the cell's centre, Mm
    # Mean position over the image's pixels, which are 0.2 Mm: putting a cell's shadow into
    # pixels moves its mean by a small fraction of one.
    assert (image.sum(axis=0) * columns).sum() / image.sum() == pytest.approx(
        height * np.sin(angle.to_value(u.rad)), abs=0.01)
    assert (image.sum(axis=1) * rows).sum() / image.sum() == pytest.approx(0.0, abs=1e-12)


def test_a_tall_column_near_the_limb_is_seen_without_gaps():
    """Cells taller than a pixel's width on the sky fill the pixels between them."""
    nz, n = 6, 21
    log_t = np.full((nz, n, n), 4.0)
    log_t[:, n // 2, n // 2] = 6.0
    frame, _ = disk_frame(930 * u.arcsec, 0 * u.arcsec)
    sky = SkyView(frame, _edges(n, 0.2), _edges(n, 0.2), np.arange(nz + 1) * 0.5 * u.Mm,
                  False, "exact")
    dem, _ = compute_dem(log_t, np.full(log_t.shape, 9.0), 1e7, LOGT, "z", sky=sky)
    streak = dem.sum(axis=(0, 2))
    lit = np.flatnonzero(streak > 0)
    assert lit.size > nz + 2
    assert np.all(streak[lit[0]:lit[-1] + 1] > 0)


@pytest.mark.parametrize("seed", range(6))
def test_a_cells_shares_are_the_parts_of_it_in_front_of_each_pixel(seed):
    """Against counting points spread evenly through the cell."""
    rng = np.random.default_rng(seed)
    frame, _ = disk_frame(*rng.uniform(-650, 650, 2) * u.arcsec, rng.uniform(0, 360) * u.deg)
    pitch = rng.uniform(0.5, 1.5, 2)
    linear = np.vstack([frame[:2] / pitch[:, np.newaxis], frame[2]])
    cell = rng.uniform(0.3, 2.0, 3)
    centre = rng.uniform(0, 1, (1, 2))
    faces, reach = _cell_faces(linear, cell)
    shares = {(int(p[0]), int(q[0])): float(share[0])
              for p, q, share in _overlaps(faces, reach, centre)}
    assert sum(shares.values()) == pytest.approx(1.0, rel=1e-12)
    m = 80
    steps = (np.arange(m) + 0.5) / m - 0.5
    points = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), -1).reshape(-1, 3) * cell
    seen = points @ linear[:2].T + centre
    counted = {}
    for p, q in np.floor(seen).astype(int):
        counted[(p, q)] = counted.get((p, q), 0.0) + 1.0 / len(points)
    for pixel in set(shares) | set(counted):
        assert shares.get(pixel, 0.0) == pytest.approx(counted.get(pixel, 0.0), abs=0.01)


def test_a_flat_patchs_shares_are_its_area_in_each_pixel():
    side_a, side_b = np.array([0.9, 0.3]), np.array([-0.2, 1.1])
    faces, reach = _patch_faces(side_a, side_b)
    shares = {(int(p[0]), int(q[0])): float(share[0])
              for p, q, share in _overlaps(faces, reach, np.array([[0.4, 0.7]]))}
    assert sum(shares.values()) == pytest.approx(1.0, rel=1e-12)
    m = 400
    steps = (np.arange(m) + 0.5) / m - 0.5
    a, b = np.meshgrid(steps, steps, indexing="ij")
    seen = np.outer(a.ravel(), side_a) + np.outer(b.ravel(), side_b) + [0.4, 0.7]
    for (p, q), share in shares.items():
        inside = (np.floor(seen[:, 0]) == p) & (np.floor(seen[:, 1]) == q)
        assert share == pytest.approx(inside.mean(), abs=5e-3)


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_the_emission_measure_by_velocity_follows_the_same_pixels(projection):
    log_t, log_n, xe, ye, ze, dz_cm = _slab(nz=10, ny=8, nx=8)
    log_t = log_t + np.linspace(-0.2, 0.2, 8)
    velocity = np.zeros(log_t.shape)
    frame, _ = disk_frame(300 * u.arcsec, 200 * u.arcsec, 40 * u.deg)
    sky = SkyView(frame, xe, ye, ze, True, projection)
    em_sq_dh = 1e18 * dz_cm * np.ones(log_t.shape)
    em_tv = build_em_tv(log_t, velocity, LOGT, GRID, em_sq_dh, "z", sky=sky)
    dem, _ = compute_dem(log_t, log_n, dz_cm, LOGT, "z", sky=sky)
    assert em_tv.sum(axis=-1) == pytest.approx(dem * 0.1, rel=1e-12, abs=1e-3)


def _flat_goft(lines, **kwargs):
    """Contribution function of 1 everywhere, in place of fiasco."""
    logT_grid = np.linspace(5.0, 7.0, 21)
    logN_grid = np.linspace(8.0, 10.0, 21)
    return ({"Fe12_195.1190": {"wl0": (195.119 * u.Angstrom).to(u.cm),
                               "g_tn": np.ones((logN_grid.size, logT_grid.size)),
                               "atom": 26, "ion": 12, "hdf5_dbase_root": None}},
            logT_grid, logN_grid)


def _synthesise(tmp_path, monkeypatch, position, flow, extra=()):
    """Synthesise a uniform box with a uniform `flow` (x, y, z) in km/s, seen from `position`."""
    import sys

    from eclipse import synthesis
    from eclipse.atmosphere import Atmosphere, write_atmosphere

    shape = (8, 6, 10)
    velocities = {f"velocity_{axis}": np.full(shape, value) * u.km / u.s
                  for axis, value in zip("xyz", flow) if value is not None}
    atmosphere = Atmosphere(x_edges=_edges(shape[2], 0.2), y_edges=_edges(shape[1], 0.2),
                            z_edges=np.arange(shape[0] + 1) * 0.1 * u.Mm,
                            temperature=np.full(shape, 1.0e6) * u.K,
                            electron_density=np.full(shape, 1.0e9) / u.cm**3, **velocities)
    path = write_atmosphere(atmosphere, tmp_path / "box.h5")
    monkeypatch.setattr(sys, "argv", [
        "synthesise-spectra", "--atmosphere", str(path), "--lines", "Fe12_195.1190",
        "--output-dir", str(tmp_path), "--output-name", "seen.h5",
        "--position", *position, *extra])
    monkeypatch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    synthesis.main()
    return tmp_path / "seen.h5"


def test_the_doppler_shift_is_the_flow_along_the_slanted_line_of_sight(tmp_path, monkeypatch):
    """West of centre, a westward flow moves away from the observer and an upflow towards."""
    from eclipse.synthesis_file import load_synthesis

    position = ("500 arcsec", "0 arcsec")
    saved = load_synthesis(_synthesise(tmp_path, monkeypatch, position, (12.0, None, 8.0)))
    frame, angle = disk_frame(500 * u.arcsec, 0 * u.arcsec)
    expected = -(frame[2, 0] * 12.0 + frame[2, 2] * 8.0)
    theta = angle.to_value(u.rad)
    assert expected == pytest.approx(12.0 * np.sin(theta) - 8.0 * np.cos(theta), rel=1e-12)
    em_tv = saved["em_tv"].sum(axis=(0, 1, 2))
    velocity = saved["vel_grid"].to_value(u.km / u.s)
    assert (em_tv * velocity).sum() / em_tv.sum() == pytest.approx(expected, rel=1e-6)
    assert [u.Quantity(value) for value in saved["config"]["position"]] == [
        500 * u.arcsec, 0 * u.arcsec]


def test_a_position_needs_the_velocity_along_the_slanted_line_of_sight(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="velocity_x"):
        _synthesise(tmp_path, monkeypatch, ("500 arcsec", "0 arcsec"), (None, None, 8.0))


def test_a_position_sets_the_line_of_sight_itself(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="--integration-axis"):
        _synthesise(tmp_path, monkeypatch, ("0 arcsec", "500 arcsec"), (None, 3.0, 8.0),
                    extra=("--integration-axis", "x"))


def test_the_image_is_where_the_emission_was_put(tmp_path, monkeypatch):
    """The synthesis's pixel edges are the sky view's, centred on the box's centre."""
    from eclipse.synthesis_file import read_synthesis

    saved = read_synthesis(_synthesise(tmp_path, monkeypatch, ("0 arcsec", "600 arcsec"),
                                       (None, 2.0, 1.0)))
    frame, _ = disk_frame(0 * u.arcsec, 600 * u.arcsec)
    sky = SkyView(frame, _edges(10, 0.2), _edges(6, 0.2), np.arange(9) * 0.1 * u.Mm, wrap=True)
    assert u.allclose(saved.x_edges, sky.x_edges, atol=1e-9 * u.Mm)
    assert u.allclose(saved.y_edges, sky.y_edges, atol=1e-9 * u.Mm)


@pytest.mark.parametrize("option", [("--no-wrap",), ("--rotation", "30 deg"),
                                    ("--projection", "exact")])
def test_the_view_options_are_only_for_a_position(tmp_path, monkeypatch, option):
    import sys

    from eclipse import synthesis

    monkeypatch.setattr(sys, "argv", ["synthesise-spectra", "--atmosphere", "a.h5",
                                      "--lines", "Fe12_195.1190", *option])
    with pytest.raises(ValueError, match=f"{option[0]} is for a box seen from a position"):
        synthesis.main()


def test_a_rotation_must_be_an_angle(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="--rotation"):
        _synthesise(tmp_path, monkeypatch, ("0 arcsec", "500 arcsec"), (None, 3.0, 8.0),
                    extra=("--rotation", "3 Mm"))


@pytest.mark.parametrize("extra, wrap", [((), True), (("--no-wrap",), False),
                                         (("--crop-x", "-0.6 Mm", "0.6 Mm"), False)])
def test_the_view_is_recorded(tmp_path, monkeypatch, extra, wrap):
    """The position, rotation, projection and whether the box repeats are in the config."""
    from eclipse.synthesis_file import load_synthesis, read_synthesis

    path = _synthesise(tmp_path, monkeypatch, ("300 arcsec", "-200 arcsec"), (1.0, 2.0, 3.0),
                       extra=("--rotation", "30 deg", "--projection", "exact", *extra))
    config = load_synthesis(path)["config"]
    assert u.Quantity(config["rotation"]) == 30 * u.deg
    assert config["projection"] == "exact"
    assert config["wrap"] is wrap
    frame, _ = disk_frame(300 * u.arcsec, -200 * u.arcsec, 30 * u.deg)
    shape = (8, 6, 6) if "--crop-x" in extra else (8, 6, 10)
    x_edges = _edges(shape[2], 0.2)
    sky = SkyView(frame, x_edges, _edges(6, 0.2), np.arange(9) * 0.1 * u.Mm, wrap, "exact")
    assert u.allclose(read_synthesis(path).x_edges, sky.x_edges, atol=1e-9 * u.Mm)


def test_an_explicit_view_from_above_is_not_taken_with_a_position(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="--integration-axis"):
        _synthesise(tmp_path, monkeypatch, ("0 arcsec", "500 arcsec"), (None, 3.0, 8.0),
                    extra=("--integration-axis", "z"))


@pytest.mark.parametrize("x, y", [(600, 0), (-300, 450), (200, -700)])
def test_the_frame_agrees_with_sunpys_coordinates(x, y):
    """Moving a point up, or north on the Sun, moves it on the sky as the frame says."""
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import HeliographicStonyhurst, Helioprojective

    observer = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, const.au, obstime="2026-01-01")
    sky = Helioprojective(observer=observer, obstime="2026-01-01")
    on_disk = HeliographicStonyhurst(obstime="2026-01-01")
    point = SkyCoord(x * u.arcsec, y * u.arcsec,
                     frame=sky).make_3d()
    surface = SkyCoord(point).transform_to(on_disk)
    step = 1e-3 * u.Mm

    def seen(lon, lat, radius):
        moved = SkyCoord(lon, lat, radius, frame=on_disk).transform_to(sky)
        return np.array([moved.Tx.to_value(u.rad), moved.Ty.to_value(u.rad)])

    start = seen(surface.lon, surface.lat, surface.radius)
    up = seen(surface.lon, surface.lat, surface.radius + step) - start
    north = seen(surface.lon, surface.lat + (step / surface.radius).decompose() * u.rad,
                 surface.radius) - start
    frame, _ = disk_frame(x * u.arcsec, y * u.arcsec)
    # A step at the Sun subtends step / distance, from the observer to the point.
    scale = (step / point.distance).decompose().value
    # To within the finite step.
    assert up / scale == pytest.approx(frame[:2, 2], abs=1e-5)
    assert north / scale == pytest.approx(frame[:2, 1], abs=1e-5)
