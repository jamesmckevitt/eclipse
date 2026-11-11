"""A box seen from a position on the solar disk, along a slanted line of sight."""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from eclipse.synthesis import SkyView, build_em_tv, compute_dem, disk_frame
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


def test_disk_centre_gives_the_view_from_above():
    rng = np.random.default_rng(2)
    log_t = rng.uniform(5.9, 6.3, (6, 5, 7))
    log_n = rng.uniform(8.5, 9.5, log_t.shape)
    frame, _ = disk_frame(0 * u.arcsec, 0 * u.arcsec)
    sky = SkyView(frame, _edges(7, 0.2), _edges(5, 0.2), np.arange(7) * 0.1 * u.Mm, wrap=True)
    above, _ = compute_dem(log_t, log_n, 1e7, LOGT, "z")
    seen, _ = compute_dem(log_t, log_n, 1e7, LOGT, "z", sky=sky)
    # The sky's grid has a pixel to spare on each side.
    assert seen[1:-1, 1:-1] == pytest.approx(above, rel=1e-12)
    assert seen.sum() == pytest.approx(above.sum(), rel=1e-12)


@pytest.mark.parametrize("x, y", [(500, 0), (0, -400), (450, 350)])
def test_a_slanted_line_of_sight_crosses_more_plasma(x, y):
    """Every cell's emission is kept, and a pixel of a repeating slab sees 1 / cos of it."""
    log_t, log_n, xe, ye, ze, dz_cm = _slab()
    frame, angle = disk_frame(x * u.arcsec, y * u.arcsec)
    sky = SkyView(frame, xe, ye, ze, wrap=True)
    dem, _ = compute_dem(log_t, log_n, dz_cm, LOGT, "z", sky=sky)
    em = dem.sum(axis=-1) * 0.1                       # cm^-5, LOGT is 0.1 apart
    column = 1e18 * dz_cm * log_t.shape[0]            # n_e^2 times the slab's height
    # Each cell's emission goes to the sky's pixels, which are as big as the cells across.
    assert em.sum() == pytest.approx(column * log_t.shape[1] * log_t.shape[2], rel=1e-12)
    lit = em > 0.5 * column
    # The median leaves out the part-filled pixels round the edge of the image.
    assert np.median(em[lit]) == pytest.approx(column / np.cos(angle.to_value(u.rad)), rel=0.02)


def test_a_raised_feature_is_seen_outward_from_disk_centre():
    """A cell 2 Mm up, west of centre, is seen 2 sin(angle) Mm further west."""
    nz, n = 21, 41
    log_t = np.full((nz, n, n), 4.0)
    log_t[20, n // 2, n // 2] = 6.0
    frame, angle = disk_frame(700 * u.arcsec, 0 * u.arcsec)
    sky = SkyView(frame, _edges(n, 0.2), _edges(n, 0.2), np.arange(nz + 1) * 0.1 * u.Mm,
                  wrap=False)
    dem, _ = compute_dem(log_t, np.full(log_t.shape, 9.0), 1e7, LOGT, "z", sky=sky)
    image = dem.sum(axis=-1)
    columns = 0.5 * (sky.x_edges[1:] + sky.x_edges[:-1]).to_value(u.Mm)
    rows = 0.5 * (sky.y_edges[1:] + sky.y_edges[:-1]).to_value(u.Mm)
    height = 2.05                                      # the cell's centre, Mm
    assert (image.sum(axis=0) * columns).sum() / image.sum() == pytest.approx(
        height * np.sin(angle.to_value(u.rad)), rel=1e-9)
    assert (image.sum(axis=1) * rows).sum() / image.sum() == pytest.approx(0.0, abs=1e-12)


def test_the_emission_measure_by_velocity_follows_the_same_pixels():
    log_t, log_n, xe, ye, ze, dz_cm = _slab(nz=10, ny=8, nx=8)
    velocity = np.zeros(log_t.shape)
    frame, _ = disk_frame(300 * u.arcsec, 200 * u.arcsec)
    sky = SkyView(frame, xe, ye, ze, wrap=True)
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


def test_no_wrap_is_only_for_a_position(tmp_path, monkeypatch):
    import sys

    from eclipse import synthesis

    monkeypatch.setattr(sys, "argv", ["synthesise-spectra", "--atmosphere", "a.h5",
                                      "--lines", "Fe12_195.1190", "--no-wrap"])
    with pytest.raises(ValueError, match="--position"):
        synthesis.main()


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
