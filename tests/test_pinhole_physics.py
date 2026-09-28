"""Where the light through a filter pinhole lands.

The filter is in the beam converging on the detector, so an EUV pinhole passes
a share of the light of every point whose cone covers it, a half disc of the
image beside it, and diffracts part of it into its Airy pattern about each of
those points, the rest staying in the image by interference with the light
through the foil. Visible light is taken to reach the hole head-on, and its
pattern is the hole's near-field one. These pin down each piece against a
result worked out independently: known closed forms, a brute-force
Rayleigh-Sommerfeld integral, and brute-force pixel integration.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube
from scipy.special import j0, j1

from euvst_response.config import AluminiumFilter, Detector_SWC, Simulation, Telescope_EUVST
from euvst_response.pinhole_diffraction import (
    _airy_irradiance,
    _converging_kernel,
    _head_on_irradiance,
    apply_euv_pinhole_diffraction,
    beam_footprint_radius,
    half_disc_fractions,
    head_on_pinhole_fractions,
)

EUV = 195.119e-10
VISIBLE = 600e-9
DISTANCE = 0.25
PIXEL = 13.5e-6


def test_the_beam_at_the_filter_is_the_distance_over_twice_the_f_number():
    """f/62.5 from the 0.159 arcsec, 13.5 micron pixels and the 280 mm primary: 2 mm at 250 mm."""
    focal_length = PIXEL / (0.159 * np.pi / 180 / 3600)
    expected = DISTANCE / (2 * focal_length / 0.28)
    assert beam_footprint_radius(Detector_SWC(), Telescope_EUVST()).to_value(u.m) == \
        pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(2.0e-3, rel=1e-2)


def test_the_half_disc_is_the_pixels_share_of_its_area():
    radius = 10.3
    shares = half_disc_fractions((60, 50), (30.2, 20.0), radius, True)
    assert shares.sum() == pytest.approx(np.pi * radius**2 / 2, rel=1e-12)
    assert shares.min() >= 0 and shares.max() <= 1 + 1e-12
    # The cut runs through the middle of column 20; nothing on the other side.
    assert np.all(shares[:, :20] == 0) and shares[30, 20] == pytest.approx(0.5)
    other = half_disc_fractions((60, 50), (30.2, 20.0), radius, False)
    assert np.all(other[:, 21:] == 0)
    assert other.sum() == pytest.approx(shares.sum(), rel=1e-12)
    # Brute force: the share of a fine grid of points in each pixel inside the half disc.
    fine = (np.arange(200) + 0.5) / 200 - 0.5
    for row, column in ((30, 29), (38, 26), (21, 28)):
        y, x = np.meshgrid(row + fine - 30.2, column + fine - 20.0, indexing="ij")
        inside = ((x**2 + y**2 <= radius**2) & (x >= 0)).mean()
        assert shares[row, column] == pytest.approx(inside, abs=5e-4)


@pytest.mark.parametrize("wavelength, diameter", [
    (VISIBLE, 1e-6), (VISIBLE, 50e-6), (VISIBLE, 400e-6), (VISIBLE, 1000e-6),
    (EUV, 50e-6), (EUV, 200e-6)])
def test_under_the_hole_the_near_field_is_the_known_one(wavelength, diameter):
    """4 sin^2(pi N_F / 2) of the light falling on the hole, per unit area of it."""
    radius = diameter / 2
    fresnel_number = radius**2 / (wavelength * DISTANCE)
    expected = 4 * np.sin(np.pi * fresnel_number / 2) ** 2 / (np.pi * radius**2)
    assert _head_on_irradiance(0.0, radius, wavelength, DISTANCE)[0] == \
        pytest.approx(expected, rel=1e-10)


def test_a_small_hole_is_in_the_far_field():
    """A 5 micron hole at 600 nm, N_F = 4e-5: the Airy pattern at the exact angle, everywhere."""
    r = np.array([0.0, 1e-3, 1e-2, 3e-2])
    near = _head_on_irradiance(r, 2.5e-6, VISIBLE, DISTANCE)
    far = _airy_irradiance(r, 2.5e-6, VISIBLE, DISTANCE)
    assert near == pytest.approx(far, rel=1e-4)


def _rayleigh_sommerfeld(r, radius, wavelength, n_rho=600, n_phi=1600):
    """The irradiance per unit transmitted power by direct integration over the hole, no expansion."""
    k = 2 * np.pi / wavelength
    rho, w_rho = np.polynomial.legendre.leggauss(n_rho)
    rho, w_rho = radius * (rho + 1) / 2, radius * w_rho / 2
    phi, w_phi = np.polynomial.legendre.leggauss(n_phi)
    phi, w_phi = np.pi * (phi + 1), np.pi * w_phi
    s = np.sqrt(DISTANCE**2 + r**2 + rho[:, None] ** 2 - 2 * r * rho[:, None] * np.cos(phi))
    field = (w_rho * rho) @ (DISTANCE / s * np.exp(1j * k * s) / s) @ w_phi / wavelength
    cos_theta = DISTANCE / np.hypot(DISTANCE, r)
    return abs(field) ** 2 * cos_theta / (np.pi * radius**2)


@pytest.mark.parametrize("r", [0.0, 2e-4, 5e-3, 3e-2])
def test_a_large_hole_lit_head_on_matches_the_full_diffraction_integral(r):
    """400 microns at 600 nm is N_F = 0.27, where the far field is 6% out under the hole."""
    radius = 200e-6
    assert _head_on_irradiance(r, radius, VISIBLE, DISTANCE)[0] == \
        pytest.approx(_rayleigh_sommerfeld(r, radius, VISIBLE), rel=5e-4)


def _brute_force_kernel(m, n, radius, wavelength, samples=48):
    """Light spread evenly over one pixel landing on the pixel (m, n) away, by midpoint sums."""
    fine = (np.arange(samples) + 0.5) / samples
    source = np.stack(np.meshgrid(fine, fine, indexing="ij"), -1).reshape(-1, 2)
    target = source + [m, n]
    total = 0.0
    for point in source:
        r = PIXEL * np.hypot(*(target - point).T)
        total += _airy_irradiance(r, radius, wavelength, DISTANCE).mean()
    return total / len(source) * PIXEL**2


@pytest.mark.parametrize("diameter, offsets", [(20e-6, [(0, 0), (3, 4), (40, 2)]),
                                               (200e-6, [(0, 0), (1, 0), (2, 1)])])
def test_the_pixel_kernel_is_the_airy_pattern_integrated_over_both_pixels(diameter, offsets):
    kernel = _converging_kernel(50, 10, diameter / 2, EUV, DISTANCE, PIXEL)
    for m, n in offsets:
        assert kernel[m, n] == pytest.approx(_brute_force_kernel(m, n, diameter / 2, EUV),
                                             rel=2e-3, abs=1e-9)


def test_the_pixel_kernel_holds_all_the_light_that_lands_near():
    """A 200 micron hole's Airy disc is 30 microns across; within 100 pixels lands all but its far wings."""
    kernel = _converging_kernel(101, 101, 100e-6, EUV, DISTANCE, PIXEL)
    everything = kernel.sum() * 4 - kernel[0].sum() * 2 - kernel[:, 0].sum() * 2 + kernel[0, 0]
    # The share of an Airy pattern beyond v is about 2 / (pi v); at 100 pixels v is about 170.
    v = np.pi * 200e-6 * 100 * PIXEL / (EUV * DISTANCE)
    assert 1 - 2 * 2 / (np.pi * v) < everything < 1


def test_the_filters_amplitude_transmission_has_its_throughput_and_its_layers_phase():
    flt = AluminiumFilter()
    wavelength = np.array([17.1, 19.5119, 21.1]) * u.nm
    t = flt.amplitude_transmission(wavelength)
    assert np.abs(t) ** 2 == pytest.approx(flt.total_throughput(wavelength).value, rel=1e-12)
    delta = {}
    for name in ("aluminium", "aluminium_oxide", "carbon"):
        table = np.loadtxt(getattr(flt, {"aluminium": "al_index_table",
                                          "aluminium_oxide": "oxide_index_table",
                                          "carbon": "c_index_table"}[name]), skiprows=2)
        delta[name] = np.interp(wavelength.value, table[:, 0], table[:, 1])
    path = (delta["aluminium"] * flt.al_thickness.to_value(u.nm)
            + delta["aluminium_oxide"] * flt.oxide_thickness.to_value(u.nm)
            + delta["carbon"] * flt.c_thickness.to_value(u.nm))
    assert np.angle(t) == pytest.approx(np.angle(np.exp(-2j * np.pi * path / wavelength.value)),
                                        abs=1e-12)


def _window(n_slit, n_scan, n_spectral, photons, increasing=True):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "SOLX", "SOLY"]
    wcs.wcs.cunit = ["cm", "Mm", "Mm"]
    step = 16.9e-11 if increasing else -16.9e-11
    wcs.wcs.crval = [195.119e-8, 0.0, 0.0]
    wcs.wcs.crpix = [(n_spectral + 1) / 2, 1, 1]
    wcs.wcs.cdelt = [step, 0.1, 0.1]
    return NDCube(np.full((n_slit, n_scan, n_spectral), photons), wcs=wcs, unit=u.photon / u.pix,
                  meta={"rest_wav": 195.119 * u.AA})


@pytest.mark.parametrize("increasing", [True, False])
def test_an_euv_pinhole_adds_what_the_foil_took_from_the_light_through_it_beside_it(increasing):
    """A 50 micron hole in the middle of a window that holds the beam's half disc.

    The filter is put 50 mm from the detector, where the beam is 30 pixels in
    radius, so that the window can be small.
    """
    det, tel = Detector_SWC(filter_distance=50 * u.mm), Telescope_EUVST()
    # The hole a quarter of the way from the short-wavelength end, so that the
    # half disc it serves is in the window.
    sim = Simulation(instrument="SWC", enable_pinholes=True, pinhole_sizes=[50 * u.um],
                     pinhole_positions=[0.5],
                     pinhole_positions_spectral=[0.25 if increasing else 0.75])
    n_rows, n_columns = 121, 61
    hole = 15.0 if increasing else 45.0
    cube = _window(n_rows, 2, n_columns, 1000.0, increasing)
    added = apply_euv_pinhole_diffraction(cube, det, sim, tel).data - cube.data
    assert np.array_equal(added[:, 0], added[:, 1])
    added = added[:, 0]

    wavelength = cube.axis_world_coords(2)[0]
    t = tel.filter.amplitude_transmission(wavelength)
    throughput = np.abs(t) ** 2
    footprint = beam_footprint_radius(det, tel).to_value(u.m) / PIXEL
    assert footprint == pytest.approx(29.6, abs=0.1)
    share = (25e-6 / (footprint * PIXEL)) ** 2 * 2 * half_disc_fractions(
        (n_rows, n_columns), (60.0, hole), footprint, increasing)
    through = 1000.0 / throughput * share

    # The light the hole adds: 2 Re[conj(t) (1 - t)] of what crosses it
    # stays, and |1 - t|^2 is diffracted, of which the window holds what the
    # kernel puts in it.
    stays = 2 * np.real(np.conj(t) * (1 - t))
    diffracted = np.abs(1 - t) ** 2
    in_window = np.empty((n_rows, n_columns))
    rows, columns = np.arange(n_rows), np.arange(n_columns)
    for column in columns:
        kernel = _converging_kernel(n_rows, n_columns, 25e-6,
                                    float(wavelength[column].to_value(u.m)), 0.05, PIXEL)
        in_window[:, column] = kernel[np.abs(rows[:, None] - rows)][:, :, np.abs(columns - column)].sum(
            axis=(1, 2))
    expected = (through * (stays + diffracted * in_window)).sum()
    assert added.sum() == pytest.approx(expected, rel=1e-9)
    # And that is 1 - T of the light through it, but for the Airy pattern's wings beyond the window.
    assert 0.98 * (through * (1 - throughput)).sum() < added.sum() < (through * (1 - throughput)).sum()

    # Well inside the half disc, where every neighbour sends as much light as
    # it receives, a pixel gains 1 - T of the light through the hole.
    row = 70
    inner = ((np.hypot(columns - hole, row - 60) < footprint - 10)
             & (np.abs(columns - hole) > 10) & (share[row] > 0))
    assert inner.sum() > 5
    assert added[row, inner] == pytest.approx(through[row, inner] * (1 - throughput[inner]),
                                              rel=1e-2)
    # On the short-wavelength side, only the Airy pattern's wings, which carry
    # J0^2 + J1^2 of the light beyond v = pi D r / (lambda L).
    short = columns < hole - 10 if increasing else columns > hole + 10
    v = np.pi * 50e-6 * 10 * PIXEL / (EUV * 0.05)
    assert added[:, short].max() < (through * diffracted).max() * (j0(v) ** 2 + j1(v) ** 2)


def _kernel_in_window(n_rows, n_columns, radius, wavelength, distance):
    """For each column, the share of its pixels' diffracted light the window holds."""
    in_window = np.empty((n_rows, n_columns))
    rows, columns = np.arange(n_rows), np.arange(n_columns)
    for column in columns:
        kernel = _converging_kernel(n_rows, n_columns, radius,
                                    float(wavelength[column].to_value(u.m)), distance, PIXEL)
        in_window[:, column] = kernel[np.abs(rows[:, None] - rows)][:, :, np.abs(columns - column)].sum(
            axis=(1, 2))
    return in_window


def _shift_a_column_on(cube):
    """A blur that moves each column's light to the next, as the optics' spectral blur moves some."""
    data = np.zeros_like(cube.data)
    data[..., 1:] = cube.data[..., :-1]
    return NDCube(data, wcs=cube.wcs, unit=cube.unit, meta=cube.meta)


def test_the_pinholes_light_is_weighed_at_its_own_wavelength_before_the_blur():
    """
    Moved a column on by the blur, each column's light keeps the filter's share of its own wavelength.

    Divided by the filter's share at the column it was moved to, and weighed
    by that column's, the light came out as the filter at the wrong wavelength
    would have passed it.
    """
    det, tel = Detector_SWC(filter_distance=50 * u.mm), Telescope_EUVST()
    sim = Simulation(instrument="SWC", enable_pinholes=True, pinhole_sizes=[50 * u.um],
                     pinhole_positions=[0.5], pinhole_positions_spectral=[0.25])
    n_rows, n_columns = 121, 61
    before = _window(n_rows, 1, n_columns, 1000.0)
    after = _shift_a_column_on(before)
    added = (apply_euv_pinhole_diffraction(after, det, sim, tel, unfocused=before,
                                           focus=_shift_a_column_on).data - after.data)[:, 0]

    wavelength = before.axis_world_coords(2)[0]
    t = tel.filter.amplitude_transmission(wavelength)
    footprint = beam_footprint_radius(det, tel).to_value(u.m) / PIXEL
    share = (25e-6 / (footprint * PIXEL)) ** 2 * 2 * half_disc_fractions(
        (n_rows, n_columns), (60.0, 15.0), footprint, True)
    in_window = _kernel_in_window(n_rows, n_columns, 25e-6, wavelength, 0.05)
    # The light of column j, without the filter, reaching column j + 1.
    through = 1000.0 / np.abs(t[:-1]) ** 2 * share[:, 1:]
    stays = 2 * np.real(np.conj(t[:-1]) * (1 - t[:-1]))
    diffracted = np.abs(1 - t[:-1]) ** 2
    expected = (through * (stays + diffracted * in_window[:, 1:])).sum()
    assert added.sum() == pytest.approx(expected, rel=1e-9)
    # Weighed at the column it reached, it would differ.
    t_there = t[1:]
    wrong = (1000.0 / np.abs(t_there) ** 2 * share[:, 1:]
             * (2 * np.real(np.conj(t_there) * (1 - t_there))
                + np.abs(1 - t_there) ** 2 * in_window[:, 1:])).sum()
    assert abs(wrong / expected - 1) > 1e-6


def test_windows_of_the_same_numbers_in_other_shapes_are_each_their_own():
    """The kept answer was found by the photons' bytes, which two shapes can share."""
    from euvst_response.pinhole_diffraction import _LAST_ADDED

    det, tel = Detector_SWC(filter_distance=50 * u.mm), Telescope_EUVST()
    sim = Simulation(instrument="SWC", enable_pinholes=True, pinhole_sizes=[50 * u.um],
                     pinhole_positions=[0.5], pinhole_positions_spectral=[0.25])
    apply_euv_pinhole_diffraction(_window(60, 2, 61, 1000.0), det, sim, tel)
    second = apply_euv_pinhole_diffraction(_window(120, 1, 61, 1000.0), det, sim, tel)
    _LAST_ADDED.clear()
    fresh = apply_euv_pinhole_diffraction(_window(120, 1, 61, 1000.0), det, sim, tel)
    assert second.data.shape == (120, 1, 61)
    assert np.array_equal(second.data, fresh.data)
