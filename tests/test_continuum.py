"""The continuum under the lines: free-free, free-bound and two-photon emission.

ECLIPSE synthesised only the lines named for a run. The continuum is now
worked out too, from the same emission measure, and kept in the synthesis
file as entries of its own, one for each group of overlapping windows, so
that a window observed adds it once.
"""
import dataclasses
import sys

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from euvst_response import synthesis
from euvst_response.atmosphere import Atmosphere, write_atmosphere
from euvst_response.continuum import (compute_continuum_fiasco, continuum_spectra,
                                      continuum_windows, is_continuum)
from euvst_response.synthesis_file import (SpectralLine, Synthesis, load_synthesis,
                                           read_synthesis, read_synthesis_products)

C_CM_S = const.c.to_value(u.cm / u.s)


def _window(rest, half_width=300.0, step=5.0):
    velocity = np.arange(-half_width, half_width + step / 2, step) * 1e5
    return (rest * (1 + velocity / C_CM_S)) * u.AA


def test_overlapping_windows_share_one_continuum_and_others_have_their_own():
    windows = continuum_windows([_window(195.119), _window(195.179), _window(171.073)])
    assert len(windows) == 2
    names = sorted(windows)
    assert all(is_continuum(name) for name in names)
    blend = windows[[name for name in names if "195" in name][0]]
    # Evenly spaced, at the finest spacing of the two, across both windows.
    steps = np.diff(blend.to_value(u.AA))
    assert np.allclose(steps, steps[0], rtol=1e-9)
    assert steps[0] == pytest.approx(np.min(np.diff(_window(195.119).value)))
    # Its bins cover the windows' bins, each half a spacing beyond its ends.
    first, last = _window(195.119).to_value(u.AA), _window(195.179).to_value(u.AA)
    edges = blend.to_value(u.AA)
    assert edges[0] - steps[0] / 2 <= first[0] - (first[1] - first[0]) / 2 + 1e-12
    assert edges[-1] + steps[0] / 2 >= last[-1] + (last[-1] - last[-2]) / 2 - 1e-12


def test_the_continuum_takes_each_pixels_emission_measure_and_density():
    logN = np.array([8.0, 9.0, 10.0])
    free = np.array([[1.0, 2.0], [3.0, 4.0]])  # (nT, n_wavelength)
    two_photon = np.zeros((3, 2, 2))
    two_photon[:, 1, 0] = [10.0, 20.0, 40.0]  # the second temperature, first wavelength
    em = np.array([[[1.0, 0.0], [0.0, 2.0]]])  # (1 row, 2 columns, nT)
    density = np.array([[[1e9, 1e9], [1e9, 10 ** 9.5]]])
    spectra = continuum_spectra(em, density, logN, free, two_photon)
    assert spectra.shape == (1, 2, 2)
    assert np.allclose(spectra[0, 0], free[0])
    # Half way between 10^9 and 10^10 in log10 n_e, the two-photon table is half way too.
    assert np.allclose(spectra[0, 1], 2.0 * (free[1] + np.array([30.0, 0.0])))
    # Off the grid of densities, as for the lines, the two-photon emission is zero.
    off = continuum_spectra(em, np.full(em.shape, 1e12), logN, free, two_photon)
    assert np.allclose(off[0, 1], 2.0 * free[1])


def test_a_window_adds_the_continuum_once_however_many_lines_reach_it():
    wavelength = np.linspace(195.0, 195.3, 31) * u.AA
    image = (1, 1, wavelength.size)
    one = SpectralLine(np.full(image, 1.0) * synthesis_unit(), wavelength, 195.119 * u.AA)
    two = SpectralLine(np.full(image, 2.0) * synthesis_unit(), wavelength, 195.179 * u.AA)
    grid = continuum_windows([wavelength])
    name, continuum_wavelength = next(iter(grid.items()))
    continuum = SpectralLine(np.full((1, 1, continuum_wavelength.size), 0.5) * synthesis_unit(),
                             continuum_wavelength,
                             continuum_wavelength[continuum_wavelength.size // 2])
    lines = {"Fe12_195.1190": one, "Fe12_195.1790": two, name: continuum}
    edges = np.array([0.0, 1.0]) * u.Mm
    summed = Synthesis(lines=lines, x_edges=edges, y_edges=edges).summed("Fe12_195.1190")
    assert np.allclose(summed.to_value(synthesis_unit()), 3.5)


def synthesis_unit():
    return u.erg / (u.s * u.cm**2 * u.sr * u.cm)


def _flat_goft(lines, **kwargs):
    """A contribution function of 1 everywhere, in place of fiasco."""
    logT_grid = np.linspace(5.0, 7.0, 21)
    logN_grid = np.linspace(8.0, 10.0, 21)
    goft = {name: {"wl0": (float(name.split("_")[1]) * u.AA).to(u.cm),
                   "g_tn": np.ones((logN_grid.size, logT_grid.size)),
                   "atom": 26, "ion": 12, "hdf5_dbase_root": None} for name in lines}
    return goft, logT_grid, logN_grid


FLAT = 1e-30  # erg cm3 / (s sr cm) per n_e^2 dh, at every temperature and wavelength


def _flat_continuum(wavelength, logT_grid, logN_grid, **kwargs):
    """A continuum of FLAT everywhere, with no two-photon part, in place of fiasco."""
    n = np.atleast_1d(wavelength).size
    return (np.full((len(logT_grid), n), FLAT),
            np.zeros((len(logN_grid), len(logT_grid), n)))


def _synthesise(tmp_path, monkeypatch, lines, continuum):
    tmp_path.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = 4, 3, 2
    density = (1.0e9 / u.cm**3 * 1.2 * const.u).to(u.g / u.cm**3)
    atmosphere = Atmosphere(
        temperature=np.full((nz, ny, nx), 1e6) * u.K,
        mass_density=np.full((nz, ny, nx), density.value) * density.unit,
        velocity_z=np.zeros((nz, ny, nx)) * u.km / u.s,
        x_edges=np.arange(nx + 1) * 0.1 * u.Mm, y_edges=np.arange(ny + 1) * 0.1 * u.Mm,
        z_edges=np.arange(nz + 1) * 0.1 * u.Mm)
    path = write_atmosphere(atmosphere, tmp_path / "atmosphere.h5")
    argv = ["synthesise-spectra", "--atmosphere", str(path), "--output-dir", str(tmp_path),
            "--output-name", "out.h5", "--mass-per-electron", "1.2", "--lines", *lines]
    if continuum:
        argv.append("--continuum")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    monkeypatch.setattr(synthesis, "compute_continuum_fiasco", _flat_continuum)
    synthesis.main()
    return tmp_path / "out.h5"


def test_a_synthesis_with_the_continuum_keeps_it_beside_the_lines(tmp_path, monkeypatch):
    lines = ["Fe12_195.1190", "Fe12_195.1790", "Fe09_171.0730"]
    path = _synthesise(tmp_path, monkeypatch, lines, continuum=True)
    synthesis_read = read_synthesis(path)
    entries = [name for name in synthesis_read.lines if is_continuum(name)]
    assert len(entries) == 2 and set(synthesis_read.lines) == set(lines) | set(entries)
    products = read_synthesis_products(path)
    assert set(products["goft"]) == set(lines) and products["config"]["continuum"] is True
    # The continuum of each pixel is its emission measure, whatever its
    # temperature and velocity, times the flat continuum.
    em = products["em_tv"].sum(axis=(-1, -2))
    for name in entries:
        radiance = synthesis_read.lines[name].radiance().to_value(synthesis_unit())
        assert np.allclose(radiance, (em * FLAT)[..., np.newaxis], rtol=1e-6)
    # A window adds it once: with the lines' own spectra set to zero, what is
    # left of the window is the continuum.
    only = {name: line if is_continuum(name)
            else dataclasses.replace(line, intensity=0 * line.intensity)
            for name, line in synthesis_read.lines.items()}
    added = dataclasses.replace(synthesis_read, lines=only).summed("Fe12_195.1190")
    assert np.allclose(added.to_value(synthesis_unit()), (em * FLAT)[..., np.newaxis], rtol=1e-9)
    # The lines' own spectra are as without it.
    without = read_synthesis(_synthesise(tmp_path / "plain", monkeypatch, lines, continuum=False))
    for name in lines:
        assert np.array_equal(synthesis_read.lines[name].intensity, without.lines[name].intensity)
    # Every entry is evenly spaced, so the file loads as line cubes.
    assert set(load_synthesis(path)["line_cubes"]) == set(synthesis_read.lines)


@pytest.mark.chianti
def test_the_continuum_is_fiascos_per_emission_measure_and_per_steradian():
    import fiasco

    logT = np.array([5.5, 6.0, 6.5])
    logN = np.array([8.0, 10.0])
    wavelength = np.array([170.0, 190.0, 210.0]) * u.AA
    free, two_photon = compute_continuum_fiasco(wavelength, logT, logN, elements=["C", "Fe"],
                                                n_workers=1)
    temperature = 10 ** logT * u.K
    per_cm = 1e8
    abundance = "sun_coronal_2021_chianti"
    ratio = fiasco.Ion("H 1", temperature, abundance=abundance).proton_electron_ratio.value
    elements = [fiasco.Element(symbol, temperature, abundance=abundance) for symbol in ("C", "Fe")]
    unit = "erg cm3 s-1 AA-1"
    expected_free = sum((element.free_free(wavelength) + element.free_bound(wavelength))
                        .to_value(unit) for element in elements)
    expected_two = sum(element.two_photon(wavelength, 10 ** logN * u.cm**-3).to_value(unit)
                       for element in elements)
    scale = ratio / (4 * np.pi) * per_cm
    assert np.allclose(free, expected_free * scale[:, np.newaxis], rtol=1e-10)
    assert two_photon.shape == (logN.size, logT.size, wavelength.size)
    assert np.allclose(two_photon, np.moveaxis(expected_two * scale[:, None, None], 1, 0),
                       rtol=1e-10)
    assert np.all(free > 0)
