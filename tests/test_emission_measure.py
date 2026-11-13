"""Synthesis from an emission measure file: a DEM, or with velocities a VDEM."""
import sys
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
import yaml

import eclipse
from eclipse import band as band_module
from eclipse import synthesis
from eclipse.band import BandLines, IonLines
from eclipse.emission_measure import (EmissionMeasure, is_emission_measure_file,
                                      read_emission_measure, write_emission_measure)
from eclipse.synthesis_file import read_synthesis, read_synthesis_products

LINE, REST = "Fe12_195.1190", 195.119
LOGT = np.linspace(5.6, 6.6, 21)
G0 = 1e-24  # erg cm3 / s at 1e7 cm^-3, and that much more with each decade of density
RADIANCE = u.erg / (u.s * u.cm**2 * u.sr)


def _edges(n):
    return (np.arange(n + 1) - n / 2) * u.arcsec


def _dem(ny=3, nx=4, **fields):
    """DEM of 1e27 cm^-5 at 1 MK and half that at 2 MK, brighter towards +x."""
    em = np.zeros((ny, nx, LOGT.size))
    em[:, :, 8], em[:, :, 14] = 1e27, 5e26
    em *= (1 + np.arange(nx))[None, :, None]
    return EmissionMeasure(emission_measure=em / u.cm**5, temperature=10 ** LOGT * u.K,
                           x_edges=_edges(nx), y_edges=_edges(ny), **fields)


def _g_at(log_density):
    # Linear in log10 n_e, as G is interpolated between the grid densities.
    return G0 * (1.0 + (log_density - 7.0))


def _flat_goft(lines, logT_min, logT_max, nT, logN_min, logN_max, nN, **kwargs):
    """G that depends on the density alone, on the grids asked for, in place of fiasco."""
    logT, logN = np.linspace(logT_min, logT_max, nT), np.linspace(logN_min, logN_max, nN)
    g = np.broadcast_to(_g_at(logN)[:, None], (nN, nT)).copy()
    return ({name: {"wl0": (REST * u.AA).to(u.cm), "g_tn": g, "atom": 26, "ion": 12,
                    "transition": 1, "hdf5_dbase_root": "test"} for name in lines}, logT, logN)


def _band_lines(band, logT, logN, **kwargs):
    g = np.broadcast_to(_g_at(np.asarray(logN))[None, :, None], (1, len(logN), len(logT)))
    ion = IonLines(name="Fe 12", atom=26, stage=12, mass=55.845, wavelength=np.array([REST]),
                   observed=np.array([True]), temperatures=np.arange(len(logT)), g=g.copy())
    return BandLines(band=tuple(band), logT_grid=np.asarray(logT, dtype=float),
                     logN_grid=np.asarray(logN, dtype=float),
                     abundance=kwargs.get("abundance", "sun_coronal_2021_chianti"),
                     hdf5_dbase_root="test", ions=[ion])


@pytest.fixture
def chianti_free(monkeypatch):
    monkeypatch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    monkeypatch.setattr(band_module, "band_contribution_functions", _band_lines)


def _synthesise(tmp_path, measure, *words, name="synthesis.h5", monkeypatch):
    path = write_emission_measure(measure, tmp_path / "measure.h5")
    monkeypatch.setattr(sys, "argv", [
        "synthesise-spectra", "--emission-measure", str(path), "--output-dir", str(tmp_path),
        "--output-name", name, *words])
    synthesis.main()
    return tmp_path / name


def _light(line):
    """Each pixel's radiance summed over wavelength, and its mean wavelength in Angstrom."""
    wavelength = line.wavelength.to_value(u.AA)
    radiance = line.radiance().to_value(RADIANCE / u.AA)
    step = np.gradient(wavelength)
    total = (radiance * step).sum(axis=2)
    return total, (radiance * step * wavelength).sum(axis=2) / total


def test_an_emission_measure_file_reads_back_as_written(tmp_path):
    measure = _dem(electron_density=np.full((3, 4), 2e9) / u.cm**3, source="An inversion",
                   time=30 * u.s)
    path = write_emission_measure(measure, tmp_path / "dem.h5")
    assert is_emission_measure_file(path) and not is_emission_measure_file(tmp_path / "no.h5")
    back = read_emission_measure(path)
    assert back.shape == (3, 4) and back.source == "An inversion" and back.time == 30 * u.s
    for name in ("emission_measure", "temperature", "x_edges", "y_edges", "electron_density"):
        assert np.array_equal(getattr(back, name), getattr(measure, name))
        assert getattr(back, name).unit == getattr(measure, name).unit
    assert back.velocity is None


@pytest.mark.parametrize("change, message", [
    (dict(emission_measure=np.ones((3, 4, 5)) / u.cm**5), "y along the slit first"),
    (dict(emission_measure=-np.ones((3, 4, 21)) / u.cm**5), "negative"),
    (dict(emission_measure=np.ones((3, 4, 21)) / u.cm**3), "unit such as"),
    (dict(temperature=np.linspace(1e5, 1e7, 21) * u.K), "log10 of temperature"),
    (dict(velocity=np.arange(-2, 3) * 10 * u.km / u.s), "A VDEM has a velocity axis"),
    (dict(emission_measure=np.ones((3, 4, 21, 5)) / u.cm**5), "A VDEM has a velocity axis"),
    (dict(emission_measure=np.ones((3, 4, 21, 5)) / u.cm**5,
          velocity=np.arange(4) * 10 * u.km / u.s), "5 velocity bins"),
    (dict(electron_density=np.ones(4) / u.cm**3), "one value, or shaped"),
    (dict(electron_density=0 / u.cm**3), "above zero"),
    (dict(emission_measure=np.ones((3, 4, 21, 3)) / u.cm**5,
          velocity=[0, 20, 10] * u.km / u.s), "velocity must increase"),
])
def test_an_emission_measure_is_checked(change, message):
    fields = dict(emission_measure=np.ones((3, 4, 21)) / u.cm**5, temperature=10 ** LOGT * u.K,
                  x_edges=_edges(4), y_edges=_edges(3))
    with pytest.raises((ValueError, u.UnitConversionError), match=message):
        EmissionMeasure(**{**fields, **change})


def test_a_dem_gives_each_line_its_emission_measure_times_g_at_rest(tmp_path, monkeypatch,
                                                                    chianti_free):
    path = _synthesise(tmp_path, _dem(), "--lines", LINE, "--electron-density", "1e9 cm-3",
                       monkeypatch=monkeypatch)
    held = read_synthesis(path)
    assert held.x_edges.unit == u.arcsec and held.shape == (3, 4)
    total, centre = _light(held.lines[LINE])
    em = 1.5e27 * (1 + np.arange(4))
    assert np.allclose(total, em * _g_at(9.0) / (4 * np.pi), rtol=1e-6)
    assert np.allclose(centre, REST, rtol=0, atol=1e-6)
    products = read_synthesis_products(path)
    assert np.allclose(products["logT_grid"], LOGT)
    assert np.allclose(products["electron_density"], 1e9)
    # A DEM's emission is in one velocity bin, at rest, and each line's wavelengths are
    # those of the default velocity grid.
    assert products["em_tv"].shape == (3, 4, 21, 1)
    assert products["vel_grid"].to_value(u.km / u.s) == pytest.approx([0.0])
    assert held.lines[LINE].wavelength.size == 121
    # The DEM is per unit log T, as the synthesis of an atmosphere keeps it.
    assert np.allclose(products["dem_map"].sum(axis=2) * 0.05, em[None, :])
    assert products["config"]["emission_measure"].endswith("measure.h5")


def test_the_density_is_the_files_a_density_given_or_one_from_a_pressure(tmp_path, monkeypatch,
                                                                         chianti_free):
    def total(measure, *words):
        path = _synthesise(tmp_path, measure, "--lines", LINE, *words, monkeypatch=monkeypatch)
        return _light(read_synthesis(path).lines[LINE])[0][0, 0]

    em = 1.5e27
    # The file's density, one for each pixel.
    in_file = _dem(electron_density=np.full((3, 4), 1e10) / u.cm**3)
    assert total(in_file) == pytest.approx(em * _g_at(10.0) / (4 * np.pi), rel=1e-6)
    # A density given to the synthesis replaces the file's.
    assert total(in_file, "--electron-density", "1e8 cm-3") == pytest.approx(
        em * _g_at(8.0) / (4 * np.pi), rel=1e-6)
    # At a pressure of 1e15 K cm^-3, the plasma at 1 MK is at 1e9 cm^-3 and that at 2 MK at
    # half of it.
    expected = (1e27 * _g_at(9.0) + 5e26 * _g_at(np.log10(1e15 / 10 ** LOGT[14]))) / (4 * np.pi)
    assert total(_dem(), "--pressure", "1e15 K cm-3") == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError, match="needs? an electron density"):
        total(_dem())
    with pytest.raises(ValueError, match="not both"):
        total(_dem(), "--electron-density", "1e9 cm-3", "--pressure", "1e15 K cm-3")
    for words in (("--electron-density", "1e9 K"), ("--pressure", "0 K cm-3"),
                  ("--electron-density", "[1e9, 2e9] cm-3")):
        with pytest.raises(ValueError, match="takes one value above zero"):
            total(_dem(), *words)
    with pytest.raises(ValueError, match="name ending in .h5"):
        _synthesise(tmp_path, _dem(), "--lines", LINE, "--electron-density", "1e9 cm-3",
                    name="synthesis.pkl", monkeypatch=monkeypatch)


def test_a_vdem_shifts_each_line_by_its_velocities(tmp_path, monkeypatch, chianti_free):
    velocity = np.arange(-60, 61) * 5.0 * u.km / u.s
    em = np.zeros((1, 2, LOGT.size, velocity.size))
    em[0, 0, 8, 60] = 1e27           # at rest
    em[0, 1, 8, 80] = 2e27           # receding at 100 km/s
    measure = EmissionMeasure(emission_measure=em / u.cm**5, temperature=10 ** LOGT * u.K,
                              x_edges=_edges(2), y_edges=_edges(1), velocity=velocity,
                              electron_density=1e9 / u.cm**3)
    path = _synthesise(tmp_path, measure, "--lines", LINE, monkeypatch=monkeypatch)
    total, centre = _light(read_synthesis(path).lines[LINE])
    assert np.allclose(total[0], np.array([1e27, 2e27]) * _g_at(9.0) / (4 * np.pi), rtol=1e-6)
    shifted = REST * (1 + (100 * u.km / u.s / const.c).to_value(""))
    assert centre[0] == pytest.approx([REST, shifted], abs=1e-6)
    assert np.allclose(read_synthesis_products(path)["vel_grid"].to_value(u.km / u.s),
                       velocity.value)


@pytest.mark.parametrize("words", [(), ("--precision", "float32")])
def test_a_vdem_keeps_its_light_whatever_its_velocity_bins(tmp_path, monkeypatch, chianti_free,
                                                           words):
    """Bins 40 km/s apart are three thermal widths of the line at 1 MK, and one bin alone is
    a VDEM too. Each line is integrated over its wavelengths, so its light is kept."""
    for velocity in (np.array([-80.0, -40.0, 0.0, 40.0, 80.0]), np.array([40.0])):
        em = np.zeros((1, 1, LOGT.size, velocity.size))
        em[0, 0, [0, 8, 20], -1] = 1e27      # at the first and last temperatures too
        measure = EmissionMeasure(emission_measure=em / u.cm**5, temperature=10 ** LOGT * u.K,
                                  x_edges=_edges(1), y_edges=_edges(1),
                                  velocity=velocity * u.km / u.s, electron_density=1e9 / u.cm**3)
        path = _synthesise(tmp_path, measure, "--lines", LINE, *words, monkeypatch=monkeypatch)
        total, centre = _light(read_synthesis(path).lines[LINE])
        assert total[0, 0] == pytest.approx(3e27 * _g_at(9.0) / (4 * np.pi), rel=1e-5)
        shifted = REST * (1 + (velocity[-1] * u.km / u.s / const.c).to_value(""))
        assert centre[0, 0] == pytest.approx(shifted, abs=1e-5)


def test_plasma_beyond_the_lines_wavelengths_is_warned_of(tmp_path, monkeypatch, chianti_free):
    velocity = np.array([0.0, 400.0]) * u.km / u.s
    em = np.zeros((1, 1, LOGT.size, 2))
    em[0, 0, 8] = 1e27
    measure = EmissionMeasure(emission_measure=em / u.cm**5, temperature=10 ** LOGT * u.K,
                              x_edges=_edges(1), y_edges=_edges(1), velocity=velocity,
                              electron_density=1e9 / u.cm**3)
    with pytest.warns(UserWarning, match="plasma at up to 400 km/s, beyond the 300 km/s"):
        _synthesise(tmp_path, measure, "--lines", LINE, monkeypatch=monkeypatch)


def test_a_dem_is_synthesised_over_a_band_and_a_strip_of_the_image(tmp_path, monkeypatch,
                                                                   chianti_free):
    words = ("--band", "194 Angstrom", "196.5 Angstrom", "--electron-density", "1e9 cm-3")
    whole = read_synthesis(_synthesise(tmp_path, _dem(), *words, monkeypatch=monkeypatch))
    (name,) = whole.lines
    assert name == "band_194.0010-196.4990"
    total, centre = _light(whole.lines[name])
    assert np.allclose(total, 1.5e27 * (1 + np.arange(4)) * _g_at(9.0) / (4 * np.pi), rtol=1e-6)
    assert np.allclose(centre, REST, rtol=0, atol=1e-5)
    strip = read_synthesis(_synthesise(tmp_path, _dem(), *words, "--image-x", "-0.5 arcsec",
                                       "1.5 arcsec", name="strip.h5", monkeypatch=monkeypatch))
    assert np.allclose(strip.x_edges.to_value(u.arcsec), [-1, 0, 1, 2])
    assert np.allclose(strip.lines[name].intensity.value, whole.lines[name].intensity.value[:, 1:4])


@pytest.mark.parametrize("words, message", [
    (("--integration-axis=x",), "--integration-axis would not be used"),
    (("--crop-x", "0 Mm", "1 Mm", "--downsample", "2"), "--crop-x, --downsample"),
    (("--mass-per-electron", "1.2"), "--mass-per-electron"),
    (("--atmosphere", "box.h5"), "--atmosphere"),
])
def test_an_emission_measure_takes_no_options_of_an_atmosphere(tmp_path, monkeypatch,
                                                               chianti_free, words, message):
    with pytest.raises(ValueError, match=message):
        _synthesise(tmp_path, _dem(), "--lines", LINE, "--electron-density", "1e9 cm-3", *words,
                    monkeypatch=monkeypatch)


def test_a_density_or_pressure_is_refused_with_an_atmosphere(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["synthesise-spectra", "--atmosphere", "box.h5", "--lines",
                                      LINE, "--electron-density", "1e9 cm-3"])
    with pytest.raises(ValueError, match="--electron-density is for an emission measure file"):
        synthesis.main()


def _job(path, **settings):
    return {"instrument": "SWC", "emission_measure": str(path), "n_iter": 2, "ncpu": 1,
            "fit_signals": "dn", "synthesis": {"lines": [LINE], "electron_density": "1e9 cm-3"},
            "simulation": {"expos": "5 s", "psf": False}, **settings}


def test_a_run_synthesises_its_emission_measure_then_observes_it(tmp_path, monkeypatch,
                                                                 chianti_free):
    from eclipse.analysis import load_instrument_response_results

    monkeypatch.chdir(tmp_path)
    cell = _edges(8) * 0.2
    em = np.zeros((6, 8, LOGT.size))
    em[:, :, 8] = 1e27
    measure = write_emission_measure(EmissionMeasure(
        emission_measure=em / u.cm**5, temperature=10 ** LOGT * u.K, x_edges=cell,
        y_edges=_edges(6) * 0.2), tmp_path / "dem.h5")
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(_job(measure)))
    [output] = eclipse.run(path)
    assert output.synthesis == Path("run/input/job.h5")
    assert output.results == Path("run/result/job.h5")
    results = load_instrument_response_results(output.results)
    assert results is not None
    # The same run again reuses the synthesis, and a new density makes it again.
    made = output.synthesis.stat().st_mtime_ns
    eclipse.synthesise(path)
    assert output.synthesis.stat().st_mtime_ns == made
    eclipse.synthesise(path, settings=["synthesis.electron_density='2e9 cm-3'"])
    assert output.synthesis.stat().st_mtime_ns != made
    with pytest.raises(ValueError, match="one 'atmosphere' or one 'emission_measure'"):
        eclipse.run(_job(measure, atmosphere="box.h5"), name="job")
    with pytest.raises(FileNotFoundError, match="Emission measure file not found"):
        eclipse.synthesise(_job(tmp_path / "gone.h5"), name="gone")
    # A mistake in the synthesis settings is found before anything is synthesised.
    for synthesis_section, message in (
            ({"lines": [LINE]}, "need an electron density"),
            ({"lines": [LINE], "electron_density": "1e9 cm-3", "crop_x": ["0 Mm", "1 Mm"]},
             "--crop-x would not be used")):
        with pytest.raises(ValueError, match=message):
            eclipse.run(_job(measure, synthesis=synthesis_section), name="mistake")
    assert not Path("run/input/mistake.h5").exists()


def test_a_run_makes_a_frame_from_an_emission_measure(tmp_path, monkeypatch, chianti_free):
    import h5py

    monkeypatch.chdir(tmp_path)
    measure = write_emission_measure(_dem(ny=20, nx=40, electron_density=1e9 / u.cm**3),
                                     tmp_path / "dem.h5")
    job = {"instrument": "SWC", "emission_measure": str(measure),
           "simulation": {"expos": "10 s", "psf": False, "noise": False,
                          "slit_width": "0.4 arcsec"},
           "frame": {"slit_position": "5.5 arcsec", "band": ["194 Angstrom", "196.5 Angstrom"]}}
    [output] = eclipse.make_frame(job, name="job")
    strip = read_synthesis("run/input/job_band.h5")
    # The slit lies within the pixel from 5 to 6 arcsec.
    assert np.allclose(strip.x_edges.to_value(u.arcsec), [5, 6])
    with h5py.File(output.frame) as f:
        rate = f["left/rate"][()]
        assert [name.decode() for name in f["spectra"][()]] == ["band_194.0010-196.4990"]
    assert rate.any()
