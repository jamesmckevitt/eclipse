"""A configuration that cannot be run is refused before anything runs.

Each of these once ran to completion and saved results that meant nothing,
or failed part-way through a sweep and lost what came before: a quantity
with no unit, or of the wrong kind; a value out of its physical range; a
repeated YAML key, which PyYAML lets the last one win; an empty value; a
sweep value that could not be run, found only when its turn came.
"""
import sys

import astropy.units as u
import numpy as np
import pytest
import yaml

from euvst_response.atmosphere import Atmosphere
from euvst_response.config import (AluminiumFilter, Detector_EIS, Detector_SWC, Simulation,
                                   Telescope_EIS, Telescope_EUVST, _load_throughput_table)
from euvst_response.fitting import FitComponent, FitConfig
from euvst_response.main import main

UNIFORM = {"instrument": "SWC", "n_iter": 1, "uniform_intensity": "5000 erg / (s cm2 sr)"}


def _run(tmp_path, monkeypatch, config, text=None):
    path = tmp_path / "run.yaml"
    path.write_text(text if text is not None else yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(path)])
    monkeypatch.chdir(tmp_path)
    main()


@pytest.mark.parametrize("build, message", [
    (lambda: Telescope_EUVST(D_ap=0.28), "telescope.D_ap needs a unit, of length"),
    (lambda: Telescope_EUVST(D_ap=28 * u.arcsec), "telescope.D_ap must be in a unit of length"),
    (lambda: Detector_SWC(qe_euv=76), "detector.qe_euv is a fraction"),
    (lambda: Detector_SWC(gain_e_per_dn=0 * u.electron / u.DN), "gain_e_per_dn must be more than zero"),
    (lambda: Detector_SWC(ccd_temperature=-60 * u.K), "above absolute zero.*-60 C"),
    (lambda: Detector_EIS(material="silicn"), "detector.material must be one of"),
    (lambda: AluminiumFilter(mesh_throughput=80), "filter.mesh_throughput is a fraction"),
    (lambda: AluminiumFilter(al_thickness=-5 * u.AA), "al_thickness cannot be negative"),
    (lambda: Simulation(expos=0 * u.s), "simulation.expos must be more than zero"),
    (lambda: Simulation(expos=-5 * u.s), "simulation.expos must be more than zero"),
    (lambda: Simulation(noise=None), "simulation.noise must be true or false"),
    (lambda: Simulation(n_iter=0), "n_iter must be a whole number of iterations"),
    (lambda: Simulation(instrument="EIS", slit_width=4 * u.arcsec), "1 or 2 arcsec, its two slits"),
    (lambda: Simulation(enable_pinholes=True, pinhole_sizes=[-5 * u.um], pinhole_positions=[0.5]),
     r"pinhole_sizes\[0\] must be a diameter"),
    (lambda: Telescope_EIS(calibration="dz2025", date="03-Jun-2012"),
     "telescope.date '03-Jun-2012' is not a date ECLIPSE can read"),
    (lambda: Telescope_EIS(calibration="dz2025", date="2012-06-03+garbage"),
     r"telescope.date '2012-06-03\+garbage' is not a date ECLIPSE can read"),
    (lambda: Telescope_EIS(psf_slit_width=np.inf * u.arcsec),
     "psf_slit_width must be a finite angle above zero"),
    (lambda: FitComponent(195.119), "wavelength must be the wavelength of a line"),
    (lambda: FitConfig(components=[FitComponent(195.119 * u.AA), FitComponent(195.179 * u.AA)],
                       primary_component=1.0), "primary_component is 1.0"),
])
def test_a_setting_of_the_wrong_kind_or_out_of_range_is_refused(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_a_table_named_in_a_configuration_reads_whatever_its_header(tmp_path):
    """Named as text, a path failed to parse as a quantity; a table with one header line lost a row."""
    rows = "17.0 0.097\n17.2 0.103\n17.4 0.110\n"
    for name, text in (("one.dat", "# reflectance\n" + rows), ("none.dat", rows)):
        (tmp_path / name).write_text(text)
        wavelength, _ = _load_throughput_table(tmp_path / name)
        assert wavelength.to_value(u.nm).tolist() == [17.0, 17.2, 17.4]
    telescope = Telescope_EUVST(pm_table=str(tmp_path / "one.dat"))
    assert np.isfinite(telescope.primary_mirror_efficiency(172.0 * u.AA))


def test_a_table_inside_an_archive_is_read_as_a_packaged_one_may_be(tmp_path):
    """importlib.resources gives a zipped package's tables as resources that read themselves but are not paths."""
    import zipfile

    with zipfile.ZipFile(tmp_path / "package.zip", "w") as archive:
        archive.writestr("reflectance.dat", "Wavelength (nm), Reflectance\n17.0 0.097\n17.4 0.110\n")
    table = zipfile.Path(tmp_path / "package.zip", "reflectance.dat")
    telescope = Telescope_EUVST(pm_table=table)
    assert telescope.pm_table is table
    assert telescope.primary_mirror_efficiency(172.0 * u.AA) == pytest.approx(0.1035)


def test_a_slip_in_a_tables_data_is_refused_not_skipped(tmp_path):
    """Skipped, 0.l03 would have left the curve interpolated across the row it was in."""
    (tmp_path / "slip.dat").write_text("# reflectance\n17.0 0.097\n17.2 0.l03\n17.4 0.110\n")
    with pytest.raises(ValueError, match=r"slip.dat, line 3: '17.2 0.l03' is not a wavelength"):
        _load_throughput_table(tmp_path / "slip.dat")
    (tmp_path / "empty.dat").write_text("# nothing here\n")
    with pytest.raises(ValueError, match="has no lines of a wavelength and a throughput"):
        _load_throughput_table(tmp_path / "empty.dat")
    # A column of numbers more is read; a word after the numbers is a slip.
    (tmp_path / "three.dat").write_text("17.0 0.097 0.001\n17.2 0.103 0.001\n")
    assert _load_throughput_table(tmp_path / "three.dat")[1].tolist() == [0.097, 0.103]
    (tmp_path / "word.dat").write_text("17.0 0.097\n17.2 0.103 garbage\n")
    with pytest.raises(ValueError, match="line 2: '17.2 0.103 garbage'"):
        _load_throughput_table(tmp_path / "word.dat")


def test_a_line_before_the_data_that_begins_with_a_number_is_said_to_be_taken_as_a_header(
        tmp_path):
    """It may be the first line of data with a slip in it; the packaged tables' headers are words."""
    import shutil
    import warnings
    from importlib.resources import files

    (tmp_path / "first.dat").write_text("17.0 0.l03\n17.2 0.103\n17.4 0.110\n")
    with pytest.warns(UserWarning, match=r"line 1: '17.0 0.l03' is taken as a header, but it "
                                         r"begins with a number"):
        assert _load_throughput_table(tmp_path / "first.dat")[1].tolist() == [0.103, 0.110]
    for table in (files("euvst_response") / "data" / "throughput").iterdir():
        if not table.name.endswith(".dat"):
            continue
        copy = tmp_path / table.name
        shutil.copyfile(table, copy)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _load_throughput_table(copy)


@pytest.mark.parametrize("build, message", [
    (lambda: Telescope_EUVST(psf_params=[2.66, 2.54]), "psf_params must be two FWHMs in pixels"),
    (lambda: Telescope_EUVST(psf_params=[2.66 * u.pix]), "psf_params must be two FWHMs in pixels"),
    (lambda: Telescope_EIS(psf_params=[3 * u.pix, -3 * u.pix]), "psf_params must be two FWHMs"),
    (lambda: Telescope_EUVST(psf_type="airy"), "psf_type must be 'gaussian'"),
    (lambda: Telescope_EIS(psf_type="moffat"), "psf_type must be 'gaussian'"),
    (lambda: Telescope_EUVST(psf_across_slit=1.0), "psf_across_slit must be a FWHM in an angle"),
    (lambda: Telescope_EUVST(psf_across_slit=2 * u.pix), "psf_across_slit must be a FWHM"),
    (lambda: Telescope_EIS(psf_across_slit=-1 * u.arcsec), "psf_across_slit must be a FWHM"),
    (lambda: Telescope_EUVST(psf_across_slit=np.inf * u.arcsec), "psf_across_slit must be"),
    (lambda: Telescope_EUVST(pm_table=3), "telescope.pm_table must be the path of a table"),
    (lambda: AluminiumFilter(al_table=None), "filter.al_table must be the path of a table"),
])
def test_the_psf_widths_and_the_tables_are_checked_when_built(build, message):
    """They were set by a default factory, which the checks of other settings pass by."""
    with pytest.raises(ValueError, match=message):
        build()


def test_a_list_of_tables_is_refused_rather_than_swept(tmp_path, monkeypatch):
    """A result's parameters leave the tables out, so a sweep would give both tables one key."""
    with pytest.raises(ValueError, match="'telescope.pm_table' names one table"):
        _run(tmp_path, monkeypatch, {**UNIFORM, "telescope": {"pm_table": ["a.dat", "b.dat"]}})


def test_a_yaml_merge_key_is_read_as_yaml_reads_it():
    from euvst_response.utils import load_yaml_config

    text = "a: &a {x: 1, y: 1}\nb:\n  <<: *a\n  y: 2\n"
    assert load_yaml_config(text) == yaml.safe_load(text) == {"a": {"x": 1, "y": 1},
                                                              "b": {"x": 1, "y": 2}}
    with pytest.raises(yaml.constructor.ConstructorError, match="'y' is given twice"):
        load_yaml_config(text + "  y: 3\n")


@pytest.mark.parametrize("text, message", [
    ("instrument: SWC\nuniform_intensity: 5000 erg / (s cm2 sr)\n"
     "simulation:\n  expos: 10 s\nsimulation:\n  slit_width: 0.4 arcsec\n", "'simulation' is given twice"),
    ("instrument: SWC\nuniform_intensity: 5000 erg / (s cm2 sr)\n"
     "simulation:\n  expos: [5 s, 10 s]\n  expos: 10 s\n", "'expos' is given twice"),
])
def test_a_key_given_twice_is_refused(tmp_path, monkeypatch, text, message):
    with pytest.raises(yaml.constructor.ConstructorError, match=message):
        _run(tmp_path, monkeypatch, None, text)


@pytest.mark.parametrize("section, message", [
    ({"expos": []}, "'simulation.expos' is empty"),
    ({"noise": None}, "'simulation.noise' is empty"),
])
def test_an_empty_value_is_refused(tmp_path, monkeypatch, section, message):
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, monkeypatch, {**UNIFORM, "simulation": section})


@pytest.mark.parametrize("extra, message", [
    ({"n_iter": 0}, "'n_iter' must be a whole number"),
    ({"n_iter": "1e3"}, "'n_iter' must be a whole number"),
    ({"instrument": None}, "'instrument' must be SWC or EIS"),
    ({"offchip_bin_slit": 2.7}, "'offchip_bin_slit' must be whole numbers"),
    ({"thermal_width": "-20 km/s"}, "'thermal_width' must be a positive speed"),
    ({"thermal_width": "0 km/s"}, "'thermal_width' must be a positive speed"),
    ({"uniform_intensity": "-5000 erg / (s cm2 sr)"}, "'uniform_intensity' must be a positive"),
])
def test_a_top_level_value_that_cannot_be_run_is_refused(tmp_path, monkeypatch, extra, message):
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, monkeypatch, {**UNIFORM, **extra})


def test_every_combination_of_a_sweep_is_checked_before_the_first_runs(tmp_path, monkeypatch,
                                                                         capsys):
    """0.3 arcsec is no slit; found when its turn came, it ended the sweep and lost the rest."""
    with pytest.raises(ValueError, match="slit_width must be 0.2, 0.4, 0.8, or 1.6"):
        _run(tmp_path, monkeypatch, {**UNIFORM, "simulation": {
            "expos": ["5 s", "10 s"], "slit_width": ["0.2 arcsec", "0.3 arcsec"]}})
    assert "Combination 1" not in capsys.readouterr().out


@pytest.mark.parametrize("config, message", [
    # At 0.05 arcsec a pixel, the 0.2 arcsec slit psf_params is for is 4
    # pixels wide, more than its 2.54-pixel spectral FWHM.
    ({**UNIFORM, "detector": {"plate_scale_angle": ["0.159 arcsec / pix", "0.05 arcsec / pix"]}},
     "leaves nothing for the optics"),
    ({**UNIFORM, "instrument": "EIS", "simulation": {
        "slit_width": "1 arcsec", "psf": True, "spectral_psf": ["quadrature", "convolution"]}},
     "Telescope_EIS has no psf_slit_width"),
])
def test_a_spectral_psf_the_telescope_and_detector_cannot_give_is_refused_before_the_first_runs(
        tmp_path, monkeypatch, capsys, config, message):
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, monkeypatch, config)
    assert "Combination 1" not in capsys.readouterr().out


def test_off_chip_binning_beyond_the_scene_is_refused_before_the_first_combination_runs(
        tmp_path, monkeypatch, capsys):
    """A scene one row long can be binned by 1 but not 2; 2 was found only when its turn came."""
    from euvst_response.synthesis_file import SpectralLine, Synthesis, write_synthesis

    edges = np.arange(3) * 0.1 * u.Mm
    wavelength = 195.119 * u.AA + np.arange(-30, 31) * 0.003 * u.AA
    line = SpectralLine(intensity=np.ones((2, 2, 61)) * 1e13 * u.erg / (u.s * u.cm**2 * u.sr * u.cm),
                        wavelength=wavelength, rest_wavelength=195.119 * u.AA)
    write_synthesis(Synthesis(lines={"Fe12_195.1190": line}, x_edges=edges, y_edges=edges),
                    tmp_path / "file.h5")
    with pytest.raises(ValueError, match="offchip_bin_slit 2 bins more rows than the 1"):
        _run(tmp_path, monkeypatch, {"instrument": "SWC", "n_iter": 1,
                                     "synthesis_file": str(tmp_path / "file.h5"),
                                     "offchip_bin_slit": [1, 2]})
    assert "Combination 1" not in capsys.readouterr().out


def test_the_line_of_a_uniform_intensity_is_said_to_be_ignored_elsewhere(tmp_path, monkeypatch):
    from euvst_response.synthesis_file import SpectralLine, Synthesis, write_synthesis

    edges = np.arange(3) * 0.1 * u.Mm
    wavelength = 195.119 * u.AA + np.arange(-30, 31) * 0.003 * u.AA
    line = SpectralLine(intensity=np.ones((2, 2, 61)) * 1e13 * u.erg / (u.s * u.cm**2 * u.sr * u.cm),
                        wavelength=wavelength, rest_wavelength=195.119 * u.AA)
    write_synthesis(Synthesis(lines={"Fe12_195.1190": line}, x_edges=edges, y_edges=edges),
                    tmp_path / "file.h5")
    with pytest.warns(UserWarning, match="'thermal_width' is ignored"):
        _run(tmp_path, monkeypatch, {"instrument": "SWC", "n_iter": 1,
                                     "synthesis_file": str(tmp_path / "file.h5"),
                                     "thermal_width": "30 km/s"})


def test_an_atmosphere_below_absolute_zero_or_of_negative_density_is_refused():
    """Such cells were dropped from the synthesis without a word."""
    edges = {axis: np.arange(3) * u.Mm for axis in ("x_edges", "y_edges", "z_edges")}
    temperature = np.full((2, 2, 2), 1e6) * u.K
    density = np.full((2, 2, 2), 1e9) / u.cm**3
    with pytest.raises(ValueError, match="temperature must be above zero"):
        Atmosphere(temperature=-temperature, electron_density=density, **edges)
    with pytest.raises(ValueError, match="electron_density cannot be negative"):
        Atmosphere(temperature=temperature, electron_density=-density, **edges)
    # An empty cell may have no density.
    Atmosphere(temperature=temperature, electron_density=0 * density, **edges)


def test_a_line_name_that_cannot_be_read_is_refused_before_the_atmosphere_is(monkeypatch):
    from euvst_response import synthesis

    def never(*args, **kwargs):
        raise AssertionError("the atmosphere was read")

    monkeypatch.setattr(synthesis, "read_atmosphere", never)
    monkeypatch.setattr(sys, "argv", ["synthesise-spectra", "--atmosphere", "box.h5",
                                      "--lines", "FeXII_195.119"])
    with pytest.raises(ValueError, match="Cannot parse line name 'FeXII_195.119'"):
        synthesis.main()


def test_a_gaussian_named_in_capitals_and_a_blur_across_the_slit_are_accepted():
    Telescope_EUVST(psf_type="Gaussian", psf_across_slit=1 * u.arcsec)
    Telescope_EIS(psf_across_slit=0.5 * u.arcmin)
