"""Both pipeline stages write ASDF and read it back.

The unit tests in test_io_asdf.py check the encoder against objects built for
the purpose. These two run the real thing end to end, which is the only way
to find out whether the tree ECLIPSE actually produces survives the trip.
"""
import sys

import astropy.units as u
import numpy as np
import pytest

from euvst_response.analysis import load_instrument_response_results
from euvst_response.data_processing import load_atmosphere
from euvst_response.io import is_asdf, save_results
from euvst_response.synthesis import (create_atmosphere_ndcube,
                                      create_line_cube, synthesise_spectra)

REST = 195.119 * u.Angstrom
INTENSITY_UNIT = u.erg / u.s / u.cm**2 / u.sr / u.cm

CONFIG = """
instrument: SWC
uniform_intensity: 5000 erg / (s cm2 sr)
rest_wavelength: 195.119 AA
n_iter: 2
ncpu: 1
simulation:
  slit_width: 0.2 arcsec
  expos: 10 s
"""


def test_an_instrument_run_writes_asdf_and_reads_back(tmp_path, monkeypatch):
    from euvst_response.main import main

    config_path = tmp_path / "run.yaml"
    config_path.write_text(CONFIG)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config_path)])

    main()

    written = tmp_path / "run" / "result" / "run.asdf"
    assert written.exists(), "the result file should carry the .asdf suffix"
    assert is_asdf(written)
    assert not (tmp_path / "run" / "result" / "run.pkl").exists()

    results = load_instrument_response_results(written)

    assert results["instrument"] == "SWC"
    assert results["software_version"]
    combos = results["results"]["all_combinations"]
    assert len(combos) == 1

    combo = next(iter(combos.values()))
    # The parameter key is a tuple of pairs, which ASDF cannot use as a
    # mapping key, so this is the part most likely to have been flattened.
    key = next(iter(combos))
    assert isinstance(key, tuple)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in key)

    assert combo["parameters"]["simulation.expos"] == 10 * u.s
    assert combo["config_objects"]["detector"].qe_euv > 0
    assert combo["dn_fit_stats"]["mean_data"].shape[-1] == 4

    # load_instrument_response_results rebuilds these from the stored arrays,
    # so they exercise the unit and the WCS together.
    signal = combo["first_dn_signal"]
    assert signal.unit.is_equivalent(u.DN / u.pix)
    assert np.all(np.isfinite(signal.data))

    # Checked as coordinates rather than as CUNIT: astropy normalises a WCS
    # to SI in place the first time it is used to compute coordinates, so by
    # the time a run is saved its spectral axis is already in m whatever it
    # was built in. What has to survive is where the axis points.
    wavelengths = signal.axis_world_coords(-1)[0].to_value(u.Angstrom)
    assert wavelengths.min() < 195.119 < wavelengths.max()


def test_a_synthesis_file_round_trips_through_load_atmosphere(tmp_path):
    """The synthesis stage writes line cubes, which the next stage reads."""
    nx, ny, n_temp = 2, 2, 2
    logT_grid = np.array([6.0, 6.2])
    vel_grid = np.arange(-50.0, 50.0 + 25.0, 25.0) * u.km / u.s

    em_tv = np.zeros((nx, ny, n_temp, vel_grid.size))
    em_tv[:, :, 0, vel_grid.size // 2] = 1.0e27
    goft = {"Fe12_195.1190": {"wl0": REST.to(u.cm),
                              "g": np.ones((nx, ny, n_temp)),
                              "atom": 26, "ion": 12}}
    synthesise_spectra(goft, em_tv, vel_grid.to(u.cm / u.s), logT_grid)

    reference = create_atmosphere_ndcube(
        np.zeros((nx, ny, 2)) * u.K, 1 * u.Mm, 1 * u.Mm, 1 * u.Mm)
    line_cubes = {name: create_line_cube(name, info, reference,
                                         INTENSITY_UNIT, integration_axis="z")
                  for name, info in goft.items()}

    path = save_results(tmp_path / "synth.asdf", {
        "line_cubes": line_cubes,
        "dynamic_mode": {"enabled": False},
        "vel_grid": vel_grid.to(u.cm / u.s),
        "logT_grid": logT_grid,
        "goft": goft,
        "config": {"lines": ["Fe12_195.1190"], "abundance": "sun_coronal"},
    })
    assert is_asdf(path)

    cube, dynamic = load_atmosphere(str(path), "Fe12_195.1190")

    assert dynamic == {"enabled": False}
    assert cube.data.shape == (nx, ny, vel_grid.size)
    assert np.allclose(cube.data,
                       line_cubes["Fe12_195.1190"].data, rtol=1e-12)
    assert cube.meta["rest_wav"] == REST.to(u.cm)
    assert cube.meta["combined_lines"] == ["Fe12_195.1190"]
    # The wavelength axis has to survive, because everything downstream
    # builds photon energies from it.
    before = line_cubes["Fe12_195.1190"].axis_world_coords(-1)[0]
    after = cube.axis_world_coords(-1)[0]
    assert np.allclose(after.to_value(u.cm), before.to_value(u.cm),
                       rtol=1e-12, atol=0.0)


def test_a_legacy_synthesis_pickle_still_loads(tmp_path):
    """Existing synthesis files predate the format change."""
    import dill

    nx, ny = 2, 2
    reference = create_atmosphere_ndcube(
        np.zeros((nx, ny, 2)) * u.K, 1 * u.Mm, 1 * u.Mm, 1 * u.Mm)
    line_data = {"si": np.ones((nx, ny, 3)),
                 "wl_grid": np.array([195.0, 195.1, 195.2]) * u.Angstrom,
                 "wl0": REST.to(u.cm), "atom": 26, "ion": 12}
    cube = create_line_cube("Fe12_195.1190", line_data, reference,
                            INTENSITY_UNIT, integration_axis="z")

    path = tmp_path / "old_synth.pkl"
    with open(path, "wb") as handle:
        dill.dump({"line_cubes": {"Fe12_195.1190": cube},
                   "dynamic_mode": {"enabled": False}}, handle)

    with pytest.warns(UserWarning, match="pickle"):
        loaded, _ = load_atmosphere(str(path), "Fe12_195.1190")

    assert np.allclose(loaded.data, cube.data)
