"""Every line in a band, for a full-CCD frame, and the density a synthesis keeps for it.

A frame of a whole detector holds every line in its band, not only the lines
named for a run. Their contribution functions are worked out in ECLIPSE's
convention, on a synthesis's own grids, and the synthesis file now keeps the
density at each temperature of each pixel, which they are taken at.
"""
import logging
import sys

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from euvst_response import synthesis
from euvst_response.atmosphere import Atmosphere, write_atmosphere
from euvst_response.band import (BandLines, IonLines, _solve_dropping_unfed_levels,
                                 _unfed_levels_dropped, band_contribution_functions,
                                 read_band_lines, write_band_lines)
from euvst_response.synthesis_file import read_synthesis_products


def _rates_with_an_unfed_level():
    """A three-level ion whose level 2 nothing populates, normalised as fiasco gives it."""
    matrix = np.array([[-2.0, 3.0, 0.0],
                       [2.0, -3.0, 0.0],
                       [1.0, 1.0, 1.0]])
    rhs = np.array([0.0, 0.0, 1.0])
    return matrix, rhs


def test_a_level_nothing_populates_is_left_out_and_the_rest_solved_exactly():
    matrix, rhs = _rates_with_an_unfed_level()
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(matrix, rhs)
    populations, dropped = _solve_dropping_unfed_levels(matrix, rhs)
    assert dropped == 1
    # Level 0 loses 2 to level 1, which returns 3: they hold 3/5 and 2/5.
    assert np.allclose(populations, [0.6, 0.4, 0.0])


def test_a_matrix_with_no_unfed_level_is_not_solved_some_other_way():
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        _solve_dropping_unfed_levels(singular, np.array([0.0, 1.0]))


def test_the_fallback_solves_each_matrix_and_leaves_the_others_alone():
    unfed, rhs = _rates_with_an_unfed_level()
    plain = np.array([[-1.0, 2.0, 0.5], [1.0, -2.0, 0.5], [1.0, 1.0, 1.0]])
    dropped = {}
    with _unfed_levels_dropped(dropped):
        solutions = np.linalg.solve(np.stack([plain, unfed]), rhs)
    assert dropped == {1: 1}
    assert np.allclose(solutions[0], np.linalg.solve(plain, rhs))
    assert np.allclose(solutions[1], [0.6, 0.4, 0.0])
    # fiasco 0.8 gives each matrix its own right-hand side, as a column.
    dropped = {}
    with _unfed_levels_dropped(dropped):
        columns = np.linalg.solve(np.stack([plain, unfed]), np.stack([rhs, rhs])[..., np.newaxis])
    assert dropped == {1: 1} and columns.shape == (2, 3, 1)
    assert np.allclose(columns[..., 0], solutions)
    # Outside it, numpy's own solve is back.
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(np.zeros((2, 2)), np.ones(2))


def test_a_bands_lines_survive_a_round_trip_through_their_file(tmp_path):
    ion = IonLines(name="Fe 12", atom=26, stage=12, mass=55.845,
                   wavelength=np.array([195.119, 195.179]), observed=np.array([True, False]),
                   temperatures=np.array([2, 3]), g=np.arange(2 * 3 * 2.0).reshape(2, 3, 2))
    lines = BandLines(band=(194.0, 196.0), logT_grid=np.linspace(5, 7, 5),
                      logN_grid=np.array([8.0, 9.0, 10.0]), abundance="sun_coronal_2021_chianti",
                      hdf5_dbase_root="/somewhere", ions=[ion])
    back = read_band_lines(write_band_lines(lines, tmp_path / "band.h5"))
    assert back.band == lines.band and back.abundance == lines.abundance
    assert np.array_equal(back.logT_grid, lines.logT_grid)
    (read,) = back.ions
    assert read.name == "Fe 12" and read.atom == 26 and read.stage == 12
    for field in ("wavelength", "observed", "temperatures", "g"):
        assert np.array_equal(getattr(read, field), getattr(ion, field))


def test_a_synthesis_file_keeps_the_density_at_each_temperature(tmp_path, monkeypatch):
    def flat_goft(lines, **kwargs):
        logT_grid, logN_grid = np.linspace(5.0, 7.0, 21), np.linspace(8.0, 10.0, 21)
        return ({"Fe12_195.1190": {"wl0": (195.119 * u.AA).to(u.cm),
                                   "g_tn": np.ones((logN_grid.size, logT_grid.size)),
                                   "atom": 26, "ion": 12, "hdf5_dbase_root": None}},
                logT_grid, logN_grid)

    nz, ny, nx = 4, 3, 2
    density = (1.0e9 / u.cm**3 * 1.2 * const.u).to(u.g / u.cm**3)
    atmosphere = Atmosphere(
        temperature=np.full((nz, ny, nx), 1e6) * u.K,
        mass_density=np.full((nz, ny, nx), density.value) * density.unit,
        velocity_z=np.zeros((nz, ny, nx)) * u.km / u.s,
        x_edges=np.arange(nx + 1) * 0.1 * u.Mm, y_edges=np.arange(ny + 1) * 0.1 * u.Mm,
        z_edges=np.arange(nz + 1) * 0.1 * u.Mm)
    path = write_atmosphere(atmosphere, tmp_path / "atmosphere.h5")
    monkeypatch.setattr(sys, "argv", [
        "synthesise-spectra", "--atmosphere", str(path), "--output-dir", str(tmp_path),
        "--output-name", "out.h5", "--mass-per-electron", "1.2", "--lines", "Fe12_195.1190"])
    monkeypatch.setattr(synthesis, "compute_goft_fiasco", flat_goft)
    synthesis.main()
    products = read_synthesis_products(tmp_path / "out.h5")
    kept = products["electron_density"]
    assert kept.shape == products["dem_map"].shape
    lit = products["dem_map"] > 0
    assert np.allclose(kept[lit], 1e9, rtol=1e-6) and np.all(kept[~lit] == 0)


@pytest.mark.chianti
def test_a_line_of_the_band_has_the_contribution_function_the_synthesis_gives_it():
    logT = np.linspace(5.8, 6.6, 9)
    logN = np.array([8.5, 9.5])
    band = band_contribution_functions((195.10, 195.13), logT, logN, elements=["Fe"],
                                       n_workers=1)
    (fe12,) = [ion for ion in band.ions if ion.name == "Fe 12"]
    line = int(np.argmin(np.abs(fe12.wavelength - 195.119)))
    goft, _, _ = synthesis.compute_goft_fiasco(
        ["Fe12_195.1190"], logT_min=logT[0], logT_max=logT[-1], nT=logT.size,
        logN_min=logN[0], logN_max=logN[-1], nN=logN.size, n_workers=1)
    expected = goft["Fe12_195.1190"]["g_tn"][:, fe12.temperatures]
    assert np.allclose(fe12.g[line], expected, rtol=1e-10)
    assert fe12.mass == pytest.approx(55.845, rel=1e-3)


@pytest.mark.chianti
def test_an_ion_whose_data_are_missing_is_left_out_and_named(monkeypatch, capsys):
    from fiasco.util.exceptions import MissingDatasetException

    from euvst_response import band as band_module

    def missing(*args, **kwargs):
        raise MissingDatasetException("no such data")

    monkeypatch.setattr(band_module, "_contribution_function", missing)
    logger = logging.getLogger("fiasco")
    monkeypatch.setattr(logger, "level", logging.INFO)
    band = band_contribution_functions((195.10, 195.13), np.linspace(5.8, 6.6, 9),
                                       np.array([9.0]), elements=["Fe"], n_workers=1)
    assert band.ions == []
    # Worked out in this process, fiasco's warnings are back on after.
    assert logger.level == logging.INFO
    assert "Fe 12 (no such data)" in capsys.readouterr().out


@pytest.mark.chianti
def test_a_band_with_no_lines_still_names_its_database():
    import fiasco

    band = band_contribution_functions((100.0, 100.0001), np.linspace(5.8, 6.6, 9),
                                       np.array([9.0]), elements=["H"], n_workers=1)
    assert band.ions == []
    assert band.hdf5_dbase_root == str(fiasco.defaults["hdf5_dbase_root"])
