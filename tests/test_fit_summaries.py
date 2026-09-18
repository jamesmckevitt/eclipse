"""Fitted components are stored by name, with statistics for each of them.

The results used to hold only the mean and standard deviation of every
fitted parameter in one unlabelled vector, and the analysis only turned the
primary component's centre and width into a velocity and a width. These tests
pin down what replaces that: intensity, velocity and width for every
component under its own name, failed fits left out of the statistics and
counted, an optional small-sample correction, and analysis that picks the
component by name.
"""
import sys
import warnings

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube
from sunpy.util.exceptions import SunpyMetadataWarning

from euvst_response import fitting
from euvst_response.analysis import (
    analyse_fit_statistics,
    create_sunpy_maps_from_combo,
    list_fit_components,
    load_instrument_response_results,
    summary_table,
)
from euvst_response.config import Detector_SWC, Simulation, Telescope_EUVST
from euvst_response.fitting import (
    FitComponent,
    FitConfig,
    fit_cube_gauss,
    fit_quantities,
    spectral_pixel_width,
    summarise_fits,
)
from euvst_response.monte_carlo import monte_carlo

REST0 = 195.119 * u.AA
REST1 = 195.179 * u.AA
STEP = 0.0169 * u.AA
C_KMS = const.c.to_value(u.km / u.s)
WHEN = "2024-03-20T00:00:00"


def _blend_config(**kwargs):
    """Fe XII 195.119, named, and 195.179 tied to it and left unnamed."""
    return FitConfig(components=[
        FitComponent(wavelength=REST0, name="Fe XII 195.119"),
        FitComponent(wavelength=REST1, tie_center=0, tie_width=0),
    ], **kwargs)


def _blend_units():
    return [u.DN / u.pix, u.cm, u.cm] * 2 + [u.DN / u.pix]


def _blend_params(v_kms, peak0=100.0, peak1=20.0, sigma=0.03 * u.AA, back=5.0):
    """One fitted parameter vector of the blend, in the units fit_cube_gauss uses."""
    shift = 1.0 + v_kms / C_KMS
    return [peak0, (REST0 * shift).to_value(u.cm), sigma.to_value(u.cm),
            peak1, (REST1 * shift).to_value(u.cm), sigma.to_value(u.cm), back]


def _single_params(peak, sigma, v_kms=0.0, back=5.0):
    return [peak, (REST0 * (1.0 + v_kms / C_KMS)).to_value(u.cm),
            sigma.to_value(u.cm), back]


SINGLE_UNITS = [u.DN / u.pix, u.cm, u.cm, u.DN / u.pix]


# --- component names and FitConfig checks --------------------------------

def test_components_are_named_or_named_after_their_wavelength():
    assert fitting.component_names(_blend_config(), REST0) == [
        "Fe XII 195.119", "195.1790 Angstrom"]
    assert fitting.component_names(None, REST0) == ["195.1190 Angstrom"]


@pytest.mark.parametrize("components, message", [
    ([FitComponent(wavelength=REST0), FitComponent(wavelength=REST0)],
     "distinct names"),
    ([FitComponent(wavelength=REST0, name="a"),
      FitComponent(wavelength=REST1, name="a")], "distinct names"),
    ([FitComponent(wavelength=REST0, tie_center=2),
      FitComponent(wavelength=REST1)], "index of another component"),
    ([FitComponent(wavelength=REST0, tie_width=0),
      FitComponent(wavelength=REST1)], "index of another component"),
    ([FitComponent(wavelength=REST0, name=" "),
      FitComponent(wavelength=REST1)], "non-empty string"),
], ids=["same wavelength", "same name", "tie out of range", "tied to itself",
        "blank name"])
def test_components_that_cannot_be_told_apart_or_tied_are_refused(components,
                                                                  message):
    with pytest.raises(ValueError, match=message):
        FitConfig(components=components)


@pytest.mark.parametrize("key", ["bessel_correction", "save_iterations"])
def test_the_switches_have_to_be_true_or_false(key):
    with pytest.raises(ValueError, match="true or false"):
        FitConfig(**{key: "yes"})


# --- what each fit means -----------------------------------------------------

def test_each_component_has_its_own_rest_wavelength():
    """10 km/s is 0.0065 A at 195.119 and 0.0065 A at 195.179, not the same shift."""
    fit_data = np.array([_blend_params(10.0)] * 2)
    quantities = fit_quantities(fit_data, _blend_units(), STEP, REST0,
                                _blend_config())
    components = quantities["components"]

    assert list(components) == ["Fe XII 195.119", "195.1790 Angstrom"]
    for name, peak in [("Fe XII 195.119", 100.0), ("195.1790 Angstrom", 20.0)]:
        assert components[name]["velocity"].to_value(u.km / u.s) == \
            pytest.approx(10.0, rel=1e-9)
        assert components[name]["width"].to_value(u.AA) == pytest.approx(0.03)
        # The fitted Gaussian summed over spectral pixels, in counts.
        assert components[name]["intensity"].unit == u.DN
        assert components[name]["intensity"].value == pytest.approx(
            np.sqrt(2 * np.pi) * peak * 0.03 / 0.0169)
    assert quantities["background"].to_value(u.DN / u.pix) == pytest.approx(5.0)


# --- statistics over iterations ---------------------------------------------

def _single_run(peaks, sigmas, failed):
    """(n_iter, 1, len(peaks[0])) single-Gaussian fits from per-iteration values."""
    fit_data = np.array([[[_single_params(p, s * u.AA)
                           for p, s in zip(peak_row, sigma_row)]]
                         for peak_row, sigma_row in zip(peaks, sigmas)])
    return fit_data, np.array(failed)[:, np.newaxis, :]


def test_failed_fits_are_left_out_and_counted():
    peaks = [[10.0, 50.0], [12.0, 50.0], [1e9, 50.0], [14.0, 50.0]]
    sigmas = [[0.03, 0.03]] * 4
    fit_data, failed = _single_run(
        peaks, sigmas, [[False, True], [False, True], [True, True], [False, True]])

    summary = summarise_fits(fit_data, failed, SINGLE_UNITS, STEP, REST0)

    assert summary["n_iterations"] == 4
    assert summary["failed_fits"].tolist() == [[1, 4]]
    assert summary["mean_data"][0, 0, 0] == pytest.approx(12.0)
    assert summary["std_data"][0, 0, 0] == pytest.approx(np.std([10.0, 12.0, 14.0]))
    # Every fit of the second pixel failed, so it has no statistics at all.
    assert np.all(np.isnan(summary["mean_data"][0, 1]))

    intensity = summary["components"]["195.1190 Angstrom"]["intensity"]
    per_fit = np.sqrt(2 * np.pi) * np.array([10.0, 12.0, 14.0]) * 0.03 / 0.0169
    assert intensity["mean"][0, 0].value == pytest.approx(per_fit.mean())
    assert intensity["first"][0, 0].value == pytest.approx(per_fit[0])
    assert np.isnan(intensity["first"][0, 1].value)
    assert np.isnan(intensity["mean"][0, 1].value)


def test_bessel_correction_divides_by_n_minus_1():
    peaks = [[10.0], [12.0], [17.0]]
    fit_data, failed = _single_run(peaks, [[0.03]] * 3, [[False]] * 3)

    plain = summarise_fits(fit_data, failed, SINGLE_UNITS, STEP, REST0)
    corrected = summarise_fits(fit_data, failed, SINGLE_UNITS, STEP, REST0,
                               FitConfig(bessel_correction=True))

    assert not plain["bessel_correction"] and corrected["bessel_correction"]
    assert plain["std_data"][0, 0, 0] == pytest.approx(np.std([10, 12, 17]))
    assert corrected["std_data"][0, 0, 0] == pytest.approx(
        np.std([10, 12, 17], ddof=1))
    name = "195.1190 Angstrom"
    ratio = (corrected["components"][name]["intensity"]["std"]
             / plain["components"][name]["intensity"]["std"])
    assert ratio[0, 0].value == pytest.approx(np.sqrt(3 / 2))


def test_intensity_spread_includes_peak_and_width_moving_together():
    """The per-parameter spreads cannot give this; the per-fit intensities can.

    Peak and width vary in opposite directions here so that their product,
    and so the intensity, is the same in every iteration.
    """
    fit_data, failed = _single_run([[10.0], [20.0], [40.0]],
                                   [[0.04], [0.02], [0.01]], [[False]] * 3)
    summary = summarise_fits(fit_data, failed, SINGLE_UNITS, STEP, REST0)

    assert summary["std_data"][0, 0, 0] > 0 and summary["std_data"][0, 0, 2] > 0
    intensity = summary["components"]["195.1190 Angstrom"]["intensity"]
    assert intensity["std"][0, 0].value == pytest.approx(0.0, abs=1e-9)


def test_ties_and_the_primary_component_are_recorded_by_name():
    fit_data = np.array([[[_blend_params(v)]] for v in (5.0, 7.0)])
    failed = np.zeros((2, 1, 1), dtype=bool)
    summary = summarise_fits(fit_data, failed, _blend_units(), STEP, REST0,
                             _blend_config(primary_component=1))

    assert summary["primary_component"] == "195.1790 Angstrom"
    components = summary["components"]
    assert components["Fe XII 195.119"]["tied"] == {"velocity": None, "width": None}
    assert components["195.1790 Angstrom"]["tied"] == {
        "velocity": "Fe XII 195.119", "width": "Fe XII 195.119"}
    assert components["195.1790 Angstrom"]["rest_wavelength"] == REST1
    assert components["195.1790 Angstrom"]["velocity"]["mean"][0, 0].to_value(
        u.km / u.s) == pytest.approx(6.0)


def test_every_iteration_is_kept_only_when_asked():
    fit_data, failed = _single_run([[10.0], [12.0]], [[0.03]] * 2, [[False]] * 2)
    assert "iterations" not in summarise_fits(fit_data, failed, SINGLE_UNITS,
                                              STEP, REST0)
    kept = summarise_fits(fit_data, failed, SINGLE_UNITS, STEP, REST0,
                          FitConfig(save_iterations=True))["iterations"]
    assert np.array_equal(kept["fit_data"], fit_data)
    assert np.array_equal(kept["failed"], failed)


# --- failures from the fitters themselves ------------------------------------

def _line_cube(n_wave=25):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [STEP.to_value(u.cm), 0.4, 0.16]
    wcs.wcs.crpix = [(n_wave + 1) / 2, 1.0, 1.0]
    wcs.wcs.crval = [REST0.to_value(u.cm), 0.0, 0.0]
    wave = (REST0 + (np.arange(n_wave) - (n_wave - 1) / 2) * STEP).to_value(u.cm)
    centre = (REST0 + 0.37 * STEP).to_value(u.cm)
    sigma = (2.3 * STEP).to_value(u.cm)
    profile = 100.0 * np.exp(-0.5 * ((wave - centre) / sigma) ** 2) + 5.0
    return NDCube(np.tile(profile, (2, 3, 1)), wcs=wcs, unit=u.DN / u.pix,
                  meta={"rest_wav": REST0})


def test_a_fit_that_runs_out_of_iterations_is_reported_as_failed():
    cube = _line_cube()
    _, _, starved = fit_cube_gauss(cube, n_jobs=1, fit_config=FitConfig(max_iter=1),
                                   return_failed=True)
    _, _, ample = fit_cube_gauss(cube, n_jobs=1, return_failed=True)
    assert starved.shape == (2, 3) and starved.all()
    assert not ample.any()


def test_return_failed_has_to_be_named():
    """Keyword-only, so the two return shapes cannot be mixed up by position."""
    with pytest.raises(TypeError):
        fit_cube_gauss(_line_cube(), 1, None, True)


def test_mpfit_running_out_of_iterations_counts_as_failed():
    """Status 5 is mpfit's out-of-iterations code; scipy raises instead."""
    params = np.array([1.0, 2.0, 3.0, 4.0])
    assert fitting._mpfit_succeeded(1, params)
    assert not fitting._mpfit_succeeded(5, params)
    assert not fitting._mpfit_succeeded(0, params)
    assert not fitting._mpfit_succeeded(1, np.array([1.0, np.nan, 3.0, 4.0]))


def test_the_ground_truth_is_nan_where_its_fit_failed():
    cube = _line_cube()
    failed = fitting.ground_truth_summary(cube, FitConfig(max_iter=1), n_jobs=1)
    assert failed["failed"].all()
    assert np.all(np.isnan(failed["components"]["195.1190 Angstrom"]["velocity"]))

    truth = fitting.ground_truth_summary(cube, n_jobs=1)
    assert not truth["failed"].any()
    velocity = truth["components"]["195.1190 Angstrom"]["velocity"]
    expected = (0.37 * STEP / REST0 * const.c).to_value(u.km / u.s)
    assert velocity.to_value(u.km / u.s) == pytest.approx(np.full((2, 3), expected),
                                                          rel=1e-4)
    assert "intensity" not in truth["components"]["195.1190 Angstrom"]


def test_spectral_pixel_width_is_read_from_the_cube():
    assert spectral_pixel_width(_line_cube()).to_value(u.AA) == pytest.approx(0.0169)


# --- the Monte Carlo keeps them ----------------------------------------------

def _intensity_cube(n_slit=3, n_scan=2, n_wave=16, peak=2.0e4):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [STEP.to_value(u.AA), 0.2, 0.16]
    wcs.wcs.crpix = [n_wave / 2.0, 1.0, n_slit / 2.0]
    wcs.wcs.crval = [REST0.to_value(u.AA), 0.0, 0.0]
    lam = np.arange(n_wave) - n_wave / 2.0
    data = np.tile(peak * np.exp(-0.5 * (lam / 1.5) ** 2), (n_slit, n_scan, 1))
    return NDCube(data, wcs=wcs, unit=u.erg / (u.cm**2 * u.s * u.sr * u.cm),
                  meta={"rest_wav": REST0})


def test_the_monte_carlo_stores_named_statistics_and_reports_failures(capsys):
    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=0.2 * u.arcsec, ncpu=1)
    fit_config = FitConfig(bessel_correction=True, save_iterations=True)

    np.random.seed(7)
    first_dn, dn_stats, _, photon_stats = monte_carlo(
        _intensity_cube(), 40 * u.s, det, tel, sim, n_iter=4,
        fit_config=fit_config, fit_signals="dn")

    assert photon_stats is None
    n_failed = int(dn_stats["failed_fits"].sum())
    out = capsys.readouterr().out
    if n_failed:
        assert f"DN fits: {n_failed} of 24 failed, in " in out
    else:
        assert "DN fits: none of 24 failed" in out

    name = "195.1190 Angstrom"
    assert dn_stats["primary_component"] == name
    assert list(dn_stats["components"]) == [name]
    assert dn_stats["n_iterations"] == 4
    assert dn_stats["failed_fits"].shape == (3, 2)

    kept = dn_stats["iterations"]
    assert kept["fit_data"].shape == (4, 3, 2, 4)
    quantities = fit_quantities(kept["fit_data"], dn_stats["units"],
                                spectral_pixel_width(first_dn), REST0)
    velocity = np.where(kept["failed"], np.nan,
                        quantities["components"][name]["velocity"].value)
    stored = dn_stats["components"][name]["velocity"]
    assert np.allclose(stored["mean"].value, np.nanmean(velocity, axis=0),
                       equal_nan=True)
    assert np.allclose(stored["std"].value, np.nanstd(velocity, axis=0, ddof=1),
                       equal_nan=True)


def test_the_run_says_how_many_fits_failed_and_in_how_many_pixels(capsys):
    from euvst_response.monte_carlo import _fit_results

    fit_data, failed = _single_run(
        [[10.0, 50.0]] * 4, [[0.03, 0.03]] * 4,
        [[True, False], [True, False], [True, True], [True, False]])

    _fit_results("DN", fit_data, failed, SINGLE_UNITS, _line_cube(), REST0, None)
    assert capsys.readouterr().out.strip() == (
        "DN fits: 5 of 8 failed, in 2 of 2 pixels, and are left out of the "
        "statistics; no fit succeeded in 1 of them")

    _fit_results("Photon", fit_data, np.zeros_like(failed), SINGLE_UNITS,
                 _line_cube(), REST0, None)
    assert capsys.readouterr().out.strip() == "Photon fits: none of 8 failed"


# --- the whole run, from the config file to the maps --------------------------

UNIFORM_RUN = """
instrument: SWC
uniform_intensity: 5000 erg / (s cm2 sr)
n_iter: 3
ncpu: 1
fit_signals: dn
fitting:
  bessel_correction: true
  save_iterations: true
"""


def test_a_run_writes_named_results_that_the_analysis_reads(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    from euvst_response.main import main

    config = tmp_path / "uniform.yaml"
    config.write_text(UNIFORM_RUN)
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    monkeypatch.chdir(tmp_path)
    main()
    assert "DN fits: " in capsys.readouterr().out

    results = load_instrument_response_results(tmp_path / "run/result/uniform.pkl")
    combination = next(iter(results["results"]["all_combinations"].values()))

    assert list_fit_components(combination) == ["195.1190 Angstrom"]
    stats = combination["dn_fit_stats"]
    assert stats["bessel_correction"] is True
    assert stats["iterations"]["fit_data"].shape == (3, 1, 1, 4)
    assert "195.1190 Angstrom" in combination["ground_truth"]["components"]

    analysis = analyse_fit_statistics(combination)
    assert analysis["component"] == "195.1190 Angstrom"
    assert analysis["i_mean"].unit == u.DN
    assert np.all(analysis["i_mean"].value > 0)

    maps = create_sunpy_maps_from_combo(combination, date_obs=WHEN)
    for key in ("intensity_from_fit", "intensity_mean", "intensity_std",
                "failed_fits", "velocity_mean", "line_width_std"):
        assert key in maps, key
    assert maps["intensity_mean"].data == pytest.approx(analysis["i_mean"].value)

    summary_table(results)
    assert "195.1190 Angstrom (primary)" in capsys.readouterr().out


# --- analysis chooses the component --------------------------------------------

def _blend_combination():
    """Results for a two-component fit, with each component at its own velocity."""
    fit_config = _blend_config()
    units = _blend_units()
    fit_data = np.array([[[_blend_params(v, peak1=p1)
                           for v, p1 in zip((4.0, 6.0), (20.0, 30.0))]]
                         for _ in range(3)])
    # These fits are made up, so give the second component a velocity 2 km/s
    # above the first, to tell the two apart in the analysis.
    fit_data[..., 4] = (REST1 * (1.0 + (np.array([4.0, 6.0]) + 2.0) / C_KMS)).to_value(u.cm)
    failed = np.zeros(fit_data.shape[:3], dtype=bool)
    failed[2, 0, 1] = True

    truth_data = fit_data[0]
    truth = fit_quantities(truth_data, units, STEP, REST0, fit_config)
    ground_truth = {
        "fit_truth_data": truth_data,
        "fit_truth_units": units,
        "failed": np.zeros(truth_data.shape[:2], dtype=bool),
        "components": {name: {key: values[key] for key in ("velocity", "width")}
                       for name, values in truth["components"].items()},
    }

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [STEP.to_value(u.AA), 0.2, 0.16]
    wcs.wcs.crpix = [10.0, 1.0, 1.0]
    wcs.wcs.crval = [REST0.to_value(u.AA), 0.0, 0.0]
    signal = np.ones((1, 2, 20))
    return {
        "first_signal_wcs": wcs,
        "first_dn_signal": NDCube(signal, wcs=wcs, unit=u.DN / u.pix),
        "first_photon_signal": NDCube(signal, wcs=wcs, unit=u.photon / u.pix),
        "dn_fit_stats": summarise_fits(fit_data, failed, units, STEP, REST0,
                                       fit_config),
        "ground_truth": ground_truth,
    }


def test_the_analysis_picks_the_component_by_name():
    combination = _blend_combination()
    assert list_fit_components(combination) == ["Fe XII 195.119",
                                                 "195.1790 Angstrom"]

    primary = analyse_fit_statistics(combination)
    second = analyse_fit_statistics(combination, component="195.1790 Angstrom")

    assert primary["component"] == "Fe XII 195.119"
    assert primary["v_mean"].to_value(u.km / u.s) == pytest.approx(np.array([[4.0, 6.0]]))
    assert second["v_mean"].to_value(u.km / u.s) == pytest.approx(np.array([[6.0, 8.0]]))
    assert second["rest_wavelength"] == REST1
    assert second["tied"] == {"velocity": "Fe XII 195.119", "width": "Fe XII 195.119"}
    assert second["failed_fits"].tolist() == [[0, 1]]
    assert second["v_err"].to_value(u.km / u.s) == pytest.approx(np.array([[0.0, 0.0]]), abs=1e-9)

    ratio = second["i_mean"] / primary["i_mean"]
    assert ratio.value == pytest.approx(np.array([[20.0 / 100.0, 30.0 / 100.0]]))


def test_maps_are_made_for_the_chosen_component():
    combination = _blend_combination()
    maps = create_sunpy_maps_from_combo(combination, date_obs=WHEN,
                                        component="195.1790 Angstrom")
    assert maps["velocity_mean"].data == pytest.approx(np.array([[6.0, 8.0]]))
    assert maps["failed_fits"].data.tolist() == [[0.0, 1.0]]
    assert maps["intensity_mean"].unit == u.DN
    assert maps["failed_fits"].unit == u.dimensionless_unscaled

    # The new maps carry the same metadata as the old ones, so reading them
    # the way plotting does raises no warnings either.
    with warnings.catch_warnings():
        warnings.simplefilter("error", SunpyMetadataWarning)
        for name in ("intensity_from_fit", "intensity_mean", "intensity_std",
                     "failed_fits"):
            maps[name].wcs


def test_a_rest_wavelength_that_disagrees_with_the_component_is_refused():
    combination = _blend_combination()
    analyse_fit_statistics(combination, rest_wavelength=REST0)  # agrees
    with pytest.raises(ValueError, match="was fitted at"):
        analyse_fit_statistics(combination, rest_wavelength=REST1)


def test_an_unknown_component_is_refused_with_the_names():
    with pytest.raises(ValueError, match="195.1790 Angstrom"):
        analyse_fit_statistics(_blend_combination(), component="Fe XII 195.18")


def test_older_results_files_still_analyse_but_cannot_choose_a_component():
    combination = _blend_combination()
    stats = combination["dn_fit_stats"]
    combination["dn_fit_stats"] = {key: stats[key] for key in
                                   ("first_fit_data", "mean_data", "std_data", "units")}

    old = analyse_fit_statistics(combination, REST0)
    assert old["v_mean"].to_value(u.km / u.s) == pytest.approx(np.array([[4.0, 6.0]]))
    assert "i_mean" not in old
    with pytest.raises(ValueError, match="stored by name"):
        analyse_fit_statistics(combination, REST0, component="Fe XII 195.119")
    with pytest.raises(ValueError, match="stored by name"):
        list_fit_components(combination)
