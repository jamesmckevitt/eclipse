"""A fitting block without components now configures the single-Gaussian fit.

It used to be dropped whole unless it had at least two components, so there
was no way to change max_iter or the backend for a single Gaussian, and a
block with exactly one component did nothing at all (issue #70).
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response import fitting
from euvst_response.fitting import FitComponent, FitConfig, fit_cube_gauss
from euvst_response.main import _parse_fitting_config

REST = 195.119 * u.Angstrom
N_WAVE = 25
STEP = 0.0169 * u.Angstrom
# Between pixels and away from the window centre, so the initial guess, which
# puts the centre on the brightest pixel, is not already the answer.
TRUE = {"peak": 100.0, "centre": REST + 0.37 * STEP, "sigma": 2.3 * STEP,
        "back": 5.0}


def _line_cube():
    """A noiseless Gaussian in every pixel of a (2, 3, N_WAVE) cube."""
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [STEP.to_value(u.cm), 0.4, 0.16]
    wcs.wcs.crpix = [(N_WAVE + 1) / 2, 1.0, 1.0]
    wcs.wcs.crval = [REST.to_value(u.cm), 0.0, 0.0]
    wave = (REST + (np.arange(N_WAVE) - (N_WAVE - 1) / 2) * STEP).to_value(u.cm)
    centre = TRUE["centre"].to_value(u.cm)
    sigma = TRUE["sigma"].to_value(u.cm)
    profile = (TRUE["peak"] * np.exp(-0.5 * ((wave - centre) / sigma) ** 2)
               + TRUE["back"])
    return NDCube(np.tile(profile, (2, 3, 1)), wcs=wcs, unit=u.DN / u.pix)


def _assert_recovers_the_line(fit):
    assert fit[..., 0] == pytest.approx(TRUE["peak"], rel=1e-4)
    assert fit[..., 1] == pytest.approx(TRUE["centre"].to_value(u.cm),
                                        abs=1e-4 * STEP.to_value(u.cm))
    assert fit[..., 2] == pytest.approx(TRUE["sigma"].to_value(u.cm), rel=1e-4)
    assert fit[..., 3] == pytest.approx(TRUE["back"], rel=1e-3)


# --- parsing the fitting block ----------------------------------------------

def test_no_fitting_block_gives_no_fit_config():
    assert _parse_fitting_config({"instrument": "SWC"}) is None


def test_a_block_without_components_configures_the_single_gaussian_fit():
    """The case from the issue: this block used to be dropped."""
    fit_config = _parse_fitting_config(
        {"fitting": {"max_iter": 5000, "backend": "mpfit"}})
    assert fit_config.is_single
    assert fit_config.max_iter == 5000
    assert fit_config.backend == "mpfit"


def test_a_single_component_is_refused():
    config = {"fitting": {"components": [{"wavelength": "195.119 angstrom"}]}}
    with pytest.raises(ValueError, match="one entry"):
        _parse_fitting_config(config)
    with pytest.raises(ValueError, match="one entry"):
        FitConfig(components=[FitComponent(wavelength=REST)])


def test_two_components_still_build_a_multi_component_fit():
    fit_config = _parse_fitting_config({"fitting": {
        "primary_component": 1,
        "components": [{"wavelength": "195.119 angstrom"},
                       {"wavelength": "195.179 angstrom", "tie_center": 0}],
    }})
    assert not fit_config.is_single
    assert fit_config.primary_component == 1
    assert fit_config.components[1].tie_center == 0


@pytest.mark.parametrize("settings, message", [
    ({"primary_component": 1}, "there are none"),
    ({"constrain_positive_intensity": True}, "no components"),
], ids=["primary_component", "constrain_positive_intensity"])
def test_multi_component_settings_without_components_are_refused(settings,
                                                                 message):
    """Accepting them would look effective and change nothing."""
    with pytest.raises(ValueError, match=message):
        _parse_fitting_config({"fitting": settings})


def test_a_primary_component_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="there are 2 components"):
        FitConfig(components=[FitComponent(wavelength=REST),
                              FitComponent(wavelength=195.179 * u.AA)],
                  primary_component=2)


@pytest.mark.parametrize("settings, message", [
    ({"backend": "lmfit"}, "Unknown fitting backend"),
    ({"max_iter": 0}, "positive integer"),
    ({"max_iter": 10.5}, "positive integer"),
    ({"max_iter": True}, "positive integer"),
])
def test_bad_backend_and_max_iter_are_refused_without_components(settings,
                                                                 message):
    """Checked in FitConfig now, so they no longer need components to apply."""
    with pytest.raises(ValueError, match=message):
        _parse_fitting_config({"fitting": settings})


# --- the settings reaching the single-Gaussian fit --------------------------

def test_the_default_single_gaussian_fit_recovers_the_line():
    fit, _ = fit_cube_gauss(_line_cube(), n_jobs=1)
    _assert_recovers_the_line(fit)


def test_max_iter_reaches_the_single_gaussian_fit():
    """With one iteration scipy runs out and returns its initial guess."""
    cube = _line_cube()
    wave = cube.axis_world_coords(2)[0].to_value(u.cm)
    guess = fitting._guess_params(wave, cube.data[0, 0].copy())

    starved, _ = fit_cube_gauss(cube, n_jobs=1, fit_config=FitConfig(max_iter=1))
    assert np.array_equal(starved[0, 0], guess)

    ample, _ = fit_cube_gauss(cube, n_jobs=1, fit_config=FitConfig(max_iter=5000))
    _assert_recovers_the_line(ample)


def test_the_mpfit_backend_reaches_the_single_gaussian_fit(monkeypatch):
    calls = []
    real_mpfit = fitting.mpfit

    def recording_mpfit(*args, **kwargs):
        calls.append(kwargs["maxiter"])
        return real_mpfit(*args, **kwargs)

    monkeypatch.setattr(fitting, "mpfit", recording_mpfit)
    fit, units = fit_cube_gauss(
        _line_cube(), n_jobs=1,
        fit_config=FitConfig(backend="mpfit", max_iter=777))

    assert calls == [777] * 6
    assert fit.shape == (2, 3, 4)
    assert len(units) == 4
    _assert_recovers_the_line(fit)
