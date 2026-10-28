"""A blend started at its rest wavelengths, and limits on each component's velocity and width.

Started this way, as eispac's templates are, a faint blend stays on its line.
"""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
import yaml
from astropy.wcs import WCS
from ndcube import NDCube

from eclipse.fitting import FitComponent, FitConfig, _guess_multi_params, fit_cube_gauss
from eclipse.results_file import load_results, save_results

STEP = 0.0169 * u.Angstrom
N_WAVE = 31
CENTRE = 195.13 * u.Angstrom
STRONG, FAINT = 195.119 * u.Angstrom, 195.179 * u.Angstrom
SIGMA = 0.027 * u.Angstrom
C_KM_S = const.c.to_value(u.km / u.s)
# eispac's fe_12_195_119.2c template: the centre within 0.04 Angstrom of rest
# and the width between 0.0191 and 0.0510 Angstrom.
EISPAC = dict(velocity_limits=(-61.5 * u.km / u.s, 61.5 * u.km / u.s),
              width_limits=(0.0191 * u.AA, 0.0510 * u.AA))


def _wave():
    return CENTRE + (np.arange(N_WAVE) - (N_WAVE - 1) / 2) * STEP


def _cube(spectra):
    """Spectra, shaped (n, N_WAVE), as a (1, n, N_WAVE) cube."""
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [STEP.to_value(u.cm), 0.4, 0.16]
    wcs.wcs.crpix = [(N_WAVE + 1) / 2, 1.0, 1.0]
    wcs.wcs.crval = [CENTRE.to_value(u.cm), 0.0, 0.0]
    return NDCube(np.asarray(spectra)[np.newaxis], wcs=wcs, unit=u.DN / u.pix)


def _blend(velocity=0.0, strong=100.0, ratio=0.06, background=5.0):
    """Fe XII 195.119 and a 195.179 `ratio` as bright, at `velocity` km/s."""
    wave = _wave()
    return background + sum(
        peak * np.exp(-0.5 * ((wave - rest * (1 + velocity / C_KM_S)) / SIGMA)
                      .decompose().value ** 2)
        for peak, rest in ((strong, STRONG), (strong * ratio, FAINT)))


def _components(primary=None, **limits):
    return [FitComponent(STRONG, **(primary or {}), **limits),
            FitComponent(FAINT, tie_center=0, tie_width=0)]


def _velocity(params):
    return (params[..., 1] * u.cm / STRONG.to(u.cm) - 1).value * C_KM_S


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
def test_a_noisy_faint_blend_started_at_rest_and_limited_stays_on_its_line(backend):
    """
    With a few dozen counts in the line, a search start lands some fits a
    line spacing, 92 km/s, from the truth; at rest within eispac's limits
    none do.
    """
    rng = np.random.default_rng(3)
    truth = _blend(strong=30.0)
    spectra = rng.poisson(np.tile(truth, (400, 1))).astype(float)
    cube = _cube(spectra)

    def far(**settings):
        config = FitConfig(components=_components(**settings.pop("limits", {})),
                           constrain_positive_intensity=True, backend=backend, **settings)
        data, _, failed = fit_cube_gauss(cube, n_jobs=1, fit_config=config, return_failed=True)
        velocity = _velocity(data[0])[~failed[0]]
        return np.mean(np.abs(velocity) > 45)

    searched = far()
    rested = far(start="rest", limits=EISPAC)
    assert searched > 0.02
    assert rested < searched / 4


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
def test_a_line_beyond_its_velocity_limit_is_fitted_at_the_limit(backend):
    """
    The line is to the blue. To the red, the faint component, 92 km/s to the
    red of the other, could reach it from inside the limits.
    """
    config = FitConfig(components=_components(**EISPAC), start="rest", backend=backend)
    data, _, failed = fit_cube_gauss(_cube([_blend(velocity=-80.0)]), n_jobs=1,
                                     fit_config=config, return_failed=True)
    assert _velocity(data[0, 0]) == pytest.approx(-61.5, abs=0.05)
    width = (data[0, 0, 2] * u.cm).to_value(u.AA)
    assert 0.0191 - 1e-9 <= width <= 0.0510 + 1e-9


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
def test_limits_that_hold_the_truth_change_nothing_in_a_noiseless_fit(backend):
    spectrum = _cube([_blend(velocity=20.0, ratio=0.2)])
    fitted = {}
    for name, settings in (("free", {}), ("limited", {"start": "rest"})):
        limits = EISPAC if name == "limited" else {}
        config = FitConfig(components=_components(**limits), backend=backend, **settings)
        fitted[name] = fit_cube_gauss(spectrum, n_jobs=1, fit_config=config)[0][0, 0]
    assert _velocity(fitted["limited"]) == pytest.approx(20.0, abs=1e-3)
    assert np.allclose(fitted["limited"], fitted["free"], rtol=1e-5, atol=0.0)


def test_a_rest_start_puts_each_component_at_its_rest_wavelength():
    """A search finds the shifted blend; a rest start does not look, but stays within limits."""
    spectrum = _blend(velocity=40.0, ratio=0.3)
    wave = _wave().to_value(u.cm)
    searched = _guess_multi_params(wave, spectrum, FitConfig(components=_components()))
    rested = _guess_multi_params(wave, spectrum, FitConfig(components=_components(),
                                                           start="rest"))
    assert _velocity(searched) == pytest.approx(40.0, abs=30.0)
    assert rested[1] == pytest.approx(STRONG.to_value(u.cm), rel=1e-12)
    assert rested[4] == pytest.approx(FAINT.to_value(u.cm), rel=1e-12)

    narrow = FitConfig(components=_components(width_limits=(0.05 * u.AA, 0.06 * u.AA),
                                              velocity_limits=(10 * u.km / u.s,
                                                               20 * u.km / u.s)),
                       start="rest")
    clipped = _guess_multi_params(wave, spectrum, narrow)
    assert _velocity(clipped) == pytest.approx(10.0, rel=1e-9)
    assert (clipped[2] * u.cm).to_value(u.AA) == pytest.approx(0.05, rel=1e-12)


@pytest.mark.parametrize("settings, message", [
    ({"start": "middle"}, "fitting.start must be 'search' or 'rest'"),
    ({"components": [FitComponent(STRONG), FitComponent(
        FAINT, tie_center=0, velocity_limits=(-60 * u.km / u.s, 60 * u.km / u.s))]},
     "velocity_limits and tie_center"),
    ({"components": [FitComponent(STRONG), FitComponent(
        FAINT, tie_width=0, width_limits=(0.02 * u.AA, 0.05 * u.AA))]},
     "width_limits and tie_width"),
    ({"start": "rest", "components": []}, "only applied to multi-component fits"),
])
def test_what_cannot_apply_is_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        FitConfig(**{"components": _components(), **settings})


@pytest.mark.parametrize("limits", [
    {"velocity_limits": (60 * u.km / u.s, -60 * u.km / u.s)},
    {"velocity_limits": (-60 * u.AA, 60 * u.AA)},
    {"velocity_limits": (-60 * u.km / u.s,)},
    {"width_limits": (0 * u.AA, 0.05 * u.AA)},
    {"width_limits": (0.02 * u.km / u.s, 0.05 * u.km / u.s)},
])
def test_limits_that_are_not_two_ordered_values_of_the_right_kind_are_refused(limits):
    with pytest.raises(ValueError, match="must be two"):
        FitComponent(STRONG, **limits)


def test_a_configuration_file_gives_them_with_units(tmp_path):
    from eclipse.main import _parse_fitting_config

    text = """
fitting:
  start: rest
  constrain_positive_intensity: true
  components:
    - wavelength: 195.119 angstrom
      velocity_limits: [-60 km/s, 60 km/s]
      width_limits: [0.019 angstrom, 0.051 angstrom]
    - wavelength: 195.179 angstrom
      tie_center: 0
      tie_width: 0
"""
    config = _parse_fitting_config(yaml.safe_load(text))
    assert config.start == "rest"
    low, high = config.components[0].velocity_limits
    assert low == -60 * u.km / u.s and high == 60 * u.km / u.s
    assert config.components[0].width_limits[1] == 0.051 * u.AA


def test_the_settings_are_kept_in_a_results_file(tmp_path):
    config = FitConfig(components=_components(**EISPAC), start="rest")
    save_results(tmp_path / "run.h5", {"config_objects": {"fit": config}})
    restored = load_results(tmp_path / "run.h5")["config_objects"]["fit"]
    assert restored.start == "rest"
    assert restored.components[0].velocity_limits == EISPAC["velocity_limits"]
    assert restored.components[0].width_limits == EISPAC["width_limits"]


def test_a_results_file_from_before_the_settings_loads_as_it_was_run(tmp_path):
    """Earlier runs searched and had no limits, which is what the settings give, without a warning."""
    import json
    import warnings

    import h5py

    from eclipse import results_file

    config = FitConfig(components=_components())
    path = save_results(tmp_path / "old.h5", {"instrument": "SWC"})
    encoded = results_file._jsonable(config)
    del encoded["fields"]["start"]
    for component in encoded["fields"]["components"]:
        del component["fields"]["velocity_limits"], component["fields"]["width_limits"]
    with h5py.File(path, "r+") as f:
        f.attrs["fit"] = json.dumps(encoded)
        f.attrs["eclipse_order"] = json.dumps(["instrument", "fit"])
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        restored = load_results(path)["fit"]
    assert not seen, [str(w.message) for w in seen]
    assert repr(restored) == repr(config)


def test_a_search_within_limits_stays_within_them():
    """The comb search tries only shifts that keep a limited component within its limits."""
    spectrum = _blend(velocity=-40.0, ratio=0.06, strong=1.0) + _blend(velocity=52.0, ratio=0.0,
                                                                       strong=1.2, background=0)
    config = FitConfig(components=_components(velocity_limits=(-60 * u.km / u.s,
                                                                 0 * u.km / u.s)))
    guess = _guess_multi_params(_wave().to_value(u.cm), spectrum, config)
    assert -60.0 <= _velocity(guess) <= 0.0
    assert _velocity(guess) == pytest.approx(-40.0, abs=10.0)


@pytest.mark.parametrize("value", ["60 km/s", 60, ["-60 km/s", "60 kms"]])
def test_a_mistyped_limit_in_a_configuration_file_says_what_it_should_be(value):
    from eclipse.main import _parse_fitting_config

    fitting = {"components": [{"wavelength": "195.119 angstrom", "velocity_limits": value},
                              {"wavelength": "195.179 angstrom", "tie_center": 0}]}
    with pytest.raises(ValueError, match="velocity_limits must be two"):
        _parse_fitting_config({"fitting": fitting})


def test_a_component_given_limits_after_it_is_made_is_fitted_within_them():
    component = FitComponent(STRONG)
    component.velocity_limits = (-10 * u.km / u.s, 10 * u.km / u.s)
    assert component.centre_limits()[1] == pytest.approx(
        STRONG.to_value(u.cm) * (1 + 10 / C_KM_S), rel=1e-12)
