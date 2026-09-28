"""Chains of ties and of amplitude orderings in a multi-component fit.

A tie to a component that was itself tied read a value the fit had not set
yet, which on the scipy path was whatever memory held and on mpfit's lagged
a step behind; an amplitude held below one that was itself held below
another was capped, or never fitted, depending on the order the components
were listed in; and a primary held fainter than another started outside its
bounds, so that every fit failed. Loops, which no fit can meet, are refused.
"""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.fitting import FitComponent, FitConfig, fit_cube_gauss

STEP = 0.0169 * u.Angstrom
N_WAVE = 41
CENTRE = 195.13 * u.Angstrom
RESTS = (195.05 * u.Angstrom, 195.119 * u.Angstrom, 195.20 * u.Angstrom)
SIGMA = 0.025 * u.Angstrom
VELOCITY = 10 * u.km / u.s
# The noiseless spectra are fitted to a part in about 1e6 either way.
CLOSE = 1e-4


def _cube(peaks):
    """Three Gaussians at VELOCITY with *peaks*, in every pixel of a (1, 2, N_WAVE) cube."""
    wave = CENTRE + (np.arange(N_WAVE) - (N_WAVE - 1) / 2) * STEP
    profile = 10.0 + sum(peak * np.exp(-0.5 * ((wave - rest * (1 + VELOCITY / const.c))
                                                / SIGMA).decompose().value ** 2)
                         for peak, rest in zip(peaks, RESTS))
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [STEP.to_value(u.cm), 0.4, 0.16]
    wcs.wcs.crpix = [(N_WAVE + 1) / 2, 1.0, 1.0]
    wcs.wcs.crval = [CENTRE.to_value(u.cm), 0.0, 0.0]
    return NDCube(np.tile(profile, (1, 2, 1)), wcs=wcs, unit=u.DN / u.pix)


def _fit(peaks, components, **settings):
    config = FitConfig(components=components, **settings)
    data, _, failed = fit_cube_gauss(_cube(peaks), n_jobs=1, fit_config=config,
                                     return_failed=True)
    assert not failed.any()
    return data[0, 0]


def _velocities(params):
    return [((params[3 * i + 1] * u.cm / rest.to(u.cm) - 1) * const.c).to_value(u.km / u.s)
            for i, rest in enumerate(RESTS)]


@pytest.mark.parametrize("settings", [{"backend": "mpfit"},
                                      {"constrain_positive_intensity": True}],
                         ids=["mpfit", "scipy-trf"])
def test_a_chain_of_ties_fits_as_ties_to_its_free_end(settings):
    """2 is tied to 1, which is tied to 0, in centre and in width."""
    components = [FitComponent(RESTS[0]),
                  FitComponent(RESTS[1], tie_center=0, tie_width=0),
                  FitComponent(RESTS[2], tie_center=1, tie_width=1)]
    first = _fit((500.0, 1000.0, 300.0), components, **settings)
    assert _velocities(first) == pytest.approx([VELOCITY.value] * 3, abs=CLOSE * 10)
    assert first[2::3][:3] == pytest.approx([SIGMA.to_value(u.cm)] * 3, rel=CLOSE)
    # The same again, not what the last fit left in memory.
    assert np.array_equal(_fit((500.0, 1000.0, 300.0), components, **settings), first)


@pytest.mark.parametrize("settings", [{"backend": "mpfit"}, {}], ids=["mpfit", "scipy"])
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0)], ids=["listed", "reversed"])
def test_a_chain_of_amplitude_orders_fits_whatever_the_listing(settings, order):
    """
    Each component is held brighter than the next along *order*.

    Reversed, each parent is listed after its child. The brightest is the
    primary either way, which the guess starts at the peak.
    """
    peaks = np.zeros(3)
    peaks[list(order)] = (900.0, 400.0, 100.0)
    components = [FitComponent(rest) for rest in RESTS]
    for brighter, fainter in zip(order, order[1:]):
        components[brighter].amplitude_greater_than = fainter
    fitted = _fit(tuple(peaks), components, primary_component=order[0], **settings)
    assert fitted[0:9:3] == pytest.approx(peaks, rel=CLOSE)


def test_a_primary_held_fainter_than_another_is_fitted():
    """The guess starts the primary brightest, which put its ratio above 1."""
    components = [FitComponent(RESTS[0]), FitComponent(RESTS[1], amplitude_greater_than=0),
                  FitComponent(RESTS[2])]
    fitted = _fit((300.0, 1000.0, 200.0), components)
    assert fitted[0:9:3] == pytest.approx([300.0, 1000.0, 200.0], rel=CLOSE)


@pytest.mark.parametrize("ties, message", [
    ({0: {"tie_center": 1}, 1: {"tie_center": 0}}, r"\[0, 1, 0\] make a loop of tie_center"),
    ({0: {"tie_width": 2}, 2: {"tie_width": 1}, 1: {"tie_width": 0}}, "loop of tie_width"),
    ({1: {"amplitude_greater_than": 0}, 0: {"amplitude_greater_than": 1}},
     "loop of amplitude_greater_than"),
    ({1: {"amplitude_greater_than": 0}, 2: {"amplitude_greater_than": 0}},
     r"\[0\] are each held fainter than more than one"),
])
def test_what_no_fit_can_meet_is_refused(ties, message):
    components = [FitComponent(rest) for rest in RESTS]
    for index, settings in ties.items():
        for key, value in settings.items():
            setattr(components[index], key, value)
    with pytest.raises(ValueError, match=message):
        FitConfig(components=components)


def test_constrain_positive_intensity_must_be_true_or_false():
    """A quoted 'false' is a string, which is true."""
    components = [FitComponent(rest) for rest in RESTS]
    with pytest.raises(ValueError, match="constrain_positive_intensity must be true or false"):
        FitConfig(components=components, constrain_positive_intensity="false")
