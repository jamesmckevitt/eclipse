"""The DN fits are weighted by each pixel's uncertainty, and the photon fits are of greatest likelihood.

Unweighted, every pixel of a spectrum counted the same, though the pixels at
a line's peak are noisier than those on its wings, which fix where the line
is. A Gaussian line's velocity then scattered by 1.24 times the least its
photons allow. eispac weights its fits by each pixel's uncertainty, worked
out from its own signal, and so do the DN fits now. The photons, a Poisson
count with no read noise, are fitted by their Poisson likelihood, which
needs no uncertainty and reaches the least scatter they allow.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.config import Detector_EIS, Detector_SWC, Simulation, Telescope_EUVST
from euvst_response.data_processing import create_uniform_intensity_cube
from euvst_response.fitting import FitComponent, FitConfig, _poisson_residuals, fit_cube_gauss
from euvst_response.monte_carlo import expected_dn_uncertainty
from euvst_response.radiometric import dn_variance, to_dn, to_electrons

REST = 195.119 * u.AA
STEP = 0.0223 * u.AA


def _wcs(n_wave):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [STEP.to_value(u.cm), 0.4, 0.16]
    wcs.wcs.crpix = [(n_wave + 1) / 2, 1.0, 1.0]
    wcs.wcs.crval = [REST.to_value(u.cm), 0.0, 0.0]
    return wcs


def _observed(photons, det, t_exp):
    """Photons, arriving as a Poisson count, through the detector to DN."""
    arrivals = np.random.poisson(photons).astype(float)
    cube = NDCube(arrivals, wcs=_wcs(arrivals.shape[-1]), unit=u.photon / u.pixel,
                  meta={"rest_wav": REST})
    return to_dn(to_electrons(cube, t_exp, det), det).data


@pytest.mark.parametrize("det", [Detector_SWC(), Detector_EIS()], ids=["SWC", "EIS"])
@pytest.mark.parametrize("photons", [0.0, 5.0, 50.0, 500.0])
def test_the_variance_worked_out_from_a_pixels_dn_is_that_of_the_detector(det, photons):
    """Each photon's many electrons vary together, so the noise is per photon, not per electron."""
    t_exp = 20 * u.s
    np.random.seed(11)
    dn = _observed(np.full((1, 1, 200_000), photons), det, t_exp)
    predicted = dn_variance(np.mean(dn), REST, t_exp, det)
    assert np.var(dn) == pytest.approx(predicted, rel=0.02)


def test_pixels_summed_after_read_out_each_bring_their_own_dark_read_noise_and_rounding():
    det, t_exp = Detector_SWC(), 20 * u.s
    np.random.seed(12)
    rows = _observed(np.full((3, 1, 100_000), 40.0), det, t_exp)
    summed = rows.sum(axis=0)
    predicted = dn_variance(np.mean(summed), REST, t_exp, det, n_binned=3)
    assert np.var(summed) == pytest.approx(predicted, rel=0.02)


def _line(n_wave=31, peak=200.0, background=10.0):
    x = np.arange(n_wave) - (n_wave - 1) / 2
    return peak * np.exp(-0.5 * (x / 2.0) ** 2) + background


def test_a_pixel_with_a_large_uncertainty_hardly_moves_a_weighted_fit():
    profile = _line()
    profile[18] += 150.0
    cube = NDCube(profile[np.newaxis, np.newaxis, :], wcs=_wcs(profile.size),
                  unit=u.DN / u.pixel, meta={"rest_wav": REST})
    uncertainty = np.ones_like(profile)
    uncertainty[18] = 1e6
    plain, _ = fit_cube_gauss(cube, n_jobs=1)
    weighted, _ = fit_cube_gauss(cube, n_jobs=1, uncertainty=uncertainty)
    centre = REST.to_value(u.cm)
    step = STEP.to_value(u.cm)
    assert abs(weighted[0, 0, 1] - centre) < 1e-4 * step
    assert abs(plain[0, 0, 1] - centre) > 1e-2 * step


def test_weighting_by_each_pixels_own_uncertainty_measures_a_bright_line_more_precisely():
    """For a line of several thousand photons, as eispac's weighting does for EIS."""
    det, t_exp = Detector_SWC(), 1 * u.s
    np.random.seed(13)
    n_spectra = 400
    photons = np.tile(_line(peak=600.0, background=2.0), (n_spectra, 1, 1))
    dn = _observed(photons, det, t_exp)
    cube = NDCube(dn, wcs=_wcs(dn.shape[-1]), unit=u.DN / u.pixel, meta={"rest_wav": REST})
    uncertainty = np.sqrt(dn_variance(dn, REST, t_exp, det))
    plain, _, plain_failed = fit_cube_gauss(cube, n_jobs=1, return_failed=True)
    weighted, _, weighted_failed = fit_cube_gauss(cube, n_jobs=1, return_failed=True,
                                                  uncertainty=uncertainty)
    assert not plain_failed.any() and not weighted_failed.any()
    ratio = np.std(weighted[:, 0, 1]) / np.std(plain[:, 0, 1])
    assert ratio < 0.92


@pytest.mark.parametrize("uncertainty, message", [
    (np.zeros(31), "finite and above zero"),
    (np.full(31, np.nan), "finite and above zero"),
    (np.ones(30), "does not match the spectra"),
])
def test_an_uncertainty_a_fit_cannot_weight_by_is_refused(uncertainty, message):
    cube = NDCube(_line()[np.newaxis, np.newaxis, :], wcs=_wcs(31), unit=u.DN / u.pixel,
                  meta={"rest_wav": REST})
    with pytest.raises(ValueError, match=message):
        fit_cube_gauss(cube, n_jobs=1, uncertainty=uncertainty)


def test_the_ground_truths_weights_are_those_of_the_dn_with_no_noise():
    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(slit_width=0.4 * u.arcsec, expos=10 * u.s)
    cube = create_uniform_intensity_cube(5000 * u.erg / (u.s * u.cm**2 * u.sr), REST,
                                         20 * u.km / u.s, det, sim, n_slit_pixels=2, tel=tel)
    expected = expected_dn_uncertainty(cube, sim.expos, det, tel, sim, offchip_bin_slit=2,
                                       uniform_mode=True)
    assert expected.shape == (1,) + cube.data.shape[1:]
    # With no line, two pixels' dark current, read noise and rounding.
    floor = np.sqrt(dn_variance(0.0, REST, sim.expos, det, n_binned=2))
    assert expected.min() == pytest.approx(floor, rel=1e-6)
    assert np.argmax(expected[0, 0]) == np.argmax(cube.data[0, 0])


def test_the_poisson_residuals_add_up_to_the_deviance():
    counts = np.array([0.0, 1.0, 4.0, 9.0, 30.0])
    model = np.array([0.5, 2.0, 4.0, 6.0, 33.0])
    residuals = _poisson_residuals(counts, model)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(counts > 0, counts * np.log(counts / model), 0.0)
    assert np.sum(residuals ** 2) == pytest.approx(2 * np.sum(model - counts + log_term))
    assert np.array_equal(np.sign(residuals), np.sign(counts - model))
    assert residuals[0] == pytest.approx(-1.0)  # no counts: -sqrt(2 m)


def _counts_cube(counts):
    return NDCube(counts, wcs=_wcs(counts.shape[-1]), unit=u.photon / u.pixel,
                  meta={"rest_wav": REST})


def _cramer_rao_centre(peak, width, background, n_wave=31):
    """The least variance of the centre, in pixels squared, that Poisson counts allow."""
    x = np.arange(n_wave) - (n_wave - 1) / 2
    g = np.exp(-0.5 * (x / width) ** 2)
    model = peak * g + background
    jacobian = np.stack([g, peak * g * x / width**2, peak * g * x**2 / width**3,
                         np.ones_like(x)], axis=1)
    fisher = jacobian.T @ (jacobian / model[:, np.newaxis])
    return np.linalg.inv(fisher)[1, 1]


def test_a_poisson_fit_of_photons_measures_a_line_as_precisely_as_its_photons_allow():
    peak, width, background = 40.0, 2.0, 1.0
    np.random.seed(14)
    x = np.arange(31) - 15
    expected = peak * np.exp(-0.5 * (x / width) ** 2) + background
    cube = _counts_cube(np.random.poisson(np.tile(expected, (600, 1, 1))).astype(float))
    poisson, _, poisson_failed = fit_cube_gauss(cube, n_jobs=1, return_failed=True,
                                                poisson=True)
    plain, _, plain_failed = fit_cube_gauss(cube, n_jobs=1, return_failed=True)
    assert not poisson_failed.any() and not plain_failed.any()
    pixels = lambda fits: (fits[:, 0, 1] - REST.to_value(u.cm)) / STEP.to_value(u.cm)  # noqa: E731
    least = np.sqrt(_cramer_rao_centre(peak, width, background))
    assert np.std(pixels(poisson)) == pytest.approx(least, rel=0.08)
    assert np.std(pixels(plain)) > 1.08 * np.std(pixels(poisson))
    assert abs(np.mean(pixels(poisson))) < 3 * least / np.sqrt(600)


def test_a_poisson_fit_takes_pixels_with_no_photons_in_its_stride():
    np.random.seed(15)
    x = np.arange(31) - 15
    expected = 4.0 * np.exp(-0.5 * (x / 2.0) ** 2)
    counts = np.random.poisson(np.tile(expected, (200, 1, 1))).astype(float)
    assert (counts == 0).any(axis=-1).all()
    fits, _, failed = fit_cube_gauss(_counts_cube(counts), n_jobs=1, return_failed=True,
                                     poisson=True)
    assert failed.mean() < 0.02
    assert np.all(fits[~failed[:, 0], 0, 0] >= 0) and np.all(fits[~failed[:, 0], 0, 3] >= 0)


@pytest.mark.parametrize("components", [[], [FitComponent(REST), FitComponent(REST + 6 * STEP)]],
                         ids=["one line", "a blend"])
def test_scipy_and_mpfit_find_the_same_poisson_fit(components):
    x = np.arange(31) - 15
    expected = 300.0 * np.exp(-0.5 * (x / 1.5) ** 2) + 2.0
    if components:
        expected += 150.0 * np.exp(-0.5 * ((x - 6) / 1.5) ** 2)
    np.random.seed(16)
    cube = _counts_cube(np.random.poisson(np.tile(expected, (5, 1, 1))).astype(float))
    fits = [fit_cube_gauss(cube, n_jobs=1, poisson=True,
                           fit_config=FitConfig(components=components, backend=backend))[0]
            for backend in ("scipy", "mpfit")]
    step = STEP.to_value(u.cm)
    centres = slice(1, None, 3)
    assert np.allclose(fits[0][..., centres], fits[1][..., centres], rtol=0, atol=1e-3 * step)
    assert np.allclose(fits[0][..., 0::3], fits[1][..., 0::3], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("counts, uncertainty, message", [
    (np.full(31, 3.0), np.ones(31), "give poisson or uncertainty, not both"),
    (np.full(31, -1.0), None, "counts, which are zero or more"),
])
def test_a_poisson_fit_it_cannot_make_is_refused(counts, uncertainty, message):
    with pytest.raises(ValueError, match=message):
        fit_cube_gauss(_counts_cube(counts[np.newaxis, np.newaxis, :]), n_jobs=1,
                       uncertainty=uncertainty, poisson=True)
