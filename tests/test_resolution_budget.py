"""The SW resolution budget (RSC-2022033B) on top of the optical design's PSF."""
import json

import astropy.units as u
import h5py
import numpy as np
import pytest

from eclipse import resolution_budget, results_file
from eclipse.config import Detector_SWC, Telescope_EUVST
from eclipse.radiometric import (across_slit_fwhm, spatial_psf_fwhm, spectral_blur_fwhm,
                                 spectral_line_spread, spectral_optics_fwhm,
                                 spectral_psf_fwhm)
from eclipse.results_file import load_results, save_results

DET = Detector_SWC()
PLATE_SCALE = DET.plate_scale_angle.to_value(u.arcsec / u.pix)
# RSC-2022033B converts arcsec along the dispersion to mA at 1.253 mA per
# micron over 0.0118 arcsec per micron.
MA_PER_ARCSEC = 1.253 / 0.0118


def _total(*widths):
    return np.sqrt(np.sum(np.square(widths)))


@pytest.mark.parametrize("budget, optics, total", [
    ("current_estimate", 0.161, 0.337), ("current_estimate", 0.195, 0.355),
    ("allocation", 0.161, 0.407), ("allocation", 0.195, 0.421)])
def test_the_spatial_rows_give_the_budgets_totals(budget, optics, total):
    """Table 3-1's SW totals for the 70 x 70 and 100 x 100 arcsec fields."""
    assert _total(optics, resolution_budget.along_slit_fwhm(budget)) == pytest.approx(
        total, abs=0.0015)


@pytest.mark.parametrize("optics, total", [(16.66, 34.1), (19.89, 35.8), (37.39, 47.8)])
def test_the_spectral_rows_give_the_budgets_totals(optics, total):
    """Table 4-1's SW totals in mA at 170.0, 191.15 and 212.3 A.

    With the 0.2 arcsec slit and its 10% width error, 21.34 mA.
    """
    blur = resolution_budget.spectral_fwhm("allocation") * MA_PER_ARCSEC
    assert _total(optics, 21.34, blur) == pytest.approx(total, abs=0.06)


def test_the_spectral_estimate_follows_the_budgets_rules():
    """Each allocation row is its spatial row by the rule in its note, so the estimate is too."""
    estimate = {name: row[0] for name, row in resolution_budget.SPECTRAL_ROWS.items()}
    assert estimate["3-2: residual of 1G compensation (2/3 of #5-2)"] == pytest.approx(0.024)
    assert estimate["5: grating micro-vibrations (2/3 of #20)"] == pytest.approx(0.0733, abs=1e-4)
    assert resolution_budget.spectral_fwhm("current_estimate") == pytest.approx(0.128, abs=0.001)


def test_the_blur_across_the_slit_leaves_out_the_spectrograph():
    """Rows 2 to 6 and 20 of Table 3-1, but not 10."""
    assert resolution_budget.across_slit_fwhm("current_estimate") == pytest.approx(0.2784, abs=1e-4)
    assert resolution_budget.across_slit_fwhm("allocation") == pytest.approx(0.3360, abs=1e-4)


def test_no_budget_gives_no_blur():
    for part in (resolution_budget.along_slit_fwhm, resolution_budget.across_slit_fwhm,
                 resolution_budget.spectral_fwhm):
        assert part("none") == 0.0


def test_the_default_is_the_full_scan_with_the_current_estimate():
    tel = Telescope_EUVST()
    assert tel.field_of_view == "280x280" and tel.resolution_budget == "current_estimate"
    assert tel.psf_params == [2.66 * u.pix, 2.54 * u.pix]


@pytest.mark.parametrize("field, along", [("280x280", 2.66), ("100x100", 1.23), ("70x70", 1.01)])
def test_the_field_sets_the_optical_design_along_the_slit(field, along):
    assert Telescope_EUVST(field_of_view=field).psf_params == [along * u.pix, 2.54 * u.pix]


def test_psf_params_given_alone_are_the_optical_design():
    tel = Telescope_EUVST(psf_params=[2.0 * u.pix, 2.54 * u.pix])
    assert tel.field_of_view is None and tel.psf_params[0] == 2.0 * u.pix


@pytest.mark.parametrize("settings, message", [
    ({"field_of_view": "70x70", "psf_params": [2.0 * u.pix, 2.54 * u.pix]},
     "field_of_view sets psf_params, so give one or the other"),
    ({"field_of_view": "10x10"}, "field_of_view must be one of 280x280, 100x100, 70x70"),
    ({"field_of_view": ["70x70"]}, "field_of_view must be one of"),
    ({"resolution_budget": "cbe"}, "resolution_budget must be one of current_estimate"),
    ({"resolution_budget": None}, "resolution_budget must be one of")])
def test_bad_settings_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        Telescope_EUVST(**settings)


def test_the_field_and_its_own_widths_together_are_accepted():
    """As when a results file gives back both."""
    tel = Telescope_EUVST(field_of_view="70x70", psf_params=[1.01 * u.pix, 2.54 * u.pix])
    assert tel.field_of_view == "70x70"


@pytest.mark.parametrize("budget", ["current_estimate", "allocation"])
def test_the_budget_adds_in_quadrature_to_the_optics(budget):
    tel = Telescope_EUVST(resolution_budget=budget)
    optics = Telescope_EUVST(resolution_budget="none")
    along = resolution_budget.along_slit_fwhm(budget) / PLATE_SCALE
    spectral = resolution_budget.spectral_fwhm(budget) / PLATE_SCALE
    assert spatial_psf_fwhm(tel, DET) == pytest.approx(np.hypot(2.66, along))
    for slit in (0.2, 0.4, 1.6):
        assert spectral_psf_fwhm(tel, DET, slit * u.arcsec) == pytest.approx(
            np.hypot(spectral_psf_fwhm(optics, DET, slit * u.arcsec), spectral))
    assert spectral_blur_fwhm(tel, DET) == pytest.approx(
        np.hypot(spectral_optics_fwhm(tel, DET), spectral))
    assert across_slit_fwhm(tel).to_value(u.arcsec) == pytest.approx(
        resolution_budget.across_slit_fwhm(budget))


def test_without_the_budget_the_psf_is_the_optical_design():
    tel = Telescope_EUVST(resolution_budget="none")
    assert spatial_psf_fwhm(tel, DET) == 2.66
    assert spectral_psf_fwhm(tel, DET, 0.2 * u.arcsec) == 2.54
    assert spectral_blur_fwhm(tel, DET) == spectral_optics_fwhm(tel, DET)
    assert across_slit_fwhm(tel) is None


def test_the_optics_across_the_slit_add_to_the_budget():
    tel = Telescope_EUVST(psf_across_slit=0.5 * u.arcsec)
    assert across_slit_fwhm(tel).to_value(u.arcsec) == pytest.approx(
        np.hypot(0.5, resolution_budget.across_slit_fwhm("current_estimate")))


def _rms_width(kernel):
    x = np.arange(kernel.size) - kernel.size // 2
    return np.sqrt(np.sum(kernel * x**2))


def test_the_convolved_profile_widens_with_the_budget():
    """Variances add under convolution, so the budget adds its variance to the profile's."""
    narrow = spectral_line_spread(Telescope_EUVST(resolution_budget="none"), DET, 0.4 * u.arcsec)
    wide = spectral_line_spread(Telescope_EUVST(), DET, 0.4 * u.arcsec)
    fwhm = resolution_budget.spectral_fwhm("current_estimate") / PLATE_SCALE
    sigma = fwhm / np.sqrt(8 * np.log(2))
    assert _rms_width(wide)**2 == pytest.approx(_rms_width(narrow)**2 + sigma**2, rel=0.01)


def test_a_telescope_from_before_the_budget_reads_back_without_it(tmp_path):
    """Its runs had only the optical design, so it gets none, not today's default."""
    encoded = results_file._jsonable(Telescope_EUVST(psf_params=[2.66 * u.pix, 2.54 * u.pix]))
    del encoded["fields"]["resolution_budget"], encoded["fields"]["field_of_view"]
    path = save_results(tmp_path / "hand.h5", {"instrument": "SWC"})
    with h5py.File(path, "r+") as f:
        f.attrs["telescope"] = json.dumps(encoded)
        f.attrs["eclipse_order"] = json.dumps(["instrument", "telescope"])
    with pytest.warns(UserWarning, match="Telescope_EUVST in the results file has no "
                                         "resolution_budget, which ECLIPSE added after it was "
                                         "made; it gets 'none'"):
        restored = load_results(path)["telescope"]
    assert restored.resolution_budget == "none"
    assert restored.psf_params == [2.66 * u.pix, 2.54 * u.pix]


def test_a_telescope_reads_back_the_same(tmp_path):
    tel = Telescope_EUVST(field_of_view="70x70")
    path = save_results(tmp_path / "default.h5", {"instrument": "SWC", "telescope": tel})
    restored = load_results(path)["telescope"]
    assert restored.field_of_view == "70x70" and restored.resolution_budget == "current_estimate"
    assert restored.psf_params == [1.01 * u.pix, 2.54 * u.pix]


@pytest.mark.parametrize("budget", ["current_estimate", "allocation"])
def test_the_budgets_own_jitter_changes_nothing(budget):
    own = Telescope_EUVST(resolution_budget=budget, pointing_jitter=0.110 * u.arcsec)
    tel = Telescope_EUVST(resolution_budget=budget)
    assert spatial_psf_fwhm(own, DET) == pytest.approx(spatial_psf_fwhm(tel, DET), rel=1e-12)
    assert across_slit_fwhm(own).value == pytest.approx(across_slit_fwhm(tel).value, rel=1e-12)
    # Row 5 is printed as 0.073 in the allocation, 2/3 of 0.110 to the rounding.
    assert spectral_psf_fwhm(own, DET, 0.2 * u.arcsec) == pytest.approx(
        spectral_psf_fwhm(tel, DET, 0.2 * u.arcsec), rel=2e-4)


@pytest.mark.parametrize("jitter", [0.0, 0.2, 0.5])
def test_a_jitter_replaces_the_budgets_row_20(jitter):
    """Along and across the slit, and 2/3 of it in place of row 5 in wavelength."""
    tel = Telescope_EUVST(pointing_jitter=jitter * u.arcsec)
    budget = "current_estimate"
    along = np.sqrt(resolution_budget.along_slit_fwhm(budget) ** 2 - 0.110**2 + jitter**2)
    across = np.sqrt(resolution_budget.across_slit_fwhm(budget) ** 2 - 0.110**2 + jitter**2)
    spectral = np.sqrt(resolution_budget.spectral_fwhm(budget) ** 2 - (2 / 3 * 0.110) ** 2
                       + (2 / 3 * jitter) ** 2)
    assert spatial_psf_fwhm(tel, DET) == pytest.approx(np.hypot(2.66, along / PLATE_SCALE))
    assert across_slit_fwhm(tel).to_value(u.arcsec) == pytest.approx(across)
    assert spectral_blur_fwhm(tel, DET) == pytest.approx(
        np.hypot(spectral_optics_fwhm(tel, DET), spectral / PLATE_SCALE))


def test_a_jitter_is_added_without_the_budget():
    tel = Telescope_EUVST(resolution_budget="none", pointing_jitter=0.3 * u.arcsec)
    assert spatial_psf_fwhm(tel, DET) == pytest.approx(np.hypot(2.66, 0.3 / PLATE_SCALE))
    assert across_slit_fwhm(tel).to_value(u.arcsec) == pytest.approx(0.3)
    assert across_slit_fwhm(Telescope_EUVST(resolution_budget="none",
                                            pointing_jitter=0 * u.arcsec)) is None


@pytest.mark.parametrize("jitter", [0.2, -0.1 * u.arcsec, 2 * u.pix, [0.1, 0.2] * u.arcsec,
                                    np.nan * u.arcsec])
def test_a_bad_jitter_is_refused(jitter):
    with pytest.raises(ValueError, match="pointing_jitter must be a FWHM in an angle"):
        Telescope_EUVST(pointing_jitter=jitter)


def test_a_pickled_telescope_from_before_the_budget_reads_back_without_it(tmp_path):
    """An old pickle's telescope lacks the setting, and would otherwise read the class default."""
    import dill

    tel = Telescope_EUVST(psf_params=[2.66 * u.pix, 2.54 * u.pix])
    for name in ("resolution_budget", "field_of_view", "pointing_jitter"):
        del tel.__dict__[name]
    path = tmp_path / "old.pkl"
    with open(path, "wb") as handle:
        dill.dump({"results": {"all_combinations": {"a": {"config_objects": {"telescope": tel}}},
                               "fit_weighted": True}}, handle)
    with pytest.warns(FutureWarning, match="results pickle"):
        results = load_results(path)["results"]
    restored = results["all_combinations"]["a"]["config_objects"]["telescope"]
    assert restored.resolution_budget == "none"
    assert spatial_psf_fwhm(restored, DET) == 2.66 and across_slit_fwhm(restored) is None


def test_telescopes_of_one_field_do_not_share_their_widths():
    one, other = Telescope_EUVST(field_of_view="70x70"), Telescope_EUVST(field_of_view="70x70")
    one.psf_params[0] *= 2
    assert other.psf_params[0] == 1.01 * u.pix
    assert Telescope_EUVST.FIELDS_OF_VIEW["70x70"][0] == 1.01 * u.pix
