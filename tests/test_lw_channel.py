"""
The EUVST LW channel: its throughput, its bands, and its intensified detector.

The effective areas are the current-best-estimate column of the sensitivity
budget RSC-2022035, which the chain of coating, microroughness, grating,
contamination and detector efficiency reproduces to a constant 0.05 per
cent, the budget's rounding of the geometric area (615.44 cm2 against
615.75 cm2 for a 28 cm aperture). The detector defaults are the values the
LW-IAPS specification SSD-SPC-ET004 requires.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from eclipse.config import (LW_BAND_COVERAGE, LW_BAND_TABLE_RANGE, Detector_LW,
                            Simulation, Telescope_EUVST_LW, check_lw_pair,
                            check_lw_wavelength)
from eclipse.main import _check_lw_config, _validate_config_keys
from eclipse.monte_carlo import simulate_once
from eclipse.radiometric import (add_visible_stray_light, spectral_optics_fwhm,
                                 spectral_psf_fwhm, to_dn, to_electrons)

# The budget's geometric area against ECLIPSE's, pi * 14 cm squared.
GEO_AREA_RATIO = (np.pi * 14.0 ** 2) / 615.44


def ea_including_qe(wavelength, band, order):
    tel = Telescope_EUVST_LW(band=band, order=order)
    det = Detector_LW(band=band, order=order)
    return (tel.ea_and_throughput(wavelength) * det.qe_euv_at(wavelength)).to_value(u.cm ** 2)


def _wcs(start, step, n):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [step, 0.2, 0.155]
    wcs.wcs.crpix = [1.0, 1.0, 1.0]
    wcs.wcs.crval = [start, 0.0, 0.0]
    return wcs


def _cube(value, unit, start=1204.0, step=0.5, n=25, shape=(2, 3), rest=1215.67):
    return NDCube(np.full((*shape, n), value, dtype=float), wcs=_wcs(start, step, n),
                  unit=unit, meta={"rest_wav": rest * u.AA})


class TestEffectiveArea:
    @pytest.mark.parametrize("wavelength, band, order, budget", [
        (1116, "LW3", 1, 10.1163),
        (1216, "LW3", 1, 0.2426),
        (460, "LW2", 2, 1.1179),
    ])
    def test_the_budget_is_reproduced(self, wavelength, band, order, budget):
        area = ea_including_qe(wavelength * u.AA, band, order)
        assert area / budget == pytest.approx(GEO_AREA_RATIO, rel=2e-4)

    def test_the_budget_starts_just_off_the_detectors(self):
        """It tabulates 709, 919 and 1108 A, a little short of where each detector starts."""
        for wavelength, band in ((709, "LW1"), (919, "LW2"), (1108, "LW3")):
            with pytest.raises(ValueError, match="not recorded"):
                ea_including_qe(wavelength * u.AA, band, 1)
            assert ea_including_qe((wavelength + 1) * u.AA, band, 1) > 0

    def test_lyman_alpha_and_beta(self):
        lya = ea_including_qe(1215.67 * u.AA, "LW3", 1)
        lyb = ea_including_qe(1025.72 * u.AA, "LW2", 1)
        assert lya == pytest.approx(0.2427, rel=1e-3)
        assert lyb == pytest.approx(6.8172, rel=1e-3)

    def test_the_bare_mcp_step_is_where_the_specification_puts_it(self):
        """SSD-SPC-ET004: LW3 bare from about 1210 A, 1-3 per cent at Lyman-alpha."""
        det = Detector_LW()
        assert det.qe_euv_at(1208 * u.AA) == pytest.approx(0.297)
        assert 0.01 <= det.qe_euv_at(1215.67 * u.AA) <= 0.03
        # Second-order light meets the bare region from about 605 A.
        det2 = Detector_LW(order=2)
        assert det2.qe_euv_at(605 * u.AA) == pytest.approx(0.372)
        assert det2.qe_euv_at(609 * u.AA) == pytest.approx(0.093)

    def test_the_photocathodes_meet_their_requirement(self):
        """SSD-SPC-ET004 Table 2 item 1-10: at least 23, 24 and 30 per cent."""
        for band, lo, hi, required in (("LW1", 720.36, 845, 0.23),
                                       ("LW2", 928.89, 1039, 0.24)):
            wl = np.linspace(lo, hi, 50) * u.AA
            assert Detector_LW(band=band).qe_euv_at(wl).min() >= required - 0.005

    def test_debye_waller_matches_the_budget(self):
        # CBE 0.98992 at 749 A with 0.6 nm; Worst 0.97224 with 1.0 nm.
        assert Telescope_EUVST_LW().microroughness_efficiency(749 * u.AA) == pytest.approx(
            0.98992, abs=1e-4)
        assert Telescope_EUVST_LW(microroughness_sigma=1.0 * u.nm).microroughness_efficiency(
            749 * u.AA) == pytest.approx(0.97224, abs=1e-4)

    def test_second_order_uses_its_own_grating_table(self):
        tel = Telescope_EUVST_LW(band="LW3", order=2)
        assert tel.grating_efficiency(585 * u.AA) == pytest.approx(0.16222, rel=1e-3)

    def test_arrays_give_one_value_per_wavelength(self):
        wl = np.array([1150.0, 1200.0, 1216.0]) * u.AA
        tel, det = Telescope_EUVST_LW(), Detector_LW()
        together = tel.throughput(wl) * det.qe_euv_at(wl)
        one_by_one = [tel.throughput(w) * det.qe_euv_at(w) for w in wl]
        assert np.allclose(together, one_by_one, rtol=1e-12)


class TestBands:
    def test_second_order_is_half_the_first(self):
        assert LW_BAND_COVERAGE[("LW3", 2)] == pytest.approx(
            tuple(x / 2 for x in LW_BAND_COVERAGE[("LW3", 1)]))
        assert ("LW1", 2) not in LW_BAND_COVERAGE

    def test_the_tables_lie_on_the_detectors_they_describe(self):
        for key, (t_lo, t_hi) in LW_BAND_TABLE_RANGE.items():
            lo, hi = LW_BAND_COVERAGE[key]
            assert t_hi > lo and t_lo < hi

    def test_mg_x_609_is_on_the_detector_but_not_yet_known(self):
        with pytest.raises(ValueError, match="not yet known"):
            check_lw_wavelength(609.79 * u.AA, "LW3", 2)
        with pytest.raises(ValueError, match="not yet known"):
            Telescope_EUVST_LW(order=2).throughput(609.79 * u.AA)
        with pytest.raises(ValueError, match="not yet known"):
            Detector_LW(order=2).qe_euv_at(609.79 * u.AA)
        # Its edge, 609 A, is tabulated and passes despite rounding in nm.
        check_lw_wavelength(609.0 * u.AA, "LW3", 2)

    def test_the_end_of_lw1_is_on_the_detector_but_not_yet_known(self):
        with pytest.raises(ValueError, match="not yet known"):
            Telescope_EUVST_LW(band="LW1").throughput(846 * u.AA)

    def test_a_scalar_qe_does_not_skip_the_check(self):
        with pytest.raises(ValueError, match="not yet known"):
            Detector_LW(order=2, qe_euv=0.3).qe_euv_at(609.79 * u.AA)

    def test_between_the_bands_is_off_every_detector(self):
        for band in ("LW2", "LW3"):
            with pytest.raises(ValueError, match="no EUVST LW detector"):
                check_lw_wavelength(535 * u.AA, band, 2)

    def test_the_wrong_band_names_the_right_one(self):
        with pytest.raises(ValueError, match="LW3 in order 1"):
            Telescope_EUVST_LW(band="LW2").throughput(1215.67 * u.AA)

    def test_a_window_with_one_wavelength_off_the_band_is_refused(self):
        with pytest.raises(ValueError, match="1230.000"):
            check_lw_wavelength(np.array([1220.0, 1230.0]) * u.AA, "LW3", 1)

    def test_unknown_bands(self):
        with pytest.raises(ValueError):
            Detector_LW(band="LW4")
        with pytest.raises(ValueError):
            Telescope_EUVST_LW(band="LW1", order=2)

    def test_the_band_is_written_one_way(self):
        det = Detector_LW(band="lw2", order=1)
        assert (det.band, det.order) == ("LW2", 1)


class TestDetectorDefaults:
    def test_the_specification_values(self):
        """SSD-SPC-ET004 Table 2 items 2-04, 2-06, 2-07 and 2-18."""
        det = Detector_LW()
        assert det.read_noise_rms == 40 * u.electron / u.pix
        assert det.dark_current == 2 * u.electron / (u.pix * u.s)
        assert det.aps_temperature == -40 * u.deg_C
        assert det.full_well == 86000 * u.electron / u.pix
        assert det.read_noise_pedestal == 0 * u.electron / u.pix

    def test_the_full_well_fills_before_the_digitiser(self):
        det = Detector_LW()
        assert det.full_well < det.max_dn * det.gain_e_per_dn

    def test_the_dark_current_is_known_only_at_minus_40(self):
        with pytest.raises(ValueError, match="dark_current"):
            Detector_LW(aps_temperature=-30 * u.deg_C)
        given = 5 * u.electron / (u.pix * u.s)
        assert Detector_LW(aps_temperature=-30 * u.deg_C, dark_current=given).dark_current == given
        # The same temperature in kelvin is the specified one.
        assert Detector_LW(aps_temperature=233.15 * u.K).dark_current == 2 * u.electron / (u.pix * u.s)

    def test_sampling(self):
        """RSC-2022021C p.6."""
        det = Detector_LW(band="LW1")
        assert det.wvl_res.to_value(u.mAA / u.pix) == pytest.approx(36.9)
        assert det.plate_scale_angle.to_value(u.arcsec / u.pix) == pytest.approx(0.162)
        det = Detector_LW(band="LW3", order=2)
        assert det.wvl_res.to_value(u.mAA / u.pix) == pytest.approx(15.1)


class TestIntensifier:
    def test_electrons_per_photon(self):
        det = Detector_LW()
        assert det.post_mcp_efficiency == pytest.approx(7.25 * 0.66 * 0.81 * 0.01 * 0.34)
        assert det.electrons_per_photon == pytest.approx(13.1779, rel=1e-4)

    def test_noise_off_gives_the_mean(self):
        det = Detector_LW()
        detected = np.array([0.0, 1.0, 2.5, 100.0])
        assert np.allclose(det.photons_to_electrons(detected, noise=False),
                           detected * det.electrons_per_photon)

    def test_no_photons_give_no_electrons(self):
        assert np.all(Detector_LW().photons_to_electrons(np.zeros(1000)) == 0)

    def test_the_mcp_doubles_the_variance(self):
        """An exponential gain has an excess noise factor of exactly 2.

        For a fixed number of photons the gain alone adds as much variance as
        Poisson arrivals do, so with Poisson arrivals the variance doubles.
        """
        np.random.seed(11)
        det = Detector_LW()
        n = 5.0
        gain = det.electrons_per_photon
        fixed = det.photons_to_electrons(np.full(400000, n))
        assert fixed.mean() == pytest.approx(n * gain, rel=5e-3)
        assert fixed.var() / (n * gain ** 2) == pytest.approx(1.0, rel=2e-2)
        arriving = det.photons_to_electrons(np.random.poisson(n, 400000))
        assert arriving.var() / (n * gain ** 2) == pytest.approx(2.0, rel=2e-2)


class TestDetectorChain:
    def test_each_pixel_gets_its_own_efficiency(self):
        """Across the bare-MCP step, 1208 to 1212 A, within one window."""
        det = Detector_LW()
        photons = _cube(1000.0, u.photon / u.pix)
        out = to_electrons(photons, 2 * u.s, det, noise=False)
        wl = photons.axis_world_coords(2)[0]
        expected = (1000.0 * det.qe_euv_at(wl) * det.electrons_per_photon
                    + (det.dark_current * 2 * u.s).to_value(u.electron / u.pix))
        assert np.allclose(out.data, np.broadcast_to(expected, out.data.shape))
        assert out.data[0, 0, 0] > 25 * out.data[0, 0, -1]

    def test_the_pedestal_is_added(self):
        det = Detector_LW(read_noise_pedestal=100 * u.electron / u.pix, qe_euv=0.0)
        out = to_electrons(_cube(0.0, u.photon / u.pix), 1 * u.s, det, noise=False)
        assert np.allclose(out.data, 100 + 2)

    def test_digitising(self):
        det = Detector_LW()
        electrons = _cube(600.0, u.electron / u.pix, shape=(200, 200))
        assert np.all(to_dn(electrons, det, noise=False).data == 100)
        np.random.seed(5)
        noisy = to_dn(electrons, det).data
        assert noisy.std() == pytest.approx(np.hypot(1.891, 1 / np.sqrt(12)), rel=0.02)

    def test_no_visible_stray_light(self):
        det = Detector_LW()
        electrons = _cube(5.0, u.electron / u.pix)
        sim = Simulation(instrument="LW")
        assert add_visible_stray_light(electrons, 1 * u.s, det, sim) is electrons

    @pytest.mark.parametrize("spectral_psf", ["quadrature", "convolution"])
    def test_a_whole_run(self, spectral_psf):
        """Intensity to DN, noise off, reproduces the chain written out by hand."""
        det, tel = Detector_LW(), Telescope_EUVST_LW()
        sim = Simulation(instrument="LW", expos=1 * u.s, slit_width=0.4 * u.arcsec,
                         psf=True, spectral_psf=spectral_psf, noise=False)
        intensity = _cube(1.0e12, u.erg / (u.cm ** 2 * u.s * u.sr * u.cm),
                          start=1200.0, step=0.0302, n=40, shape=(9, 2))
        steps = simulate_once(intensity, 1 * u.s, det, tel, sim)
        arrivals, electrons, dn = steps[5], steps[6], steps[-1]
        wl = arrivals.axis_world_coords(2)[0]
        expected = (arrivals.data * det.qe_euv_at(wl) * det.electrons_per_photon
                    + det.dark_current.to_value(u.electron / (u.pix * u.s)))
        assert np.allclose(electrons.data, expected)
        assert np.array_equal(dn.data, np.round(expected / 6.0))
        assert 100 < dn.data.max() < 65535


class TestPSF:
    def test_psf_params_come_from_the_optical_design(self):
        """RSC-2022021C p.7, divided by the sampling of p.6."""
        tel = Telescope_EUVST_LW()
        assert tel.psf_params[0].to_value(u.pix) == pytest.approx(0.381 / 0.155)
        assert tel.psf_params[1].to_value(u.pix) == pytest.approx(74.84 / 30.2)

    @pytest.mark.parametrize("band, order, optics_arcsec", [
        ("LW1", 1, 0.172), ("LW2", 1, 0.230), ("LW3", 1, 0.328),
        ("LW2", 2, 0.132), ("LW3", 2, 0.253),
    ])
    def test_removing_the_slit_leaves_the_optics(self, band, order, optics_arcsec):
        """What is left is the document's own optics-only FWHM (Appendix A.2)."""
        tel, det = Telescope_EUVST_LW(band=band, order=order), Detector_LW(band=band, order=order)
        plate = det.plate_scale_angle.to_value(u.arcsec / u.pix)
        assert spectral_optics_fwhm(tel, det) * plate == pytest.approx(optics_arcsec, rel=0.01)
        assert spectral_psf_fwhm(tel, det, 0.2 * u.arcsec) == pytest.approx(
            tel.psf_params[1].to_value(u.pix))


class TestConfiguration:
    def test_lw_slits_and_what_lw_cannot_have(self):
        Simulation(instrument="LW", slit_width=1.6 * u.arcsec)
        with pytest.raises(ValueError):
            Simulation(instrument="LW", slit_width=1.0 * u.arcsec)
        with pytest.raises(ValueError, match="visible"):
            Simulation(instrument="LW", vis_sl=1 * u.photon / (u.s * u.cm ** 2))
        with pytest.raises(ValueError, match="Pinholes"):
            Simulation(instrument="LW", enable_pinholes=True,
                       pinhole_sizes=[1 * u.um], pinhole_positions=[0.5])

    def test_lw_keys(self):
        config = {"instrument": "LW", "detector": {"band": "LW1", "read_noise_rms": "30 electron/pix"},
                  "telescope": {"order": 1, "microroughness_sigma": "1 nm"}}
        _validate_config_keys(config, "LW")
        with pytest.raises(ValueError):
            _validate_config_keys({"detector": {"filter_distance": "250 mm"}}, "LW")
        with pytest.raises(ValueError):
            _validate_config_keys({"detector": {"band": "LW1"}}, "SWC")

    def test_the_band_is_shared_between_sections(self):
        det, tel = {"band": "LW1"}, {"order": 1}
        _check_lw_config(det, {}, tel, {}, {}, {}, False)
        assert det == tel == {"band": "LW1", "order": 1}
        with pytest.raises(ValueError, match="same observation"):
            _check_lw_config({"band": "LW1"}, {}, {"band": "LW2"}, {}, {}, {}, False)
        with pytest.raises(ValueError, match="cannot be swept"):
            _check_lw_config({}, {"band": ["LW1", "LW2"]}, {}, {}, {}, {}, False)

    def test_pinholes_and_the_sw_slit_settings_are_refused_before_a_run(self):
        with pytest.raises(ValueError, match="Pinhole"):
            _check_lw_config({}, {}, {}, {}, {}, {}, [1 * u.um])
        with pytest.raises(ValueError, match="Pinhole"):
            _check_lw_config({}, {}, {}, {}, {"pinhole_visible_wavelength": 500 * u.nm}, {},
                             False)
        for key in ("slit_mask", "focal_plane_dispersion"):
            with pytest.raises(ValueError, match=f"simulation.{key}"):
                _check_lw_config({}, {}, {}, {}, {key: True}, {}, False)


@pytest.mark.parametrize("cls, settings, message", [
    (Telescope_EUVST_LW, {"D_ap": 0.28}, "telescope.D_ap needs a unit"),
    (Telescope_EUVST_LW, {"microroughness_sigma": 0.6}, "microroughness_sigma needs a unit"),
    (Telescope_EUVST_LW, {"contamination_factor": 2.0}, "contamination_factor is a fraction"),
    (Detector_LW, {"attenuation": 5.0}, "attenuation is a fraction"),
    (Detector_LW, {"gain_mcp": -1.0}, "gain_mcp must be more than zero"),
    (Detector_LW, {"qe_euv": 1.5}, "qe_euv is a fraction"),
    (Detector_LW, {"wvl_res": 30 * u.arcsec / u.pix}, "wvl_res must be a finite quantity"),
    (Detector_LW, {"dark_current": 2.0}, "dark_current must be a finite quantity"),
    (Detector_LW, {"aps_temperature": -40}, "aps_temperature needs a unit"),
    (Detector_LW, {"aps_temperature": -40 * u.K, "dark_current": 1 * u.electron / (u.pix * u.s)},
     "above absolute zero"),
])
def test_settings_of_the_wrong_kind_or_range_are_refused(cls, settings, message):
    with pytest.raises(ValueError, match=message):
        cls(**settings)


def test_a_numpy_number_is_a_fraction_too():
    assert Detector_LW(qe_euv=np.float32(0.3)).qe_euv == np.float32(0.3)


def test_a_detector_and_telescope_on_different_bands_are_refused():
    check_lw_pair(Detector_LW(band="LW2"), Telescope_EUVST_LW(band="LW2"))
    with pytest.raises(ValueError, match="LW3 in order 1 but the telescope to LW3 in order 2"):
        check_lw_pair(Detector_LW(), Telescope_EUVST_LW(order=2))
    with pytest.raises(ValueError, match="Detector_LW with Telescope_EUVST_LW"):
        check_lw_pair(Detector_LW(), object())


def test_the_mcp_gain_drawn_by_inverse_transform_uses_one_number_per_pixel():
    """Runs that differ only in photon flux then keep their later draws in step."""
    det = Detector_LW()
    after = []
    for photons in (np.full(5000, 3.0), np.full(5000, 300.0)):
        np.random.seed(2)
        electrons = det.photons_to_electrons(photons, inverse_transform=True)
        after.append(np.random.random())
        assert electrons.mean() == pytest.approx(photons[0] * det.electrons_per_photon, rel=0.03)
        assert electrons.var() == pytest.approx(
            photons[0] * det.electrons_per_photon ** 2, rel=0.1)
    assert after[0] == after[1]
    assert np.all(det.photons_to_electrons(np.zeros(3), inverse_transform=True) == 0)
