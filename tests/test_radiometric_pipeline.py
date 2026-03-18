"""
Tests for the radiometric pipeline and noise calculations.

These tests verify that:
1. Poisson shot noise is applied to photon counts per pixel, not intensity.
2. Dark current noise is independent per pixel.
3. The pipeline produces correct unit conversions at every stage.
4. Noise statistics match expected physics.
"""

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.wcs import WCS
from ndcube import NDCube
import pytest

from euvst_response.config import Detector_SWC, Telescope_EUVST, Simulation
from euvst_response.radiometric import (
    apply_exposure, intensity_to_photons, add_telescope_throughput,
    photons_to_pixel_counts, add_poisson, to_electrons, to_dn,
)
from euvst_response.monte_carlo import simulate_once
from euvst_response.data_processing import create_uniform_intensity_cube
from euvst_response.utils import angle_to_distance


def _make_test_cube(
    n_scan=1, n_slit=1, n_lam=51, lam0_AA=195.119,
    sigma_km_s=20.0, total_intensity_cgs=5000.0,
):
    """Create a small uniform-intensity Gaussian cube for testing."""
    det = Detector_SWC()
    sim = Simulation(expos=1.0 * u.s, slit_width=0.2 * u.arcsec, instrument="SWC")
    cube = create_uniform_intensity_cube(
        total_intensity=total_intensity_cgs * u.erg / (u.s * u.cm**2 * u.sr),
        rest_wavelength=lam0_AA * u.AA,
        thermal_width=sigma_km_s * u.km / u.s,
        det=det,
        sim=sim,
    )
    return cube, det, sim


# -------------------------------------------------------------------
# 1. apply_exposure must NOT inject Poisson noise
# -------------------------------------------------------------------
class TestApplyExposure:
    def test_deterministic(self):
        """apply_exposure should return exactly I * t_exp (no stochastic component)."""
        cube, det, sim = _make_test_cube()
        t_exp = 10.0 * u.s
        out1 = apply_exposure(cube, t_exp)
        out2 = apply_exposure(cube, t_exp)
        np.testing.assert_array_equal(out1.data, out2.data)

    def test_unit_scaling(self):
        """Output values should be input × t_exp."""
        cube, det, sim = _make_test_cube()
        t_exp = 5.0 * u.s
        out = apply_exposure(cube, t_exp)
        # Output unit includes the extra factor of seconds
        expected_unit = cube.unit * u.s
        assert out.unit.is_equivalent(expected_unit)
        np.testing.assert_allclose(
            out.data, cube.data * t_exp.value, rtol=1e-12,
        )


# -------------------------------------------------------------------
# 2. Poisson noise is applied to photon counts per pixel
# -------------------------------------------------------------------
class TestPoissonPlacement:
    """Verify that Poisson noise is applied to photon counts per pixel."""

    def test_add_poisson_returns_integers(self):
        """add_poisson should return integer-valued data (Poisson samples)."""
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ["WAVE", "HPLT-TAN", "HPLN-TAN"]
        wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
        wcs.wcs.crpix = [26, 1, 1]
        wcs.wcs.crval = [1.95e-5, 0, 0]
        wcs.wcs.cdelt = [1.69e-10, 0.159, 0.2]
        data = np.full((1, 1, 51), 100.0)  # 100 expected photons per pixel
        cube = NDCube(data=data, wcs=wcs, unit=u.photon / u.pix)
        out = add_poisson(cube)
        # All values should be non-negative integers
        assert np.all(out.data >= 0)
        assert np.all(out.data == np.floor(out.data))

    def test_add_poisson_clips_negatives(self):
        """add_poisson should handle small negative values without error."""
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ["WAVE", "HPLT-TAN", "HPLN-TAN"]
        wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
        wcs.wcs.crpix = [1, 1, 1]
        wcs.wcs.crval = [1.95e-5, 0, 0]
        wcs.wcs.cdelt = [1.69e-10, 0.159, 0.2]
        data = np.array([[[-1e-10, 0.0, 5.0]]])
        cube = NDCube(data=data, wcs=wcs, unit=u.photon / u.pix)
        out = add_poisson(cube)  # should not raise
        assert out.data[0, 0, 0] == 0  # negative clipped to 0 → Poisson(0) = 0
        assert out.data[0, 0, 1] == 0

    def test_poisson_statistics(self):
        """Mean and variance of Poisson samples should match the input rate."""
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ["WAVE", "HPLT-TAN", "HPLN-TAN"]
        wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
        wcs.wcs.crpix = [1, 1, 1]
        wcs.wcs.crval = [1.95e-5, 0, 0]
        wcs.wcs.cdelt = [1.69e-10, 0.159, 0.2]
        lam_rate = 500.0  # expected photon count
        n_trials = 10_000
        samples = np.empty(n_trials)
        for i in range(n_trials):
            data = np.array([[[lam_rate]]])
            cube = NDCube(data=data, wcs=wcs, unit=u.photon / u.pix)
            out = add_poisson(cube)
            samples[i] = out.data[0, 0, 0]
        # For Poisson: mean ≈ λ, var ≈ λ
        np.testing.assert_allclose(samples.mean(), lam_rate, rtol=0.05)
        np.testing.assert_allclose(samples.var(), lam_rate, rtol=0.1)


# -------------------------------------------------------------------
# 3. Dark current noise is per-pixel (not a single scalar)
# -------------------------------------------------------------------
class TestDarkCurrentPerPixel:
    def test_dark_current_varies_across_pixels(self):
        """Different pixels should receive different dark-current noise."""
        np.random.seed(42)
        det = Detector_SWC(ccd_temperature=0 * u.deg_C)  # warmer → higher dark current
        t_exp = 100.0 * u.s  # long exposure

        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ["WAVE", "HPLT-TAN", "HPLN-TAN"]
        wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
        wcs.wcs.crpix = [6, 5, 5]
        wcs.wcs.crval = [1.95e-5, 0, 0]
        wcs.wcs.cdelt = [1.69e-10, 0.159, 0.2]
        n_scan, n_slit, n_lam = 10, 10, 11
        # Feed zeros so the only signal comes from dark current + read noise
        data = np.zeros((n_scan, n_slit, n_lam))
        cube = NDCube(
            data=data, wcs=wcs, unit=u.photon / u.pix,
            meta={"rest_wav": 195.119 * u.AA},
        )
        out = to_electrons(cube, t_exp, det)
        # With per-pixel noise, not all pixel values should be identical
        unique_vals = np.unique(out.data)
        assert len(unique_vals) > 1, (
            "Dark current + read noise should produce different values per pixel"
        )


# -------------------------------------------------------------------
# 4. Pipeline unit consistency
# -------------------------------------------------------------------
class TestPipelineUnits:
    """Check unit conversions at each pipeline step."""

    def test_intensity_to_photons_units(self):
        cube, det, sim = _make_test_cube()
        t_exp = 10.0 * u.s
        exposed = apply_exposure(cube, t_exp)
        photons = intensity_to_photons(exposed)
        # Should be photon / (cm2 sr cm)
        assert photons.unit.is_equivalent(u.photon / u.cm**2 / u.sr / u.cm)

    def test_throughput_units(self):
        cube, det, sim = _make_test_cube()
        tel = Telescope_EUVST()
        t_exp = 10.0 * u.s
        exposed = apply_exposure(cube, t_exp)
        photons = intensity_to_photons(exposed)
        through = add_telescope_throughput(photons, tel)
        # After multiplying by collecting_area (cm2), units → photon / (sr cm)
        assert through.unit.is_equivalent(u.photon / u.sr / u.cm)

    def test_pixel_counts_units(self):
        cube, det, sim = _make_test_cube()
        tel = Telescope_EUVST()
        t_exp = 10.0 * u.s
        exposed = apply_exposure(cube, t_exp)
        photons = intensity_to_photons(exposed)
        through = add_telescope_throughput(photons, tel)
        pixel_counts = photons_to_pixel_counts(
            through, det.wvl_res, det.plate_scale_length,
            angle_to_distance(sim.slit_width),
        )
        # Should be photon / pixel
        assert pixel_counts.unit.is_equivalent(u.photon / u.pix)


# -------------------------------------------------------------------
# 5. Full pipeline integration test
# -------------------------------------------------------------------
class TestFullPipeline:
    def test_simulate_once_runs(self):
        """Smoke test: simulate_once should run without error."""
        cube, det, sim = _make_test_cube()
        tel = Telescope_EUVST()
        t_exp = 10.0 * u.s
        result = simulate_once(cube, t_exp, det, tel, sim)
        # Should return 11 elements
        assert len(result) == 11
        # DN output should be non-negative
        dn = result[-1]
        assert np.all(dn.data >= 0)

    def test_shot_noise_variance_scales_with_signal(self):
        """
        Verify that the photon shot noise (Poisson) variance ≈ mean
        when measured in photon counts per pixel.
        """
        np.random.seed(0)
        det = Detector_SWC()
        tel = Telescope_EUVST()
        sim = Simulation(expos=1.0 * u.s, slit_width=0.2 * u.arcsec, instrument="SWC")
        # Use a strong line so the photon count is large enough for good statistics
        cube = create_uniform_intensity_cube(
            total_intensity=50000 * u.erg / (u.s * u.cm**2 * u.sr),
            rest_wavelength=195.119 * u.AA,
            thermal_width=20 * u.km / u.s,
            det=det,
            sim=sim,
        )
        t_exp = 30.0 * u.s
        n_mc = 200
        # Only look at the peak pixel (index with highest expected counts)
        # Run pipeline up to photons_noisy (index 6 in the result tuple)
        peak_samples = []
        for _ in range(n_mc):
            result = simulate_once(cube, t_exp, det, tel, sim)
            photons_noisy = result[6]
            peak_samples.append(photons_noisy.data.max())

        samples = np.array(peak_samples)
        mean_val = samples.mean()
        var_val = samples.var()
        # For Poisson: var / mean ≈ 1 (within statistical tolerance)
        ratio = var_val / mean_val
        assert 0.7 < ratio < 1.3, (
            f"Poisson shot noise ratio var/mean = {ratio:.3f} "
            f"(expected ~1.0 for shot-noise-limited signal)"
        )
