"""Tests of the slit's ends and fiducials blocking the light.

With `simulation.slit_mask` on, the pixels along the slit average only the
light that gets through, blurred along the slit before the slit.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from eclipse.config import Detector_SWC, Simulation, Telescope_EUVST
from eclipse.data_processing import rebin_atmosphere
from eclipse.sampling import centred_edges, pixel_weights, pixel_weights_through
from eclipse.utils import VELOCITY_CONVENTION, _parse_section, angle_to_distance

REST = 195.119 * u.AA
RADIANCE = u.erg / (u.s * u.cm**2 * u.sr * u.cm)


def _open_share(pixels, openings):
    """Share of each pixel the openings cover."""
    low, high = pixels[:-1, np.newaxis], pixels[1:, np.newaxis]
    covered = np.clip(np.minimum(high, openings[:, 1]) - np.maximum(low, openings[:, 0]), 0, None)
    return covered.sum(axis=1) / (pixels[1:] - pixels[:-1])


def test_one_opening_past_every_pixel_gives_the_plain_weights():
    cells = centred_edges(50, 0.3)
    pixels = centred_edges(80, 0.159)
    for sigma in (0.0, 0.2):
        through = pixel_weights_through(cells, pixels, [[-100.0, 100.0]], sigma)
        assert through == pytest.approx(pixel_weights(cells, pixels, sigma), rel=1e-12, abs=1e-15)


def test_a_uniform_scene_shows_the_open_share_of_each_pixel():
    """Blurred, a uniform scene stays uniform, so each pixel holds the share of it that is open."""
    cells = centred_edges(60, 0.25)
    pixels = centred_edges(90, 0.159)
    openings = np.array([[-7.0, 0.33], [1.13, 3.0], [3.8, 7.0]])
    for sigma in (0.0, 0.18):
        seen = pixel_weights_through(cells, pixels, openings, sigma, extend=True) @ np.ones(60)
        assert seen == pytest.approx(_open_share(pixels, openings), rel=1e-10, abs=1e-12)
    assert np.any(seen == 0.0) and np.any((seen > 0.0) & (seen < 1.0))


def test_the_weights_match_a_fine_grid():
    """Compare with blurring cells on a fine grid, cutting out the gap and averaging per pixel."""
    rng = np.random.default_rng(3)
    cells = centred_edges(12, 0.4)
    pixels = centred_edges(30, 0.159)
    openings = np.array([[-3.0, -0.31], [0.49, 3.0]])
    sigma = 0.15
    values = rng.uniform(0.5, 2.0, 12)

    step = 1e-4
    x = np.arange(pixels[0], pixels[-1], step) + step / 2
    from scipy.special import erf
    scale = np.sqrt(2) * sigma
    light = sum(v * 0.5 * (erf((x - lo) / scale) - erf((x - hi) / scale))
                for v, lo, hi in zip(values, cells[:-1], cells[1:]))
    inside = (x[:, np.newaxis] > openings[:, 0]) & (x[:, np.newaxis] < openings[:, 1])
    light *= np.any(inside, axis=1)
    pixel = np.searchsorted(pixels, x, side="right") - 1
    brute = np.bincount(pixel, weights=light, minlength=30) / np.bincount(pixel, minlength=30)

    exact = pixel_weights_through(cells, pixels, openings, sigma) @ values
    # The fine grid puts each edge to within one step, 1/1600 of a pixel.
    assert exact == pytest.approx(brute, rel=2e-3, abs=1e-6)


def test_the_openings_of_each_slit():
    tel = Telescope_EUVST()
    for slit, gap in ((0.2, 0.8), (0.4, 0.8), (0.8, 0.8), (1.6, 1.6)):
        openings = tel.slit_openings(slit * u.arcsec).to_value(u.arcsec)
        expected = [[-150.0, -25.0 - gap / 2], [-25.0 + gap / 2, 70.0 - gap / 2],
                    [70.0 + gap / 2, 150.0]]
        assert openings == pytest.approx(np.array(expected))
    with pytest.raises(ValueError, match="0.2, 0.4, 0.8 and 1.6 arcsec"):
        tel.slit_openings(0.3 * u.arcsec)


@pytest.mark.parametrize("settings, message", [
    (dict(fiducial_centres=[149.5 * u.arcsec]), "inside the"),
    (dict(fiducial_centres=[0 * u.arcsec, 1.0 * u.arcsec]), "apart"),
    (dict(fiducial_centres=[-25 * u.mm]), "angles along the slit"),
    (dict(fiducial_lengths=[0.8 * u.arcsec] * 3), "four lengths"),
    (dict(fiducial_lengths=[0.8 * u.arcsec] * 3 + [0 * u.arcsec]), "four lengths"),
    (dict(slit_length=-300 * u.arcsec), "more than zero"),
])
def test_fiducials_that_cannot_be_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        Telescope_EUVST(**settings)


def test_the_mask_is_off_by_default_and_for_swc_only():
    assert Simulation().slit_mask is False
    with pytest.raises(ValueError, match="slit_mask is for SWC"):
        Simulation(instrument="EIS", slit_width=1 * u.arcsec, slit_mask=True)


def test_the_fiducial_lists_are_one_setting_not_a_sweep():
    fixed, sweep = _parse_section({"fiducial_centres": ["-25 arcsec", "70 arcsec"],
                                   "fiducial_lengths": ["0.8 arcsec", "0.8 arcsec",
                                                        "0.8 arcsec", "1.6 arcsec"]}, "telescope")
    assert not sweep
    assert fixed["fiducial_centres"] == [-25 * u.arcsec, 70 * u.arcsec]
    Telescope_EUVST(**fixed)


def _scene(data, cell):
    ny, nx, nl = data.shape
    size = angle_to_distance(cell).to_value(u.Mm)
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "SOLX", "SOLY"]
    wcs.wcs.cunit = ["Angstrom", "Mm", "Mm"]
    wcs.wcs.crpix = [(nl + 1) / 2, (nx + 1) / 2, (ny + 1) / 2]
    wcs.wcs.crval = [REST.to_value(u.AA), 0.0, 0.0]
    wcs.wcs.cdelt = [0.0169 / 10, size, size]
    return NDCube(data, wcs=wcs, unit=RADIANCE, meta={"rest_wav": REST})


@pytest.mark.parametrize("psf", [False, True])
def test_the_rebinned_scene_is_dark_past_the_ends_and_in_the_fiducials(psf):
    """A uniform scene 24 arcsec long through a 20 arcsec slit with gaps at -3 and +5 arcsec."""
    det = Detector_SWC()
    tel = Telescope_EUVST(slit_length=20 * u.arcsec,
                          fiducial_centres=[-3 * u.arcsec, 5 * u.arcsec])
    scene = _scene(np.ones((240, 4, 21)), 0.1 * u.arcsec)
    common = dict(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, psf=psf)
    plain = rebin_atmosphere(scene, det, Simulation(**common), tel=tel)
    masked = rebin_atmosphere(scene, det, Simulation(**common, slit_mask=True), tel=tel)

    rows = plain.data.shape[0]
    pixels = centred_edges(rows, det.plate_scale_angle.to_value(u.arcsec / u.pix))
    share = _open_share(pixels, tel.slit_openings(0.4 * u.arcsec).to_value(u.arcsec))
    # With the blur along the slit replicating the edges, the plain scene is
    # uniform along it, so the mask scales each row by its open share.
    ratio = masked.data.sum(axis=(1, 2)) / plain.data.sum(axis=(1, 2))
    assert ratio == pytest.approx(share, rel=1e-9, abs=1e-12)
    assert np.count_nonzero(share == 0.0) > 0


def test_a_rebinned_scene_says_whether_it_was_masked():
    det, tel = Detector_SWC(), Telescope_EUVST()
    scene = _scene(np.ones((40, 4, 21)), 0.1 * u.arcsec)
    common = dict(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1)
    assert rebin_atmosphere(scene, det, Simulation(**common, slit_mask=True),
                            tel=tel).meta["slit_mask_applied"] is True
    assert "slit_mask_applied" not in rebin_atmosphere(scene, det, Simulation(**common),
                                                       tel=tel).meta


def test_the_mask_is_refused_for_a_cube_it_was_not_applied_to():
    """Such as a uniform intensity from Python, which would otherwise come out unmasked."""
    from eclipse.data_processing import create_uniform_intensity_cube
    from eclipse.monte_carlo import monte_carlo, simulate_once

    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, slit_mask=True)
    cube = create_uniform_intensity_cube(total_intensity=1000 * u.erg / (u.s * u.cm**2 * u.sr),
                                         rest_wavelength=REST, thermal_width=20 * u.km / u.s,
                                         det=det, sim=sim, n_slit_pixels=1, tel=tel)
    with pytest.raises(ValueError, match="slit's ends and fiducials were not cut out"):
        simulate_once(cube, 1 * u.s, det, tel, sim, uniform_mode=True)
    with pytest.raises(ValueError, match="slit's ends and fiducials were not cut out"):
        monte_carlo(cube, 1 * u.s, det, tel, sim, n_iter=1, uniform_mode=True)


@pytest.mark.parametrize("psf", [False, True])
def test_an_instrument_run_with_the_mask_on(tmp_path, monkeypatch, psf):
    """End to end from a synthesis file, binned off-chip or not, so the mask reaches the fits."""
    import sys

    import astropy.constants as const
    import yaml

    from eclipse.analysis import load_instrument_response_results
    from eclipse.main import main
    from eclipse.synthesis_file import write_line_cubes

    n_spec, n_cells = 121, 8
    step = (5 * u.km / u.s / const.c * REST).to_value(u.cm)
    sigma = (20 * u.km / u.s / const.c * REST).to_value(u.cm)
    offsets = (np.arange(n_spec) - n_spec // 2) * step
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "SOLX", "SOLY"]
    wcs.wcs.cunit = ["cm", "Mm", "Mm"]
    wcs.wcs.cdelt = [step, 0.192, 0.192]
    wcs.wcs.crpix = [(n_spec + 1) / 2, (n_cells + 1) / 2, (n_cells + 1) / 2]
    wcs.wcs.crval = [REST.to_value(u.cm), 0.0, 0.0]
    line = 1e13 * np.exp(-0.5 * (offsets / sigma) ** 2)
    cube = NDCube(np.tile(line, (n_cells, n_cells, 1)), wcs=wcs,
                  unit=u.erg / (u.s * u.cm**3 * u.sr),
                  meta={"rest_wav": REST, "integration_axis": "z",
                        "velocity_convention": VELOCITY_CONVENTION})
    synthesis_file = write_line_cubes({"Fe12_195.1190": cube}, tmp_path / "synthesis.h5")
    config = tmp_path / "masked.yaml"
    config.write_text(yaml.safe_dump({
        "instrument": "SWC", "synthesis_file": str(synthesis_file),
        "reference_line": "Fe12_195.1190", "n_iter": 1, "ncpu": 1,
        "offchip_bin_slit": [1, 2],
        "simulation": {"slit_width": "0.4 arcsec", "expos": "10 s", "psf": psf,
                       "noise": False, "slit_mask": True},
    }))
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    monkeypatch.chdir(tmp_path)
    main()
    results = load_instrument_response_results(tmp_path / "run" / "result" / "masked.h5")
    assert len(results["results"]["all_combinations"]) == 2


def test_a_cube_masked_without_the_psf_is_not_blurred_afterwards():
    """The blur comes before the slit, so masking first and blurring later is refused."""
    from eclipse.monte_carlo import monte_carlo, simulate_once

    det, tel = Detector_SWC(), Telescope_EUVST()
    scene = _scene(np.ones((40, 4, 21)), 0.1 * u.arcsec)
    common = dict(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, slit_mask=True)
    cube = rebin_atmosphere(scene, det, Simulation(**common, psf=False), tel=tel)
    blurred = Simulation(**common, psf=True)
    with pytest.raises(ValueError, match="masked without the PSF"):
        simulate_once(cube, 1 * u.s, det, tel, blurred)
    with pytest.raises(ValueError, match="masked without the PSF"):
        monte_carlo(cube, 1 * u.s, det, tel, blurred, n_iter=1)
    # Masked with the PSF, it is accepted.
    together = rebin_atmosphere(scene, det, blurred, tel=tel)
    simulate_once(together, 1 * u.s, det, tel, blurred)


def test_the_mask_does_not_move_the_simulations_positional_arguments():
    """It is keyword only, so a call passing `noise` and the rest by position is unchanged."""
    import inspect

    positional = [p.name for p in inspect.signature(Simulation).parameters.values()
                  if p.kind is p.POSITIONAL_OR_KEYWORD]
    assert positional == ["expos", "n_iter", "slit_width", "ncpu", "instrument", "vis_sl", "psf",
                          "psf_boundary", "spectral_psf", "noise", "enable_pinholes",
                          "pinhole_sizes", "pinhole_positions", "pinhole_positions_spectral"]
    assert "slit_mask" not in positional
