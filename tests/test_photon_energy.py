"""Each photon is counted, and frees electrons, at its own wavelength.

A pixel's electrons are its photons times the electrons per photon at the
wavelength of their mean energy, spread as set by the wavelength of their
root mean square energy.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.config import Detector_SWC, Simulation, Telescope_EUVST
from euvst_response.data_processing import rebin_atmosphere
from euvst_response.monte_carlo import simulate_once
from euvst_response.pinhole_diffraction import apply_euv_pinhole_diffraction
from euvst_response.radiometric import (_vectorized_fano_noise, electrons_per_photon,
                                         photons_per_energy, to_electrons)
from euvst_response.sampling import light_onto_pixels, photons_onto_pixels
from euvst_response.utils import angle_to_distance, binned_photon_wavelengths, rebin_slit_offchip

REST = 195.119 * u.AA
RADIANCE = u.erg / (u.s * u.cm**2 * u.sr * u.cm)


def _scene(data, cell, wavelength_step, rest=REST):
    """Cube laid out as the synthesis writes it, with square cells `cell` across."""
    ny, nx, nl = data.shape
    size = angle_to_distance(cell).to_value(u.Mm)
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "SOLX", "SOLY"]
    wcs.wcs.cunit = ["Angstrom", "Mm", "Mm"]
    wcs.wcs.crpix = [(nl + 1) / 2, (nx + 1) / 2, (ny + 1) / 2]
    wcs.wcs.crval = [rest.to_value(u.AA), 0.0, 0.0]
    wcs.wcs.cdelt = [wavelength_step.to_value(u.AA), size, size]
    return NDCube(data, wcs=wcs, unit=RADIANCE, meta={"rest_wav": rest})


def _broad_line(n_fine=401, step=0.0169 * u.AA / 10, offset=0.05 * u.AA, sigma=0.04 * u.AA,
                rest=REST):
    """Line off the rest wavelength in every cell, broad enough that photon energies differ."""
    wavelength = rest + (np.arange(n_fine) - (n_fine - 1) / 2) * step
    profile = np.exp(-0.5 * ((wavelength - rest - offset) / sigma).decompose().value ** 2)
    return _scene(np.broadcast_to(profile * 1e13, (8, 24, n_fine)).copy(), 0.1 * u.arcsec,
                  step, rest), wavelength


def test_a_pixel_holds_its_cells_photons_and_the_wavelengths_of_their_mean_and_rms_energies():
    # Two cells wholly in the first pixel, one with twice the photons.
    share = light_onto_pixels([1.0, 3.0], [1.5, 3.5], [0.0, 4.0, 8.0])
    on, mean, rms = photons_onto_pixels(share, np.array([200.0, 100.0]),
                                        np.array([190.0, 200.0]))
    assert on == pytest.approx([300.0, 0.0])
    assert mean[0] == pytest.approx(300.0 / (200.0 / 190.0 + 100.0 / 200.0), rel=1e-14)
    assert rms[0] == pytest.approx(np.sqrt(300.0 / (200.0 / 190.0**2 + 100.0 / 200.0**2)),
                                   rel=1e-14)
    assert np.isnan(mean[1]) and np.isnan(rms[1])


def test_a_pixel_whose_photons_vanish_keeps_its_own_wavelength():
    """
    In a line's far wings the photons can round to zero, or to below the
    smallest normal float, where the photons over their wavelengths, in cm,
    do not. Such pixels keep their own wavelength, not 0 or a ratio of
    rounding errors.
    """
    from euvst_response.sampling import _carry_photon_wavelengths

    wavelength = 1.95e-6  # cm
    # Photons just above the smallest normal float keep their wavelength.
    photons = np.array([[1.0, 1e-322, 1e-310, 1.1 * np.finfo(float).tiny]])

    def carry(cells):
        return cells * np.array([1.0, 0.01, 1.0, 1.0])

    own = np.array([1.9e-6, 2e-6, 2.1e-6, 2.2e-6])
    moved, mean, rms = _carry_photon_wavelengths(carry, photons, wavelength, wavelength, own)
    assert moved[0, 1] == 0.0 and 0.0 < moved[0, 2] < np.finfo(float).tiny
    assert mean[0, [0, 3]] == pytest.approx(wavelength, rel=1e-15)
    assert rms[0, [0, 3]] == pytest.approx(wavelength, rel=1e-15)
    assert list(mean[0, 1:3]) == list(own[1:3]) and list(rms[0, 1:3]) == list(own[1:3])


def test_a_blur_can_differ_from_cell_to_cell_and_be_none():
    edges = np.linspace(0.0, 10.0, 41)
    share = light_onto_pixels([4.0, 6.0], [4.1, 6.1], edges, sigma=[0.0, 0.3],
                              width=[0.0, 0.2]).toarray()
    # The unblurred first cell stays in its own pixel.
    assert share[:, 0] == pytest.approx(np.eye(40)[16])
    # The blurred second cell spreads but keeps its light.
    assert np.count_nonzero(share[:, 1]) > 3
    assert share[:, 1].sum() == pytest.approx(1.0, rel=1e-12)


def test_the_detector_frees_each_photons_own_electrons():
    det = Detector_SWC()
    counts, wavelengths = np.array([200.0, 100.0]), np.array([190.0, 200.0])
    share = light_onto_pixels([1.0, 3.0], [1.5, 3.5], [0.0, 4.0])
    on, mean, _ = photons_onto_pixels(share, counts, wavelengths)
    cube = NDCube(on.reshape(1, 1, 1), wcs=WCS(naxis=3), unit=u.photon / u.pix,
                  meta={"rest_wav": REST, "photon_wavelength": mean.reshape(1, 1, 1) * u.AA})
    electrons = to_electrons(cube, 0 * u.s, det, noise=False)
    expected = det.qe_euv * (counts * electrons_per_photon(wavelengths * u.AA, det)).sum()
    assert electrons.data.item() == pytest.approx(expected, rel=1e-12)


def _two_energies(det):
    """A pixel holding photons of two wavelengths.

    Returns their counts and electrons per photon, and the pixel's photon
    count and mean and rms wavelengths.
    """
    counts, wavelengths = np.array([200.0, 100.0]), np.array([170.0, 212.0])
    share = light_onto_pixels([1.0, 3.0], [1.5, 3.5], [0.0, 4.0])
    on, mean, rms = photons_onto_pixels(share, counts, wavelengths)
    return counts, electrons_per_photon(wavelengths * u.AA, det), on.item(), mean.item(), rms.item()


def test_a_pixels_electrons_spread_as_its_photons_own_electrons_do():
    # Each photon frees electrons by its own energy, with a Fano spread. The
    # variance for N photons is N times that of one photon: its Fano spread
    # plus the spread of electrons per photon across the photons.
    det = Detector_SWC()
    counts, own, n, mean, rms = _two_energies(det)
    m = (counts * own).sum() / n
    variance = n * ((counts * own**2).sum() / n - m**2 + det.si_fano * m)
    np.random.seed(20261005)
    drawn = _vectorized_fano_noise(np.full(5, n), mean * u.AA, det, every_pixel=True,
                                   rms_wavelength=rms * u.AA)
    np.random.seed(20261005)
    standard = np.random.standard_normal(5)
    assert (drawn - n * m) / standard == pytest.approx(np.full(5, np.sqrt(variance)), rel=1e-9)


def test_the_detector_draws_the_spread_of_a_pixels_photons_energies():
    # Every photon caught and no read noise, so the electrons vary only by the
    # spread of electrons per photon and the Fano factor.
    det = Detector_SWC(qe_euv=1.0)
    det.read_noise_rms = 0 * u.electron / u.pix
    counts, own, n, mean, rms = _two_energies(det)
    shape = (1, 1, 100000)
    cube = NDCube(np.full(shape, n), wcs=WCS(naxis=3), unit=u.photon / u.pix,
                  meta={"rest_wav": REST, "photon_wavelength": np.full(shape, mean) * u.AA,
                        "photon_rms_wavelength": np.full(shape, rms) * u.AA})
    np.random.seed(20261005)
    electrons = to_electrons(cube, 0 * u.s, det).data
    m = (counts * own).sum() / n
    spread = n * ((counts * own**2).sum() / n - m**2)
    fano = n * det.si_fano * m
    assert electrons.var() == pytest.approx(spread + fano, rel=2e-2)
    # Taken all at the mean energy, the electrons would vary less than half as
    # much.
    assert fano < 0.5 * (spread + fano)


@pytest.mark.parametrize("psf", [False, True])
def test_every_photon_reaches_the_detector_and_frees_its_own_electrons(psf):
    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, psf=psf, noise=False)
    scene, wavelength = _broad_line()
    through = rebin_atmosphere(scene, det, sim, tel=tel)
    steps = simulate_once(through, 1 * u.s, det, tel, sim)
    photons, electrons = steps[5], steps[6]
    row, column = photons.data.shape[0] // 2, photons.data.shape[1] // 2
    dark = (det.dark_current * 1 * u.s).to_value(u.electron / u.pix)
    # The scene's photons at each wavelength, and their electrons.
    emitted = scene.data[4, 12] * photons_per_energy(tel, wavelength).value
    per_photon = (emitted * electrons_per_photon(wavelength, det)).sum() / emitted.sum()
    read_out = (electrons.data[row, column] - dark).sum() / det.qe_euv
    assert read_out / photons.data[row, column].sum() == pytest.approx(per_photon, rel=1e-9)
    # For a line this far off the rest wavelength, using the rest wavelength
    # would give a part in 1e4 more electrons per photon.
    at_rest = electrons_per_photon(REST, det)
    assert abs(at_rest / per_photon - 1) > 1e-4
    # The photons' mean square energy, which sets the spread of their
    # electrons, reaches the detector as their number and energy do.
    reaching = photons.data[row, column]
    rms = u.Quantity(photons.meta["photon_rms_wavelength"])[row, column].to_value(u.AA)
    square = (emitted / wavelength.to_value(u.AA) ** 2).sum() / emitted.sum()
    assert (reaching / rms**2).sum() / reaching.sum() == pytest.approx(square, rel=1e-9)


def test_a_cube_with_no_photon_wavelength_counts_each_pixels_photons_at_its_own():
    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, psf=False, noise=False)
    scene, _ = _broad_line()
    cube = rebin_atmosphere(scene, det, sim)
    assert "photon_wavelength" not in cube.meta
    steps = simulate_once(cube, 1 * u.s, det, tel, sim)
    photons, electrons = steps[5], steps[6]
    dark = (det.dark_current * 1 * u.s).to_value(u.electron / u.pix)
    own = electrons_per_photon(cube.axis_world_coords_values(2)[0], det)
    assert electrons.data - dark == pytest.approx(det.qe_euv * photons.data * own, rel=1e-12)


def test_binned_pixels_take_the_wavelengths_of_their_photons_mean_and_rms_energies():
    data = np.array([200.0, 100.0, 0.0, 0.0]).reshape(4, 1, 1)
    mean = np.array([190.0, 200.0, 195.0, 196.0]).reshape(4, 1, 1) * u.AA
    rms = np.array([189.5, 199.0, 195.0, 196.0]).reshape(4, 1, 1) * u.AA
    photons = NDCube(data, wcs=WCS(naxis=3), unit=u.photon / u.pix,
                     meta={"rest_wav": REST, "photon_wavelength": mean,
                           "photon_rms_wavelength": rms})
    binned_mean, binned_rms = binned_photon_wavelengths(photons, 2)
    assert binned_mean.to_value(u.AA).ravel() == pytest.approx(
        [300.0 / (200.0 / 190.0 + 100.0 / 200.0), 195.0], rel=1e-14)
    assert binned_rms.to_value(u.AA).ravel() == pytest.approx(
        [np.sqrt(300.0 / (200.0 / 189.5**2 + 100.0 / 199.0**2)), 195.0], rel=1e-14)
    # The binned cube does not carry its rows' wavelengths.
    binned = rebin_slit_offchip(photons, 2).meta
    assert "photon_wavelength" not in binned and "photon_rms_wavelength" not in binned


def test_with_the_psf_off_a_run_still_observes_through_the_telescope(tmp_path, monkeypatch):
    import importlib
    import sys

    import yaml

    from euvst_response.data_processing import rebin_spectra
    from euvst_response.synthesis_file import (SpectralLine, Synthesis, read_synthesis,
                                               write_synthesis)

    edges = np.arange(5) * 0.3 * u.Mm
    wavelength = REST + np.arange(-60, 61) * 0.003 * u.AA
    profile = np.exp(-0.5 * ((wavelength - REST) / (0.03 * u.AA)).decompose() ** 2)
    line = SpectralLine(intensity=np.ones((4, 4, 1)) * profile.value * 1e13 * RADIANCE,
                        wavelength=wavelength, rest_wavelength=REST)
    write_synthesis(Synthesis(lines={"Fe12_195.1190": line}, x_edges=edges, y_edges=edges),
                    tmp_path / "file.h5")
    (tmp_path / "run.yaml").write_text(yaml.safe_dump({
        "instrument": "SWC", "n_iter": 1, "ncpu": 1, "synthesis_file": str(tmp_path / "file.h5"),
        "simulation": {"expos": "5 s", "slit_width": "0.4 arcsec", "psf": False}}))
    main_module = importlib.import_module("euvst_response.main")
    observed = []
    real = main_module.monte_carlo

    def recording(cube, *args, **kwargs):
        observed.append(cube)
        return real(cube, *args, **kwargs)

    monkeypatch.setattr(main_module, "monte_carlo", recording)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(tmp_path / "run.yaml")])
    main_module.main()

    (cube,) = observed
    sim = Simulation(expos=1 * u.s, n_iter=1, slit_width=0.4 * u.arcsec, ncpu=1,
                     instrument="SWC", psf=False)
    expected = rebin_spectra(read_synthesis(tmp_path / "file.h5"), "Fe12_195.1190",
                             Detector_SWC(), sim, tel=Telescope_EUVST())
    assert cube.data == pytest.approx(expected.data, rel=1e-10, abs=0)
    assert "psf_applied" not in cube.meta
    assert np.shape(cube.meta["photon_wavelength"]) == cube.data.shape
    assert np.shape(cube.meta["photon_rms_wavelength"]) == cube.data.shape


def test_rows_binned_off_the_chip_vary_as_their_own_photons_electrons_add_up():
    from euvst_response.radiometric import dn_variance

    det = Detector_SWC()
    # Two rows, each holding photons of two wavelengths.
    counts = np.array([200.0, 100.0, 50.0, 150.0])
    wavelengths = np.array([170.0, 175.0, 205.0, 210.0])
    share = light_onto_pixels([1.0, 3.0, 5.0, 7.0], [1.5, 3.5, 5.5, 7.5], [0.0, 4.0, 8.0])
    on, mean, rms = photons_onto_pixels(share, counts, wavelengths)
    photons = NDCube(on.reshape(2, 1, 1), wcs=WCS(naxis=3), unit=u.photon / u.pix,
                     meta={"rest_wav": REST, "photon_wavelength": mean.reshape(2, 1, 1) * u.AA,
                           "photon_rms_wavelength": rms.reshape(2, 1, 1) * u.AA})
    own = electrons_per_photon(wavelengths * u.AA, det)
    # The photons are a Poisson count, so each adds its electrons squared,
    # plus the Fano factor times its electrons, to the variance. The photons
    # of every row add up.
    gain = det.gain_e_per_dn.to_value(u.electron / u.DN)
    read = det.read_noise_rms.to_value(u.electron / u.pixel) ** 2
    each = counts * (own**2 + det.si_fano * own)
    for n_bin, signal, spread in ((1, (counts * own).reshape(2, 2).sum(axis=1),
                                   each.reshape(2, 2).sum(axis=1)),
                                  (2, (counts * own).sum(keepdims=True), each.sum(keepdims=True))):
        binned_mean, binned_rms = binned_photon_wavelengths(photons, n_bin)
        variance = dn_variance(signal / gain, binned_mean.ravel(), 0 * u.s, det, 0.0, n_bin,
                               rms_wavelength=binned_rms.ravel())
        expected = (spread + n_bin * read) / gain**2 + n_bin / 12
        assert variance == pytest.approx(expected, rel=1e-12)


# ----------------------------------------------------------------------
# Pinholes pass the light the filter blocked, at each photon's wavelength
# ----------------------------------------------------------------------
PINHOLES = dict(enable_pinholes=True, pinhole_sizes=[20 * u.um], pinhole_positions=[0.5])
# On the filter edge the transmission rises from 0.12 to 0.51 between 170.4
# and 170.8 Angstrom, so it changes across a pixel's photons.
EDGE = 170.5 * u.AA


class _OpenFilter:
    """Filter that blocks no EUV."""

    def total_throughput(self, wl0):
        return np.ones(np.shape(wl0)) * u.dimensionless_unscaled


def _pinhole_weights(photons, det, sim, tel):
    """Share of the light through a pinhole that lands in each pixel."""
    shape = photons.data.shape
    unit = NDCube(np.ones(shape), wcs=photons.wcs, unit=photons.unit,
                  meta={"rest_wav": photons.meta["rest_wav"],
                        "photon_filter_transmission": np.full(shape, 0.5)})
    return apply_euv_pinhole_diffraction(unit, det, sim, tel).data - 1.0


@pytest.mark.parametrize("psf", [False, True])
def test_a_pinhole_lets_through_the_light_the_filter_blocked_from_each_photon(psf):
    from euvst_response.monte_carlo import _photons_on_the_detector

    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, psf=psf, noise=False,
                     **PINHOLES)
    scene, _ = _broad_line(rest=EDGE)
    steps = _photons_on_the_detector(rebin_atmosphere(scene, det, sim, tel=tel), 1 * u.s, det,
                                     tel, sim)
    passed, through = steps[4], steps[5]
    # The same scene through an open filter gives each pixel's photons before
    # the filter, each counted at its own wavelength.
    open_tel = Telescope_EUVST()
    open_tel.filter = _OpenFilter()
    no_pinholes = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, psf=psf,
                             noise=False)
    before = _photons_on_the_detector(rebin_atmosphere(scene, det, no_pinholes, tel=open_tel),
                                      1 * u.s, det, open_tel, no_pinholes)[5]
    blocked = before.data - passed.data
    weights = _pinhole_weights(passed, det, sim, tel)
    added = through.data - passed.data
    assert added.max() > 0
    assert added == pytest.approx(weights * blocked, rel=1e-9, abs=1e-12 * added.max())
    # Undoing the filter at each pixel's own wavelength would be wrong here.
    own = passed.axis_world_coords_values(2)[0]
    at_pixels = weights * passed.data * (1 / tel.filter.total_throughput(own).value - 1)
    lit = added > 1e-6 * added.max()
    assert np.abs(at_pixels[lit] / added[lit] - 1).max() > 1e-3
    # The pinhole photons have the energies of the blocked light.

    def per(cube, key, power=1):
        return cube.data / u.Quantity(cube.meta[key]).to_value(u.AA) ** power

    for key, power in (("photon_wavelength", 1), ("photon_rms_wavelength", 2)):
        energy = per(passed, key, power) + weights * (per(before, key, power)
                                                      - per(passed, key, power))
        assert per(through, key, power) == pytest.approx(energy, rel=1e-9,
                                                         abs=1e-12 * energy.max())
    assert "photon_filter_transmission" not in through.meta


def test_photons_counted_at_their_own_wavelengths_need_the_filter_for_the_pinholes():
    from euvst_response.monte_carlo import _photons_on_the_detector

    det, tel = Detector_SWC(), Telescope_EUVST()
    no_pinholes = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, noise=False)
    cube = rebin_atmosphere(_broad_line(rest=EDGE)[0], det, no_pinholes, tel=tel)
    sim = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, noise=False,
                     **PINHOLES)
    with pytest.raises(ValueError, match="filter passed"):
        _photons_on_the_detector(cube, 1 * u.s, det, tel, sim)


def test_photons_counted_at_their_pixels_wavelength_passed_the_filter_there():
    from euvst_response.monte_carlo import _photons_on_the_detector

    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, noise=False,
                     **PINHOLES)
    cube = rebin_atmosphere(_broad_line(rest=EDGE)[0], det, sim)
    steps = _photons_on_the_detector(cube, 1 * u.s, det, tel, sim)
    passed, through = steps[4], steps[5]
    own = passed.axis_world_coords_values(2)[0]
    blocked = passed.data * (1 / tel.filter.total_throughput(own).value - 1)
    weights = _pinhole_weights(passed, det, sim, tel)
    assert through.data - passed.data == pytest.approx(weights * blocked, rel=1e-9,
                                                       abs=1e-12 * through.data.max())


def test_the_pinholes_refuse_a_filter_that_passes_no_euv():
    # The pinhole light is worked out from the light the filter passes, and
    # here there is none.
    from euvst_response.config import AluminiumFilter
    from euvst_response.monte_carlo import _photons_on_the_detector

    det = Detector_SWC()
    opaque = Telescope_EUVST(filter=AluminiumFilter(mesh_throughput=0.0))
    sim = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, noise=False,
                     **PINHOLES)
    scene = _broad_line(rest=EDGE)[0]
    with pytest.raises(ValueError, match="passes no EUV"):
        rebin_atmosphere(scene, det, sim, tel=opaque)
    with pytest.raises(ValueError, match="passes no EUV"):
        _photons_on_the_detector(rebin_atmosphere(scene, det, sim), 1 * u.s, det, opaque, sim)
