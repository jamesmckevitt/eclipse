"""Each photon is counted at its own wavelength, and frees the electrons its own energy gives.

The telescope's throughput and the photon's energy are taken at the
wavelength the photon has, before the spectrograph's blur moves where it
lands. A pixel then holds photons of a spread of wavelengths, and the
electrons it reads out are each photon's own: its photons times the
electrons of a photon at the wavelength of their mean energy. The run used
the line's rest wavelength for every photon.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.config import Detector_SWC, Simulation, Telescope_EUVST
from euvst_response.data_processing import rebin_atmosphere
from euvst_response.monte_carlo import simulate_once
from euvst_response.radiometric import electrons_per_photon, photons_per_energy, to_electrons
from euvst_response.sampling import light_onto_pixels, photons_onto_pixels
from euvst_response.utils import angle_to_distance, binned_photon_wavelength, rebin_slit_offchip

REST = 195.119 * u.AA
RADIANCE = u.erg / (u.s * u.cm**2 * u.sr * u.cm)


def _scene(data, cell, wavelength_step):
    """A cube as the synthesis writes it, with square cells *cell* across."""
    ny, nx, nl = data.shape
    size = angle_to_distance(cell).to_value(u.Mm)
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "SOLX", "SOLY"]
    wcs.wcs.cunit = ["Angstrom", "Mm", "Mm"]
    wcs.wcs.crpix = [(nl + 1) / 2, (nx + 1) / 2, (ny + 1) / 2]
    wcs.wcs.crval = [REST.to_value(u.AA), 0.0, 0.0]
    wcs.wcs.cdelt = [wavelength_step.to_value(u.AA), size, size]
    return NDCube(data, wcs=wcs, unit=RADIANCE, meta={"rest_wav": REST})


def _broad_line(n_fine=401, step=0.0169 * u.AA / 10, offset=0.05 * u.AA, sigma=0.04 * u.AA):
    """A line off the rest wavelength, broad enough that its photons' energies differ, in every cell."""
    wavelength = REST + (np.arange(n_fine) - (n_fine - 1) / 2) * step
    profile = np.exp(-0.5 * ((wavelength - REST - offset) / sigma).decompose().value ** 2)
    return _scene(np.broadcast_to(profile * 1e13, (8, 24, n_fine)).copy(), 0.1 * u.arcsec,
                  step), wavelength


def test_a_pixel_holds_its_cells_photons_and_the_wavelength_of_their_mean_energy():
    # Two cells wholly in the first pixel, one with twice the photons.
    share = light_onto_pixels([1.0, 3.0], [1.5, 3.5], [0.0, 4.0, 8.0])
    on, wavelength = photons_onto_pixels(share, np.array([200.0, 100.0]),
                                         np.array([190.0, 200.0]))
    assert on == pytest.approx([300.0, 0.0])
    assert wavelength[0] == pytest.approx(300.0 / (200.0 / 190.0 + 100.0 / 200.0), rel=1e-14)
    assert np.isnan(wavelength[1])


def test_a_blur_can_differ_from_cell_to_cell_and_be_none():
    edges = np.linspace(0.0, 10.0, 41)
    share = light_onto_pixels([4.0, 6.0], [4.1, 6.1], edges, sigma=[0.0, 0.3],
                              width=[0.0, 0.2]).toarray()
    # Unblurred, the first cell lands in the one pixel it is in.
    assert share[:, 0] == pytest.approx(np.eye(40)[16])
    # Blurred, the second spreads, keeping its light.
    assert np.count_nonzero(share[:, 1]) > 3
    assert share[:, 1].sum() == pytest.approx(1.0, rel=1e-12)


def test_the_detector_frees_each_photons_own_electrons():
    det = Detector_SWC()
    counts, wavelengths = np.array([200.0, 100.0]), np.array([190.0, 200.0])
    share = light_onto_pixels([1.0, 3.0], [1.5, 3.5], [0.0, 4.0])
    on, mean = photons_onto_pixels(share, counts, wavelengths)
    cube = NDCube(on.reshape(1, 1, 1), wcs=WCS(naxis=3), unit=u.photon / u.pix,
                  meta={"rest_wav": REST, "photon_wavelength": mean.reshape(1, 1, 1) * u.AA})
    electrons = to_electrons(cube, 0 * u.s, det, noise=False)
    expected = det.qe_euv * (counts * electrons_per_photon(wavelengths * u.AA, det)).sum()
    assert electrons.data.item() == pytest.approx(expected, rel=1e-12)


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
    # The scene's photons at each of its wavelengths, and their electrons.
    emitted = scene.data[4, 12] * photons_per_energy(tel, wavelength).value
    per_photon = (emitted * electrons_per_photon(wavelength, det)).sum() / emitted.sum()
    read_out = (electrons.data[row, column] - dark).sum() / det.qe_euv
    assert read_out / photons.data[row, column].sum() == pytest.approx(per_photon, rel=1e-9)
    # Taken at the rest wavelength, a photon of a line this far off it
    # would free a part in 1e4 more electrons than it does.
    at_rest = electrons_per_photon(REST, det)
    assert abs(at_rest / per_photon - 1) > 1e-4


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


def test_binned_pixels_take_the_wavelength_of_their_photons_mean_energy():
    data = np.array([200.0, 100.0, 0.0, 0.0]).reshape(4, 1, 1)
    wavelength = np.array([190.0, 200.0, 195.0, 196.0]).reshape(4, 1, 1) * u.AA
    photons = NDCube(data, wcs=WCS(naxis=3), unit=u.photon / u.pix,
                     meta={"rest_wav": REST, "photon_wavelength": wavelength})
    binned = binned_photon_wavelength(photons, 2)
    assert binned.to_value(u.AA).ravel() == pytest.approx(
        [300.0 / (200.0 / 190.0 + 100.0 / 200.0), 195.0], rel=1e-14)
    # The binned cube does not carry its rows' wavelengths.
    assert "photon_wavelength" not in rebin_slit_offchip(photons, 2).meta


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


def test_rows_binned_off_the_chip_vary_as_their_own_electrons_add_up():
    from euvst_response.monte_carlo import _variance_wavelength
    from euvst_response.radiometric import dn_variance

    det = Detector_SWC()
    data = np.array([200.0, 100.0]).reshape(2, 1, 1)
    wavelength = np.array([170.0, 210.0]).reshape(2, 1, 1) * u.AA
    photons = NDCube(data, wcs=WCS(naxis=3), unit=u.photon / u.pix,
                     meta={"rest_wav": REST, "photon_wavelength": wavelength})
    per_photon = electrons_per_photon(wavelength, det).ravel()
    electrons = data.ravel() * per_photon
    # Each row's EUV electrons vary by its electrons per photon, plus the
    # Fano factor, times their number, and the rows add up.
    gain = det.gain_e_per_dn.to_value(u.electron / u.DN)
    read = det.read_noise_rms.to_value(u.electron / u.pixel) ** 2
    expected = (((per_photon + det.si_fano) * electrons).sum() + 2 * read) / gain**2 + 2 / 12
    variance = dn_variance(np.array([electrons.sum() / gain]),
                           _variance_wavelength(photons, 2).ravel(), 0 * u.s, det, 0.0, 2)
    assert variance.item() == pytest.approx(expected, rel=1e-12)
