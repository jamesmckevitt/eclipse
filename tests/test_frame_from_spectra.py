"""A full-CCD frame holds the spectra of the synthesis file, from ECLIPSE or another code."""
import sys

import astropy.constants as const
import astropy.units as u
import h5py
import numpy as np
import pytest

from eclipse import band as band_module
from eclipse import synthesis
from eclipse.atmosphere import Atmosphere, write_atmosphere
from eclipse.band import (BandLines, IonLines, band_lines_for, band_spectra, band_wavelengths,
                          write_band_lines)
from eclipse.config import Detector_SWC, Telescope_EUVST
from eclipse.frame import photons_from_lines, photons_from_spectrum
from eclipse.full_frame import full_ccd_frame
from eclipse.readout import FocalPlane_SWC
from eclipse.synthesis_file import (SpectralLine, Synthesis, read_synthesis,
                                    read_synthesis_products, write_synthesis)

RADIANCE = u.erg / (u.s * u.cm**2 * u.sr * u.AA)
SHARP = dict(slit_position=0 * u.arcsec, exposure=1 * u.s, psf=False, noise=False,
             telescope=Telescope_EUVST(resolution_budget="none"))


def _write(path, lines, nx=3, ny=40):
    """Synthesis file of `lines`, name to (wavelength, spectrum), the same in every pixel."""
    held = {name: SpectralLine(intensity=np.broadcast_to(spectrum, (ny, nx, spectrum.size)).copy()
                               * spectrum.unit, wavelength=wavelength,
                               rest_wavelength=wavelength[wavelength.size // 2])
            for name, (wavelength, spectrum) in lines.items()}
    edges = lambda n: (np.arange(n + 1) - n / 2) * 0.2 * u.arcsec
    return write_synthesis(Synthesis(lines=held, x_edges=edges(nx), y_edges=edges(ny)), path)


def _middle_rate(frame, ccd="left"):
    return frame.rate[ccd][:, frame.rate[ccd].shape[1] // 2]


def _rows(ccd="left"):
    focal_plane = FocalPlane_SWC.from_detector(Detector_SWC())
    return focal_plane.wavelength(np.arange(focal_plane.n_rows), ccd).to_value(u.AA)


@pytest.mark.parametrize("psf", [False, True])
def test_a_smooth_spectrum_lands_on_the_rows_as_photons_from_spectrum_lays_it(tmp_path, psf):
    wavelength = np.linspace(180.0, 184.0, 2001) * u.AA
    spectrum = 50.0 * (wavelength.value / 182.0) * RADIANCE
    path = _write(tmp_path / "smooth.h5", {"window": (wavelength, spectrum)})
    frame = full_ccd_frame(path, **{**SHARP, "psf": psf})
    det, tel = Detector_SWC(), SHARP["telescope"]
    expected = photons_from_spectrum(
        FocalPlane_SWC.from_detector(det), "left", tel, 0.2 * u.arcsec, wavelength, spectrum,
        det=det, spectral_psf="quadrature" if psf else None).to_value(1 / u.s)
    # Away from the ends of the window, where the two differ by half a bin.
    inside = (_rows() > 180.3) & (_rows() < 183.7)
    assert expected[inside].min() > 0
    assert np.allclose(_middle_rate(frame)[inside], expected[inside], rtol=1e-5, atol=0)
    assert frame.settings["spectra"] == ["window"]


def test_a_line_keeps_its_light_whatever_its_sampling(tmp_path):
    """A line narrower than its bins still gives the rows all its light."""
    sigma, centre, total = 0.004, 195.119, 800.0     # Angstrom, and erg / (s cm2 sr)
    det, tel = Detector_SWC(), SHARP["telescope"]
    focal_plane = FocalPlane_SWC.from_detector(det)
    exact = photons_from_lines(focal_plane, "left", tel, 0.2 * u.arcsec, [centre] * u.AA,
                               [total] * u.erg / (u.s * u.cm**2 * u.sr),
                               [sigma] * u.AA).to_value(1 / u.s)
    for step, tolerance in ((0.0005, 2e-3), (0.01, 0.25)):
        wavelength = band_wavelengths((194.6, 195.6), step)
        edges = np.append(wavelength - step / 2, wavelength[-1] + step / 2)
        from scipy.stats import norm
        mean = total * np.diff(norm.cdf(edges, centre, sigma)) / step
        path = _write(tmp_path / f"line_{step}.h5", {"line": (wavelength * u.AA, mean * RADIANCE)})
        rate = _middle_rate(full_ccd_frame(path, **SHARP))
        assert rate.sum() == pytest.approx(exact.sum(), rel=1e-5)
        assert np.allclose(rate, exact, rtol=0, atol=tolerance * exact.max())


def test_a_frame_adds_every_entry_of_the_file(tmp_path):
    """Entries are added where they overlap, and an entry on the other CCD lands there."""
    first = (np.linspace(194.9, 195.3, 201) * u.AA, np.full(201, 30.0) * RADIANCE)
    second = (np.linspace(195.1, 195.6, 126) * u.AA, np.full(126, 20.0) * RADIANCE)
    right = (np.linspace(203.0, 203.4, 101) * u.AA, np.full(101, 10.0) * RADIANCE)
    frames = {name: full_ccd_frame(_write(tmp_path / f"{name}.h5", entries), **SHARP)
              for name, entries in (("first", {"a": first}), ("second", {"b": second}),
                                    ("right", {"c": right}),
                                    ("all", {"a": first, "b": second, "c": right}))}
    assert frames["first"].rate["left"].any() and frames["second"].rate["left"].any()
    assert frames["right"].rate["right"].any() and not frames["right"].rate["left"].any()
    assert np.allclose(frames["all"].rate["left"],
                       frames["first"].rate["left"] + frames["second"].rate["left"],
                       rtol=1e-12, atol=0)
    assert np.allclose(frames["all"].rate["right"], frames["right"].rate["right"],
                       rtol=1e-12, atol=0)
    assert frames["all"].settings["spectra"] == ["a", "b", "c"]


def test_a_frame_takes_any_unit_of_spectral_radiance_and_uneven_wavelengths(tmp_path):
    # Finer in the middle, as another code may give it.
    wavelength = np.unique(np.round(np.concatenate([np.linspace(190.0, 191.0, 41),
                                                    np.linspace(190.4, 190.6, 81)]), 6)) * u.AA
    spectrum = 40.0 * np.exp(-0.5 * ((wavelength.value - 190.5) / 0.05) ** 2) * RADIANCE
    photons = spectrum.to(u.ph / (u.s * u.cm**2 * u.sr * u.Hz),
                          equivalencies=u.spectral_density(wavelength))
    in_si = spectrum.to(u.W / (u.m**2 * u.sr * u.nm))
    rates = [_middle_rate(full_ccd_frame(_write(tmp_path / f"{i}.h5", {"line": (grid, values)}),
                                         **SHARP))
             for i, (grid, values) in enumerate(((wavelength, spectrum), (wavelength, photons),
                                                 (wavelength.to(u.nm), in_si)))]
    assert rates[0].sum() > 0
    for other in rates[1:]:
        assert np.allclose(other, rates[0], rtol=1e-9, atol=0)
    # Each wavelength holds the light of the bin halfway to its neighbours.
    edges = np.concatenate([[wavelength[0].value], (wavelength.value[1:] + wavelength.value[:-1]) / 2,
                            [wavelength[-1].value]])
    edges[0] -= edges[1] - edges[0]
    edges[-1] += edges[-1] - edges[-2]
    det, tel = Detector_SWC(), SHARP["telescope"]
    from eclipse.frame import pixel_solid_angle
    from eclipse.radiometric import photons_per_energy

    solid_angle = pixel_solid_angle(FocalPlane_SWC.from_detector(det), 0.2 * u.arcsec)
    light = (spectrum * np.diff(edges) * u.AA * solid_angle
             * photons_per_energy(tel, wavelength)).to_value(1 / u.s)
    # The frame takes the throughput at the middle of each bin, which is not quite its
    # wavelength where the spacing changes.
    assert rates[0].sum() == pytest.approx(light.sum(), rel=1e-6)


def test_wavelengths_outside_the_throughput_tables_give_no_light(tmp_path):
    wavelength = np.linspace(160.0, 230.0, 701) * u.AA
    path = _write(tmp_path / "wide.h5", {"wide": (wavelength, np.full(701, 5.0) * RADIANCE)})
    frame = full_ccd_frame(path, **SHARP)
    for ccd in ("left", "right"):
        assert np.all(np.isfinite(frame.rate[ccd])) and frame.rate[ccd].any()
        dark = (_rows(ccd) < 169.9) | (_rows(ccd) > 212.4)
        assert dark.any() and not frame.rate[ccd][dark].any()


# The band of one made-up ion, for syntheses without CHIANTI.
LINES = np.array([194.9, 195.119, 195.4])
G_LINES = np.array([1e-24, 3e-24, 2e-24])


def _band_lines(logT_grid, logN_grid, band=(194.0, 196.5), **kwargs):
    g = np.broadcast_to(G_LINES[:, None, None], (LINES.size, logN_grid.size, logT_grid.size))
    # G rises with density, by its value at 1e7 cm^-3 for each decade, so the density the
    # lines are taken at shows.
    g = g * (1 + (np.asarray(logN_grid, dtype=float) - 7.0))[None, :, None]
    ion = IonLines(name="Fe 12", atom=26, stage=12, mass=55.845, wavelength=LINES,
                   observed=np.ones(LINES.size, dtype=bool),
                   temperatures=np.arange(logT_grid.size), g=g)
    return BandLines(band=tuple(band), logT_grid=np.asarray(logT_grid, dtype=float),
                     logN_grid=np.asarray(logN_grid, dtype=float),
                     abundance=kwargs.get("abundance", "sun_coronal_2021_chianti"),
                     hdf5_dbase_root="test", ions=[ion])


def test_the_bands_spectrum_holds_each_lines_light_at_its_doppler_shift():
    logT, logN = np.array([5.8, 6.0, 6.2]), np.array([8.0, 9.0, 10.0])
    lines = _band_lines(logT, logN)
    velocity = np.array([-100.0, 0.0, 100.0]) * u.km / u.s
    em = np.zeros((2, 1, 3, 3))
    em[0, 0, 1, 1] = 4e27          # at rest, at 1 MK
    em[1, 0, 2, 2] = 1e27          # receding at 100 km/s, at 1.6 MK
    # The first pixel is at a grid density, the second halfway, in log, between two.
    density = np.array([[[1e9] * 3], [[10 ** 9.5] * 3]])
    wavelength = band_wavelengths(lines.band, 0.002) * u.AA
    spectra = band_spectra(lines, em, density, velocity, wavelength, n_workers=1)
    assert spectra.shape == (2, 1, wavelength.size) and spectra.unit == u.erg / (u.s * u.cm**2 * u.sr * u.cm)
    per_bin = (spectra * 0.002 * u.AA).to_value(u.erg / (u.s * u.cm**2 * u.sr))
    # G at 1e9 is three times G_LINES, and halfway to 1e10 it is 3.5 times.
    assert per_bin[0].sum() == pytest.approx(4e27 * 3 * G_LINES.sum() / (4 * np.pi), rel=1e-9)
    assert per_bin[1].sum() == pytest.approx(1e27 * 3.5 * G_LINES.sum() / (4 * np.pi), rel=1e-9)
    # Each line's light is centred on its wavelength, moved by the Doppler shift.
    for pixel, shift in ((0, 1.0), (1, 1.0 + (100 * u.km / u.s / const.c).to_value(""))):
        for line, g in zip(LINES, G_LINES):
            near = np.abs(wavelength.value - line * shift) < 0.08
            centre = (per_bin[pixel, 0, near] * wavelength.value[near]).sum() / per_bin[pixel, 0, near].sum()
            assert centre == pytest.approx(line * shift, abs=1e-5)
            assert per_bin[pixel, 0, near].sum() / per_bin[pixel].sum() == pytest.approx(
                g / G_LINES.sum(), rel=1e-6)
    # A density off the lines' grid gives no light, as the contribution functions are zero there.
    off = band_spectra(lines, em, 1e12, velocity, wavelength, n_workers=1)
    assert not off.value.any()
    # Weights give each row the weighted sum of its columns' spectra.
    wide = np.concatenate([em, 3 * em], axis=1)
    both = np.concatenate([density, density], axis=1)
    each = band_spectra(lines, wide, both, velocity, wavelength, n_workers=1)
    summed = band_spectra(lines, wide, both, velocity, wavelength, n_workers=1,
                          column_weights=[0.25, 0.5])
    assert summed.shape == (2, wavelength.size)
    assert np.allclose(summed, 0.25 * each[:, 0] + 0.5 * each[:, 1], rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="2 columns, but 3 weights"):
        band_spectra(lines, wide, both, velocity, wavelength, column_weights=[1, 1, 1])
    with pytest.raises(ValueError, match="temperatures"):
        band_spectra(lines, em[:, :, :2], density[..., :2], velocity, wavelength, n_workers=1)
    with pytest.raises(ValueError, match="velocity bins"):
        band_spectra(lines, em, density, velocity[:2], wavelength, n_workers=1)


def _atmosphere(tmp_path, view="z"):
    """Corona at 1 MK and 1e9 cm^-3, brighter towards +x, 4.4 Mm along the slit."""
    nz, ny, nx = 4, 44, 6
    if view.endswith("x"):
        nz, ny, nx = 44, 6, 4
    density = np.full((nz, ny, nx), 1.0e9)
    # Brighter along the axis the image's columns run on.
    columns = {"z": 2, "x": 1, "-x": 1}[view]
    shape = [1, 1, 1]
    shape[columns] = -1
    density = density * (1 + 0.1 * np.arange(density.shape[columns])).reshape(shape)
    mass = (density / u.cm**3 * 1.2 * const.u).to(u.g / u.cm**3)
    edges = lambda n: np.arange(n + 1) * 0.1 * u.Mm
    atmosphere = Atmosphere(
        temperature=np.full((nz, ny, nx), 1e6) * u.K, mass_density=mass,
        **{f"velocity_{axis}": np.zeros((nz, ny, nx)) * u.km / u.s for axis in "xyz"},
        x_edges=edges(nx), y_edges=edges(ny), z_edges=edges(nz))
    return write_atmosphere(atmosphere, tmp_path / "atmosphere.h5")


def _synthesise(tmp_path, name, *words, view="z", monkeypatch=None):
    patch = monkeypatch or pytest.MonkeyPatch()
    atmosphere = tmp_path / "atmosphere.h5"
    if not atmosphere.exists():
        _atmosphere(tmp_path, view)
    patch.setattr(sys, "argv", [
        "synthesise-spectra", "--atmosphere", str(atmosphere), "--output-dir", str(tmp_path),
        "--output-name", name, "--mass-per-electron", "1.2",
        *([] if "--position" in words else [f"--integration-axis={view}"]),
        *words])
    patch.setattr(band_module, "band_contribution_functions",
                  lambda band, logT, logN, **kwargs: _band_lines(logT, logN, band, **kwargs))
    try:
        synthesis.main()
    finally:
        if monkeypatch is None:
            patch.undo()
    return tmp_path / name


BAND = ("--band", "194 Angstrom", "196.5 Angstrom")


def test_a_band_is_synthesised_as_one_spectrum_and_its_lines_kept_in_a_file(tmp_path):
    path = _synthesise(tmp_path, "band.h5", *BAND)
    held = read_synthesis(path)
    assert list(held.lines) == ["band_194.0010-196.4990"]
    line = held.lines["band_194.0010-196.4990"]
    assert np.allclose(np.diff(line.wavelength.to_value(u.AA)), 0.002, rtol=1e-6)
    products = read_synthesis_products(path)
    assert products["config"]["band"] == [194.0, 196.5] and products["config"]["lines"] == []
    assert products["goft"] == {}
    # Each pixel's light is its emission measure times G at its density.
    em, density = products["em_tv"].sum(axis=(2, 3)), products["electron_density"].max(axis=2)
    expected = em * (1 + (np.log10(density) - 7.0)) * G_LINES.sum() / (4 * np.pi)
    light = (line.radiance() * 0.002 * u.AA).to_value(u.erg / (u.s * u.cm**2 * u.sr)).sum(axis=2)
    assert np.allclose(light, expected, rtol=1e-6, atol=0) and expected.min() > 0
    # The lines are kept beside the synthesis, and read back by the next one.
    assert (tmp_path / "band_lines.h5").is_file()
    patch = pytest.MonkeyPatch()
    patch.setattr(band_module, "band_contribution_functions",
                  lambda *args, **kwargs: pytest.fail("the lines were worked out again"))
    again = _synthesise(tmp_path, "again.h5", *BAND, monkeypatch=patch)
    patch.undo()
    assert np.array_equal(read_synthesis(again).lines["band_194.0010-196.4990"].intensity,
                          line.intensity)


def test_a_synthesis_names_lines_or_a_band(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        _synthesise(tmp_path, "both.h5", *BAND, "--lines", "Fe12_195.1190")
    with pytest.raises(ValueError, match="Name the --lines"):
        _synthesise(tmp_path, "neither.h5")
    with pytest.raises(ValueError, match="--band takes two wavelengths"):
        _synthesise(tmp_path, "unitless.h5", "--band", "194", "196.5")
    with pytest.raises(ValueError, match="shorter wavelength"):
        _synthesise(tmp_path, "backwards.h5", "--band", "196 Angstrom", "194 Angstrom")


@pytest.mark.parametrize("view", ["z", "x", "-x"])
def test_a_strip_of_the_image_holds_the_whole_images_columns(tmp_path, view):
    whole = read_synthesis(_synthesise(tmp_path, "whole.h5", *BAND, view=view))
    edges = whole.x_edges.to_value(u.Mm)
    # From inside the second column to inside the fourth.
    limits = (f"{edges[1] + 0.01} Mm", f"{edges[4] - 0.01} Mm")
    strip = read_synthesis(_synthesise(tmp_path, "strip.h5", *BAND, "--image-x", *limits,
                                       view=view))
    assert np.allclose(strip.x_edges.to_value(u.Mm), edges[1:5], rtol=0, atol=1e-9)
    assert np.allclose(strip.y_edges.to_value(u.Mm), whole.y_edges.to_value(u.Mm))
    (name,) = whole.lines
    full, part = whole.lines[name].intensity.value, strip.lines[name].intensity.value
    assert np.allclose(part, full[:, 1:4], rtol=1e-12, atol=0)
    # The columns differ, so the strip is the right one.
    assert not np.allclose(full[:, 1], full[:, 3], rtol=1e-3)
    with pytest.raises(ValueError, match="No part of the image"):
        _synthesise(tmp_path, "off.h5", *BAND, "--image-x", "50 Mm", "60 Mm", view=view)
    with pytest.raises(ValueError, match="two positions"):
        _synthesise(tmp_path, "unitless.h5", *BAND, "--image-x", "1", "2", view=view)


def test_the_deprecated_lines_give_the_frame_of_the_band_synthesised_into_a_file(tmp_path):
    band = _synthesise(tmp_path, "band.h5", *BAND)
    products = read_synthesis_products(band)
    lines = _band_lines(products["logT_grid"], products["logN_grid"])
    settings = dict(slit_position=0.3 * u.Mm, exposure=1 * u.s, noise=False,
                    telescope=Telescope_EUVST(resolution_budget="none"))
    held = full_ccd_frame(band, **settings)
    with pytest.warns(FutureWarning, match="synthesise the band into it"):
        worked_out = full_ccd_frame(band, lines, n_workers=1, **settings)
    assert held.rate["left"].any()
    # The file's wavelengths are the band's to rounding.
    assert np.allclose(worked_out.rate["left"], held.rate["left"], rtol=1e-6,
                       atol=1e-9 * held.rate["left"].max())
    assert held.settings["spectra"] == ["band_194.0010-196.4990"]
    # Lines on the same temperatures to a float32 file's rounding serve too.
    rounded = BandLines(band=lines.band, logT_grid=lines.logT_grid.astype(np.float32),
                        logN_grid=lines.logN_grid, abundance=lines.abundance,
                        hdf5_dbase_root="test", ions=lines.ions)
    with pytest.warns(FutureWarning):
        again = full_ccd_frame(band, rounded, n_workers=1, **settings)
    assert np.array_equal(again.rate["left"], worked_out.rate["left"])
    # A file from another code keeps no emission measure to work a band out from.
    other = _write(tmp_path / "other.h5", {"line": (np.linspace(195.0, 195.2, 51) * u.AA,
                                                    np.ones(51) * RADIANCE)})
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="Leave out lines"):
        full_ccd_frame(other, lines, **settings)


def test_a_file_of_a_bands_lines_is_read_back_only_if_it_serves(tmp_path, monkeypatch):
    logT, logN = np.linspace(5.0, 7.0, 21), np.linspace(8.2, 9.4, 5)
    path = write_band_lines(_band_lines(logT, np.linspace(7.0, 13.0, 21)), tmp_path / "lines.h5")
    monkeypatch.setattr(band_module, "band_contribution_functions",
                        lambda *args, **kwargs: pytest.fail("the lines were worked out"))
    # Densities within the file's, and a band inside its own.
    lines = band_lines_for(path, (194.5, 196.0), logT, logN)
    assert lines.band == (194.0, 196.5) and lines.logN_grid.size == 21
    # A file that is not there is worked out on the wider grid, if one is given.
    made = []
    monkeypatch.setattr(band_module, "band_contribution_functions",
                        lambda band, logT, logN, **kwargs: made.append(logN) or _band_lines(
                            logT, logN, band))
    wide = np.linspace(7.0, 13.0, 21)
    new = band_lines_for(tmp_path / "new.h5", (194.5, 196.0), logT, logN, wider_grid=wide)
    assert np.array_equal(made[0], wide) and new.logN_grid.size == 21
    assert (tmp_path / "new.h5").is_file()
    with pytest.raises(ValueError, match="wider_grid must have every density"):
        band_lines_for(tmp_path / "other.h5", (194.5, 196.0), logT, logN + 0.1, wider_grid=wide)
    monkeypatch.setattr(band_module, "band_contribution_functions",
                        lambda *args, **kwargs: pytest.fail("the lines were worked out"))
    for settings, message in (
            (dict(band=(193.0, 196.0)), "holds the band 194 to 196.5"),
            (dict(abundance="sun_coronal_1992_feldman"), "abundances"),
            (dict(logT_grid=logT + 0.01), "another grid of temperature"),
            (dict(logN_grid=logN + 0.05), "do not have all"),
            (dict(hdf5_dbase_root="/another/database"), "CHIANTI database")):
        given = {"band": (194.5, 196.0), "logT_grid": logT, "logN_grid": logN, **settings}
        with pytest.raises(ValueError, match=message):
            band_lines_for(path, **given)
    assert path.is_file()


def test_a_run_files_frame_of_the_band_under_the_slit_is_the_whole_images(tmp_path, monkeypatch):
    """The blur across the slit reaches about 2 arcsec either side of it, and the image is
    8.3 arcsec across, so the run synthesises a strip, which gives the frame the whole image
    gives."""
    import eclipse

    monkeypatch.chdir(tmp_path)
    nz, ny, nx = 3, 20, 60
    # Brighter towards +x, so the columns the slit takes light from matter.
    density = np.broadcast_to(1e9 * (1 + 0.05 * np.arange(nx)), (nz, ny, nx)) / u.cm**3
    edges = lambda n: (np.arange(n + 1) - n / 2) * 0.1 * u.Mm
    write_atmosphere(Atmosphere(
        temperature=np.full((nz, ny, nx), 1e6) * u.K, electron_density=density,
        velocity_z=np.zeros((nz, ny, nx)) * u.km / u.s, x_edges=edges(nx), y_edges=edges(ny),
        z_edges=np.arange(nz + 1) * 0.1 * u.Mm), tmp_path / "box.h5")
    monkeypatch.setattr(band_module, "band_contribution_functions",
                        lambda band, logT, logN, **kwargs: _band_lines(logT, logN, band, **kwargs))
    job = {"instrument": "SWC", "atmosphere": "box.h5",
           "simulation": {"expos": "10 s", "psf": True, "noise": False},
           "frame": {"slit_position": "0.4 arcsec", "band": ["194 Angstrom", "196.5 Angstrom"]}}
    [output] = eclipse.make_frame(job, name="job")
    strip = read_synthesis("run/input/job_band.h5")
    assert 20 < strip.shape[1] < nx
    with h5py.File(output.frame) as f:
        rate = f["left/rate"][()]
        assert [name.decode() for name in f["spectra"][()]] == ["band_194.0010-196.4990"]
    monkeypatch.setattr(sys, "argv", [
        "synthesise-spectra", "--atmosphere", "box.h5", "--output-dir", "whole",
        "--output-name", "band.h5", "--band", "194 Angstrom", "196.5 Angstrom",
        "--band-lines", "run/input/band_lines.h5"])
    synthesis.main()
    whole = full_ccd_frame("whole/band.h5", slit_position=0.4 * u.arcsec, exposure=10 * u.s,
                           noise=False)
    assert read_synthesis("whole/band.h5").shape[1] == nx
    assert rate.any()
    assert np.allclose(rate, whole.rate["left"], rtol=1e-9, atol=0)


def test_a_strip_of_a_view_from_a_position_holds_the_whole_images_columns(tmp_path):
    """Seen from a position, the image is cut once it is made, so the strip is the whole
    image's columns, as for a view from above."""
    words = ("--position", "300 arcsec", "100 arcsec")
    whole = read_synthesis(_synthesise(tmp_path, "whole.h5", *BAND, *words))
    edges = whole.x_edges.to_value(u.Mm)
    limits = (f"{edges[1] + 0.01} Mm", f"{edges[4] - 0.01} Mm")
    strip = read_synthesis(_synthesise(tmp_path, "strip.h5", *BAND, *words, "--image-x",
                                       *limits))
    assert np.allclose(strip.x_edges.to_value(u.Mm), edges[1:5], rtol=0, atol=1e-9)
    (name,) = whole.lines
    full, part = whole.lines[name].intensity.value, strip.lines[name].intensity.value
    assert full[:, 1:4].any()
    assert np.allclose(part, full[:, 1:4], rtol=1e-12, atol=0)
