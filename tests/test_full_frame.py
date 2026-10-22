"""The full-CCD frame of the SW camera, from a synthesis and every line in the band."""
import inspect
import sys
import warnings

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from eclipse import synthesis
from eclipse.atmosphere import Atmosphere, write_atmosphere
from eclipse.band import BandLines, IonLines
from eclipse.config import Detector_SWC, Simulation, Telescope_EUVST
from eclipse.frame import photons_from_lines
from eclipse.full_frame import (_density_shares, _settings_from_simulation,
                                full_ccd_frame, lay_out_register, write_full_frame)
from eclipse.readout import FocalPlane_SWC
from eclipse.synthesis_file import read_synthesis_products

REST = 195.119
G0 = 1e-24  # erg cm3 / s, at every temperature and density


def _flat_goft(lines, **kwargs):
    logT_grid, logN_grid = np.linspace(5.0, 7.0, 21), np.linspace(8.0, 10.0, 21)
    return ({"Fe12_195.1190": {"wl0": (REST * u.AA).to(u.cm),
                               "g_tn": np.ones((logN_grid.size, logT_grid.size)),
                               "atom": 26, "ion": 12, "hdf5_dbase_root": None}},
            logT_grid, logN_grid)


def _synthesise(tmp_path, temperature, velocity=0.0):
    """Synthesis of a corona at 1e9 cm^-3, 6 arcsec along the slit, moving at `velocity` km/s."""
    nz, ny, nx = temperature.size, 44, 6
    density = (1.0e9 / u.cm**3 * 1.2 * const.u).to(u.g / u.cm**3)
    atmosphere = Atmosphere(
        temperature=np.broadcast_to(temperature[:, None, None], (nz, ny, nx)).copy() * u.K,
        mass_density=np.full((nz, ny, nx), density.value) * density.unit,
        velocity_z=np.full((nz, ny, nx), velocity) * u.km / u.s,
        x_edges=np.arange(nx + 1) * 0.1 * u.Mm, y_edges=np.arange(ny + 1) * 0.1 * u.Mm,
        z_edges=np.arange(nz + 1) * 0.1 * u.Mm)
    path = write_atmosphere(atmosphere, tmp_path / "atmosphere.h5")
    patch = pytest.MonkeyPatch()
    patch.setattr(sys, "argv", [
        "synthesise-spectra", "--atmosphere", str(path), "--output-dir", str(tmp_path),
        "--output-name", "synthesis.h5", "--mass-per-electron", "1.2", "--lines",
        "Fe12_195.1190"])
    patch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    synthesis.main()
    patch.undo()
    return tmp_path / "synthesis.h5"


@pytest.fixture(scope="module")
def uniform_synthesis(tmp_path_factory):
    """Uniform corona at 1 MK."""
    return _synthesise(tmp_path_factory.mktemp("frame"), np.full(4, 1e6))


@pytest.fixture(scope="module")
def fast_synthesis(tmp_path_factory):
    """Uniform corona at 1 MK moving at 500 km/s, beyond the velocity grid of +/-300 km/s."""
    return _synthesise(tmp_path_factory.mktemp("fast"), np.full(4, 1e6), velocity=500.0)


@pytest.fixture(scope="module")
def two_temperature_synthesis(tmp_path_factory):
    """Corona at 1 MK in front and 2 MK behind, so the workers have temperatures to share."""
    return _synthesise(tmp_path_factory.mktemp("frame2"), np.array([1e6, 1e6, 2e6, 2e6]))


@pytest.fixture(scope="module")
def sharp_frame(uniform_synthesis):
    """Frame of the uniform corona for 1 s, without the PSF or noise. Tests must not change it."""
    products = read_synthesis_products(uniform_synthesis)
    return full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1 * u.s,
                          telescope=Telescope_EUVST(resolution_budget="none"), psf=False,
                          noise=False, n_workers=1)


@pytest.fixture(scope="module")
def blurred_frame(uniform_synthesis):
    """Frame of the uniform corona with the PSF and no noise. Tests must not change it.

    The exposure is long, 1000 s, so the read-out smear adds little light. The
    telescope leaves out the resolution budget, whose blur across the slit would
    carry light past the edges of this small scene.
    """
    products = read_synthesis_products(uniform_synthesis)
    return full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1000 * u.s,
                          telescope=Telescope_EUVST(resolution_budget="none"), noise=False,
                          n_workers=1)


def _one_line(products, logN=None, g_off_the_synthesis=None):
    logT = products["logT_grid"]
    logN = products["logN_grid"] if logN is None else logN
    ion = IonLines(name="Fe 12", atom=26, stage=12, mass=55.845, wavelength=np.array([REST]),
                   observed=np.array([True]), temperatures=np.arange(logT.size),
                   g=np.full((1, logN.size, logT.size), G0))
    if g_off_the_synthesis is not None:
        # Densities the synthesis lacks get a different G, which must not
        # reach the frame.
        off = ~np.isin(np.round(logN, 6), np.round(products["logN_grid"], 6))
        ion.g[:, off, :] = g_off_the_synthesis
    return BandLines(band=(194.0, 196.0), logT_grid=logT, logN_grid=logN,
                     abundance=products["config"]["abundance"], hdf5_dbase_root="test",
                     ions=[ion])


@pytest.mark.parametrize("psf", [False, True])
def test_a_line_lands_on_the_rows_as_photons_from_lines_lays_it(uniform_synthesis, psf, request):
    products = read_synthesis_products(uniform_synthesis)
    # The rate does not depend on the exposure, so the 1000 s frame serves.
    frame = request.getfixturevalue("blurred_frame" if psf else "sharp_frame")
    det, tel = Detector_SWC(), Telescope_EUVST(resolution_budget="none")
    focal_plane = FocalPlane_SWC.from_detector(det)
    # A column in the middle of the scene sees one pixel's emission measure,
    # all at 1 MK and at rest.
    em = products["em_tv"][0, 0]                                     # (nT, nv)
    temperature = 10 ** products["logT_grid"]
    width = (REST * np.sqrt(const.k_B * temperature * u.K / (55.845 * const.u)) / const.c
             ).to_value(u.dimensionless_unscaled) * u.AA
    expected = sum(
        photons_from_lines(focal_plane, "left", tel, 0.2 * u.arcsec, [REST] * u.AA,
                           [em[t].sum() * G0 / (4 * np.pi)] * u.erg / (u.s * u.cm**2 * u.sr),
                           [width[t].value] * u.AA, det=det,
                           spectral_psf="quadrature" if psf else None).to_value(1 / u.s)
        for t in range(temperature.size) if em[t].sum() > 0)
    middle = focal_plane.n_columns // 2
    # atol allows for the far wings, where one integral gives 1e-20 and the other 0.
    assert np.allclose(frame.rate["left"][:, middle], expected, rtol=1e-9,
                       atol=1e-12 * expected.max())
    assert expected.sum() > 0
    # The right CCD does not reach 195 Angstrom, and the scene does not reach
    # the columns at the ends of the slit.
    assert not frame.rate["right"].any()
    assert not frame.rate["left"][:, 0].any()


def test_the_frame_is_read_out_with_its_prescan_and_overscan(sharp_frame, tmp_path):
    det = Detector_SWC()
    frame = sharp_frame
    rows = det.n_rows + det.parallel_overscan_rows
    samples = 2 * (det.serial_prescan + det.n_columns // 2 + det.serial_overscan)
    assert frame.dn["left"].shape == (rows, samples)
    path = write_full_frame(frame, tmp_path / "frame.h5")
    import h5py

    with h5py.File(path) as f:
        assert f["left/dn"].shape == (rows, samples) and f.attrs["noise"] == np.False_
        assert f["left/photon_wavelength"].shape == (rows, det.n_columns)
        assert f["left/photon_rms_wavelength"].shape == (rows, det.n_columns)
        assert f.attrs["cells_per_row"] == 16
        assert f.attrs["psf_boundary"] == "replicate" and f.attrs["slit_mask"] == np.False_
        assert (f.attrs["serial_prescan"], f.attrs["serial_overscan"],
                f.attrs["parallel_overscan_rows"]) == (det.serial_prescan, det.serial_overscan,
                                                       det.parallel_overscan_rows)


def test_the_prescan_and_overscan_hold_only_the_read_noise():
    det = Detector_SWC()
    image = np.full((5, 8), 7.0)
    laid = lay_out_register(image, det, noise=False)
    pre, over = det.serial_prescan, det.serial_overscan
    assert laid.shape == (5, 8 + 2 * (pre + over))
    assert np.all(laid[:, pre:pre + 4] == 7.0)
    assert np.all(laid[:, pre + 4 + 2 * over:pre + 8 + 2 * over] == 7.0)
    assert np.all(laid[:, :pre] == 0.0) and np.all(laid[:, -pre:] == 0.0)


def test_the_overscan_holds_what_the_register_leaves_behind():
    det = Detector_SWC()
    image = np.full((5, 8), 7.0)
    pre, over = det.serial_prescan, det.serial_overscan
    overscan = np.arange(5 * 2 * over, dtype=float).reshape(5, 2 * over)
    laid = lay_out_register(image, det, noise=False, overscan_dn=overscan)
    assert np.array_equal(laid[:, pre + 4:pre + 4 + 2 * over], overscan)
    with pytest.raises(ValueError, match="overscan_dn must be shaped"):
        lay_out_register(image, det, noise=False, overscan_dn=overscan[:, 1:])


def test_a_frame_with_imperfect_serial_transfer_trails_into_the_overscan(uniform_synthesis):
    """The line and the dark charge in each half's last image pixels trail into the overscan.

    A long exposure gives the line hundreds of DN. In rows the line does not
    reach, the overscan holds the dark charge alone.
    """
    from eclipse.readout import ReadoutSequence, dark_current_time, serial_dark_shares

    products = read_synthesis_products(uniform_synthesis)
    det = Detector_SWC(cte_serial=0.999)
    frame = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 500 * u.s,
                           det=det, noise=False, n_workers=1)
    perfect = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 500 * u.s,
                             noise=False, n_workers=1)
    pre, over, half = det.serial_prescan, det.serial_overscan, det.n_columns // 2
    sequence = ReadoutSequence.from_detector(det)
    dark = (det.dark_current * dark_current_time(500 * u.s, sequence, det.n_rows)).to_value(
        u.electron / u.pix)[:det.n_rows]
    _, shares = serial_dark_shares(sequence, det.n_columns)
    gain = det.gain_e_per_dn.to_value(u.electron / u.DN)
    photons = perfect.photons["left"][:det.n_rows]
    for column, sample, share in ((half - 1, pre + half, shares[0]),
                                  (half, pre + half + 2 * over - 1, shares[-1])):
        assert photons[:, column].max() > 0
        dn = frame.dn["left"][:det.n_rows, sample]
        dark_only = np.round(dark * share / gain)
        lit = photons[:, column] >= photons[:, column].max() / 2
        assert np.all(dn[lit] > dark_only[lit])
        unlit = photons[:, column] == 0
        assert np.array_equal(dn[unlit], dark_only[unlit])
        assert np.all(perfect.dn["left"][:, sample] == 0)


def test_a_pixels_emission_measure_is_shared_between_the_densities_either_side():
    grid = np.array([8.0, 9.0, 10.0])
    lower, upper, low_share, high_share = _density_shares(np.array([8.25, 10.0, 11.0, np.nan]),
                                                          grid)
    assert list(lower[:2]) == [0, 2] and list(upper[:2]) == [1, 2]
    assert np.allclose(low_share, [0.75, 1.0, 0.0, 0.0])
    assert np.allclose(high_share, [0.25, 0.0, 0.0, 0.0])


def test_lines_on_a_wider_density_grid_give_the_same_frame(uniform_synthesis, sharp_frame):
    products = read_synthesis_products(uniform_synthesis)
    grid = products["logN_grid"]
    step = grid[1] - grid[0]
    wider = np.concatenate([grid[0] - step * np.arange(3, 0, -1), grid,
                            grid[-1] + step * np.arange(1, 4)])
    frame = full_ccd_frame(uniform_synthesis,
                           _one_line(products, wider, g_off_the_synthesis=5 * G0),
                           0.3 * u.Mm, 1 * u.s, psf=False, noise=False, n_workers=1)
    assert np.array_equal(frame.rate["left"], sharp_frame.rate["left"])


def test_a_frame_needs_the_synthesis_grids_and_abundances_and_no_tilt(uniform_synthesis):
    products = read_synthesis_products(uniform_synthesis)
    lines = _one_line(products)
    other = BandLines(band=lines.band, logT_grid=lines.logT_grid + 0.01,
                      logN_grid=lines.logN_grid, abundance=lines.abundance,
                      hdf5_dbase_root="test", ions=lines.ions)
    with pytest.raises(ValueError, match="another grid of temperature"):
        full_ccd_frame(uniform_synthesis, other, 0.3 * u.Mm, 1 * u.s, n_workers=1)
    # Half a step off the synthesis's densities.
    shifted = _one_line(products, products["logN_grid"] + 0.05)
    with pytest.raises(ValueError, match="not all on the band's grid"):
        full_ccd_frame(uniform_synthesis, shifted, 0.3 * u.Mm, 1 * u.s, n_workers=1)
    feldman = BandLines(band=lines.band, logT_grid=lines.logT_grid,
                        logN_grid=lines.logN_grid, abundance="sun_coronal_1992_feldman",
                        hdf5_dbase_root="test", ions=lines.ions)
    with pytest.raises(ValueError, match="abundances"):
        full_ccd_frame(uniform_synthesis, feldman, 0.3 * u.Mm, 1 * u.s, n_workers=1)
    wavelength = np.linspace(190.0, 200.0, 11) * u.AA
    continuum = (wavelength, np.zeros((lines.logT_grid.size, 11)),
                 np.zeros((lines.logN_grid.size - 1, lines.logT_grid.size, 11)))
    with pytest.raises(ValueError, match="work it out on the band's grids"):
        full_ccd_frame(uniform_synthesis, lines, 0.3 * u.Mm, 1 * u.s, continuum=continuum,
                       n_workers=1)
    descending = (wavelength[::-1], np.zeros((lines.logT_grid.size, 11)),
                  np.zeros((lines.logN_grid.size, lines.logT_grid.size, 11)))
    with pytest.raises(ValueError, match="finite and increase"):
        full_ccd_frame(uniform_synthesis, lines, 0.3 * u.Mm, 1 * u.s, continuum=descending,
                       n_workers=1)
    for settings in ({"exposure": -1 * u.s}, {"exposure": 1.0}, {"slit_width": 0 * u.arcsec}):
        settings = {"exposure": 1 * u.s, **settings}
        with pytest.raises(ValueError, match="above zero"):
            full_ccd_frame(uniform_synthesis, lines, 0.3 * u.Mm, **settings, n_workers=1)
    with pytest.raises(ValueError, match="one position"):
        full_ccd_frame(uniform_synthesis, lines, [0.3, 0.4] * u.Mm, 1 * u.s, n_workers=1)
    smaller = FocalPlane_SWC.from_detector(Detector_SWC(n_columns=1024))
    with pytest.raises(ValueError, match="they must match"):
        full_ccd_frame(uniform_synthesis, lines, 0.3 * u.Mm, 1 * u.s, focal_plane=smaller,
                       n_workers=1)
    tilted = FocalPlane_SWC.from_detector(Detector_SWC(), slit_image_tilt=(0.5, 0.0))
    with pytest.raises(ValueError, match="tilted"):
        full_ccd_frame(uniform_synthesis, lines, 0.3 * u.Mm, 1 * u.s, focal_plane=tilted,
                       n_workers=1)
    # The detector and the focal plane must agree on the focal plane settings.
    shifted = FocalPlane_SWC.from_detector(Detector_SWC(), wavelength_offset=0.1 * u.AA)
    with pytest.raises(ValueError, match="different wavelength_offset"):
        full_ccd_frame(uniform_synthesis, lines, 0.3 * u.Mm, 1 * u.s, focal_plane=shifted,
                       n_workers=1)
    det = Detector_SWC(dispersion=(199.0, 1.25), lit_band=(171 * u.AA, 211 * u.AA))
    with pytest.raises(ValueError, match="different dispersion, lit_band"):
        full_ccd_frame(uniform_synthesis, lines, 0.3 * u.Mm, 1 * u.s, det=det,
                       focal_plane=FocalPlane_SWC(), n_workers=1)


def test_each_pixel_takes_the_wavelength_of_the_photons_reaching_it(uniform_synthesis,
                                                                    blurred_frame):
    from scipy.stats import norm

    from eclipse.frame import _response

    products = read_synthesis_products(uniform_synthesis)
    det, tel = Detector_SWC(), Telescope_EUVST(resolution_budget="none")
    frame = blurred_frame
    focal_plane = FocalPlane_SWC.from_detector(det)
    middle = focal_plane.n_columns // 2
    peak = int(np.argmax(frame.rate["left"][:, middle]))
    edges = focal_plane.row_edges("left").to_value(u.AA)
    row = peak + 3
    low, high = sorted(edges[row:row + 2])
    # The plasma is at one temperature, so the line has one thermal width.
    (t,) = np.flatnonzero(products["em_tv"][0, 0].sum(axis=-1))
    thermal = REST * np.sqrt(const.k_B * 10 ** products["logT_grid"][t] * u.K
                             / (55.845 * const.u)).to_value(u.km / u.s) / const.c.to_value(u.km / u.s)
    psf, _ = _response(focal_plane, "left", None, tel, det, 0.2 * u.arcsec, "quadrature",
                       np.array([REST]))
    total = np.hypot(thermal, psf[0])
    # The row three away gets the photons the blur brings from the line,
    # more of them from its nearer side. Their mean wavelength is the line's,
    # moved toward the row by the thermal share of the line's spread.
    a, b = (low - REST) / total, (high - REST) / total
    landed = REST + total * (norm.pdf(a) - norm.pdf(b)) / (norm.cdf(b) - norm.cdf(a))
    expected = REST + (thermal / total) ** 2 * (landed - REST)
    reaching = frame.photon_wavelength["left"][row, middle].to_value(u.AA)
    assert frame.wavelength["left"][row].to_value(u.AA) == pytest.approx((low + high) / 2)
    assert reaching == pytest.approx(expected, abs=0.01 * abs(expected - REST))
    assert abs(reaching - REST) < 0.5 * abs((low + high) / 2 - REST)
    # The root mean square energy gives the spread of their wavelengths: the
    # thermal share of the blur's spread, plus how the place they land in
    # the row moves their mean.
    z = norm.cdf(b) - norm.cdf(a)
    landing = total**2 * (1 + (a * norm.pdf(a) - b * norm.pdf(b)) / z
                          - ((norm.pdf(a) - norm.pdf(b)) / z) ** 2)
    spread = (thermal * psf[0] / total) ** 2 + (thermal / total) ** 4 * landing
    rms = frame.photon_rms_wavelength["left"][row, middle].to_value(u.AA)
    assert 1 / rms**2 - 1 / reaching**2 == pytest.approx(spread / expected**4, rel=2e-2)


@pytest.mark.parametrize("psf", [False, True])
def test_the_continuum_lands_on_the_rows_as_photons_from_spectrum_lays_it(uniform_synthesis,
                                                                           psf):
    from eclipse.frame import photons_from_spectrum

    products = read_synthesis_products(uniform_synthesis)
    grid = products["logN_grid"]
    step = grid[1] - grid[0]
    wider = np.concatenate([grid[0] - step * np.arange(3, 0, -1), grid,
                            grid[-1] + step * np.arange(1, 4)])
    lines = _one_line(products, wider)
    lines.ions[0].g[:] = 0.0
    wavelength = np.linspace(194.0, 196.0, 801)
    n_t = lines.logT_grid.size
    free = np.tile(1e-30 * wavelength / REST, (n_t, 1))
    # Two-photon emission differs at each density of the band's grid, so only
    # the synthesis's own density gives the expected frame.
    two_photon = (1e-30 * (1 + np.arange(wider.size))[:, None, None]
                  * (REST / wavelength) ** 2 * np.ones((1, n_t, 1)))
    frame = full_ccd_frame(uniform_synthesis, lines, 0.3 * u.Mm, 1 * u.s, psf=psf,
                           continuum=(wavelength * u.AA, free, two_photon), noise=False,
                           n_workers=1, telescope=Telescope_EUVST(resolution_budget="none"))
    det, tel = Detector_SWC(), Telescope_EUVST(resolution_budget="none")
    focal_plane = FocalPlane_SWC.from_detector(det)
    # A column in the middle of the scene sees one pixel's emission measure,
    # all at 1e9 cm^-3, from the DEM per unit log T. The continuum is per unit
    # emission measure and per cm.
    logT = products["logT_grid"]
    em = products["dem_map"][0, 0] * (logT[1] - logT[0])            # (nT,)
    density = int(np.argmin(np.abs(wider - 9.0)))
    radiance = (em[:, None] * (free + two_photon[density])).sum(axis=0) * 1e-8
    expected = photons_from_spectrum(
        focal_plane, "left", tel, 0.2 * u.arcsec, wavelength * u.AA,
        radiance * u.erg / (u.s * u.cm**2 * u.sr * u.AA), det=det,
        spectral_psf="quadrature" if psf else None).to_value(1 / u.s)
    middle = focal_plane.n_columns // 2
    assert np.allclose(frame.rate["left"][:, middle], expected, rtol=1e-9,
                       atol=1e-12 * expected.max())
    assert expected.sum() > 0


def test_the_workers_give_the_frame_one_process_gives(two_temperature_synthesis, monkeypatch):
    import multiprocessing

    products = read_synthesis_products(two_temperature_synthesis)
    lines = _one_line(products)
    settings = {"psf": False, "noise": False}
    one = full_ccd_frame(two_temperature_synthesis, lines, 0.3 * u.Mm, 1 * u.s, n_workers=1,
                         **settings)
    two = full_ccd_frame(two_temperature_synthesis, lines, 0.3 * u.Mm, 1 * u.s, n_workers=2,
                         **settings)
    assert np.count_nonzero(products["em_tv"][0, 0].sum(axis=-1)) > 1
    assert one.rate["left"].any()
    for name in ("rate", "photon_wavelength", "photon_rms_wavelength"):
        assert np.array_equal(u.Quantity(getattr(two, name)["left"]).value,
                              u.Quantity(getattr(one, name)["left"]).value)
    # Where processes cannot be forked, one process computes every temperature.
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    with pytest.warns(UserWarning, match="cannot be forked"):
        alone = full_ccd_frame(two_temperature_synthesis, lines, 0.3 * u.Mm, 1 * u.s,
                               n_workers=2, **settings)
    assert np.array_equal(alone.rate["left"], one.rate["left"])


def test_the_continuum_holds_plasma_faster_than_the_velocity_grid(uniform_synthesis,
                                                                  fast_synthesis):
    products = read_synthesis_products(uniform_synthesis)
    fast = read_synthesis_products(fast_synthesis)
    # The fast plasma is beyond the velocity grid, so it has no lines.
    assert not fast["em_tv"].any() and products["em_tv"].any()
    lines = _one_line(products)
    lines.ions[0].g[:] = 0.0
    wavelength = np.linspace(194.0, 196.0, 201)
    n_t, n_n = lines.logT_grid.size, lines.logN_grid.size
    continuum = (wavelength * u.AA, np.full((n_t, wavelength.size), 1e-30),
                 np.full((n_n, n_t, wavelength.size), 1e-30))
    settings = {"psf": False, "noise": False, "continuum": continuum, "n_workers": 1}
    at_rest = full_ccd_frame(uniform_synthesis, lines, 0.3 * u.Mm, 1 * u.s, **settings)
    moving = full_ccd_frame(fast_synthesis, lines, 0.3 * u.Mm, 1 * u.s, **settings)
    assert at_rest.rate["left"].any()
    assert np.allclose(moving.rate["left"], at_rest.rate["left"], rtol=1e-12, atol=0)


def test_a_frame_takes_its_settings_from_a_simulation(uniform_synthesis, blurred_frame):
    """A `Simulation` with the keyword arguments' values gives the same frame."""
    products = read_synthesis_products(uniform_synthesis)
    sim = Simulation(slit_width=0.2 * u.arcsec, psf=True, noise=False)
    frame = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1000 * u.s,
                           telescope=Telescope_EUVST(resolution_budget="none"), sim=sim,
                           n_workers=1)
    _same_frames(frame, blurred_frame)


def _same_frames(a, b):
    assert a.settings == b.settings
    assert a.slit_width == b.slit_width
    for name in ("rate", "photons", "dn", "wavelength", "photon_wavelength",
                 "photon_rms_wavelength"):
        for ccd in ("left", "right"):
            assert np.array_equal(u.Quantity(getattr(a, name)[ccd]).value,
                                  u.Quantity(getattr(b, name)[ccd]).value)


def test_a_frame_from_a_non_default_simulation_matches_the_keyword_arguments(
        uniform_synthesis):
    """With the same seed, noise, convolution, a dark boundary and a 0.4 arcsec slit give the
    same frame."""
    products = read_synthesis_products(uniform_synthesis)
    lines = _one_line(products)
    common = dict(telescope=Telescope_EUVST(resolution_budget="none"), n_workers=1)
    sim = Simulation(slit_width=0.4 * u.arcsec, psf=True, psf_boundary="zero",
                     spectral_psf="convolution", noise=True)
    np.random.seed(170)
    from_sim = full_ccd_frame(uniform_synthesis, lines, 0.3 * u.Mm, 10 * u.s, sim=sim, **common)
    np.random.seed(170)
    from_keywords = full_ccd_frame(uniform_synthesis, lines, 0.3 * u.Mm, 10 * u.s,
                                   slit_width=0.4 * u.arcsec, spectral_psf="convolution",
                                   psf_boundary="zero", **common)
    assert from_sim.slit_width == 0.4 * u.arcsec
    assert from_sim.settings["spectral_psf"] == "convolution" and from_sim.settings["noise"]
    assert from_sim.settings["psf_boundary"] == "zero"
    _same_frames(from_sim, from_keywords)


def test_psf_none_gives_the_frame_without_the_psf(uniform_synthesis, sharp_frame):
    """`psf=None` is taken as false and recorded as given."""
    products = read_synthesis_products(uniform_synthesis)
    frame = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1 * u.s,
                           telescope=Telescope_EUVST(resolution_budget="none"), psf=None,
                           noise=False, n_workers=1)
    assert frame.settings["psf"] is None
    frame.settings["psf"] = False
    _same_frames(frame, sharp_frame)


def test_the_signature_keeps_its_plain_defaults():
    """The defaults read from the signature are those of a call without `sim`."""
    parameters = inspect.signature(full_ccd_frame).parameters
    assert parameters["slit_width"].default == 0.2 * u.arcsec
    assert parameters["psf"].default is True
    assert parameters["spectral_psf"].default == "quadrature"
    assert parameters["noise"].default is True
    assert parameters["n_workers"].default == 0
    assert parameters["sim"].default is None
    assert parameters["psf_boundary"].default == "replicate"
    assert parameters["psf_boundary"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["slit_mask"].default is False
    assert parameters["slit_mask"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["psf"].annotation == "bool"


@pytest.mark.parametrize("settings, message", [
    ({"enable_pinholes": True, "pinhole_sizes": [5 * u.um], "pinhole_positions": [0.5]},
     "pinholes"),
    ({"vis_sl": 1e3 * u.photon / (u.s * u.cm**2)}, "vis_sl"),
    ({"instrument": "EIS", "slit_width": 1 * u.arcsec}, "instrument 'EIS'"),
])
def test_a_frame_refuses_a_simulation_asking_for_what_it_does_not_model(settings, message):
    """The settings are checked before the synthesis is read, so no file is needed."""
    with pytest.raises(ValueError, match=message):
        full_ccd_frame("no_such_file.h5", None, 0.3 * u.Mm, 1 * u.s,
                       sim=Simulation(**settings))


# Arguments of `full_ccd_frame` that a `Simulation` sets, at their defaults.
_DEFAULTS = {"slit_width": 0.2 * u.arcsec, "psf": True, "spectral_psf": "quadrature",
             "noise": True, "psf_boundary": "replicate", "slit_mask": False}


def test_one_error_lists_every_unsupported_setting_and_conflict():
    sim = Simulation(psf=True, vis_sl=1 * u.photon / (u.s * u.cm**2))
    with pytest.raises(ValueError) as error:
        full_ccd_frame("no_such_file.h5", None, 0.3 * u.Mm, 1 * u.s, sim=sim,
                       noise=False, spectral_psf="convolution", psf_boundary="zero",
                       slit_mask=True)
    for name in ("psf_boundary", "slit_mask", "vis_sl", "noise", "spectral_psf"):
        assert name in str(error.value)


def test_a_simulation_field_the_frame_does_not_know_is_refused():
    """A field a subclass adds is refused whatever its value, as the frame cannot tell
    whether it changes the frame."""
    import dataclasses

    @dataclasses.dataclass
    class Extended(Simulation):
        new_effect: bool = dataclasses.field(default=False, kw_only=True)

    for sim in (Extended(), Extended(new_effect=True)):
        with pytest.raises(ValueError, match="new_effect"):
            _settings_from_simulation(sim, _DEFAULTS, 1 * u.s)


def test_a_subclass_cannot_hide_a_setting_by_changing_its_default():
    import dataclasses

    @dataclasses.dataclass
    class Lit(Simulation):
        vis_sl: u.Quantity = 1 * u.photon / (u.s * u.cm**2)
        expos: u.Quantity = 5 * u.s

    dark = 0 * u.photon / (u.s * u.cm**2)
    with pytest.raises(ValueError, match="vis_sl"):
        _settings_from_simulation(Lit(), _DEFAULTS, 1 * u.s)
    with pytest.warns(UserWarning, match="sim.expos"):
        _settings_from_simulation(Lit(vis_sl=dark), _DEFAULTS, 1 * u.s)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _settings_from_simulation(Lit(vis_sl=dark), _DEFAULTS, 5 * u.s)


def test_settings_a_frame_does_not_use_are_accepted():
    """Pinholes switched on without any, the PSF boundary with the PSF off, and the
    iterations and CPUs of a run change nothing in a frame."""
    sim = Simulation(enable_pinholes=True, psf_boundary="replicate", n_iter=3, ncpu=2,
                     pinhole_positions_spectral=[], pinhole_visible_wavelength=500 * u.nm)
    assert _settings_from_simulation(sim, _DEFAULTS, 1 * u.s)["psf"] is False


def test_a_simulation_gives_the_frame_its_slit_width_psf_and_noise():
    sim = Simulation(slit_width=0.4 * u.arcsec, psf=True, psf_boundary="zero",
                     spectral_psf="convolution", noise=False)
    assert _settings_from_simulation(sim, _DEFAULTS, 1 * u.s) == {
        "slit_width": 0.4 * u.arcsec, "psf": True, "spectral_psf": "convolution", "noise": False,
        "psf_boundary": "zero", "slit_mask": False}


def test_an_argument_set_both_ways_must_agree_with_the_simulation():
    """An argument at its default takes the `Simulation`'s value; one set otherwise must
    match it."""
    sim = Simulation(slit_width=0.4 * u.arcsec, psf=False, spectral_psf="convolution",
                     noise=False)
    expected = {"slit_width": 0.4 * u.arcsec, "psf": False, "spectral_psf": "convolution",
                "noise": False, "psf_boundary": "replicate", "slit_mask": False}
    assert _settings_from_simulation(sim, _DEFAULTS, 1 * u.s) == expected
    assert _settings_from_simulation(sim, {**_DEFAULTS, "slit_width": 400 * u.mas, "psf": False,
                                           "noise": False}, 1 * u.s) == expected
    for given in ({"slit_width": 0.8 * u.arcsec}, {"psf": None}, {"noise": None},
                  {"slit_width": None}, {"slit_width": [0.2] * u.arcsec}):
        with pytest.raises(ValueError, match=f"{next(iter(given))} is"):
            full_ccd_frame("no_such_file.h5", None, 0.3 * u.Mm, 1 * u.s, sim=sim, **given)
    with pytest.raises(TypeError, match="must be a Simulation"):
        full_ccd_frame("no_such_file.h5", None, 0.3 * u.Mm, 1 * u.s, sim={"psf": False})


def test_a_frame_warns_that_the_simulation_exposure_is_not_used():
    with pytest.warns(UserWarning, match="sim.expos"):
        _settings_from_simulation(Simulation(expos=5 * u.s), _DEFAULTS, 10 * u.s)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _settings_from_simulation(Simulation(expos=10 * u.s), _DEFAULTS, 10 * u.s)
        _settings_from_simulation(Simulation(), _DEFAULTS, 10 * u.s)


def test_the_frame_api_is_exported_from_the_package():
    """The frame API is importable from both the package and its modules."""
    import eclipse
    from eclipse import band, full_frame, readout

    for module, names in ((full_frame, ("full_ccd_frame", "write_full_frame", "FullFrame")),
                          (readout, ("FocalPlane_SWC", "ReadoutSequence")),
                          (band, ("BandLines", "IonLines", "band_contribution_functions",
                                  "read_band_lines", "write_band_lines"))):
        for name in names:
            assert getattr(eclipse, name) is getattr(module, name)
            assert name in eclipse.__all__


def _columns_in_the_scene(focal_plane, openings=None):
    """Fraction of each column along the slit inside the 4.4 Mm scene, centred on the field,
    and inside `openings` (arcsec) if given."""
    from eclipse.utils import distance_to_angle

    half = distance_to_angle(4.4 * u.Mm).to_value(u.arcsec) / 2
    edges = focal_plane.field_angle(np.arange(focal_plane.n_columns + 1) - 0.5).to_value(u.arcsec)
    inside = 0.0
    for low, high in [(-half, half)] if openings is None else openings:
        low, high = max(low, -half), min(high, half)
        inside = inside + np.clip(np.minimum(edges[1:], high) - np.maximum(edges[:-1], low),
                                  0.0, None)
    return inside / np.diff(edges)


def test_replicate_gives_the_ends_of_a_uniform_scene_the_middle_rate(blurred_frame):
    """With the Sun beyond the scene like its edges, the blur along the slit brings in as much
    light as it takes out, so every column inside the scene gets the middle column's rate and
    a column part inside gets that share of it."""
    focal_plane = FocalPlane_SWC.from_detector(Detector_SWC())
    rate = blurred_frame.rate["left"]
    fraction = _columns_in_the_scene(focal_plane)
    middle = rate[:, focal_plane.n_columns // 2]
    assert middle.sum() > 0 and 0 < fraction[fraction < 1].max()
    assert np.allclose(rate, middle[:, np.newaxis] * fraction, rtol=1e-9,
                       atol=1e-12 * middle.max())
    assert blurred_frame.settings["psf_boundary"] == "replicate"


def test_zero_takes_light_out_of_the_ends_of_the_scene(uniform_synthesis, blurred_frame):
    """With the Sun beyond the scene dark, the end columns lose light to the columns beyond,
    and the blur keeps the total."""
    products = read_synthesis_products(uniform_synthesis)
    frame = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1000 * u.s,
                           telescope=Telescope_EUVST(resolution_budget="none"), noise=False,
                           n_workers=1, psf_boundary="zero")
    focal_plane = FocalPlane_SWC.from_detector(Detector_SWC())
    fraction = _columns_in_the_scene(focal_plane)
    zero, replicate = frame.rate["left"].sum(axis=0), blurred_frame.rate["left"].sum(axis=0)
    middle = focal_plane.n_columns // 2
    assert zero[middle] == pytest.approx(replicate[middle], rel=1e-12)
    ends = np.flatnonzero(fraction == 1)[[0, -1]]
    assert np.all(zero[ends] < 0.8 * replicate[ends])
    beyond = np.flatnonzero(fraction > 0)[[0, -1]] + [-1, 1]
    assert np.all(zero[beyond] > 0) and not replicate[fraction == 0].any()
    assert zero.sum() == pytest.approx(replicate.sum(), rel=1e-6)
    assert frame.settings["psf_boundary"] == "zero"
    with pytest.raises(ValueError, match="psf_boundary must be"):
        full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1 * u.s,
                       psf_boundary="reflect")


def test_replicate_keeps_the_light_the_blur_across_the_slit_takes_past_the_scene(
        uniform_synthesis):
    """The scene is 0.8 arcsec across, so a 1 arcsec blur across the slit carries light past its
    edges. With the edges extended a uniform scene gives the slit what it gives without blur,
    even with the slit half past the edge."""
    from eclipse.full_frame import read_slit_scene

    sharp = read_slit_scene(uniform_synthesis, 0.3 * u.Mm, 0.2 * u.arcsec).emission_measure
    zero, replicate = (read_slit_scene(uniform_synthesis, 0.3 * u.Mm, 0.2 * u.arcsec,
                                       1.0 * u.arcsec, extend=extend).emission_measure
                       for extend in (False, True))
    assert sharp.sum() > 0
    assert np.allclose(replicate, sharp, rtol=1e-12, atol=0)
    assert zero.sum() < 0.95 * sharp.sum()
    edge = read_slit_scene(uniform_synthesis, 0.6 * u.Mm, 0.2 * u.arcsec, 1.0 * u.arcsec,
                           extend=True).emission_measure
    assert np.allclose(edge, sharp, rtol=1e-12, atol=0)


# A slit 4 arcsec long with fiducials 1 arcsec either side of its centre, so the 6 arcsec
# scene reaches both its ends and both fiducials.
_SHORT_SLIT = dict(resolution_budget="none", slit_length=4 * u.arcsec,
                   fiducial_centres=[-1 * u.arcsec, 1 * u.arcsec])


def test_the_slit_mask_darkens_the_columns_beyond_the_slit_and_behind_the_fiducials(
        uniform_synthesis, blurred_frame):
    """The blur comes before the slit, so in a uniform scene with its edges extended each column
    gets the middle column's rate times its open part inside the scene."""
    products = read_synthesis_products(uniform_synthesis)
    telescope = Telescope_EUVST(**_SHORT_SLIT)
    frame = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1000 * u.s,
                           telescope=telescope, noise=False, n_workers=1, slit_mask=True)
    focal_plane = FocalPlane_SWC.from_detector(Detector_SWC())
    openings = telescope.slit_openings(0.2 * u.arcsec).to_value(u.arcsec)
    fraction = _columns_in_the_scene(focal_plane, openings)
    # The blurred frame's middle column is open, as the slit's centre is between the fiducials.
    middle = blurred_frame.rate["left"][:, focal_plane.n_columns // 2]
    assert np.allclose(frame.rate["left"], middle[:, np.newaxis] * fraction, rtol=1e-9,
                       atol=1e-12 * middle.max())
    scene = _columns_in_the_scene(focal_plane)
    assert np.count_nonzero((fraction == 0) & (scene == 1)) >= 6
    assert np.count_nonzero((fraction < 1) & (fraction > 0)) == 6
    open_columns = fraction == 1
    assert np.array_equal(frame.rate["left"][:, open_columns],
                          blurred_frame.rate["left"][:, open_columns])
    assert frame.settings["slit_mask"] is True
    with pytest.raises(ValueError, match="known for the 0.2, 0.4, 0.8 and 1.6"):
        full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1 * u.s,
                       slit_width=0.3 * u.arcsec, slit_mask=True)
    with pytest.raises(ValueError, match="slit_mask must be True or False"):
        full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1 * u.s,
                       slit_mask="yes")


def test_a_simulation_with_the_slit_mask_and_a_dark_boundary_gives_the_keyword_frame(
        uniform_synthesis):
    products = read_synthesis_products(uniform_synthesis)
    common = dict(telescope=Telescope_EUVST(**_SHORT_SLIT), n_workers=1)
    sim = Simulation(psf=True, psf_boundary="zero", slit_mask=True, noise=False)
    from_sim = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1000 * u.s,
                              sim=sim, **common)
    from_keywords = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm,
                                   1000 * u.s, noise=False, psf_boundary="zero", slit_mask=True,
                                   **common)
    assert from_sim.settings["slit_mask"] is True and from_sim.settings["psf_boundary"] == "zero"
    _same_frames(from_sim, from_keywords)
    focal_plane = FocalPlane_SWC.from_detector(Detector_SWC())
    openings = common["telescope"].slit_openings(0.2 * u.arcsec).to_value(u.arcsec)
    edges = focal_plane.field_angle(np.arange(focal_plane.n_columns + 1) - 0.5).to_value(u.arcsec)
    middles = (edges[:-1] + edges[1:]) / 2
    closed = ~np.any((middles[:, None] > openings[:, 0] - 0.08)
                     & (middles[:, None] < openings[:, 1] + 0.08), axis=1)
    assert closed.sum() > 6 and not from_sim.rate["left"][:, closed].any()
    assert from_sim.rate["left"].any()


def test_a_frame_with_the_psf_and_noise_draws_its_photons(uniform_synthesis):
    """No rate is below zero, so the Poisson draw takes them all."""
    products = read_synthesis_products(uniform_synthesis)
    np.random.seed(0)
    frame = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1 * u.s,
                           telescope=Telescope_EUVST(resolution_budget="none"), n_workers=1)
    for ccd in ("left", "right"):
        assert (frame.rate[ccd] >= 0).all()
        lit = frame.rate[ccd] > 0
        image = frame.photon_wavelength[ccd][:lit.shape[0]]
        assert np.all(image[~lit] == np.broadcast_to(
            frame.wavelength[ccd].to_value(u.AA)[:, np.newaxis], lit.shape)[~lit] * u.AA)
