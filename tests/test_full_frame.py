"""The full-CCD frame of the SW camera, from a synthesis and every line in the band."""
import sys

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from euvst_response import synthesis
from euvst_response.atmosphere import Atmosphere, write_atmosphere
from euvst_response.band import BandLines, IonLines
from euvst_response.config import Detector_SWC, Telescope_EUVST
from euvst_response.frame import photons_from_lines
from euvst_response.full_frame import (_density_shares, full_ccd_frame, lay_out_register,
                                       write_full_frame)
from euvst_response.readout import FocalPlane_SWC
from euvst_response.synthesis_file import read_synthesis_products

REST = 195.119
G0 = 1e-24  # erg cm3 / s, at every temperature and density


def _flat_goft(lines, **kwargs):
    logT_grid, logN_grid = np.linspace(5.0, 7.0, 21), np.linspace(8.0, 10.0, 21)
    return ({"Fe12_195.1190": {"wl0": (REST * u.AA).to(u.cm),
                               "g_tn": np.ones((logN_grid.size, logT_grid.size)),
                               "atom": 26, "ion": 12, "hdf5_dbase_root": None}},
            logT_grid, logN_grid)


@pytest.fixture(scope="module")
def uniform_synthesis(tmp_path_factory):
    """Synthesis of a uniform corona at 1 MK and 1e9 cm^-3, 6 arcsec along the slit, at rest."""
    tmp_path = tmp_path_factory.mktemp("frame")
    nz, ny, nx = 4, 44, 6
    density = (1.0e9 / u.cm**3 * 1.2 * const.u).to(u.g / u.cm**3)
    atmosphere = Atmosphere(
        temperature=np.full((nz, ny, nx), 1e6) * u.K,
        mass_density=np.full((nz, ny, nx), density.value) * density.unit,
        velocity_z=np.zeros((nz, ny, nx)) * u.km / u.s,
        x_edges=np.arange(nx + 1) * 0.1 * u.Mm, y_edges=np.arange(ny + 1) * 0.1 * u.Mm,
        z_edges=np.arange(nz + 1) * 0.1 * u.Mm)
    path = write_atmosphere(atmosphere, tmp_path / "atmosphere.h5")
    patch = pytest.MonkeyPatch()
    patch.setattr(sys, "argv", [
        "synthesise-spectra", "--atmosphere", str(path), "--output-dir", str(tmp_path),
        "--output-name", "uniform.h5", "--mass-per-electron", "1.2", "--lines", "Fe12_195.1190"])
    patch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    synthesis.main()
    patch.undo()
    return tmp_path / "uniform.h5"


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
def test_a_line_lands_on_the_rows_as_photons_from_lines_lays_it(uniform_synthesis, psf):
    products = read_synthesis_products(uniform_synthesis)
    frame = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1 * u.s,
                           psf=psf, noise=False, n_workers=1)
    det, tel = Detector_SWC(), Telescope_EUVST()
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


def test_the_frame_is_read_out_with_its_prescan_and_overscan(uniform_synthesis, tmp_path):
    products = read_synthesis_products(uniform_synthesis)
    det = Detector_SWC()
    frame = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1 * u.s,
                           noise=False, n_workers=1)
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


def test_the_prescan_and_overscan_hold_only_the_read_noise():
    det = Detector_SWC()
    image = np.full((5, 8), 7.0)
    laid = lay_out_register(image, det, noise=False)
    pre, over = det.serial_prescan, det.serial_overscan
    assert laid.shape == (5, 8 + 2 * (pre + over))
    assert np.all(laid[:, pre:pre + 4] == 7.0)
    assert np.all(laid[:, pre + 4 + 2 * over:pre + 8 + 2 * over] == 7.0)
    assert np.all(laid[:, :pre] == 0.0) and np.all(laid[:, -pre:] == 0.0)


def test_a_pixels_emission_measure_is_shared_between_the_densities_either_side():
    grid = np.array([8.0, 9.0, 10.0])
    lower, upper, low_share, high_share = _density_shares(np.array([8.25, 10.0, 11.0, np.nan]),
                                                          grid)
    assert list(lower[:2]) == [0, 2] and list(upper[:2]) == [1, 2]
    assert np.allclose(low_share, [0.75, 1.0, 0.0, 0.0])
    assert np.allclose(high_share, [0.25, 0.0, 0.0, 0.0])


def test_lines_on_a_wider_density_grid_give_the_same_frame(uniform_synthesis):
    products = read_synthesis_products(uniform_synthesis)
    grid = products["logN_grid"]
    step = grid[1] - grid[0]
    wider = np.concatenate([grid[0] - step * np.arange(3, 0, -1), grid,
                            grid[-1] + step * np.arange(1, 4)])
    settings = {"psf": False, "noise": False, "n_workers": 1}
    own = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1 * u.s,
                         **settings)
    frame = full_ccd_frame(uniform_synthesis,
                           _one_line(products, wider, g_off_the_synthesis=5 * G0),
                           0.3 * u.Mm, 1 * u.s, **settings)
    assert np.array_equal(frame.rate["left"], own.rate["left"])


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
    tilted = FocalPlane_SWC.from_detector(Detector_SWC(), slit_image_tilt=(0.5, 0.0))
    with pytest.raises(ValueError, match="tilted"):
        full_ccd_frame(uniform_synthesis, lines, 0.3 * u.Mm, 1 * u.s, focal_plane=tilted,
                       n_workers=1)


def test_each_pixel_takes_the_wavelength_of_the_photons_reaching_it(uniform_synthesis):
    from scipy.stats import norm

    from euvst_response.frame import _response

    products = read_synthesis_products(uniform_synthesis)
    det, tel = Detector_SWC(), Telescope_EUVST()
    # Long exposure, so the read-out smear adds little light.
    frame = full_ccd_frame(uniform_synthesis, _one_line(products), 0.3 * u.Mm, 1000 * u.s,
                           noise=False, n_workers=1)
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
