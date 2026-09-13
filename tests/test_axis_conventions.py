"""Pin down the cube axis conventions.

Observation cubes are (row, column, wavelength): rows run along the slit
(image vertical) and columns along the raster scan (image horizontal),
against a WCS whose FITS axes are (WAVE, HPLN, HPLT). Simulation volumes are
(z, y, x) against (SOLX, SOLY, SOLZ), so a horizontal slice is an image too.
Everything here would pass with the spatial axes swapped throughout as long
as the swap were consistent, so each test anchors one end of the chain to
something absolute: where a known feature lands in the array, which WCS
entry a physical pitch is written to, or which way a map comes out.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.analysis import create_sunpy_maps_from_combo
from euvst_response.config import Detector_SWC, Simulation, Telescope_EUVST
from euvst_response.data_processing import create_uniform_intensity_cube
from euvst_response.radiometric import apply_focusing_optics_psf
from euvst_response.synthesis import (
    apply_cube_cropping,
    create_atmosphere_ndcube,
    create_line_cube,
    load_cube,
    synthesise_spectra,
)
from euvst_response.utils import rebin_slit_offchip

REST = 195.119 * u.Angstrom


def test_load_cube_reads_the_file_into_z_y_x(tmp_path):
    """A voxel at file position (ix, iz, iy) must land at data[iz, iy, ix].

    The value planted in each voxel encodes its own position, so the check
    reads the transpose back out of the data. Distinct voxel sizes then pin
    which WCS entry each physical axis is written to.
    """
    nx, nz, ny = 3, 4, 5
    file_arr = np.empty((nx, nz, ny), dtype=np.float32)
    for ix in range(nx):
        for iz in range(nz):
            for iy in range(ny):
                file_arr[ix, iz, iy] = ix + 10 * iy + 100 * iz

    path = tmp_path / "eosT.0000000"
    file_arr.ravel(order="F").tofile(path)

    cube = load_cube(path, shape=(nx, nz, ny), unit=u.K,
                     voxel_dx=1.5 * u.Mm, voxel_dy=2.5 * u.Mm,
                     voxel_dz=3.5 * u.Mm, create_ndcube=True)

    assert cube.data.shape == (nz, ny, nx)
    for ix in range(nx):
        for iz in range(nz):
            for iy in range(ny):
                assert cube.data[iz, iy, ix] == ix + 10 * iy + 100 * iz

    assert list(cube.wcs.wcs.ctype) == ["SOLX", "SOLY", "SOLZ"]
    assert cube.wcs.wcs.cdelt[0] == pytest.approx(1.5)
    assert cube.wcs.wcs.cdelt[1] == pytest.approx(2.5)
    assert cube.wcs.wcs.cdelt[2] == pytest.approx(3.5)


def test_cropping_keeps_the_requested_world_box():
    """Crop points are world coordinates in (x, y, z) order, matching the WCS."""
    vol = np.zeros((6, 5, 4))
    tc, rc, vc = apply_cube_cropping(
        *(create_atmosphere_ndcube(vol * unit, voxel_dx=1.5 * u.Mm,
                                   voxel_dy=2.5 * u.Mm, voxel_dz=3.5 * u.Mm)
          for unit in (u.K, u.g / u.cm**3, u.cm / u.s)),
        crop_x=["-1.6 Mm", "1.6 Mm"],
        crop_y=["-2.6 Mm", "2.6 Mm"],
        crop_z=["0 Mm", "8 Mm"],
    )

    assert tc.data.shape == (3, 3, 4)
    zc, yc, xc = (tc.axis_world_coords(k)[0].to_value(u.Mm) for k in (0, 1, 2))
    assert np.allclose(xc, [-2.25, -0.75, 0.75, 2.25])
    assert np.allclose(yc, [-2.5, 0.0, 2.5])
    assert np.allclose(zc, [0.0, 3.5, 7.0])


def test_line_cube_is_row_column_wavelength():
    """A feature at scene (row j, column i) must land at cube.data[j, i],
    with X on FITS axis 2 of both the volume and the line cube.

    Built through synthesise_spectra with a hand-made emission measure whose
    value encodes its own position, so the check reads the position back out
    of the data rather than trusting any labelling. Distinct x and y voxel
    sizes then pin which spatial axis the WCS calls X.
    """
    nx, ny = 3, 5
    logT_grid = np.array([6.0, 6.2])
    vel_grid = np.array([-50.0e5, 0.0, 50.0e5]) * (u.cm / u.s)

    weight = 1.0 + 10.0 * np.arange(nx)[None, :] + 100.0 * np.arange(ny)[:, None]
    em_tv = np.zeros((ny, nx, len(logT_grid), len(vel_grid)))
    em_tv[:, :, 0, 1] = weight

    goft = {"Fe12_195.1190": {
        "wl0": REST.to(u.cm),
        "g": np.ones((ny, nx, len(logT_grid))),
        "atom": 26,
        "ion": 12,
    }}
    synthesise_spectra(goft, em_tv, vel_grid, logT_grid)

    reference = create_atmosphere_ndcube(
        np.zeros((2, ny, nx)) * u.K,
        voxel_dx=2.0 * u.Mm, voxel_dy=1.0 * u.Mm, voxel_dz=0.5 * u.Mm,
    )
    assert list(reference.wcs.wcs.ctype) == ["SOLX", "SOLY", "SOLZ"]
    assert reference.wcs.wcs.cdelt[0] == pytest.approx(2.0)
    assert reference.wcs.wcs.cdelt[2] == pytest.approx(0.5)

    cube = create_line_cube("Fe12_195.1190", goft["Fe12_195.1190"], reference,
                            u.erg / u.s / u.cm**2 / u.sr / u.cm,
                            integration_axis="z")

    assert cube.data.shape == (ny, nx, len(vel_grid))

    # The emission is linear in the emission measure, so the total intensity
    # image divided by the weight of the pixel it claims to be is constant
    # exactly when every value stayed at (row=j, col=i).
    image = cube.data.sum(axis=-1)
    ratio = image / weight
    assert np.allclose(ratio, ratio[0, 0], rtol=1e-12)

    # And the WCS says the same thing: FITS axis 2 is X with the 2 Mm pitch,
    # FITS axis 3 is Y with the 1 Mm pitch.
    assert list(cube.wcs.wcs.ctype) == ["WAVE", "SOLX", "SOLY"]
    assert cube.wcs.wcs.cdelt[1] == pytest.approx(2.0)
    assert cube.wcs.wcs.cdelt[2] == pytest.approx(1.0)


@pytest.mark.parametrize("axis,ctypes,cdelt_col,cdelt_row", [
    ("x", ["WAVE", "SOLY", "SOLZ"], 1.0, 0.5),
    ("y", ["WAVE", "SOLX", "SOLZ"], 2.0, 0.5),
])
def test_side_view_line_cubes_are_row_column_wavelength(axis, ctypes,
                                                        cdelt_col, cdelt_row):
    """The x and y integration views obey the same (row, column) layout.

    Looking along x or y, the image rows are height Z and the columns the
    surviving horizontal axis. Same position-encoding check as the top-down
    view, with the voxel sizes pinning which axis the WCS calls which.
    """
    nx, ny, nz = 3, 4, 5
    n_cols = ny if axis == "x" else nx
    n_rows = nz
    logT_grid = np.array([6.0, 6.2])
    vel_grid = np.array([-50.0e5, 0.0, 50.0e5]) * (u.cm / u.s)

    weight = (1.0 + 10.0 * np.arange(n_cols)[None, :]
              + 100.0 * np.arange(n_rows)[:, None])
    em_tv = np.zeros((n_rows, n_cols, len(logT_grid), len(vel_grid)))
    em_tv[:, :, 0, 1] = weight

    goft = {"Fe12_195.1190": {
        "wl0": REST.to(u.cm),
        "g": np.ones((n_rows, n_cols, len(logT_grid))),
        "atom": 26,
        "ion": 12,
    }}
    synthesise_spectra(goft, em_tv, vel_grid, logT_grid)

    reference = create_atmosphere_ndcube(
        np.zeros((nz, ny, nx)) * u.K,
        voxel_dx=2.0 * u.Mm, voxel_dy=1.0 * u.Mm, voxel_dz=0.5 * u.Mm,
    )
    cube = create_line_cube("Fe12_195.1190", goft["Fe12_195.1190"], reference,
                            u.erg / u.s / u.cm**2 / u.sr / u.cm,
                            integration_axis=axis)

    assert cube.data.shape == (n_rows, n_cols, len(vel_grid))
    image = cube.data.sum(axis=-1)
    ratio = image / weight
    assert np.allclose(ratio, ratio[0, 0], rtol=1e-12)

    assert list(cube.wcs.wcs.ctype) == ctypes
    assert cube.wcs.wcs.cdelt[1] == pytest.approx(cdelt_col)
    assert cube.wcs.wcs.cdelt[2] == pytest.approx(cdelt_row)


def test_uniform_cube_puts_the_slit_on_the_first_axis():
    """Slit pixels stack down axis 0, and the pitches land in the right WCS slots."""
    det = Detector_SWC()
    sim = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec)
    cube = create_uniform_intensity_cube(
        total_intensity=5000 * u.erg / (u.s * u.cm**2 * u.sr),
        rest_wavelength=REST,
        thermal_width=20 * u.km / u.s,
        det=det, sim=sim, n_slit_pixels=4,
    )

    assert cube.data.shape[0] == 4
    assert cube.data.shape[1] == 1
    assert list(cube.wcs.wcs.ctype) == ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    # HPLN (the scan direction) carries the slit width, HPLT the plate scale.
    assert cube.wcs.wcs.cdelt[1] == pytest.approx(0.4)
    assert cube.wcs.wcs.cdelt[2] == pytest.approx(
        (det.plate_scale_angle * u.pix).to_value(u.arcsec))


def _detector_wcs(n_spec, cdelt_x=1.0, cdelt_y=0.5):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [0.02, cdelt_x, cdelt_y]
    wcs.wcs.crpix = [n_spec / 2.0, 1.0, 1.0]
    wcs.wcs.crval = [REST.to_value(u.Angstrom), 0.0, 0.0]
    return wcs


def test_rebin_slit_offchip_sums_down_the_slit():
    """Binning by 2 adds slit-adjacent rows and doubles the HPLT pixel scale."""
    n_slit, n_scan, n_spec = 4, 2, 3
    data = np.arange(n_slit * n_scan * n_spec, dtype=float).reshape(
        (n_slit, n_scan, n_spec))
    cube = NDCube(data, wcs=_detector_wcs(n_spec, cdelt_x=1.0, cdelt_y=0.3),
                  unit=u.photon / u.pix, meta={"rest_wav": REST})

    out = rebin_slit_offchip(cube, 2)

    assert out.data.shape == (2, n_scan, n_spec)
    assert np.array_equal(out.data, data[0::2] + data[1::2])

    # The slit is HPLT, FITS axis 3: its pitch doubles, the scan pitch does
    # not. astropy normalises angles to degrees somewhere along the way, so
    # compare with the units attached rather than the raw cdelt numbers.
    def cdelt_arcsec(wcs, k):
        return (wcs.wcs.cdelt[k] * u.Unit(str(wcs.wcs.cunit[k]))).to_value(
            u.arcsec)

    assert cdelt_arcsec(out.wcs, 2) == pytest.approx(0.6)
    assert cdelt_arcsec(out.wcs, 1) == pytest.approx(1.0)


def test_psf_blurs_within_one_scan_position():
    """The focusing PSF acts on a detector frame, which is (slit, wavelength).

    Scan positions are exposed one after another, so no optical blur can move
    light between them. A point source at one scan position must spread along
    the slit and the spectral axis and leave the neighbouring scan positions
    dark.
    """
    n_slit, n_scan, n_spec = 9, 3, 33
    data = np.zeros((n_slit, n_scan, n_spec))
    data[4, 1, 16] = 1.0
    cube = NDCube(data, wcs=_detector_wcs(n_spec), unit=u.photon / u.pix,
                  meta={"rest_wav": REST})

    tel = Telescope_EUVST(psf_params=[2.0 * u.pix, 2.0 * u.pix])
    out = apply_focusing_optics_psf(cube, tel)

    assert out.data[:, 0, :].sum() == 0.0
    assert out.data[:, 2, :].sum() == 0.0
    assert out.data[3, 1, 16] > 0.0
    assert out.data[4, 1, 15] > 0.0
    # The kernel fits inside the frame here, so no flux leaves it either.
    assert out.data[:, 1, :].sum() == pytest.approx(1.0, rel=1e-12)


def test_maps_come_out_the_right_way_up():
    """A pixel at (row=j, col=i) of the cube is (row=j, col=i) of the map.

    The value planted at each pixel encodes its own position, so a transposed
    map cannot pass. The x pixel pitch is 2 arcsec against 0.5 in y, which
    also pins the WCS: SunPy reads its x scale from CDELT1, so the axes being
    merely consistently swapped would show up here as a 0.5 arcsec x scale.
    """
    ny, nx, n_spec = 5, 3, 4
    value = (10.0 * np.arange(ny)[:, None] + np.arange(nx)[None, :]) + 1.0
    data = np.repeat(value[:, :, np.newaxis], n_spec, axis=2)
    wcs = _detector_wcs(n_spec, cdelt_x=2.0, cdelt_y=0.5)

    fits = np.zeros((ny, nx, 4))
    fits[..., 0] = 1.0
    fits[..., 1] = REST.to_value(u.Angstrom)
    fits[..., 2] = 0.06
    units = [u.DN / u.pix, u.Angstrom, u.Angstrom, u.DN / u.pix]

    combination_results = {
        "first_signal_wcs": wcs,
        "first_photon_signal": NDCube(data, wcs=wcs, unit=u.photon / u.pix),
        "first_dn_signal": NDCube(data, wcs=wcs, unit=u.DN / u.pix),
        "dn_fit_stats": {
            "first_fit_data": fits,
            "mean_data": fits,
            "std_data": np.zeros_like(fits),
            "units": units,
        },
        "ground_truth": {
            "fit_truth_data": fits,
            "fit_truth_units": units,
        },
    }

    maps = create_sunpy_maps_from_combo(combination_results,
                                        rest_wavelength=REST,
                                        data_type="dn")

    total = maps["total_dn"]
    assert total.data.shape == (ny, nx)
    assert np.allclose(total.data, n_spec * value, rtol=1e-12)
    assert total.wcs.wcs.ctype[0].startswith("HPLN")
    assert total.scale[0].to_value(u.arcsec / u.pix) == pytest.approx(2.0)
    assert total.scale[1].to_value(u.arcsec / u.pix) == pytest.approx(0.5)
