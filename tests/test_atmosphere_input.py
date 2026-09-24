"""An atmosphere from any code, given as one HDF5 file, is what the synthesis reads.

The synthesis needs temperature, density, the velocity along the line of
sight and the cell sizes. An ECLIPSE atmosphere file carries those with
units, so a simulation from any code can be fed in without a reader for it.
These check that the file round-trips, that it is validated, that MURaM's own
files, which the deprecated route still reads, synthesise as the same values
in a file do, that an electron density is used as given and otherwise
derived from the abundances, and that a stretched line of sight integrates
the true size of every cell.

The synthesis runs use a flat contribution function in place of fiasco, so
the intensity of a cell is just its emission measure.
"""
import sys

import astropy.constants as const
import astropy.units as u
import dill
import h5py
import numpy as np
import pytest
from mendeleev import element

from euvst_response import synthesis
from euvst_response.atmosphere import (
    Atmosphere,
    edges_from_centres,
    main as eclipse_atmosphere,
    mass_per_electron_from_abundances,
    read_atmosphere,
    write_atmosphere,
)
from euvst_response.synthesis import (
    along_line_of_sight,
    apply_cube_cropping,
    create_atmosphere_ndcube,
    load_cube,
)

LINE = "Fe12_195.1190"
REST = 195.119 * u.Angstrom
TEMPERATURE = 1.0e6 * u.K
ELECTRON_DENSITY = 1.0e9 / u.cm**3
MASS_PER_ELECTRON = 1.29
SHAPE = (6, 5, 4)  # (nz, ny, nx)
SPACING = {"x": 0.1 * u.Mm, "y": 0.15 * u.Mm, "z": 0.05 * u.Mm}
# Six cells of growing size, and four of growing size about zero.
STRETCHED_Z = np.array([0.0, 0.05, 0.2, 0.45, 0.8, 1.25, 1.75]) * u.Mm
STRETCHED_X = np.array([-0.25, -0.15, -0.05, 0.1, 0.3]) * u.Mm


def _edges(shape=SHAPE, spacing=SPACING):
    """Cell edges with x and y centred on zero and the bottom cell centred on z = 0."""
    nz, ny, nx = shape
    return {
        "x_edges": (np.arange(nx + 1) - nx / 2) * spacing["x"],
        "y_edges": (np.arange(ny + 1) - ny / 2) * spacing["y"],
        "z_edges": (np.arange(nz + 1) - 0.5) * spacing["z"],
    }


def _mass_density(electron_density=ELECTRON_DENSITY):
    return (electron_density * MASS_PER_ELECTRON * const.u).to(u.g / u.cm**3)


def _atmosphere(shape=SHAPE, **overrides):
    """A uniform coronal box with no flow, with any field replaced."""
    fields = {
        "temperature": np.full(shape, TEMPERATURE.value) * TEMPERATURE.unit,
        "mass_density": np.full(shape, _mass_density().value) * _mass_density().unit,
        "velocity_z": np.zeros(shape) * u.cm / u.s,
        **_edges(shape),
    }
    fields.update(overrides)
    return Atmosphere(**fields)


def _flat_goft(lines, **kwargs):
    """A contribution function of 1 everywhere, in place of fiasco."""
    logT_grid = np.linspace(5.0, 7.0, 21)
    logN_grid = np.linspace(8.0, 10.0, 21)
    goft = {LINE: {
        "wl0": REST.to(u.cm),
        "g_tn": np.ones((logN_grid.size, logT_grid.size)),
        "atom": 26,
        "ion": 12,
        "hdf5_dbase_root": None,
    }}
    return goft, logT_grid, logN_grid


def _synthesise(tmp_path, monkeypatch, name, *options):
    """Run synthesis main() with *options* and return what it saved."""
    argv = ["synthesise-spectra", "--output-dir", str(tmp_path / "out"),
            "--output-name", f"{name}.pkl", "--lines", LINE, *options]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    synthesis.main()
    with open(tmp_path / "out" / f"{name}.pkl", "rb") as f:
        return dill.load(f)


def _intensity(saved):
    """Wavelength-summed intensity of every pixel."""
    return saved["line_cubes"][LINE].data.sum(axis=-1)


def _world(cube, ctype):
    """The world coordinates in Mm along the axis of *cube* named *ctype*."""
    wcs = cube.wcs.wcs
    axis = list(wcs.ctype).index(ctype)
    assert wcs.cunit[axis] == "Mm"
    # The WCS axes run the other way round from the array's.
    n = cube.data.shape[::-1][axis]
    return wcs.crval[axis] + (np.arange(n) + 1 - wcs.crpix[axis]) * wcs.cdelt[axis]


def _write_muram_cube(path, data):
    """A cube in MURaM's own (nx, nz, ny) Fortran layout, from (nz, ny, nx)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(data, dtype=np.float32).transpose(2, 0, 1).ravel(order="F").tofile(path)


def _write_muram_files(root, atmosphere, suffix="0270000"):
    """*atmosphere* as the MURaM files the deprecated route and dynamic mode read."""
    _write_muram_cube(root / "temp" / f"eosT.{suffix}",
                      atmosphere.temperature.to_value(u.K))
    _write_muram_cube(root / "rho" / f"result_prim_0.{suffix}",
                      atmosphere.mass_density.to_value(u.g / u.cm**3))
    for axis, name in {"x": "vx/result_prim_1", "y": "vy/result_prim_3",
                       "z": "vz/result_prim_2"}.items():
        velocity = getattr(atmosphere, f"velocity_{axis}")
        if velocity is not None:
            _write_muram_cube(root / f"{name}.{suffix}", velocity.to_value(u.cm / u.s))


# ----------------------------------------------------------------------
# The file
# ----------------------------------------------------------------------
def test_a_file_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    atmosphere = _atmosphere(
        temperature=rng.uniform(5e5, 3e6, SHAPE) * u.K,
        electron_density=rng.uniform(1e8, 1e10, SHAPE) / u.cm**3,
        velocity_x=rng.normal(0, 10, SHAPE) * u.km / u.s,
        time=1234.5 * u.s, source="a test box")
    path = write_atmosphere(atmosphere, tmp_path / "box.h5")

    back = read_atmosphere(path)

    for name in ("temperature", "mass_density", "electron_density",
                 "velocity_x", "velocity_z", "x_edges", "y_edges", "z_edges", "time"):
        original = getattr(atmosphere, name)
        assert getattr(back, name).unit == original.unit
        assert np.array_equal(getattr(back, name).value, original.value)
    assert back.velocity_y is None
    assert back.source == "a test box"


def test_the_reader_reads_only_the_velocity_asked_for(tmp_path):
    atmosphere = _atmosphere(velocity_x=np.ones(SHAPE) * u.km / u.s)
    path = write_atmosphere(atmosphere, tmp_path / "box.h5")

    only_x = read_atmosphere(path, velocities=("x",))
    assert only_x.velocity_x is not None
    assert only_x.velocity_z is None

    with pytest.raises(ValueError, match="no velocity_y"):
        read_atmosphere(path, velocities=("y",))


def test_the_reader_refuses_files_that_are_not_atmospheres(tmp_path):
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f.create_dataset("temperature", data=np.ones(SHAPE))
    with pytest.raises(ValueError, match="not an ECLIPSE atmosphere file"):
        read_atmosphere(other)

    unitless = tmp_path / "unitless.h5"
    write_atmosphere(_atmosphere(), unitless)
    with h5py.File(unitless, "a") as f:
        del f["temperature"].attrs["unit"]
    with pytest.raises(ValueError, match="'temperature'.*no 'unit'"):
        read_atmosphere(unitless)

    newer = tmp_path / "newer.h5"
    write_atmosphere(_atmosphere(), newer)
    with h5py.File(newer, "a") as f:
        f.attrs["version"] = 99
    with pytest.raises(ValueError, match="version 99"):
        read_atmosphere(newer)


def test_the_version_must_be_the_one_this_eclipse_reads(tmp_path):
    for version, message in ((None, "no 'version'"), (1.5, "version 1.5"),
                             (2, "version 2"), ("one", "version one")):
        path = write_atmosphere(_atmosphere(), tmp_path / "box.h5")
        with h5py.File(path, "a") as f:
            if version is None:
                del f.attrs["version"]
            else:
                f.attrs["version"] = version
        with pytest.raises(ValueError, match=message):
            read_atmosphere(path)


def test_info_describes_a_file_without_loading_its_cubes(tmp_path, capsys):
    """The cubes are the size of the simulation; the description must not need them."""
    from euvst_response.atmosphere import describe_atmosphere_file
    path = tmp_path / "huge.h5"
    nz, ny, nx = SHAPE
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "eclipse-atmosphere"
        f.attrs["version"] = 1
        f.attrs["source"] = "a box whose cubes are elsewhere"
        for axis, edges in _edges().items():
            dataset = f.create_dataset(axis, data=edges.value)
            dataset.attrs["unit"] = str(edges.unit)
        # A dataset whose bytes live in a file that does not exist: its shape
        # and attributes read fine, its values cannot.
        missing = f.create_dataset("temperature", shape=SHAPE, dtype="f4",
                                   external=[(str(tmp_path / "missing.bin"), 0,
                                              4 * nz * ny * nx)])
        missing.attrs["unit"] = "K"
        density = f.create_dataset("mass_density", data=np.ones(SHAPE, dtype="f4"))
        density.attrs["unit"] = "g / cm3"

    description = describe_atmosphere_file(path)
    assert f"Shape (nz, ny, nx): {SHAPE}" in description
    assert "Density: mass_density" in description
    assert "Source: a box whose cubes are elsewhere" in description

    eclipse_atmosphere(["info", str(path)])
    assert f"Shape (nz, ny, nx): {SHAPE}" in capsys.readouterr().out

    with pytest.raises((OSError, RuntimeError)):
        read_atmosphere(path)


def test_the_mass_per_electron_must_be_finite_and_positive(tmp_path, monkeypatch):
    from euvst_response.atmosphere import require_mass_per_electron
    for bad in (0.0, -1.0, float("nan"), float("inf"), "many"):
        with pytest.raises(ValueError, match="finite, positive"):
            require_mass_per_electron(bad)
    assert require_mass_per_electron(1.16) == 1.16

    atmosphere = _atmosphere()
    with pytest.raises(ValueError, match="finite, positive"):
        atmosphere.electron_density_from(0.0)
    # Not needed, so not checked, when the electron density is given.
    given = _atmosphere(mass_density=None,
                        electron_density=np.ones(SHAPE) * ELECTRON_DENSITY)
    assert given.electron_density_from(0.0) is given.electron_density

    path = write_atmosphere(atmosphere, tmp_path / "box.h5")
    for bad in ("0", "-1.2", "nan"):
        with pytest.raises(ValueError, match="--mass-per-electron.*finite, positive"):
            _synthesise(tmp_path, monkeypatch, "bad", "--atmosphere", str(path),
                        "--mass-per-electron", bad)


def test_the_downsampling_factor_must_be_a_whole_number_of_one_or_more(tmp_path, monkeypatch):
    from euvst_response.utils import require_downsample_divides
    # True is equal to 1, so it has to be refused before any shortcut for 1.
    for bad in (0, -2, 2.0, True):
        with pytest.raises(ValueError, match="whole number of 1 or more"):
            require_downsample_divides((4, 4, 4), bad)
        with pytest.raises(ValueError, match="whole number of 1 or more"):
            _atmosphere((4, 4, 4)).downsampled(bad)
    require_downsample_divides((4, 4, 4), 2)

    path = write_atmosphere(_atmosphere(), tmp_path / "box.h5")
    with pytest.raises(ValueError, match="--downsample must be 1 or more"):
        _synthesise(tmp_path, monkeypatch, "bad", "--atmosphere", str(path),
                    "--downsample", "0")


@pytest.mark.parametrize("field, value, message", [
    ("temperature", None, "needs a temperature"),
    ("temperature", np.ones((4, 5, 6)) * u.K, "shape"),
    ("temperature", np.ones(SHAPE), "Quantity"),
    ("temperature", np.ones(SHAPE) * u.m, "convertible"),
    ("mass_density", None, "needs a mass_density or an electron_density"),
    ("z_edges", np.arange(SHAPE[0] + 1)[::-1] * u.Mm, "must increase"),
    ("x_edges", np.arange(2) * u.Mm, "at least 2 cells"),
    ("time", np.arange(2) * u.s, "0 dimensions"),
])
def test_an_atmosphere_checks_what_it_is_given(field, value, message):
    with pytest.raises((ValueError, TypeError, u.UnitConversionError), match=message):
        _atmosphere(**{field: value})


def test_a_velocity_the_atmosphere_lacks_is_asked_for_by_name():
    with pytest.raises(ValueError, match="no velocity_x"):
        _atmosphere().velocity("x")


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
def test_cell_sizes_come_from_the_edges():
    atmosphere = _atmosphere()
    for axis in ("x", "y", "z"):
        assert atmosphere.is_uniform(axis)
        assert atmosphere.spacing(axis).to_value(u.Mm) == pytest.approx(
            SPACING[axis].to_value(u.Mm))
        assert np.allclose(atmosphere.cell_thickness(axis).to_value(u.Mm),
                           SPACING[axis].to_value(u.Mm))
    assert np.allclose(atmosphere.coordinate("z").to_value(u.Mm),
                       np.arange(SHAPE[0]) * 0.05)
    assert np.allclose(atmosphere.coordinate("x").to_value(u.Mm),
                       [-0.15, -0.05, 0.05, 0.15])

    stretched = _atmosphere(z_edges=STRETCHED_Z)
    assert not stretched.is_uniform("z")
    thickness = stretched.cell_thickness("z").to_value(u.Mm)
    assert np.allclose(thickness, [0.05, 0.15, 0.25, 0.35, 0.45, 0.5])
    assert np.allclose(stretched.coordinate("z").to_value(u.Mm),
                       [0.025, 0.125, 0.325, 0.625, 1.025, 1.5])
    with pytest.raises(ValueError, match="evenly spaced"):
        stretched.spacing("z")


def test_edges_are_placed_halfway_between_centres():
    even = edges_from_centres(np.arange(4) * 0.5 * u.Mm)
    assert np.allclose(even.to_value(u.Mm), [-0.25, 0.25, 0.75, 1.25, 1.75])

    stretched = edges_from_centres(np.array([0.0, 1.0, 3.0]) * u.Mm)
    assert np.allclose(stretched.to_value(u.Mm), [-0.5, 0.5, 2.0, 4.0])
    assert stretched.unit == u.Mm

    with pytest.raises(ValueError, match="At least 2"):
        edges_from_centres(np.array([1.0]) * u.Mm)


@pytest.mark.parametrize("crop_x, crop_y, crop_z", [
    (["-0.16 Mm", "0.16 Mm"], ["-0.26 Mm", "0.26 Mm"], ["0 Mm", "0.2 Mm"]),
    (["-0.14 Mm", "0.14 Mm"], None, ["0.12 Mm", "0.3 Mm"]),
    (["-0.05 Mm", "0.05 Mm"], ["-0.15 Mm", "0.15 Mm"], None),
    (None, None, ["0.024 Mm", "0.026 Mm"]),
])
def test_cropping_keeps_the_cells_ndcube_cropping_keeps(crop_x, crop_y, crop_z):
    """The same crop options select the same cells on either route."""
    atmosphere = _atmosphere()
    cubes = [create_atmosphere_ndcube(np.zeros(SHAPE) * unit, **{
        f"voxel_d{axis}": SPACING[axis] for axis in "xyz"})
        for unit in (u.K, u.g / u.cm**3, u.cm / u.s)]
    reference, _, _ = apply_cube_cropping(*cubes, crop_x, crop_y, crop_z)

    cropped = atmosphere.cropped(x=crop_x, y=crop_y, z=crop_z)

    assert cropped.shape == reference.data.shape
    for k, axis in enumerate(("z", "y", "x")):
        expected = reference.axis_world_coords(k)[0].to_value(u.Mm)
        assert np.allclose(cropped.coordinate(axis).to_value(u.Mm), expected)


def test_cropping_a_stretched_axis_keeps_every_cell_the_range_touches():
    atmosphere = _atmosphere(z_edges=STRETCHED_Z)
    # This range touches the cells spanning 0.2 to 0.45 and 0.45 to 0.8 and
    # no others.
    cropped = atmosphere.cropped(z=("0.25 Mm", "0.5 Mm"))
    assert np.allclose(cropped.z_edges.to_value(u.Mm), [0.2, 0.45, 0.8])
    assert cropped.shape == (2, SHAPE[1], SHAPE[2])
    with pytest.raises(ValueError, match="No cells lie within"):
        atmosphere.cropped(z=("2 Mm", "3 Mm"))
    with pytest.raises(ValueError, match="low < high"):
        atmosphere.cropped(z=("1 Mm", "0 Mm"))


def test_a_bound_on_a_cell_boundary_does_not_keep_that_cell_on_rounding():
    """Round bounds land on round grids; rounding noise must not decide the cell."""
    nudged = _edges()["x_edges"] + 1e-13 * u.Mm
    atmosphere = _atmosphere(x_edges=nudged)
    cropped = atmosphere.cropped(x=("-0.1 Mm", "0.1 Mm"))
    assert cropped.shape[2] == 2
    assert np.allclose(cropped.x_edges.to_value(u.Mm), [-0.1, 0.0, 0.1])
    other_way = _atmosphere(x_edges=_edges()["x_edges"] - 1e-13 * u.Mm)
    assert other_way.cropped(x=("-0.1 Mm", "0.1 Mm")).shape[2] == 2


def test_downsampling_keeps_the_extent_of_the_box():
    # y has 5 cells, which 2 does not divide.
    with pytest.raises(ValueError, match="does not divide"):
        _atmosphere().downsampled(2)

    even = _atmosphere((6, 4, 4))
    coarse = even.downsampled(2)
    assert coarse.shape == (3, 2, 2)
    for axis in ("x", "y", "z"):
        assert coarse.spacing(axis).to_value(u.Mm) == pytest.approx(
            2 * SPACING[axis].to_value(u.Mm))
        assert np.allclose(coarse.edges(axis).to_value(u.Mm),
                           even.edges(axis).to_value(u.Mm)[::2])
    # The kept cells stand for the blocks they start, so they take the
    # blocks' boundaries and sit at the blocks' centres.
    assert np.allclose(coarse.coordinate("x").to_value(u.Mm), [-0.1, 0.1])
    assert np.allclose(coarse.coordinate("z").to_value(u.Mm), [0.025, 0.125, 0.225])
    assert even.downsampled(1) is even


def test_downsampling_a_stretched_axis_adds_up_the_cells():
    atmosphere = _atmosphere((6, 4, 4), z_edges=STRETCHED_Z)
    fine = atmosphere.cell_thickness("z").to_value(u.Mm)
    coarse = atmosphere.downsampled(2)
    assert np.allclose(coarse.cell_thickness("z").to_value(u.Mm),
                       fine.reshape(3, 2).sum(axis=1))


def test_along_line_of_sight_lays_the_sizes_along_the_right_axis():
    sizes = np.array([1.0, 2.0, 3.0])
    assert along_line_of_sight(sizes, "z").shape == (3, 1, 1)
    assert along_line_of_sight(sizes, "y").shape == (1, 3, 1)
    assert along_line_of_sight(sizes, "x").shape == (1, 1, 3)
    assert along_line_of_sight(2.5, "z").shape == ()
    with pytest.raises(ValueError, match="one value per cell"):
        along_line_of_sight(np.ones((2, 2)), "z")


# ----------------------------------------------------------------------
# Mass per electron
# ----------------------------------------------------------------------
def test_the_mass_per_electron_follows_from_the_abundances():
    hydrogen = element("H").atomic_weight
    assert mass_per_electron_from_abundances({"H": 1.0}) == pytest.approx(hydrogen)

    helium = element("He").atomic_weight
    expected = (hydrogen + 0.085 * helium) / (1.0 + 0.085 * 2)
    assert mass_per_electron_from_abundances({"H": 1.0, "He": 0.085}) == pytest.approx(expected)
    # Well below the neutral-gas 1.29 that ECLIPSE 0.8.0 used, and well
    # above the 1 a pure hydrogen plasma would have.
    assert 1.1 < expected < 1.2

    with pytest.raises(ValueError, match="hydrogen"):
        mass_per_electron_from_abundances({"He": 0.085})


# ----------------------------------------------------------------------
# The synthesis
# ----------------------------------------------------------------------
def _structured_atmosphere():
    """Temperature, density and flow that vary with position."""
    nz, ny, nx = SHAPE
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    density = _mass_density() * (1.0 + 0.1 * i + 0.05 * j + 0.02 * k)
    velocity = (10.0 * (i - 1.5) + 5.0 * (j - 2)) * u.km / u.s
    return _atmosphere(
        temperature=np.full(SHAPE, TEMPERATURE.value) * (1.0 + 0.1 * k) * u.K,
        mass_density=density,
        velocity_z=velocity)


def test_the_units_in_the_file_do_not_matter(tmp_path, monkeypatch):
    """The same atmosphere in SI units gives the same spectra, flows included."""
    atmosphere = _structured_atmosphere()
    cgs = write_atmosphere(atmosphere, tmp_path / "cgs.h5")
    si = write_atmosphere(Atmosphere(
        temperature=atmosphere.temperature.to(u.MK),
        mass_density=atmosphere.mass_density.to(u.kg / u.m**3),
        velocity_z=atmosphere.velocity_z.to(u.m / u.s),
        x_edges=atmosphere.x_edges.to(u.km), y_edges=atmosphere.y_edges.to(u.m),
        z_edges=atmosphere.z_edges.to(u.km)), tmp_path / "si.h5")

    from_cgs = _synthesise(tmp_path, monkeypatch, "cgs", "--atmosphere", str(cgs),
                           "--mass-per-electron", str(MASS_PER_ELECTRON))
    from_si = _synthesise(tmp_path, monkeypatch, "si", "--atmosphere", str(si),
                          "--mass-per-electron", str(MASS_PER_ELECTRON))

    cgs_cube = from_cgs["line_cubes"][LINE]
    si_cube = from_si["line_cubes"][LINE]
    assert np.all(_intensity(from_cgs) > 0)
    # The flows shift the lines, so the spectra only agree if the velocities
    # were read in the right unit; the summed intensity would not tell.
    assert not np.allclose(cgs_cube.data[0, 0], cgs_cube.data[0, -1])
    assert si_cube.data == pytest.approx(cgs_cube.data, rel=1e-6)
    for k in range(3):
        assert np.allclose(si_cube.axis_world_coords(k)[0].value,
                           cgs_cube.axis_world_coords(k)[0].value)
    assert from_si["dem_map"] == pytest.approx(from_cgs["dem_map"], rel=1e-6)


def test_an_electron_density_is_used_as_given(tmp_path, monkeypatch):
    """With the electron density in the file, no mass per electron is needed."""
    with_mass = write_atmosphere(_atmosphere(), tmp_path / "mass.h5")
    with_electrons = write_atmosphere(
        _atmosphere(mass_density=None,
                    electron_density=np.full(SHAPE, ELECTRON_DENSITY.value) * ELECTRON_DENSITY.unit),
        tmp_path / "electrons.h5")

    def never(*args, **kwargs):
        raise AssertionError("the abundances were not needed")
    monkeypatch.setattr(synthesis, "mass_per_electron", never)

    from_electrons = _synthesise(tmp_path, monkeypatch, "electrons",
                                 "--atmosphere", str(with_electrons))
    from_mass = _synthesise(tmp_path, monkeypatch, "mass", "--atmosphere",
                            str(with_mass), "--mass-per-electron", str(MASS_PER_ELECTRON))

    assert np.all(_intensity(from_electrons) > 0)
    assert _intensity(from_electrons) == pytest.approx(_intensity(from_mass), rel=1e-6)
    assert from_electrons["config"]["mass_per_electron"] is None
    assert "electron density" in from_electrons["config"]["mass_per_electron_source"]
    assert from_electrons["atmosphere"]["electron_density_given"] is True


def test_the_mass_per_electron_comes_from_the_abundances_by_default(tmp_path, monkeypatch):
    path = write_atmosphere(_atmosphere(), tmp_path / "box.h5")
    asked = []

    def from_abundances(abundance, hdf5_dbase_root=None):
        asked.append((abundance, hdf5_dbase_root))
        return 1.17
    monkeypatch.setattr(synthesis, "mass_per_electron", from_abundances)

    derived = _synthesise(tmp_path, monkeypatch, "derived", "--atmosphere",
                          str(path), "--abundance", "sun_photospheric_2021_asplund")
    given = _synthesise(tmp_path, monkeypatch, "given", "--atmosphere",
                        str(path), "--mass-per-electron", str(MASS_PER_ELECTRON))

    assert asked == [("sun_photospheric_2021_asplund", None)]
    assert derived["config"]["mass_per_electron"] == 1.17
    assert "sun_photospheric_2021_asplund" in derived["config"]["mass_per_electron_source"]
    # The intensity goes as the electron density squared, so as the inverse
    # square of the mass per electron.
    assert _intensity(derived) == pytest.approx(
        _intensity(given) * (MASS_PER_ELECTRON / 1.17) ** 2, rel=1e-6)


def test_the_old_option_name_still_sets_the_mass_per_electron(tmp_path, monkeypatch):
    path = write_atmosphere(_atmosphere(), tmp_path / "box.h5")
    saved = _synthesise(tmp_path, monkeypatch, "old", "--atmosphere", str(path),
                        "--mean-mol-wt", "1.2")
    assert saved["config"]["mass_per_electron"] == 1.2
    assert saved["config"]["mass_per_electron_source"] == "given on the command line"


def test_a_stretched_line_of_sight_integrates_the_true_cell_sizes(tmp_path, monkeypatch):
    """The emission measure of a uniform column is its density squared times its depth."""
    stretched = _atmosphere(z_edges=STRETCHED_Z)
    path = write_atmosphere(stretched, tmp_path / "stretched.h5")

    saved = _synthesise(tmp_path, monkeypatch, "stretched", "--atmosphere",
                        str(path), "--mass-per-electron", str(MASS_PER_ELECTRON))

    depth_cm = stretched.cell_thickness("z").sum().to_value(u.cm)
    logT_grid = saved["logT_grid"]
    emission_measure = saved["dem_map"].sum(axis=-1) * (logT_grid[1] - logT_grid[0])
    assert emission_measure == pytest.approx(ELECTRON_DENSITY.value ** 2 * depth_cm, rel=1e-6)

    uniform = _synthesise(tmp_path, monkeypatch, "uniform", "--atmosphere",
                          str(write_atmosphere(_atmosphere(), tmp_path / "uniform.h5")),
                          "--mass-per-electron", str(MASS_PER_ELECTRON))
    uniform_depth = _atmosphere().cell_thickness("z").sum().to_value(u.cm)
    assert _intensity(saved) == pytest.approx(
        _intensity(uniform) * depth_cm / uniform_depth, rel=1e-6)

    assert saved["atmosphere"]["nonuniform_axes"] == ["z"]
    assert saved["voxel_sizes"]["dz"] is None
    assert saved["voxel_sizes"]["dx"].to_value(u.Mm) == pytest.approx(
        SPACING["x"].to_value(u.Mm))


def test_a_stretched_image_axis_is_refused(tmp_path, monkeypatch):
    path = write_atmosphere(_atmosphere(x_edges=STRETCHED_X,
                                        velocity_x=np.zeros(SHAPE) * u.km / u.s),
                            tmp_path / "stretched_x.h5")
    with pytest.raises(ValueError, match="x axis.*not evenly spaced"):
        _synthesise(tmp_path, monkeypatch, "refused", "--atmosphere", str(path),
                    "--mass-per-electron", str(MASS_PER_ELECTRON))
    # Along x it is the line of sight, which may be stretched.
    saved = _synthesise(tmp_path, monkeypatch, "side", "--atmosphere", str(path),
                        "--integration-axis", "x",
                        "--mass-per-electron", str(MASS_PER_ELECTRON))
    assert np.all(_intensity(saved) > 0)


def test_the_synthesis_crops_and_downsamples_an_atmosphere_file(tmp_path, monkeypatch):
    atmosphere = _structured_atmosphere()
    path = write_atmosphere(atmosphere, tmp_path / "box.h5")
    # x cells span -0.2 to 0.2 in steps of 0.1, so this keeps the middle two.
    crop = ["--crop-x", "-0.09 Mm", "0.09 Mm", "--crop-z", "0.12 Mm", "0.3 Mm"]
    # Cropping at synthesis has to give what synthesising an already cropped
    # atmosphere does, which is cropping tested against NDCube's own rule above.
    cropped = write_atmosphere(
        atmosphere.cropped(x=("-0.09 Mm", "0.09 Mm"), z=("0.12 Mm", "0.3 Mm")),
        tmp_path / "cropped.h5")

    beforehand = _synthesise(tmp_path, monkeypatch, "beforehand", "--atmosphere",
                             str(cropped), "--mass-per-electron", str(MASS_PER_ELECTRON))
    from_file = _synthesise(tmp_path, monkeypatch, "file", "--atmosphere",
                            str(path), *crop, "--mass-per-electron", str(MASS_PER_ELECTRON))
    assert _intensity(from_file).shape == (5, 2)
    assert from_file["line_cubes"][LINE].data == pytest.approx(
        beforehand["line_cubes"][LINE].data, rel=1e-6)

    even_path = write_atmosphere(_atmosphere((6, 4, 4)), tmp_path / "even.h5")
    full = _synthesise(tmp_path, monkeypatch, "full", "--atmosphere", str(even_path),
                       "--mass-per-electron", str(MASS_PER_ELECTRON))
    coarse = _synthesise(tmp_path, monkeypatch, "coarse", "--atmosphere", str(even_path),
                         "--downsample", "2", "--mass-per-electron", str(MASS_PER_ELECTRON))
    assert _intensity(coarse).shape == (2, 2)
    assert _intensity(coarse) == pytest.approx(_intensity(full)[0, 0], rel=1e-6)
    wcs = coarse["line_cubes"][LINE].wcs.wcs
    assert wcs.cdelt[1] == pytest.approx(2 * SPACING["x"].to_value(u.Mm))
    assert wcs.cdelt[2] == pytest.approx(2 * SPACING["y"].to_value(u.Mm))


def test_muram_files_given_directly_synthesise_as_the_same_atmosphere_file_with_a_warning(
        tmp_path, monkeypatch):
    """The old command line still runs until it is removed, placing the box where it always did."""
    atmosphere = _structured_atmosphere()
    _write_muram_files(tmp_path / "muram", atmosphere, suffix="0300000")
    nz, ny, nx = SHAPE
    layout = ["--data-dir", str(tmp_path / "muram"),
              "--cube-shape", str(nx), str(nz), str(ny),
              "--voxel-dx", str(SPACING["x"]), "--voxel-dy", str(SPACING["y"]),
              "--voxel-dz", str(SPACING["z"])]
    options = ["--crop-x", "-0.09 Mm", "0.19 Mm", "--crop-z", "0.12 Mm", "0.3 Mm",
               "--mass-per-electron", str(MASS_PER_ELECTRON)]

    # The same values as the MURaM files hold, in their units and precision,
    # on the grid the MURaM route has always used: x and y centred on zero
    # and the bottom cell centred on z = 0.
    as_written = write_atmosphere(Atmosphere(
        temperature=atmosphere.temperature.to(u.K).astype(np.float32),
        mass_density=atmosphere.mass_density.to(u.g / u.cm**3).astype(np.float32),
        velocity_z=atmosphere.velocity_z.to(u.cm / u.s).astype(np.float32),
        **_edges()), tmp_path / "as_written.h5")
    from_file = _synthesise(tmp_path, monkeypatch, "file", "--atmosphere",
                            str(as_written), *options)
    with pytest.warns(FutureWarning, match="deprecated.*atmosphere-files.*--atmosphere"):
        direct = _synthesise(tmp_path, monkeypatch, "direct", *layout, *options,
                             "--temp-file", "temp/eosT.0300000",
                             "--rho-file", "rho/result_prim_0.0300000",
                             "--vz-file", "vz/result_prim_2.0300000")

    direct_cube = direct["line_cubes"][LINE]
    file_cube = from_file["line_cubes"][LINE]
    assert np.all(_intensity(direct) > 0)
    assert np.array_equal(direct_cube.data, file_cube.data)
    assert list(direct_cube.wcs.wcs.ctype) == list(file_cube.wcs.wcs.ctype)
    for attribute in ("crval", "cdelt", "crpix"):
        assert np.array_equal(getattr(direct_cube.wcs.wcs, attribute),
                              getattr(file_cube.wcs.wcs, attribute))
    assert np.array_equal(direct["dem_map"], from_file["dem_map"])

    # The image axes are where the cube loader, which dynamic mode reads
    # MURaM files with, puts them.
    raw_cube = load_cube(tmp_path / "muram" / "temp" / "eosT.0300000",
                         shape=(nx, nz, ny), unit=u.K, voxel_dx=SPACING["x"],
                         voxel_dy=SPACING["y"], voxel_dz=SPACING["z"],
                         create_ndcube=True)
    kept = slice(1, 4)  # the three x cells the crop keeps
    assert _world(direct_cube, "SOLX") == pytest.approx(_world(raw_cube, "SOLX")[kept])
    assert _world(direct_cube, "SOLY") == pytest.approx(_world(raw_cube, "SOLY"))

    assert direct["config"]["atmosphere"] is None
    assert direct["config"]["data_dir"] == str(tmp_path / "muram")
    assert direct["config"]["cube_shape"] == [nx, nz, ny]
    assert direct["atmosphere"]["path"] is None
    assert direct["atmosphere"]["source"] == "MURaM"


def test_without_an_atmosphere_file_the_missing_muram_files_are_named(tmp_path, monkeypatch):
    """Forgetting --atmosphere warns that the MURaM route is deprecated before it fails."""
    with pytest.warns(FutureWarning, match="--atmosphere"):
        with pytest.raises(FileNotFoundError, match="temperature file not found"):
            _synthesise(tmp_path, monkeypatch, "none", "--data-dir", str(tmp_path / "empty"))


def test_the_atmosphere_option_excludes_the_muram_layout_options(tmp_path, monkeypatch):
    path = write_atmosphere(_atmosphere(), tmp_path / "box.h5")
    with pytest.raises(ValueError, match="--cube-shape would not be used"):
        _synthesise(tmp_path, monkeypatch, "both", "--atmosphere", str(path),
                    "--cube-shape", "4", "6", "5")
    with pytest.raises(ValueError, match="--voxel-dx.*--voxel-dz would not be used"):
        _synthesise(tmp_path, monkeypatch, "both", "--atmosphere", str(path),
                    "--voxel-dx", "0.1 Mm", "--voxel-dz", "0.1 Mm")
    with pytest.raises(ValueError, match="--temp-file would not be used"):
        _synthesise(tmp_path, monkeypatch, "both", "--atmosphere", str(path),
                    "--temp-file", "temp/eosT.0300000")
    # Typed at its default value it is still a MURaM option that goes unused.
    with pytest.raises(ValueError, match="--data-dir.*--cube-shape would not be used"):
        _synthesise(tmp_path, monkeypatch, "both", "--atmosphere", str(path),
                    "--data-dir", "data/atmosphere", "--cube-shape", "512", "768", "256")
    with pytest.raises(ValueError, match="Dynamic mode"):
        _synthesise(tmp_path, monkeypatch, "dynamic", "--atmosphere", str(path),
                    "--slit-rest-time", "40 s", "--slit-width", "0.2 arcsec")


def test_the_atmosphere_option_excludes_the_dynamic_mode_options(tmp_path, monkeypatch):
    path = write_atmosphere(_atmosphere(), tmp_path / "box.h5")
    with pytest.raises(ValueError, match="--temp-dir would not be used"):
        _synthesise(tmp_path, monkeypatch, "static", "--atmosphere", str(path),
                    "--temp-dir", "other")
    with pytest.raises(ValueError, match="--slit-width would not be used"):
        _synthesise(tmp_path, monkeypatch, "static", "--atmosphere", str(path),
                    "--slit-width", "0.2 arcsec")
    # Typed at their default values they still go unused.
    with pytest.raises(ValueError, match="--time-dir, --time-filename would not be used"):
        _synthesise(tmp_path, monkeypatch, "static", "--atmosphere", str(path),
                    "--time-dir", "header", "--time-filename", "Header")


def test_the_line_cube_refuses_a_stretched_image_axis_whatever_calls_it():
    """The guard sits on the cube, not only on the command line route."""
    atmosphere = _atmosphere(x_edges=STRETCHED_X)
    reference = atmosphere.to_ndcube(atmosphere.temperature)
    assert reference.meta["nonuniform_axes"] == ["x"]
    wl_grid = (REST + np.linspace(-0.5, 0.5, 11) * u.Angstrom).to(u.cm)
    line = {"si": np.zeros((SHAPE[1], SHAPE[2], wl_grid.size)),
            "wl_grid": wl_grid, "wl0": REST.to(u.cm), "atom": 26, "ion": 12}
    with pytest.raises(ValueError, match="x axis.*not evenly spaced"):
        synthesis.create_line_cube(LINE, line, reference,
                                   u.erg / u.s / u.cm**2 / u.sr / u.cm, "z")
