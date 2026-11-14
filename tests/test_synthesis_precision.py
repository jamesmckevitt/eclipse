"""A synthesis stores its spectra at the precision asked for."""

import sys

import astropy.constants as const
import astropy.units as u
import h5py
import numpy as np
import pytest

from eclipse import synthesis
from eclipse.atmosphere import Atmosphere, write_atmosphere

LINE = "Fe12_195.1190"


def _flat_goft(lines, **kwargs):
    """Contribution function of 1 everywhere, in place of fiasco."""
    logT_grid = np.linspace(5.0, 7.0, 21)
    logN_grid = np.linspace(8.0, 10.0, 21)
    goft = {name: {"wl0": (float(name.split("_")[1]) * u.AA).to(u.cm),
                   "g_tn": np.ones((logN_grid.size, logT_grid.size)),
                   "atom": 26, "ion": 12, "hdf5_dbase_root": None} for name in lines}
    return goft, logT_grid, logN_grid


def _synthesise(tmp_path, precision):
    nz, ny, nx = 4, 3, 2
    density = (1.0e9 / u.cm**3 * 1.2 * const.u).to(u.g / u.cm**3)
    atmosphere = Atmosphere(
        temperature=np.full((nz, ny, nx), 1e6) * u.K,
        mass_density=np.full((nz, ny, nx), density.value) * density.unit,
        velocity_z=np.linspace(-20e5, 20e5, nz * ny * nx).reshape(nz, ny, nx) * u.cm / u.s,
        x_edges=np.arange(nx + 1) * 0.1 * u.Mm, y_edges=np.arange(ny + 1) * 0.1 * u.Mm,
        z_edges=np.arange(nz + 1) * 0.1 * u.Mm)
    path = write_atmosphere(atmosphere, tmp_path / "atmosphere.h5")
    patch = pytest.MonkeyPatch()
    patch.setattr(sys, "argv", [
        "synthesise-spectra", "--atmosphere", str(path), "--output-dir", str(tmp_path),
        "--output-name", f"{precision}.h5", "--mass-per-electron", "1.2", "--lines", LINE,
        "--precision", precision])
    patch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    synthesis.main()
    patch.undo()
    with h5py.File(tmp_path / f"{precision}.h5", "r") as f:
        return f["lines"][LINE]["intensity"][()]


def test_the_spectra_are_stored_at_the_precision_asked_for(tmp_path):
    single = _synthesise(tmp_path, "float32")
    double = _synthesise(tmp_path, "float64")
    assert single.dtype == np.float32 and double.dtype == np.float64
    assert np.allclose(single, double, rtol=1e-4, atol=1e-6 * double.max())
