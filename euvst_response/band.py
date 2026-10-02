"""
Every line in a wavelength band: the contribution functions a full-CCD frame needs.

ECLIPSE's synthesis works out the lines named for a run. A frame of a whole
detector holds every line in its band, which in CHIANTI is tens of thousands
of transitions of almost two hundred ions. This works out the contribution
function, G(T, n_e), of each of them, on the grids of temperature and
density a synthesis uses, in ECLIPSE's convention: G includes n_H / n_e, so
that it multiplies an emission measure of n_e^2 dh.

An ion is solved only at the temperatures where CHIANTI's ionisation
equilibrium holds some of it; elsewhere its G is zero, so nothing is lost.
In CHIANTI 10.1, a few ions, such as N IV, have levels that nothing
populates, which leaves their equations with no single solution. Those
levels hold nothing in a steady state, so they are left out and the others
solved exactly. An ion that still cannot be solved stops the run. For some
ions, CHIANTI's recombination and ionisation rates of each level stop short
of the hottest temperatures; there, fiasco's single-ion model is used, as
it is for ions that have no such rates at all, and the run says so. An ion
whose data, or abundance, CHIANTI does not have is left out, and the run
names it.

The result is written to a file, as the whole SW band takes about two hours
on 32 cores and serves every frame on the same grids.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import astropy.units as u

FORMAT_NAME = "eclipse-band-lines"
FORMAT_VERSION = 1
G_UNIT = u.erg * u.cm**3 / u.s


@dataclass(frozen=True)
class IonLines:
    """
    One ion's lines in a band.

    Parameters
    ----------
    name : str
        The ion, such as ``"Fe 12"``.
    atom, stage : int
        The atomic number and the ionisation stage, 26 and 12 for Fe XII.
    mass : float
        The ion's mass, in atomic mass units, for its lines' thermal width.
    wavelength : np.ndarray
        Each line's wavelength in CHIANTI, in Angstrom, observed where
        CHIANTI has one and theoretical otherwise.
    observed : np.ndarray of bool
        Whether each wavelength is observed.
    temperatures : np.ndarray of int
        The indices of the temperatures of the grid where the ion exists.
    g : np.ndarray
        G at those temperatures and every density of the grid, shaped
        (n_lines, nN, n_temperatures), in erg cm3 / s.
    """

    name: str
    atom: int
    stage: int
    mass: float
    wavelength: np.ndarray
    observed: np.ndarray
    temperatures: np.ndarray
    g: np.ndarray


@dataclass(frozen=True)
class BandLines:
    """
    Every line in a band, ion by ion, on one grid of temperatures and densities.

    Parameters
    ----------
    band : tuple of float
        The band, in Angstrom.
    logT_grid, logN_grid : np.ndarray
        The grids, as log10(T / K) and log10(n_e / cm-3).
    abundance : str
        The CHIANTI abundance set.
    hdf5_dbase_root : str
        The CHIANTI database the lines came from.
    ions : list of IonLines
        The ions with lines in the band.
    """

    band: Tuple[float, float]
    logT_grid: np.ndarray
    logN_grid: np.ndarray
    abundance: str
    hdf5_dbase_root: str
    ions: List[IonLines]

    @property
    def n_lines(self) -> int:
        return sum(ion.wavelength.size for ion in self.ions)


def _solve_dropping_unfed_levels(matrix: np.ndarray, rhs: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Solve one ion's level populations with the levels nothing populates left out.

    fiasco gives the rate matrix with its last row replaced by ones, for the
    populations to add up to one. A level whose column is zero in every
    other row gains nothing from any level, so in a steady state it holds
    nothing. It is left out, the others solved exactly, and its population
    set to zero. The ground level is always kept.
    """
    balance = matrix[:-1]
    unfed = np.all(balance == 0, axis=0)
    unfed[0] = False
    keep = np.flatnonzero(~unfed)
    if keep.size == matrix.shape[0]:
        raise np.linalg.LinAlgError("the matrix is singular, with no unpopulated level to drop")
    reduced = matrix[np.ix_(keep, keep)].copy()
    reduced[-1, :] = 1.0
    reduced_rhs = np.zeros(keep.size)
    reduced_rhs[-1] = rhs[-1]
    solution = np.linalg.solve(reduced, reduced_rhs)
    populations = np.zeros(matrix.shape[0])
    populations[keep] = solution
    return populations, int(unfed.sum())


@contextlib.contextmanager
def _unfed_levels_dropped(dropped: Dict[int, int]):
    """
    While active, a level population solve that is singular drops the levels nothing populates.

    fiasco solves every temperature at once with numpy's solve; this stands
    in for it, solving each matrix as numpy would and falling back to
    :func:`_solve_dropping_unfed_levels` for one that is singular. *dropped*
    collects how many levels were left out, by the index of the matrix. It
    runs in a worker process, so it changes nothing outside it.
    """
    from unittest import mock

    original = np.linalg.solve

    def solve(a, b):
        try:
            return original(a, b)
        except np.linalg.LinAlgError:
            pass
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        stacked = a.ndim == 3
        solutions = []
        for k, matrix in enumerate(a if stacked else [a]):
            rhs = b[k] if b.ndim == 2 else b
            try:
                solutions.append(original(matrix, rhs))
            except np.linalg.LinAlgError:
                populations, n = _solve_dropping_unfed_levels(matrix, rhs)
                dropped[k] = n
                solutions.append(populations)
        return np.stack(solutions) if stacked else solutions[0]

    with mock.patch("numpy.linalg.solve", new=solve):
        yield


def _beyond_the_rates(error: ValueError) -> bool:
    """Whether fiasco stopped at a temperature beyond the rates CHIANTI tabulates."""
    return "interpolation range" in str(error)


def _contribution_function(ion, density, dropped: Dict[int, int], **kwargs):
    """
    fiasco's contribution function of an ion, with the levels nothing populates left out if need be.
    """
    try:
        return ion.contribution_function(density, **kwargs)
    except np.linalg.LinAlgError:
        pass
    try:
        with _unfed_levels_dropped(dropped):
            return ion.contribution_function(density, **kwargs)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            f"The level populations of {ion.ion_name} cannot be solved, even with the "
            f"levels nothing populates left out: {error}") from None


def _ion_band_lines(args):
    """One ion's lines in the band, for a worker process; None if it has none there."""
    (atom, stage, temperature_K, densities_cm3, band, abundance, dbase_root) = args
    import logging

    import fiasco
    from fiasco.util.exceptions import MissingDatasetException

    logging.getLogger("fiasco").setLevel(logging.ERROR)
    kwargs = {} if dbase_root is None else {"hdf5_dbase_root": dbase_root}
    try:
        ion = fiasco.Ion((atom, stage), temperature_K * u.K, abundance=abundance, **kwargs)
        transitions = ion.transitions
    except MissingDatasetException:
        return None
    bound = transitions.is_bound_bound
    wavelength = transitions.wavelength[bound].to_value(u.AA)
    observed = np.asarray(transitions.is_observed[bound])
    columns = np.flatnonzero((wavelength >= band[0]) & (wavelength <= band[1]))
    fraction = np.nan_to_num(np.asarray(ion.ionization_fraction, dtype=float))
    temperatures = np.flatnonzero(fraction > 0)
    if columns.size == 0 or temperatures.size == 0:
        return None

    try:
        present = fiasco.Ion((atom, stage), temperature_K[temperatures] * u.K, abundance=abundance,
                             **kwargs)
        density = densities_cm3 * u.cm**-3
        dropped: Dict[int, int] = {}
        single_ion = None
        try:
            g = _contribution_function(present, density, dropped)
        except ValueError as error:
            if not _beyond_the_rates(error):
                raise
            # CHIANTI's recombination and ionisation rates of each level stop
            # short of the hottest temperatures for some ions: there, fiasco's
            # single-ion model, as for ions that have no such rates at all.
            first = None
            for index, temperature in enumerate(present.temperature):
                one = fiasco.Ion((atom, stage), temperature[np.newaxis], abundance=abundance,
                                 **kwargs)
                try:
                    _contribution_function(one, density[:1], {})
                except ValueError as error:
                    if not _beyond_the_rates(error):
                        raise
                    first = index
                    break
            if first is None:
                raise
            parts = []
            if first > 0:
                below = fiasco.Ion((atom, stage), present.temperature[:first], abundance=abundance,
                                   **kwargs)
                parts.append(_contribution_function(below, density, dropped))
            above = fiasco.Ion((atom, stage), present.temperature[first:], abundance=abundance,
                               **kwargs)
            parts.append(_contribution_function(above, density, dropped, use_two_ion_model=False))
            g = np.concatenate(parts)
            single_ion = float(np.log10(present.temperature[first].to_value(u.K)))
        g = g * present.proton_electron_ratio[:, np.newaxis, np.newaxis]
    except MissingDatasetException as error:
        # Data the lines need that CHIANTI, or the abundance set, does not have.
        return {"skipped": f"{ion.ion_name} ({error})"}
    # (temperature, density, transition) to (line, density, temperature), as
    # compute_goft_fiasco keeps G, with what fiasco cannot give taken as zero.
    g = np.nan_to_num(g.to_value(G_UNIT)[..., columns], nan=0.0, posinf=0.0, neginf=0.0)
    return {"atom": atom, "stage": stage, "name": present.ion_name,
            "mass": float(ion.mass.to_value(u.u)),
            "wavelength": wavelength[columns], "observed": observed[columns],
            "temperatures": temperatures, "g": np.ascontiguousarray(np.transpose(g, (2, 1, 0))),
            "dropped": len(dropped), "single_ion": single_ion,
            "hdf5_dbase_root": str(ion.hdf5_dbase_root)}


def band_contribution_functions(
    band: Sequence[float],
    logT_grid: np.ndarray,
    logN_grid: np.ndarray,
    abundance: str = "sun_coronal_2021_chianti",
    n_workers: int = 0,
    hdf5_dbase_root=None,
    elements: Optional[Sequence[str]] = None,
) -> BandLines:
    """
    The contribution function of every CHIANTI line in a band.

    Parameters
    ----------
    band : sequence of two floats
        The band, in Angstrom. For a frame, leave room beyond the detector's
        own for Doppler shifts and the spectral PSF.
    logT_grid, logN_grid : np.ndarray
        The grids of the synthesis, as log10(T / K) and log10(n_e / cm-3).
    abundance : str, optional
        The CHIANTI abundance set. Default ``"sun_coronal_2021_chianti"``.
    n_workers : int, optional
        How many processes to work out the ions with, one ion each at a
        time. Default 0, which uses every CPU this process may use. The
        workers are spawned, so a script calling this needs an
        ``if __name__ == "__main__":`` guard. The ions with the most levels
        need about 3 GB each.
    hdf5_dbase_root : str or Path, optional
        The CHIANTI database to read. Default fiasco's own.
    elements : sequence of str, optional
        The elements, by symbol. Default every element CHIANTI holds.

    Returns
    -------
    BandLines
    """
    import fiasco

    from .atmosphere import _offer_database_build
    from .utils import element_data

    low, high = (float(value) for value in band)
    if not low < high:
        raise ValueError(f"The band must run from a shorter wavelength to a longer one, "
                         f"got {band}.")
    dbase_root = None if hdf5_dbase_root is None else str(hdf5_dbase_root)
    _offer_database_build(dbase_root)
    logT_grid = np.asarray(logT_grid, dtype=float)
    logN_grid = np.asarray(logN_grid, dtype=float)
    temperature_K = 10.0 ** logT_grid
    densities_cm3 = 10.0 ** logN_grid
    if elements is None:
        elements = fiasco.list_elements(dbase_root)
    jobs = []
    for symbol in elements:
        atom, _ = element_data(symbol)
        jobs += [(atom, stage, temperature_K, densities_cm3, (low, high), abundance, dbase_root)
                 for stage in range(1, atom + 1)]
    # The ions with the most levels take longest, so they are started first.
    jobs.sort(key=lambda job: -job[0] * (job[0] - job[1] + 1))

    if n_workers <= 0:
        n_workers = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                     else os.cpu_count() or 1)
    if n_workers > 1 and len(jobs) > 1:
        import multiprocessing as mp

        # A new worker for each ion, so that each gives its memory back.
        with mp.get_context("spawn").Pool(min(n_workers, len(jobs)), maxtasksperchild=1) as pool:
            results = pool.map(_ion_band_lines, jobs, chunksize=1)
    else:
        results = [_ion_band_lines(job) for job in jobs]

    ions, used, single_ion, skipped = [], None, [], []
    for result in results:
        if result is None:
            continue
        if "skipped" in result:
            skipped.append(result["skipped"])
            continue
        used = result["hdf5_dbase_root"]
        if dbase_root is not None and used != dbase_root:
            raise RuntimeError(f"An ion was worked out from the CHIANTI database at {used} "
                               f"instead of the requested {dbase_root}.")
        if result["dropped"]:
            print(f"  {result['name']}: levels nothing populates were left out at "
                  f"{result['dropped']} temperatures, and the rest solved exactly")
        if result["single_ion"] is not None:
            single_ion.append(f"{result['name']} from log T {result['single_ion']:.2f}")
        ions.append(IonLines(name=result["name"], atom=result["atom"], stage=result["stage"],
                             mass=result["mass"],
                             wavelength=result["wavelength"], observed=result["observed"],
                             temperatures=result["temperatures"], g=result["g"]))
    if single_ion:
        print(f"  CHIANTI's recombination and ionisation rates of each level stop short of "
              f"the hottest temperatures for {len(single_ion)} ions, whose level populations "
              f"there are worked out with fiasco's single-ion model, as for ions that have "
              f"no such rates: {', '.join(single_ion)}")
    if skipped:
        print(f"  Left out, for want of data in CHIANTI or the abundance set: "
              f"{'; '.join(skipped)}")
    ions.sort(key=lambda ion: (ion.atom, ion.stage))
    return BandLines(band=(low, high), logT_grid=logT_grid, logN_grid=logN_grid,
                     abundance=abundance, hdf5_dbase_root=used or str(dbase_root),
                     ions=ions)


def write_band_lines(lines: BandLines, path: str | Path) -> Path:
    """
    Write a band's lines to an HDF5 file, to read back with `read_band_lines`.

    Parameters
    ----------
    lines : BandLines
        The lines.
    path : str or Path
        The file to write. An existing file is replaced.

    Returns
    -------
    Path
        The file written.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = FORMAT_NAME
        f.attrs["version"] = FORMAT_VERSION
        f.attrs["band"] = np.asarray(lines.band, dtype=float)
        f.attrs["abundance"] = lines.abundance
        f.attrs["hdf5_dbase_root"] = lines.hdf5_dbase_root
        f["logT_grid"] = lines.logT_grid
        f["logN_grid"] = lines.logN_grid
        group = f.create_group("ions")
        for ion in lines.ions:
            entry = group.create_group(f"{ion.atom}_{ion.stage}")
            entry.attrs["name"] = ion.name
            entry.attrs["atom"] = ion.atom
            entry.attrs["stage"] = ion.stage
            entry.attrs["mass"] = ion.mass
            entry["wavelength"] = ion.wavelength
            entry["observed"] = ion.observed
            entry["temperatures"] = ion.temperatures
            entry.create_dataset("g", data=ion.g, compression="gzip")
            entry["g"].attrs["unit"] = str(G_UNIT)
    return path


def read_band_lines(path: str | Path) -> BandLines:
    """
    Read a band's lines from a file `write_band_lines` wrote.

    Parameters
    ----------
    path : str or Path
        The file.

    Returns
    -------
    BandLines
    """
    path = Path(path).expanduser()
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT_NAME:
            raise ValueError(f"{path} is not a file of a band's lines.")
        if int(f.attrs["version"]) != FORMAT_VERSION:
            raise ValueError(f"{path} is version {f.attrs['version']} of the band's lines "
                             f"file; this ECLIPSE reads version {FORMAT_VERSION}.")
        ions = [IonLines(name=str(entry.attrs["name"]), atom=int(entry.attrs["atom"]),
                         stage=int(entry.attrs["stage"]),
                         mass=float(entry.attrs["mass"]), wavelength=entry["wavelength"][()],
                         observed=entry["observed"][()], temperatures=entry["temperatures"][()],
                         g=entry["g"][()])
                for entry in f["ions"].values()]
        ions.sort(key=lambda ion: (ion.atom, ion.stage))
        return BandLines(band=tuple(float(v) for v in f.attrs["band"]),
                         logT_grid=f["logT_grid"][()], logN_grid=f["logN_grid"][()],
                         abundance=str(f.attrs["abundance"]),
                         hdf5_dbase_root=str(f.attrs["hdf5_dbase_root"]), ions=ions)
