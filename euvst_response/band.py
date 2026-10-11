"""
Contribution functions of every line in a wavelength band, for full-CCD frames.

The synthesis computes only the lines named for a run. A frame of a whole
detector holds every line in its band: in CHIANTI, tens of thousands of
transitions from almost two hundred ions. This module computes the
contribution function G(T, n_e) of each, on the temperature and density
grids of a synthesis. As elsewhere in ECLIPSE, G includes n_H / n_e, so it
multiplies an emission measure of n_e^2 dh.

An ion is solved only at temperatures where CHIANTI's ionisation
equilibrium gives it a nonzero fraction. Elsewhere its G is zero, so
nothing is lost. In CHIANTI 10.1 a few ions, such as N IV, have levels that
nothing populates, so their level equations have no unique solution. These
levels are empty in a steady state, so they are left out and the rest
solved exactly. An ion that still cannot be solved stops the run. For some
ions, CHIANTI's level-resolved recombination and ionisation rates stop
short of the hottest temperatures. There fiasco's single-ion model is used,
as for ions with no such rates at all, and the run reports it. An ion whose
data or abundance CHIANTI lacks is left out, and the run names it.

The result is written to a file, since the whole SW band takes about two
hours on 32 cores and serves every frame on the same grids.
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
        Ion, such as `"Fe 12"`.
    atom, stage : int
        Atomic number and ionisation stage, 26 and 12 for Fe XII.
    mass : float
        Ion mass in atomic mass units, for the thermal width of its lines.
    wavelength : np.ndarray
        Wavelength of each line in Angstrom: CHIANTI's observed value where
        it has one, and the theoretical value otherwise.
    observed : np.ndarray of bool
        Whether each wavelength is observed.
    temperatures : np.ndarray of int
        Indices of the grid temperatures where the ion exists.
    g : np.ndarray
        G at those temperatures and every grid density, shape
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
    Every line in a band, ion by ion, on one temperature and density grid.

    Parameters
    ----------
    band : tuple of float
        Band edges in Angstrom.
    logT_grid, logN_grid : np.ndarray
        Grids as log10(T / K) and log10(n_e / cm-3).
    abundance : str
        CHIANTI abundance set.
    hdf5_dbase_root : str
        CHIANTI database the lines came from.
    ions : list of IonLines
        Ions with lines in the band.
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
    One ion's level populations, solved without the levels nothing populates.

    In fiasco's rate matrix, element [i, j] is the rate from level j into
    level i, and each column sums to zero. The last row is replaced by ones
    so that the populations sum to one. A level whose row has no rate from
    any other level gains nothing, so it is empty in a steady state. Such
    levels are left out, the rest solved exactly, and the left-out
    populations set to zero. The ground level is always kept.
    """
    matrix = np.asarray(matrix, dtype=float)
    # Rebuild the last row, which the ones replaced, from each column summing
    # to zero; values within the rounding error of the sums are taken as zero.
    rebuilt = -matrix[:-1].sum(axis=0)
    rounding = matrix.shape[0] * np.finfo(float).eps * np.abs(matrix[:-1]).sum(axis=0)
    rebuilt[np.abs(rebuilt) <= rounding] = 0.0
    inflow = np.vstack([matrix[:-1], rebuilt])
    np.fill_diagonal(inflow, 0.0)
    unfed = ~np.any(inflow != 0, axis=1)
    unfed[0] = False
    keep = np.flatnonzero(~unfed)
    if keep.size == matrix.shape[0]:
        raise np.linalg.LinAlgError("the matrix is singular, with no unpopulated level to drop")
    reduced = matrix[np.ix_(keep, keep)].copy()
    reduced[-1, :] = 1.0
    rhs = np.asarray(rhs, dtype=float)
    reduced_rhs = np.zeros((keep.size,) + rhs.shape[1:])
    reduced_rhs[-1] = rhs[-1]
    solution = np.linalg.solve(reduced, reduced_rhs)
    populations = np.zeros((matrix.shape[0],) + rhs.shape[1:])
    populations[keep] = solution
    return populations, int(unfed.sum())


@contextlib.contextmanager
def _unfed_levels_dropped(dropped: Dict[int, int]):
    """
    Context in which a singular level-population solve drops the levels nothing populates.

    fiasco solves all temperatures at once with numpy's solve. This replaces
    it, solving each matrix as numpy would and falling back to
    :func:`_solve_dropping_unfed_levels` for a singular one. `dropped`
    collects the number of levels left out, keyed by matrix index. numpy's
    solve is restored on exit.
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
            # fiasco 0.6 gives one right-hand side for every matrix, fiasco 0.8
            # one for each, shaped (n_temperature, n_levels, 1).
            rhs = b[k] if stacked and b.ndim > 1 else b
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
    fiasco's contribution function of an ion, leaving out levels nothing populates if needed.
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
    import logging

    # fiasco creates its logger on first import, so import it before looking
    # the logger up.
    import fiasco  # noqa: F401

    # fiasco warns about its fallbacks once per density; the run reports them
    # once instead.
    fiasco_logger = logging.getLogger("fiasco")
    previous = fiasco_logger.level
    fiasco_logger.setLevel(logging.ERROR)
    try:
        return _band_lines_of_ion(*args)
    finally:
        fiasco_logger.setLevel(previous)


def _band_lines_of_ion(atom, stage, temperature_K, densities_cm3, band, abundance, dbase_root):
    """The work of `_ion_band_lines`."""
    import fiasco
    from fiasco.util.exceptions import MissingDatasetException

    kwargs = {} if dbase_root is None else {"hdf5_dbase_root": dbase_root}
    try:
        ion = fiasco.Ion((atom, stage), temperature_K * u.K, abundance=abundance, **kwargs)
        transitions = ion.transitions
    except MissingDatasetException:
        # CHIANTI has no levels or transitions for the ion, so no lines.
        return None
    bound = transitions.is_bound_bound
    wavelength = transitions.wavelength[bound].to_value(u.AA)
    observed = np.asarray(transitions.is_observed[bound])
    columns = np.flatnonzero((wavelength >= band[0]) & (wavelength <= band[1]))
    if columns.size == 0:
        return None
    try:
        fraction = np.nan_to_num(np.asarray(ion.ionization_fraction, dtype=float))
    except MissingDatasetException as error:
        return {"skipped": f"{ion.ion_name} ({error})"}
    temperatures = np.flatnonzero(fraction > 0)
    if temperatures.size == 0:
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
            # For some ions CHIANTI's level-resolved recombination and
            # ionisation rates stop short of the hottest temperatures. Use
            # fiasco's single-ion model there, as for ions with no such rates.
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
            # Recount the dropped levels, keyed by temperature index in the
            # full array, without those of the failed attempt.
            dropped.clear()
            parts = []
            if first > 0:
                below = fiasco.Ion((atom, stage), present.temperature[:first], abundance=abundance,
                                   **kwargs)
                parts.append(_contribution_function(below, density, dropped))
            above = fiasco.Ion((atom, stage), present.temperature[first:], abundance=abundance,
                               **kwargs)
            dropped_above: Dict[int, int] = {}
            parts.append(_contribution_function(above, density, dropped_above,
                                                use_two_ion_model=False))
            dropped.update({first + k: n for k, n in dropped_above.items()})
            g = np.concatenate(parts)
            single_ion = float(np.log10(present.temperature[first].to_value(u.K)))
        g = g * present.proton_electron_ratio[:, np.newaxis, np.newaxis]
    except MissingDatasetException as error:
        # CHIANTI or the abundance set lacks data the lines need.
        return {"skipped": f"{ion.ion_name} ({error})"}
    # Reorder (temperature, density, transition) to (line, density,
    # temperature), as in compute_goft_fiasco, with non-finite values set to 0.
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
    Contribution function of every CHIANTI line in a band.

    Parameters
    ----------
    band : sequence of two floats
        Band edges in Angstrom. For a frame, leave room beyond the detector's
        own range for Doppler shifts and the spectral PSF.
    logT_grid, logN_grid : np.ndarray
        Synthesis grids as log10(T / K) and log10(n_e / cm-3).
    abundance : str, optional
        CHIANTI abundance set. Default `"sun_coronal_2021_chianti"`.
    n_workers : int, optional
        Number of processes, each working on one ion at a time. Default 0
        uses every CPU available to this process. The workers are spawned,
        so a calling script needs an `if __name__ == "__main__":` guard. The
        whole SW band on 32 workers needs about 150 GB of memory in total;
        use fewer workers where there is less.
    hdf5_dbase_root : str or Path, optional
        CHIANTI database to read. Default fiasco's own.
    elements : sequence of str, optional
        Elements by symbol. Default every element in CHIANTI.

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

        # A new worker for each ion, so its memory is freed afterwards.
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
    if used is None:
        # No ion was computed, so record the requested database.
        used = str(fiasco.defaults["hdf5_dbase_root"] if dbase_root is None else dbase_root)
    return BandLines(band=(low, high), logT_grid=logT_grid, logN_grid=logN_grid,
                     abundance=abundance, hdf5_dbase_root=used, ions=ions)


def write_band_lines(lines: BandLines, path: str | Path) -> Path:
    """
    Write a band's lines to an HDF5 file readable by `read_band_lines`.

    Parameters
    ----------
    lines : BandLines
        Lines to write.
    path : str or Path
        Output file. An existing file is replaced.

    Returns
    -------
    Path
        File written.
    """
    from .atmosphere import _replacing

    path = Path(path).expanduser()
    with _replacing(path) as partial, h5py.File(partial, "w") as f:
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
    Read a band's lines from a file written by `write_band_lines`.

    Parameters
    ----------
    path : str or Path
        File to read.

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
