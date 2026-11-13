"""
Every line in a band, for full-CCD frames: contribution functions and spectrum.

G(T, n_e) of every CHIANTI line in the band, on a synthesis's temperature
and density grids, with n_H / n_e included as elsewhere in ECLIPSE. Levels
that nothing populates, as in a few ions such as N IV in CHIANTI 10.1, are
left out and the rest solved exactly. Where an ion's level-resolved
recombination and ionisation rates stop short of the hottest temperatures,
or it has none, fiasco's single-ion model is used. Ions without data or an
abundance are skipped and named. The whole SW band takes about two hours on
32 cores, so the result is written to a file.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import astropy.constants as const
import astropy.units as u

FORMAT_NAME = "eclipse-band-lines"
FORMAT_VERSION = 1
G_UNIT = u.erg * u.cm**3 / u.s
_C_CM_S = const.c.to_value(u.cm / u.s)
_K_B = const.k_B.cgs.value
_AMU_G = const.u.cgs.value

# Default wavelength step of a band's spectrum, in Angstrom.
BAND_STEP = 0.002


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


def _serves(path: Path, band, logT_grid, logN_grid, abundance, dbase_root) -> Optional[str]:
    """Why the band's lines in the file at `path` do not serve a synthesis, or None if they do."""
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT_NAME:
            return "is not a file of a band's lines"
        low, high = (float(value) for value in f.attrs["band"])
        if low > band[0] or high < band[1]:
            return f"holds the band {low:g} to {high:g} Angstrom"
        if str(f.attrs["abundance"]) != abundance:
            return f"was worked out with the abundances {f.attrs['abundance']}"
        if dbase_root is not None and str(f.attrs["hdf5_dbase_root"]) != str(dbase_root):
            return f"was worked out from the CHIANTI database at {f.attrs['hdf5_dbase_root']}"
        held = f["logT_grid"][()]
        if held.shape != np.shape(logT_grid) or not np.allclose(held, logT_grid, rtol=0.0,
                                                                 atol=1e-6):
            return "is on another grid of temperature"
        held = f["logN_grid"][()]
        nearest = np.abs(np.asarray(logN_grid, dtype=float)[:, np.newaxis] - held).argmin(axis=1)
        if not np.allclose(held[nearest], logN_grid, rtol=0.0, atol=1e-6):
            return (f"is on densities, log10 n_e {held[0]:.2f} to {held[-1]:.2f}, that do not "
                    f"have all of this synthesis's, {logN_grid[0]:.2f} to {logN_grid[-1]:.2f}")
    return None


def band_lines_for(path: str | Path, band: Sequence[float], logT_grid: np.ndarray,
                   logN_grid: np.ndarray, abundance: str = "sun_coronal_2021_chianti",
                   n_workers: int = 0, hdf5_dbase_root=None) -> BandLines:
    """
    A band's lines, read back from a file if it holds them and worked out and written there if not.

    A file serves if it covers the band, is on the same temperatures and
    abundances, and has every density of `logN_grid` on its grid. Lines are
    worked out on `logN_grid`, so a grid wider than one synthesis needs
    serves later ones too.

    Parameters
    ----------
    path : str or Path
        File of the band's lines, as written by `write_band_lines`.
    band, logT_grid, logN_grid, abundance, n_workers, hdf5_dbase_root
        As for `band_contribution_functions`.

    Returns
    -------
    BandLines

    Raises
    ------
    ValueError
        If the file is there and does not serve. It is not replaced, as the
        whole SW band takes hours to work out.
    """
    path = Path(path).expanduser()
    band = tuple(float(value) for value in band)
    if path.exists():
        problem = _serves(path, band, logT_grid, logN_grid, abundance, hdf5_dbase_root)
        if problem is not None:
            raise ValueError(f"{path} {problem}, so it does not serve this synthesis. Move "
                             f"it, or name another file for the band's lines.")
        print(f"Reading the band's lines from {path}")
        return read_band_lines(path)
    print(f"Working out every line from {band[0]:g} to {band[1]:g} Angstrom, kept in {path}")
    lines = band_contribution_functions(band, logT_grid, logN_grid, abundance=abundance,
                                        n_workers=n_workers, hdf5_dbase_root=hdf5_dbase_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_band_lines(lines, path)
    return lines


# Lines are laid onto the wavelengths in blocks of this many, to bound the
# memory a whole band takes. The result does not depend on it.
_LINES_AT_ONCE = 20000

# Pixels are given their light in blocks of this many, for the same reason.
_PIXELS_AT_ONCE = 2048

# Data shared by the workers of one `band_spectra` call, set in each worker
# as it starts.
_WORKER_CONTEXT: Optional[dict] = None


def band_wavelengths(band: Sequence[float], step: float) -> np.ndarray:
    """
    Evenly spaced wavelengths whose bins cover a band.

    Parameters
    ----------
    band : sequence of two floats
        Band edges in Angstrom.
    step : float
        Bin width in Angstrom.

    Returns
    -------
    np.ndarray
        Bin centres in Angstrom. The first bin starts at the band's lower
        edge, and the last ends at or just past its upper edge.
    """
    low, high = (float(value) for value in band)
    step = float(step)
    if not low < high:
        raise ValueError(f"The band must run from a shorter wavelength to a longer one, "
                         f"got {band}.")
    if not (np.isfinite(step) and step > 0):
        raise ValueError(f"The wavelength step must be above zero, got {step}.")
    return low + (np.arange(int(np.ceil((high - low) / step * (1 - 1e-12)))) + 0.5) * step


def _density_shares(log_density: np.ndarray, logN_grid: np.ndarray):
    """
    Indices of the grid densities either side of each density, and the share
    of its emission measure each takes.

    The shares are linear in log10 n_e, as the contribution functions are
    interpolated. Off the grid both shares are zero, as are the contribution
    functions.
    """
    grid = np.asarray(logN_grid, dtype=float)
    finite = np.isfinite(log_density)
    inside = finite & (log_density >= grid[0]) & (log_density <= grid[-1])
    position = np.interp(np.where(inside, log_density, grid[0]), grid,
                         np.arange(grid.size, dtype=float))
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, grid.size - 1)
    fraction = position - lower
    return lower, upper, np.where(inside, 1.0 - fraction, 0.0), np.where(inside, fraction, 0.0)


def _set_worker_context(context: dict) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _templates_in_worker(t: int):
    return _templates_at(t, _WORKER_CONTEXT)


def _templates_at(t: int, context: dict) -> np.ndarray:
    """
    Light of every line in each wavelength bin, per unit emission measure,
    from plasma at one temperature (an index into the temperature grid).

    One spectrum for each velocity bin and grid density with plasma at this
    temperature, `context["used"][t]`, in erg cm3 / s over the whole sphere,
    shape (n_velocities, n_densities, n_bins).
    """
    from .sampling import _REACH_IN_SIGMA, points_onto_pixels

    lines, edges = context["lines"], context["edges"]
    bins, densities = context["used"][t]
    templates = np.zeros((bins.size, densities.size, edges.size - 1))
    present = [ion for ion in lines.ions if t in ion.temperatures]
    if not present:
        return templates
    g = np.concatenate([ion.g[:, densities, np.searchsorted(ion.temperatures, t)]
                        for ion in present])                            # (lines, densities)
    wavelength = np.concatenate([ion.wavelength for ion in present])
    mass = np.concatenate([np.full(ion.wavelength.size, ion.mass) for ion in present])
    thermal = wavelength * np.sqrt(_K_B * 10.0 ** lines.logT_grid[t] / (mass * _AMU_G)) / _C_CM_S
    factors = 1.0 + context["velocity"][bins] / _C_CM_S
    # Lines whose light can reach the bins at one of the Doppler shifts.
    reach = _REACH_IN_SIGMA * thermal
    near = (((wavelength + reach) * factors.max() >= edges[0])
            & ((wavelength - reach) * factors.min() <= edges[-1]))
    wavelength, thermal, g = wavelength[near], thermal[near], g[near]
    for template, factor in zip(templates, factors):
        # Each line is a Gaussian of its thermal width at its Doppler-shifted
        # wavelength, shared between the bins by exact integrals.
        for start in range(0, wavelength.size, _LINES_AT_ONCE):
            block = slice(start, start + _LINES_AT_ONCE)
            template += (points_onto_pixels(wavelength[block] * factor, edges,
                                            thermal[block] * factor) @ g[block]).T
    return templates


def _add_temperature(light: np.ndarray, t: int, templates: np.ndarray, context: dict) -> None:
    """Add to `light`, (n_pixels, n_bins), the light of the plasma at one temperature."""
    bins, densities = context["used"][t]
    lower, upper, low_share, high_share = (part[:, t] for part in context["shares"])
    lit = np.flatnonzero((low_share + high_share) > 0)
    flat = templates.reshape(bins.size * densities.size, -1)
    for start in range(0, lit.size, _PIXELS_AT_ONCE):
        block = lit[start:start + _PIXELS_AT_ONCE]
        em = context["emission_measure"][block, t][:, bins]            # (pixels, velocities)
        # Each pixel's emission measure, shared between the grid densities
        # either side of its own.
        pixel = np.arange(block.size)
        weight = np.zeros((block.size, bins.size, densities.size))
        weight[pixel, :, np.searchsorted(densities, lower[block])] += (
            low_share[block, np.newaxis] * em)
        weight[pixel, :, np.searchsorted(densities, upper[block])] += (
            high_share[block, np.newaxis] * em)
        light[block] += weight.reshape(block.size, -1) @ flat


def band_spectra(lines: BandLines, emission_measure: np.ndarray, electron_density: np.ndarray,
                 velocity: u.Quantity, wavelength: u.Quantity,
                 n_workers: int = 0) -> u.Quantity:
    """
    Spectrum of every line in a band in every pixel, from the emission measure EM(T, v).

    Each temperature and velocity bin adds every line as a Gaussian with
    that temperature's thermal width, Doppler shifted by that velocity,
    scaled by the emission measure times G. Each Gaussian is integrated
    over the wavelength bins, so a line's light is kept whatever the
    spacing.

    Parameters
    ----------
    lines : BandLines
        Every line in the band, from `band_contribution_functions`, on the
        temperatures of `emission_measure`.
    emission_measure : np.ndarray
        Emission measure in each bin, in cm-5, shape (rows, columns,
        temperatures, velocities), as `em_tv` from `build_em_tv`.
    electron_density : np.ndarray or float
        Electron density in cm^-3 at each temperature of each pixel, shape
        (rows, columns, temperatures), or one density for all. G is
        interpolated linearly in log10 n_e, and is zero at densities off
        the grid of `lines`.
    velocity : u.Quantity
        Velocity bin centres.
    wavelength : u.Quantity
        Wavelengths of the spectrum, increasing. Each covers the bin halfway
        to its neighbours.
    n_workers : int, optional
        Number of processes to compute the temperatures with. Default 0 uses
        every CPU this process may use. Where processes cannot be forked, as
        on Windows, one process is used.

    Returns
    -------
    u.Quantity
        Spectral radiance, the mean over each wavelength's bin, in
        erg / (s cm2 sr cm), shape (rows, columns, n_wavelength).
    """
    import multiprocessing as mp
    import warnings

    from .utils import _bin_edges

    em = np.asarray(emission_measure, dtype=float)
    if em.ndim != 4:
        raise ValueError(f"The emission measure must be shaped (rows, columns, temperatures, "
                         f"velocities), got {em.shape}.")
    rows, columns, n_t, n_v = em.shape
    if n_t != lines.logT_grid.size:
        raise ValueError(f"The emission measure has {n_t} temperatures and the band's lines "
                         f"{lines.logT_grid.size}; work the lines out on the emission "
                         f"measure's own.")
    velocity = np.atleast_1d(u.Quantity(velocity).to_value(u.cm / u.s))
    if velocity.shape != (n_v,):
        raise ValueError(f"The emission measure has {n_v} velocity bins, but {velocity.size} "
                         f"velocities were given.")
    centres = np.atleast_1d(u.Quantity(wavelength).to_value(u.Angstrom)).astype(float)
    if centres.ndim != 1 or centres.size < 2 or not np.all(np.isfinite(centres)) \
            or np.any(np.diff(centres) <= 0):
        raise ValueError("The wavelengths must be two or more, finite and increasing.")
    edges = _bin_edges(centres)
    density = np.broadcast_to(np.asarray(electron_density, dtype=float), em.shape[:3])
    log_density = np.log10(np.where(density > 0, density, np.nan))
    shares = tuple(part.reshape(rows * columns, n_t)
                   for part in _density_shares(log_density, lines.logN_grid))
    context = {"lines": lines, "edges": edges, "velocity": velocity,
               "emission_measure": em.reshape(rows * columns, n_t, n_v), "shares": shares}
    # The velocity bins and grid densities with plasma at each temperature,
    # among pixels at a density on the grid.
    used = {}
    for t in range(n_t):
        lit = (shares[2][:, t] + shares[3][:, t]) > 0
        bins = np.flatnonzero(context["emission_measure"][lit, t].any(axis=0))
        if bins.size:
            used[t] = (bins, np.union1d(shares[0][lit, t], shares[1][lit, t]))
    context["used"] = used
    temperatures = sorted(used)
    if n_workers <= 0:
        n_workers = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                     else os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(temperatures)))
    if n_workers > 1 and "fork" not in mp.get_all_start_methods():
        warnings.warn("Processes cannot be forked here, so the temperatures are computed in "
                      "one process.", stacklevel=2)
        n_workers = 1
    light = np.zeros((rows * columns, centres.size))
    if n_workers > 1:
        # Fork, so the workers share the lines rather than being sent them.
        # The temperatures are added in order, so the spectra do not depend
        # on the number of workers.
        with mp.get_context("fork").Pool(n_workers, initializer=_set_worker_context,
                                         initargs=(context,)) as pool:
            for t, templates in zip(temperatures,
                                    pool.imap(_templates_in_worker, temperatures)):
                _add_temperature(light, t, templates, context)
    else:
        for t in temperatures:
            _add_temperature(light, t, _templates_at(t, context), context)
    # G times the emission measure is the light over the whole sphere; the
    # radiance is per steradian and per unit wavelength.
    light /= 4.0 * np.pi * np.diff(edges) * 1e-8
    return light.reshape(rows, columns, -1) * (u.erg / (u.s * u.cm**2 * u.sr * u.cm))
