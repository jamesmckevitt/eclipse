"""
The continuum under the lines: free-free, free-bound and two-photon emission.

ECLIPSE's synthesis adds up the lines named for a run. The continuum is the
light the plasma gives at every wavelength, not only at its lines: free-free
emission, from electrons passing ions, free-bound emission, from electrons
captured by them, and the two-photon emission of hydrogen- and helium-like
ions. It is worked out from CHIANTI through fiasco, for every element CHIANTI
holds, with the abundances and ionisation fractions the lines use.

Like a line's contribution function, the continuum here multiplies an
emission measure of n_e^2 dh. fiasco gives it per n_e n_H and over all
directions, so it is multiplied by n_H / n_e and divided by 4 pi. Free-free
and free-bound emission do not depend on the density; two-photon emission
does, so it is worked out on the same grid of densities as the lines.

In a synthesis file, the continuum of each group of overlapping spectral
windows is one entry, beside the lines, on evenly spaced wavelengths that
cover the windows. When the instrument simulation observes a window, it adds
up every entry that reaches it, so the continuum is added once, and the
lines' own spectra stay those of the lines.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import astropy.units as u

# What a continuum entry of a synthesis file is named: this, then the
# wavelengths in Angstrom that its window group runs between.
CONTINUUM_PREFIX = "continuum"
# The continuum's unit per emission measure: the spectral radiance per
# n_e^2 dh, with wavelengths in cm, as ECLIPSE's line spectra are.
EMISSIVITY_UNIT = u.erg * u.cm**3 / (u.s * u.sr * u.cm)


def is_continuum(name: str) -> bool:
    """Whether a synthesis file's entry *name* is the continuum rather than a line."""
    return name.startswith(CONTINUUM_PREFIX + "_")


def continuum_windows(wavelengths: Sequence[u.Quantity]) -> Dict[str, u.Quantity]:
    """
    The wavelengths of the continuum entries that cover a set of spectral windows.

    Each wavelength of a window stands for the bin halfway to its
    neighbours, and the end ones for a bin as wide again beyond them.
    Windows whose bins overlap are grouped, and each group gets one grid of
    evenly spaced wavelengths, with the smallest spacing any of them has,
    whose own bins cover all of theirs. Groups do not overlap, so no
    wavelength is in two entries.

    Parameters
    ----------
    wavelengths : sequence of u.Quantity
        The wavelengths of each line, increasing, as `synthesise_spectra`
        gives them in ``wl_grid``.

    Returns
    -------
    dict
        For each group, by its entry's name, such as
        ``"continuum_194.925-195.375"``, its wavelengths in cm.
    """
    windows = []
    for grid in wavelengths:
        values = np.asarray(u.Quantity(grid).to_value(u.cm), dtype=float)
        if values.size < 2:
            raise ValueError("A spectral window needs two or more wavelengths.")
        spacing = np.diff(values)
        if not np.all(spacing > 0):
            raise ValueError("A spectral window's wavelengths must increase.")
        # The window's bins, from half a spacing before its first wavelength
        # to half a spacing after its last.
        windows.append((float(values[0] - spacing[0] / 2), float(values[-1] + spacing[-1] / 2),
                        float(np.min(spacing))))
    def bins(low, high, step):
        """How many bins of *step* reach from *low* to *high*."""
        return int(np.ceil((high - low) / step * (1 - 1e-12)))

    windows.sort()
    groups: List[List[float]] = []
    for low, high, step in windows:
        # A group's bins can reach past its windows' last bin, so a window is
        # grouped with it if it begins before the group's bins end.
        if groups:
            first, last, spacing = groups[-1]
            if low < first + bins(first, last, spacing) * spacing:
                groups[-1] = [first, max(last, high), min(spacing, step)]
                continue
        groups.append([low, high, step])
    grids = {}
    for low, high, step in groups:
        # Bins of this spacing from the group's first bin edge to past its last.
        grid = low + (np.arange(bins(low, high, step)) + 0.5) * step
        name = f"{CONTINUUM_PREFIX}_{grid[0] * 1e8:.3f}-{grid[-1] * 1e8:.3f}"
        grids[name] = grid * u.cm
    return grids


def _element_continuum(args):
    """
    One element's free-free and free-bound, and two-photon, emission, for a worker process.

    In erg cm3 / (s Angstrom), per n_e n_H, over all directions, as fiasco
    gives them: free ``(nT, n_wavelength)`` and two-photon ``(nT, nN,
    n_wavelength)``.
    """
    symbol, temperature_K, wavelength_aa, densities_cm3, abundance, dbase_root = args
    import logging

    import fiasco
    from fiasco.util.exceptions import MissingDatasetException

    # fiasco leaves out an ion whose data a continuum needs, and logs it; the
    # ions are collected here and said once, rather than a line each.
    left_out = []

    class _LeftOut(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if " not included in " in message:
                ion, rest = message.split(" not included in ", 1)
                # An ion with no electrons has no bound states, and so no
                # free-bound emission to leave out.
                if ion.split()[-1] == str(atomic_number + 1):
                    return
                left_out.append(f"{ion} ({rest.split(' emission')[0]})")

    from .utils import element_data

    atomic_number, _ = element_data(symbol)
    logger = logging.getLogger("fiasco")
    handler, propagate = _LeftOut(), logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    single_ion = []
    try:
        kwargs = {} if dbase_root is None else {"hdf5_dbase_root": dbase_root}
        element = fiasco.Element(symbol, temperature_K * u.K, abundance=abundance, **kwargs)
        wavelength = wavelength_aa * u.AA
        unit = u.erg * u.cm**3 / (u.s * u.AA)
        free = (element.free_free(wavelength) + element.free_bound(wavelength)).to_value(unit)
        density = densities_cm3 * u.cm**-3
        two_photon = np.zeros((temperature_K.size, densities_cm3.size, wavelength_aa.size))
        for ion in element:
            # Only hydrogen- and helium-like ions give two-photon emission.
            if not (ion.hydrogenic or ion.helium_like):
                continue
            try:
                emission = _two_photon(ion, wavelength, density, unit, single_ion, kwargs,
                                       abundance)
            except MissingDatasetException:
                left_out.append(f"{ion.ion_name} (two-photon)")
                continue
            share = (u.Quantity(ion.abundance).to_value(u.dimensionless_unscaled)
                     * u.Quantity(ion.ionization_fraction).to_value(u.dimensionless_unscaled))
            two_photon += emission * share[:, np.newaxis, np.newaxis]
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate
    return (symbol, free, two_photon, str(element[0].hdf5_dbase_root), left_out,
            single_ion)


def _beyond_the_rates(error: ValueError) -> bool:
    """Whether fiasco stopped at a temperature beyond the rates CHIANTI tabulates."""
    return "interpolation range" in str(error)


def _two_photon(ion, wavelength, density, unit, single_ion: list, ion_kwargs: dict,
                abundance: str) -> np.ndarray:
    """
    One ion's two-photon emission, shaped (nT, nN, n_wavelength), in *unit*.

    fiasco is asked for one density at a time: given several, version 0.8
    fails to arrange its result. It works out the level populations with
    the recombination and ionisation rates of each level, which CHIANTI
    tabulates only up to some temperature for some ions. Above it, fiasco's
    single-ion model is used, as it is for ions that have no such rates at
    all, and the ion and the temperatures are added to *single_ion*.
    """
    import fiasco

    def at(one_density, ions):
        parts = []
        for part, single in ions:
            kwargs = {"use_two_ion_model": False} if single else {}
            parts.append(part.two_photon(wavelength, one_density, **kwargs).to_value(unit)[:, 0])
        return np.concatenate(parts)

    ions = [(ion, False)]
    columns = []
    for one_density in density:
        one_density = one_density[np.newaxis]
        try:
            columns.append(at(one_density, ions))
            continue
        except ValueError as error:
            if not _beyond_the_rates(error) or len(ions) > 1:
                raise
        # The first temperature the rates do not reach, found once.
        first = None
        for index, temperature in enumerate(ion.temperature):
            one = fiasco.Ion(ion.ion_name, temperature[np.newaxis], abundance=abundance,
                             **ion_kwargs)
            try:
                one.two_photon(wavelength, one_density)
            except ValueError as error:
                if not _beyond_the_rates(error):
                    raise
                first = index
                break
        if first is None:
            raise RuntimeError(f"fiasco could not work out the two-photon emission of "
                               f"{ion.ion_name} at its temperatures together, but could at "
                               f"each on its own.")
        below, above = ion.temperature[:first], ion.temperature[first:]
        ions = ([(fiasco.Ion(ion.ion_name, below, abundance=abundance, **ion_kwargs), False)]
                if below.size else [])
        ions.append((fiasco.Ion(ion.ion_name, above, abundance=abundance, **ion_kwargs), True))
        single_ion.append(f"{ion.ion_name} from log T "
                          f"{np.log10(above[0].to_value(u.K)):.2f}")
        columns.append(at(one_density, ions))
    return np.stack(columns, axis=1)


def compute_continuum_fiasco(
    wavelength: u.Quantity,
    logT_grid: np.ndarray,
    logN_grid: np.ndarray,
    abundance: str = "sun_coronal_2021_chianti",
    n_workers: int = 0,
    hdf5_dbase_root=None,
    elements: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The continuum per emission measure, from CHIANTI through fiasco.

    Parameters
    ----------
    wavelength : u.Quantity
        The wavelengths to work it out at.
    logT_grid : np.ndarray
        The temperatures, as log10(T / K), as the lines' contribution
        functions have them.
    logN_grid : np.ndarray
        The densities, as log10(n_e / cm-3), as the lines' contribution
        functions have them, for the two-photon emission.
    abundance : str, optional
        The CHIANTI abundance set, as for the lines. Default
        ``"sun_coronal_2021_chianti"``.
    n_workers : int, optional
        How many processes to compute the elements with. Default 0, which
        uses every CPU this process may use, up to one per element. The
        workers are spawned, so a script calling this needs an
        ``if __name__ == "__main__":`` guard.
    hdf5_dbase_root : str or Path, optional
        The CHIANTI database to read. Default fiasco's own.
    elements : sequence of str, optional
        The elements, by symbol. Default every element CHIANTI holds.

    Returns
    -------
    free : np.ndarray
        Free-free and free-bound emission, shaped (nT, n_wavelength), in
        erg cm3 / (s sr cm): the spectral radiance per n_e^2 dh.
    two_photon : np.ndarray
        Two-photon emission, shaped (nN, nT, n_wavelength), in the same unit,
        on the densities of *logN_grid*, as a line's ``g_tn`` is.
    """
    import fiasco

    from .atmosphere import _offer_database_build

    dbase_root = None if hdf5_dbase_root is None else str(hdf5_dbase_root)
    _offer_database_build(dbase_root)
    temperature_K = 10.0 ** np.asarray(logT_grid, dtype=float)
    densities_cm3 = 10.0 ** np.asarray(logN_grid, dtype=float)
    wavelength_aa = np.atleast_1d(u.Quantity(wavelength).to_value(u.AA))
    if elements is None:
        elements = fiasco.list_elements(dbase_root)
    worker_args = [(symbol, temperature_K, wavelength_aa, densities_cm3, abundance, dbase_root)
                   for symbol in elements]

    if n_workers <= 0:
        n_workers = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                     else os.cpu_count() or 1)
    if n_workers > 1 and len(worker_args) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(min(n_workers, len(worker_args))) as pool:
            results = pool.map(_element_continuum, worker_args)
    else:
        results = [_element_continuum(args) for args in worker_args]

    free = np.zeros((temperature_K.size, wavelength_aa.size))
    two_photon = np.zeros((temperature_K.size, densities_cm3.size, wavelength_aa.size))
    left_out = {ion for *_, ions, _ in results for ion in ions}
    single_ion = [entry for *_, entries in results for entry in entries]
    if single_ion:
        print(f"  CHIANTI's recombination and ionisation rates of each level stop short of "
              f"the hottest temperatures for {len(single_ion)} ions, whose two-photon "
              f"emission there is worked out with fiasco's single-ion model, as for ions "
              f"that have no such rates: {', '.join(single_ion)}")
    for kind in ("free-bound", "two-photon"):
        ions = [entry.rsplit(" (", 1)[0] for entry in left_out if entry.endswith(f"({kind})")]
        if ions:
            print(f"  CHIANTI has no {kind} data for {len(ions)} ions, which are left out of "
                  f"that part of the continuum: {_ion_list(ions)}")
    for symbol, element_free, element_two_photon, used, _, _ in results:
        if dbase_root is not None and used != dbase_root:
            raise RuntimeError(
                f"The continuum of {symbol} was worked out from the CHIANTI database at "
                f"{used} instead of the requested {dbase_root}.")
        if not (np.all(np.isfinite(element_free)) and np.all(np.isfinite(element_two_photon))):
            raise RuntimeError(f"fiasco gave a continuum for {symbol} that is not finite "
                               f"everywhere.")
        free += element_free
        two_photon += element_two_photon

    # fiasco's continuum is per n_e n_H and over all directions; the emission
    # measure is of n_e^2, and the radiance per steradian, per cm.
    kwargs = {} if dbase_root is None else {"hdf5_dbase_root": dbase_root}
    hydrogen = fiasco.Ion("H 1", temperature_K * u.K, abundance=abundance, **kwargs)
    ratio = np.asarray(hydrogen.proton_electron_ratio.to_value(u.dimensionless_unscaled))
    per_cm = (1.0 / u.AA).to_value(1.0 / u.cm)
    scale = ratio / (4.0 * np.pi) * per_cm
    free = free * scale[:, np.newaxis]
    two_photon = np.moveaxis(two_photon * scale[:, np.newaxis, np.newaxis], 1, 0)
    return free, two_photon


def continuum_spectra(
    emission_measure: np.ndarray,
    electron_density: np.ndarray,
    logN_grid: np.ndarray,
    free: np.ndarray,
    two_photon: np.ndarray,
) -> np.ndarray:
    """
    The continuum's spectral radiance in every pixel.

    Parameters
    ----------
    emission_measure : np.ndarray
        The emission measure, n_e^2 dh, in each temperature bin of each
        pixel, in cm^-5, shaped (rows, columns, nT): an ECLIPSE synthesis's
        ``em_tv`` summed over velocity, or a DEM times its bins' widths.
    electron_density : np.ndarray or float
        The electron density, in cm^-3, at each pixel and temperature,
        shaped as *emission_measure*, or one density for all. ECLIPSE's own
        synthesis gives the mean weighted by emission measure.
    logN_grid : np.ndarray
        The densities *two_photon* is tabulated on, as log10(n_e / cm-3).
    free, two_photon : np.ndarray
        As `compute_continuum_fiasco` gives them, on the temperatures of the
        emission measure.

    Returns
    -------
    np.ndarray
        The spectral radiance, (rows, columns, n_wavelength), in
        erg / (s cm2 sr cm). The two-photon emission is interpolated linearly
        in log10 n_e, as the lines' contribution functions are, and is zero at
        densities off the grid, as they are.
    """
    em = np.asarray(emission_measure, dtype=float)
    rows, columns, n_t = em.shape
    if free.shape[0] != n_t or two_photon.shape[1] != n_t:
        raise ValueError(f"The emission measure has {n_t} temperatures, but the continuum "
                         f"has {free.shape[0]}.")
    density = np.broadcast_to(np.asarray(electron_density, dtype=float), em.shape)
    flat = em.reshape(-1, n_t)
    spectra = flat @ free
    log_n = np.log10(np.where(density > 0, density, np.nan)).reshape(-1, n_t)
    grid = np.asarray(logN_grid, dtype=float)
    for t in range(n_t):
        weight = flat[:, t]
        lit = (weight > 0) & np.isfinite(log_n[:, t])
        lit &= (log_n[:, t] >= grid[0]) & (log_n[:, t] <= grid[-1])
        if not lit.any():
            continue
        position = np.interp(log_n[lit, t], grid, np.arange(grid.size, dtype=float))
        lower = np.floor(position).astype(int)
        upper = np.minimum(lower + 1, grid.size - 1)
        fraction = position - lower
        table = two_photon[:, t, :]
        spectra[lit] += weight[lit, np.newaxis] * (
            (1.0 - fraction)[:, np.newaxis] * table[lower] + fraction[:, np.newaxis] * table[upper])
    return spectra.reshape(rows, columns, -1)


def _ion_list(ions: Sequence[str]) -> str:
    """Ions such as ``["Fe 1", "Fe 2", "Fe 27"]`` as ``"Fe 1-2, 27"``, element by element."""
    stages: Dict[str, List[int]] = {}
    for ion in ions:
        symbol, stage = ion.split()
        stages.setdefault(symbol, []).append(int(stage))
    parts = []
    from .utils import element_data

    for symbol in sorted(stages, key=lambda s: element_data(s)[0]):
        numbers = sorted(set(stages[symbol]))
        runs, start = [], numbers[0]
        for previous, number in zip(numbers, numbers[1:] + [None]):
            if number is None or number != previous + 1:
                runs.append(str(start) if start == previous else f"{start}-{previous}")
                start = number
        parts.append(f"{symbol} {', '.join(runs)}")
    return "; ".join(parts)
