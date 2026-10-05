"""
A full-CCD frame of the SW camera: every line in the band, at one slit position.

ECLIPSE observes one spectral window at a time. A full-CCD frame holds the
whole SW band on both CCDs, as the camera reads it out. It is built from a
synthesis file, which keeps the emission measure in each bin of temperature
and line-of-sight velocity of every pixel, and the electron density at each
temperature, and from the contribution functions of every line in the band
(`euvst_response.band`).

The light in each row of a CCD is worked out one temperature at a time. At
each, every line of an ion that exists there is a Gaussian at each velocity
bin's Doppler shift, with its thermal width at that temperature, blurred by
the instrument's spectral response and laid onto the rows as
`euvst_response.frame.photons_from_lines` lays it: each part of a line is
counted in photons at its own wavelength, with the telescope's throughput
there. Each detector column sees the emission measure of the scene under
the slit at its place along it. Each pixel's emission measure is shared
between the two densities of the grid either side of its own, as the
synthesis does when it takes the contribution functions at it. The
continuum, if given, is added as a spectrum. The light then goes through
the exposure, the detector and the digitiser, each photon freeing the
electrons of its own energy, and the prescan and overscan samples are laid
around the image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import astropy.constants as const
import astropy.units as u

from .band import BandLines
from .frame import (_REACH_IN_SIGMA, _cells, _photons_per_energy, _response, _sorted_rows,
                    detect, digitise, expose_with_wavelength, pixel_solid_angle)
from .radiometric import photons_per_energy
from .readout import FocalPlane_SWC, ReadoutSequence, dark_current_time
from .sampling import light_onto_pixels, pixel_weights, points_onto_pixels
from .utils import _fwhm_to_sigma, distance_to_angle

CCDS = ("left", "right")
C_CM_S = const.c.to_value(u.cm / u.s)
K_B = const.k_B.cgs.value
AMU_G = const.u.cgs.value


def _density_shares(log_density: np.ndarray, logN_grid: np.ndarray):
    """
    The two densities of the grid either side of each density, and the share
    of its emission measure each takes, linearly in log10 n_e, as the
    contribution functions are interpolated. Off the grid, both shares are
    zero, as the contribution functions are there.
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


def _on_the_grid(points: np.ndarray, grid: np.ndarray) -> Optional[np.ndarray]:
    """
    Where each of a synthesis's densities is on the band's grid, or None if one is not on it.

    A synthesis's density grid is the run of points of ECLIPSE's default
    grid that its plasma needs, so the lines worked out on the default grid
    serve every synthesis whose densities it covers.
    """
    grid = np.asarray(grid, dtype=float)
    index = np.abs(np.asarray(points, dtype=float)[:, np.newaxis] - grid).argmin(axis=1)
    return index if np.allclose(grid[index], points, rtol=0.0, atol=1e-6) else None


def _angle(edges: u.Quantity) -> np.ndarray:
    """Edges on the Sun, as lengths or as angles from 1 AU, in arcsec."""
    edges = u.Quantity(edges)
    if edges.unit.is_equivalent(u.arcsec):
        return edges.to_value(u.arcsec)
    return distance_to_angle(edges).to_value(u.arcsec)


@dataclass
class _Scene:
    """What a frame takes from a synthesis file, under the slit."""

    logT_grid: np.ndarray
    logN_grid: np.ndarray
    velocity: np.ndarray            # bin centres, cm/s
    emission_measure: np.ndarray    # (rows of the image, nT, nN, nv), under the slit
    y_edges: np.ndarray             # arcsec, along the slit, from the image's centre


def read_slit_scene(path: str | Path, slit_position: u.Quantity, slit_width: u.Quantity,
                    psf_across_slit: Optional[u.Quantity] = None) -> _Scene:
    """
    The emission measure under the slit, by temperature, density and velocity, from a synthesis file.

    Parameters
    ----------
    path : str or Path
        A synthesis file from ECLIPSE's own synthesis, which keeps the
        emission measure (``em_tv``) and the electron density at each
        temperature.
    slit_position : u.Quantity
        Where the slit's centre is across it, in the file's x coordinates.
    slit_width : u.Quantity
        The slit, as an angle.
    psf_across_slit : u.Quantity, optional
        The FWHM of the telescope's blur across the slit, which brings in
        emission from beside it. Default None, for none.
    """
    path = Path(path).expanduser()
    with h5py.File(path, "r") as f:
        x_edges = u.Quantity(f["x_edges"][()], f["x_edges"].attrs["unit"])
        y_edges = u.Quantity(f["y_edges"][()], f["y_edges"].attrs["unit"])
        products = f["synthesis"]
        missing = [key for key in ("em_tv", "electron_density", "logT_grid", "logN_grid",
                                   "vel_grid") if key not in products]
        if missing:
            raise ValueError(
                f"{path} has no {', '.join(missing)}, which a full-CCD frame needs: it was "
                f"not made by this version of ECLIPSE's synthesis. Synthesise it again.")
        x = _angle(x_edges)
        centre = _angle(u.Quantity(slit_position)[np.newaxis] if
                        u.Quantity(slit_position).isscalar else slit_position)[0]
        width = u.Quantity(slit_width).to_value(u.arcsec)
        sigma = (0.0 if psf_across_slit is None
                 else _fwhm_to_sigma(u.Quantity(psf_across_slit).to_value(u.arcsec)))
        across = pixel_weights(x, np.array([centre - width / 2, centre + width / 2]), sigma)[0]
        columns = np.flatnonzero(across > 0)
        if columns.size == 0:
            raise ValueError(f"The slit at {slit_position} is off the image, which runs from "
                             f"{x_edges[0]} to {x_edges[-1]}.")
        first, last = columns[0], columns[-1] + 1
        em = products["em_tv"][:, first:last, :, :].astype(float)
        density = products["electron_density"][:, first:last, :].astype(float)
        logT = products["logT_grid"][()]
        logN = products["logN_grid"][()]
        velocity = u.Quantity(products["vel_grid"][()],
                              products["vel_grid"].attrs.get("unit", "cm / s")).to_value(u.cm / u.s)
    weights = across[first:last]
    lower, upper, low_share, high_share = _density_shares(
        np.log10(np.where(density > 0, density, np.nan)), logN)
    ny, _, n_t, n_v = em.shape
    scene = np.zeros((ny, n_t, logN.size, n_v))
    for k, weight in enumerate(weights):
        for share, index in ((low_share, lower), (high_share, upper)):
            part = weight * share[:, k, :, np.newaxis] * em[:, k, :, :]
            rows, temperatures = np.indices(index[:, k, :].shape)
            np.add.at(scene, (rows, temperatures, index[:, k, :]), part)
    y = _angle(y_edges)
    return _Scene(logT_grid=logT, logN_grid=logN, velocity=velocity, emission_measure=scene,
                  y_edges=y - 0.5 * (y[0] + y[-1]))


@dataclass
class _Layout:
    """
    How one CCD's rows take light, worked out once for every temperature.

    A line is laid onto the rows by the exact integrals of its Gaussian, with
    the telescope's throughput and the photon energy at its centre, and is
    then corrected cell by cell for how those change across it (see
    :func:`euvst_response.frame.photons_from_lines`). The cells are fixed in
    the plasma's frame; in each velocity bin they move and stretch with the
    Doppler shift, so each bin has its own way onto the rows.
    """

    edges: np.ndarray                 # the rows' edges, increasing, in Angstrom
    order: np.ndarray                 # the rows into that order
    lit: Tuple[int, int]              # the first and last lit row
    cells: np.ndarray                 # the cells' edges at rest, in Angstrom
    onto_rows: list                   # each velocity bin's cells onto the rows, (rows, cells)
    collected: np.ndarray             # photons per erg at each bin's cells, (nv, cells)
    cell_wavelength: np.ndarray       # each bin's cells' middles, (nv, cells)
    continuum: Optional[tuple] = None  # the continuum's intervals onto the rows, and theirs


def _layout(focal_plane: FocalPlane_SWC, ccd: str, telescope, det, slit_width: u.Quantity,
            blur: Optional[str], velocity: np.ndarray, solid_angle: float,
            continuum_wavelength: Optional[np.ndarray]) -> _Layout:
    """The light onto one CCD's rows, for `full_ccd_frame`."""
    edges, order = _sorted_rows(focal_plane, ccd)
    if blur is None:
        reach = 0.0
    else:
        sigma, box = _response(focal_plane, ccd, None, telescope, det, slit_width, blur,
                               edges[[0, -1]])
        reach = _REACH_IN_SIGMA * sigma.max() + box.max()
    # At rest, the cells reach as far as the fastest Doppler shift can bring
    # light onto the rows from.
    shift = (1.0 + velocity / C_CM_S)
    reach += np.abs(edges[[0, -1]]).max() * np.abs(shift - 1.0).max()
    cells = _cells(focal_plane, ccd, None, edges, reach)
    onto_rows, collected, cell_wavelength = [], [], []
    for factor in shift:
        low, high = cells[:-1] * factor, cells[1:] * factor
        middle = (low + high) / 2
        if blur is None:
            sigma = box = 0.0
        else:
            sigma, box = _response(focal_plane, ccd, None, telescope, det, slit_width, blur,
                                   middle)
        onto_rows.append(light_onto_pixels(low, high, edges, sigma, box))
        # Photons per second per pixel for a unit of G times the emission
        # measure: the radiance G EM / 4 pi through the pixel's solid angle,
        # in photons at the cell's own wavelength; none beyond the tables.
        collected.append(solid_angle / (4.0 * np.pi) * np.nan_to_num(
            photons_per_energy(telescope, middle * u.Angstrom).to_value(u.cm**2 / u.erg)))
        cell_wavelength.append(middle)
    continuum = None
    if continuum_wavelength is not None:
        # The continuum is not Doppler shifted. It is laid as a spectrum is,
        # each interval between its wavelengths holding the trapezium rule's
        # light, counted in photons at its own wavelengths.
        low, high = continuum_wavelength[:-1], continuum_wavelength[1:]
        if blur is None:
            sigma = box = 0.0
        else:
            sigma, box = _response(focal_plane, ccd, None, telescope, det, slit_width, blur,
                                   (low + high) / 2)
        # The continuum is a radiance already, per steradian.
        at_samples = solid_angle * _photons_per_energy(
            telescope, continuum_wavelength * u.Angstrom).to_value(u.cm**2 / u.erg)
        continuum = (light_onto_pixels(low, high, edges, sigma, box), at_samples)
    return _Layout(edges=edges, order=order, lit=focal_plane.lit_rows(ccd), cells=cells,
                   onto_rows=onto_rows, collected=np.array(collected),
                   cell_wavelength=np.array(cell_wavelength), continuum=continuum)


def _lines_on_rows(layout: _Layout, focal_plane: FocalPlane_SWC, ccd: str, telescope, det,
                   slit_width: u.Quantity, blur: Optional[str], wavelength: np.ndarray,
                   thermal: np.ndarray, g: np.ndarray, velocity: np.ndarray,
                   bins: np.ndarray, solid_angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    The photons per second in each row of one CCD for a unit of emission measure, at one temperature.

    Parameters
    ----------
    wavelength, thermal : np.ndarray
        Each line's rest wavelength and its thermal width, in Angstrom.
    g : np.ndarray
        Each line's contribution function at this temperature, (n_lines, nN).
    velocity : np.ndarray
        The velocity bins' centres, in cm/s.
    bins : np.ndarray
        The velocity bins with emission measure, the only ones worked out.

    Returns
    -------
    photons, per_wavelength : np.ndarray
        ``(nN, nv, n_rows)``: photons per second per pixel per cm^-5, and
        the same with each photon divided by its wavelength in Angstrom, from
        which the mean energy of a pixel's photons follows.
    """
    n_rows = layout.edges.size - 1
    photons = np.zeros((g.shape[1], velocity.size, n_rows))
    per_wavelength = np.zeros_like(photons)
    masses = points_onto_pixels(wavelength, layout.cells, thermal)          # (cells, lines)
    at_rest = masses @ g                                                     # (cells, nN)
    for v in bins:
        factor = 1.0 + velocity[v] / C_CM_S
        centre = wavelength * factor
        at_centre = solid_angle / (4.0 * np.pi) * np.nan_to_num(
            photons_per_energy(telescope, centre * u.Angstrom).to_value(u.cm**2 / u.erg))
        if blur is None:
            psf_sigma = box = np.zeros(centre.size)
        else:
            psf_sigma, box = _response(focal_plane, ccd, None, telescope, det, slit_width,
                                       blur, centre)
        # Each line, its Gaussian blurred by the response, onto the rows by
        # its exact integrals, in photons at its centre's wavelength ...
        exact = points_onto_pixels(centre, layout.edges,
                                   np.hypot(thermal * factor, psf_sigma), box)
        weighted = g * at_centre[:, np.newaxis]
        per_centre = weighted / centre[:, np.newaxis]
        rows = exact @ weighted
        rows_per = exact @ per_centre
        # ... and, cell by cell across each line, the photons at the cell's
        # own wavelength rather than at the centre's.
        collected = layout.collected[v][:, np.newaxis]
        own = layout.cell_wavelength[v][:, np.newaxis]
        change = collected * at_rest - masses @ weighted
        change_per = collected / own * at_rest - masses @ per_centre
        rows += layout.onto_rows[v] @ change
        rows_per += layout.onto_rows[v] @ change_per
        photons[:, v, layout.order] = rows.T
        per_wavelength[:, v, layout.order] = rows_per.T
    first, last = layout.lit
    for array in (photons, per_wavelength):
        array[..., :first] = 0.0
        array[..., last + 1:] = 0.0
    return photons, per_wavelength


# What the workers of `full_ccd_frame` share, set before they are forked.
_CONTEXT: Optional[dict] = None


def _rates_at(temperatures: Sequence[int]):
    """
    The photons per second in each pixel of both CCDs from the plasma at some
    temperatures, and the same with each photon divided by its wavelength in
    Angstrom, by CCD, each (n_columns, n_rows).
    """
    context = _CONTEXT
    scene, lines, along = context["scene"], context["lines"], context["along"]
    focal_plane, telescope, det = context["focal_plane"], context["telescope"], context["det"]
    slit_width, blur, continuum = context["slit_width"], context["blur"], context["continuum"]
    density_index, layouts = context["density_index"], context["layouts"]
    solid_angle = context["solid_angle"]
    n_n, n_v = scene.logN_grid.size, scene.velocity.size
    shape = (focal_plane.n_columns, focal_plane.n_rows)
    rate = {ccd: np.zeros(shape) for ccd in CCDS}
    per_wavelength = {ccd: np.zeros(shape) for ccd in CCDS}
    for t in temperatures:
        em = scene.emission_measure[:, t]                              # (image rows, nN, nv)
        em_columns = along @ em.reshape(em.shape[0], -1)               # (columns, nN * nv)
        bins = np.flatnonzero(em_columns.reshape(-1, n_n, n_v).any(axis=(0, 1)))
        present = [ion for ion in lines.ions if t in ion.temperatures]
        if present and bins.size:
            g = np.concatenate([ion.g[:, density_index, np.searchsorted(ion.temperatures, t)]
                                for ion in present])
            wavelength = np.concatenate([ion.wavelength for ion in present])
            mass = np.concatenate([np.full(ion.wavelength.size, ion.mass) for ion in present])
            thermal = (wavelength * np.sqrt(K_B * 10.0 ** scene.logT_grid[t] / (mass * AMU_G))
                       / C_CM_S)
            for ccd in CCDS:
                layout = layouts[ccd]
                # The lines whose light can reach the cells.
                near = ((wavelength + _REACH_IN_SIGMA * thermal >= layout.cells[0])
                        & (wavelength - _REACH_IN_SIGMA * thermal <= layout.cells[-1]))
                if not near.any():
                    continue
                photons, per = _lines_on_rows(
                    layout, focal_plane, ccd, telescope, det, slit_width, blur,
                    wavelength[near], thermal[near], g[near], scene.velocity, bins,
                    solid_angle)
                rate[ccd] += em_columns @ photons.reshape(n_n * n_v, -1)
                per_wavelength[ccd] += em_columns @ per.reshape(n_n * n_v, -1)
        if continuum is not None:
            wavelength, free, two_photon = continuum
            em_density = em_columns.reshape(-1, n_n, n_v).sum(axis=-1)   # (columns, nN)
            for ccd in CCDS:
                onto_rows, at_samples = layouts[ccd].continuum
                order = layouts[ccd].order
                # Photons per second per pixel per Angstrom at each sample, for
                # a unit of emission measure at each density.
                spectrum = (free[t][np.newaxis, :] + two_photon[density_index, t]) * 1e-8
                density = spectrum * at_samples
                step = np.diff(wavelength)
                light = step * (density[:, 1:] + density[:, :-1]) / 2
                per_sample = density / wavelength
                light_per = step * (per_sample[:, 1:] + per_sample[:, :-1]) / 2
                rows = np.zeros((n_n, focal_plane.n_rows))
                rows_per = np.zeros_like(rows)
                rows[:, order] = (onto_rows @ light.T).T
                rows_per[:, order] = (onto_rows @ light_per.T).T
                first, last = layouts[ccd].lit
                for array in (rows, rows_per):
                    array[:, :first] = 0.0
                    array[:, last + 1:] = 0.0
                rate[ccd] += em_density @ rows
                per_wavelength[ccd] += em_density @ rows_per
    return rate, per_wavelength


@dataclass
class FullFrame:
    """
    A full-CCD frame of both SW CCDs, at one slit position.

    For each CCD, by name, ``"left"`` and ``"right"``:

    Attributes
    ----------
    rate : dict
        Photons per second reaching each image pixel, (n_rows, n_columns).
    photons : dict
        Photons in each pixel of the frame, with the parallel overscan rows,
        the mean if *noise* was off.
    dn : dict
        The frame as read out: the prescan, the image and the overscan of
        each output, in DN, (n_rows + parallel overscan, samples per row).
    wavelength : dict
        The wavelength each image row records.
    photon_wavelength : dict
        The wavelength of the mean energy of the photons in each pixel of
        the frame, with the parallel overscan rows, which sets the electrons
        they free.
    exposure : u.Quantity
    slit_position, slit_width : u.Quantity
    """

    rate: Dict[str, np.ndarray]
    photons: Dict[str, np.ndarray]
    dn: Dict[str, np.ndarray]
    wavelength: Dict[str, u.Quantity]
    photon_wavelength: Dict[str, u.Quantity]
    exposure: u.Quantity
    slit_position: u.Quantity
    slit_width: u.Quantity
    settings: dict = field(default_factory=dict)


def lay_out_register(image_dn: np.ndarray, det, noise: bool = True) -> np.ndarray:
    """
    A frame's samples as the two outputs read them, with their prescan and overscan.

    Each output reads half the columns, with the prescan at the outer end of
    the register and the overscan at the inner end (`ReadoutSequence`), so a
    row is, in order across the CCD: the left output's prescan, the image's
    first half, the left output's overscan, the right output's overscan, the
    image's second half and the right output's prescan. The prescan and
    overscan samples hold no charge from the image area, so they read the
    read noise alone; the serial register's own dark current is not modelled,
    as it is not elsewhere.
    """
    n_rows, n_columns = image_dn.shape
    half = (n_columns + 1) // 2
    prescan, overscan = det.serial_prescan, det.serial_overscan
    empty = np.zeros((n_rows, 2 * (prescan + overscan)))
    blank = digitise(detect(empty, np.full(n_rows, 200.0) * u.AA, 0.0 * u.s, det,
                            noise=noise), det)
    pieces = np.split(blank, np.cumsum([prescan, overscan, overscan]), axis=1)
    return np.concatenate([pieces[0], image_dn[:, :half], pieces[1], pieces[2],
                           image_dn[:, half:], pieces[3]], axis=1)


def full_ccd_frame(synthesis_path: str | Path, lines: BandLines, slit_position: u.Quantity,
                   exposure: u.Quantity, *, slit_width: u.Quantity = 0.2 * u.arcsec,
                   det=None, telescope=None, focal_plane: Optional[FocalPlane_SWC] = None,
                   psf: bool = True, spectral_psf: str = "quadrature",
                   continuum: Optional[Tuple[u.Quantity, np.ndarray, np.ndarray]] = None,
                   noise: bool = True, n_workers: int = 0) -> FullFrame:
    """
    A full-CCD frame of the SW camera, from a synthesis and every line in the band.

    Parameters
    ----------
    synthesis_path : str or Path
        A synthesis file from ECLIPSE's synthesis.
    lines : BandLines
        Every line in the band, from
        `euvst_response.band.band_contribution_functions`, on the synthesis's
        temperatures, on a density grid that has the synthesis's densities,
        such as the default one, and with its abundances.
    slit_position : u.Quantity
        Where the slit's centre is across it, in the file's x coordinates.
        Along the slit, the image's centre is at the slit's.
    exposure : u.Quantity
        The exposure time.
    slit_width : u.Quantity, optional
        Default 0.2 arcsec.
    det, telescope : optional
        Default `Detector_SWC()` and `Telescope_EUVST()`.
    focal_plane : FocalPlane_SWC, optional
        Default `FocalPlane_SWC.from_detector(det)`.
    psf : bool, optional
        Blur with the telescope's PSF, along the slit, across it if the
        telescope gives that, and in wavelength. Default True.
    spectral_psf : str, optional
        ``"quadrature"`` (default) or ``"convolution"``, as in the
        configuration.
    continuum : tuple, optional
        The continuum per emission measure: wavelengths, free-free and
        free-bound emission (nT, n_wavelength) and two-photon emission (nN,
        nT, n_wavelength), as `euvst_response.continuum.compute_continuum_fiasco`
        gives them on the grids of *lines*, within the telescope's throughput
        tables. Default None, for none.
    noise : bool, optional
        With False, every random draw is replaced by its mean. Default True.
    n_workers : int, optional
        How many processes to work out the temperatures with. Default 0,
        which uses every CPU this process may use.

    Returns
    -------
    FullFrame
    """
    from . import frame as frame_module
    from .config import Detector_SWC, Telescope_EUVST

    det = Detector_SWC() if det is None else det
    telescope = Telescope_EUVST() if telescope is None else telescope
    focal_plane = FocalPlane_SWC.from_detector(det) if focal_plane is None else focal_plane
    if any(value != 0.0 for value in focal_plane.slit_image_tilt):
        raise ValueError("The slit's image is tilted, so each column's rows have their own "
                         "wavelengths, which this frame does not yet work out.")
    across = telescope.psf_across_slit if psf else None
    scene = read_slit_scene(synthesis_path, slit_position, slit_width, across)
    if not np.array_equal(scene.logT_grid, lines.logT_grid):
        raise ValueError("The band's lines are on another grid of temperature than the "
                         "synthesis; work them out on the synthesis's own.")
    density_index = _on_the_grid(scene.logN_grid, lines.logN_grid)
    if density_index is None:
        raise ValueError(
            f"The synthesis takes the contribution functions at densities, log10 n_e "
            f"{scene.logN_grid[0]:.2f} to {scene.logN_grid[-1]:.2f}, that are not all on the "
            f"band's grid, {lines.logN_grid[0]:.2f} to {lines.logN_grid[-1]:.2f}; work the "
            f"lines out on a grid that has them.")
    from .synthesis_file import read_synthesis_products

    abundance = (read_synthesis_products(synthesis_path, keys=["config"]).get("config")
                 or {}).get("abundance")
    if abundance is not None and abundance != lines.abundance:
        raise ValueError(f"The synthesis used the abundances {abundance} and the band's lines "
                         f"{lines.abundance}; work the lines out with the synthesis's.")
    if continuum is not None:
        wavelength, free, two_photon = continuum
        expected = (lines.logN_grid.size, lines.logT_grid.size, u.Quantity(wavelength).size)
        if np.shape(two_photon) != expected or np.shape(free) != expected[1:]:
            raise ValueError(
                f"The continuum is shaped {np.shape(free)} and {np.shape(two_photon)}, not "
                f"{expected[1:]} and {expected}: work it out on the band's grids.")

    # Along the slit: each detector column's share of each row of the image.
    columns = np.arange(focal_plane.n_columns + 1) - 0.5
    column_edges = focal_plane.field_angle(columns).to_value(u.arcsec)
    sigma = (_fwhm_to_sigma(telescope.psf_params[0].to_value(u.pix))
             * focal_plane.plate_scale.to_value(u.arcsec) if psf else 0.0)
    along = pixel_weights(scene.y_edges, column_edges, sigma)        # (columns, image rows)

    blur = spectral_psf if psf else None
    temperatures = [t for t in range(scene.logT_grid.size)
                    if np.any(scene.emission_measure[:, t])]
    solid_angle = pixel_solid_angle(focal_plane, slit_width).to_value(u.sr)
    if continuum is not None:
        wavelength, free, two_photon = continuum
        continuum = (u.Quantity(wavelength).to_value(u.Angstrom), free, two_photon)
    # How each CCD's rows take light, the same at every temperature, worked
    # out once before the workers share it.
    layouts = {ccd: _layout(focal_plane, ccd, telescope, det, u.Quantity(slit_width), blur,
                            scene.velocity, solid_angle,
                            None if continuum is None else continuum[0])
               for ccd in CCDS}
    context = {"scene": scene, "lines": lines, "along": along, "focal_plane": focal_plane,
               "telescope": telescope, "det": det, "slit_width": u.Quantity(slit_width),
               "blur": blur, "continuum": continuum, "density_index": density_index,
               "layouts": layouts, "solid_angle": solid_angle}
    if n_workers <= 0:
        n_workers = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                     else os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(temperatures)))
    shares = [temperatures[k::n_workers] for k in range(n_workers)]
    global _CONTEXT
    _CONTEXT = context
    try:
        if n_workers > 1:
            import multiprocessing as mp

            # Forked, so that each worker shares the scene and the lines
            # rather than being sent them.
            with mp.get_context("fork").Pool(n_workers) as pool:
                parts = pool.map(_rates_at, shares)
        else:
            parts = [_rates_at(share) for share in shares]
    finally:
        _CONTEXT = None
    rate = {ccd: sum(part[0][ccd] for part in parts) for ccd in CCDS}
    per_wavelength = {ccd: sum(part[1][ccd] for part in parts) for ccd in CCDS}

    sequence = ReadoutSequence.from_detector(det)
    dark = dark_current_time(exposure, sequence, focal_plane.n_rows)
    rows = np.arange(focal_plane.n_rows)
    frame = {"rate": {}, "photons": {}, "dn": {}, "wavelength": {}, "photon_wavelength": {}}
    for ccd in CCDS:
        image_rate = rate[ccd].T                                          # (rows, columns)
        wavelength = focal_plane.wavelength(rows, ccd)
        # The wavelength of the mean energy of the photons reaching each
        # pixel; a pixel none reach keeps its row's.
        own = np.broadcast_to(wavelength.to_value(u.Angstrom)[:, np.newaxis], image_rate.shape)
        reaching = np.divide(image_rate, per_wavelength[ccd].T, out=np.array(own, dtype=float),
                             where=per_wavelength[ccd].T > 0) * u.Angstrom
        photons, mean_wavelength = expose_with_wavelength(image_rate, reaching, exposure,
                                                          sequence)
        drawn = np.random.poisson(photons).astype(float) if noise else photons
        electrons = detect(drawn, mean_wavelength, dark, det, noise=noise)
        frame["rate"][ccd] = image_rate
        frame["photons"][ccd] = drawn
        frame["dn"][ccd] = lay_out_register(digitise(electrons, det), det, noise=noise)
        frame["wavelength"][ccd] = wavelength
        frame["photon_wavelength"][ccd] = mean_wavelength
    return FullFrame(**frame, exposure=u.Quantity(exposure),
                     slit_position=u.Quantity(slit_position), slit_width=u.Quantity(slit_width),
                     settings={"psf": psf, "spectral_psf": spectral_psf, "noise": noise,
                               "continuum": continuum is not None,
                               "cells_per_row": frame_module.CELLS_PER_ROW})


FRAME_FORMAT = "eclipse-full-ccd-frame"
FRAME_VERSION = 1


def write_full_frame(frame: FullFrame, path: str | Path) -> Path:
    """
    Write a full-CCD frame to an HDF5 file.

    The root holds the settings as attributes, and a group for each CCD,
    ``left`` and ``right``, holds ``dn``, the frame as read out with its
    prescan and overscan, ``photons``, ``rate``, ``wavelength``, the
    wavelength each image row records, and ``photon_wavelength``, the
    wavelength of the mean energy of each pixel's photons, each with its
    unit.

    Parameters
    ----------
    frame : FullFrame
        The frame.
    path : str or Path
        The file to write. An existing file is replaced.

    Returns
    -------
    Path
        The file written.
    """
    from .atmosphere import _replacing

    path = Path(path).expanduser()
    with _replacing(path) as partial, h5py.File(partial, "w") as f:
        f.attrs["format"] = FRAME_FORMAT
        f.attrs["version"] = FRAME_VERSION
        for name in ("exposure", "slit_position", "slit_width"):
            value = getattr(frame, name)
            f.attrs[name] = value.value
            f.attrs[f"{name}_unit"] = str(value.unit)
        for key, value in frame.settings.items():
            f.attrs[key] = value
        for ccd in CCDS:
            group = f.create_group(ccd)
            for name, data, unit in (("dn", frame.dn[ccd], "DN"),
                                     ("photons", frame.photons[ccd], "photon"),
                                     ("rate", frame.rate[ccd], "photon / s"),
                                     ("wavelength", frame.wavelength[ccd].to_value(u.AA),
                                      "Angstrom"),
                                     ("photon_wavelength",
                                      frame.photon_wavelength[ccd].to_value(u.AA), "Angstrom")):
                group.create_dataset(name, data=data, compression="gzip")
                group[name].attrs["unit"] = unit
    return path
