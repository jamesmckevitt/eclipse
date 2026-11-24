"""
Full-CCD frames of the SW camera from a synthesis file at one slit position.

Every entry of the file is taken as a spectral radiance on its own
wavelengths, and the entries are added. A file with three lines gives
three lines on the CCDs, and a file synthesised over the whole band
(`eclipse.band`) gives every line in it.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import astropy.units as u

from .band import BandLines
from .deprecated import _band_scene
from .config import Simulation
from .frame import (_response, _sorted_rows, detect, digitise,
                    expose_with_serial_overscan_with_wavelengths, expose_with_wavelengths,
                    pixel_solid_angle)
from .radiometric import (_poisson_inverse_transform, across_slit_fwhm, photons_per_energy,
                          spatial_psf_fwhm)
from .config import _FOCAL_PLANE_SETTINGS, _same_setting
from .data_processing import _openings_along_the_slit
from .readout import FocalPlane_SWC, ReadoutSequence, dark_current_time, serial_dark_shares
from .sampling import light_onto_pixels, pixel_weights, pixel_weights_through
from .utils import (_bin_edges, _fwhm_to_sigma, _get_software_version, distance_to_angle,
                    get_git_commit_id)

CCDS = ("left", "right")


def _angle(edges: u.Quantity) -> np.ndarray:
    """Edges on the Sun, given as lengths or angles, in arcsec seen from 1 AU."""
    edges = u.Quantity(edges)
    if edges.unit.is_equivalent(u.arcsec):
        return edges.to_value(u.arcsec)
    return distance_to_angle(edges).to_value(u.arcsec)


def _scene_weights(cell_edges: np.ndarray, pixel_edges: np.ndarray, sigma: float,
                   extend: bool = False, openings: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Weight of each scene cell in each pixel, `(n_pixels, n_cells)`.

    With `extend` and a blur, the edge cells continue past the scene, and
    each pixel takes light only over its part inside the scene, to match a
    run, whose pixels all lie inside it. With `openings`, `(n, 2)` in the
    unit of the edges, light gets through only there, after the blur.
    """
    if not (extend and sigma > 0):
        if openings is None:
            return pixel_weights(cell_edges, pixel_edges, sigma)
        return pixel_weights_through(cell_edges, pixel_edges, openings, sigma)
    low, high = cell_edges[0], cell_edges[-1]
    inside = np.clip([[low, high]] if openings is None else openings, low, high)
    inside = inside[inside[:, 1] > inside[:, 0]]
    return pixel_weights_through(cell_edges, pixel_edges, inside, sigma, extend=True)


@dataclass
class _Scene:
    """Spectra under the slit, read from a synthesis file."""

    y_edges: np.ndarray             # arcsec along the slit, from the image centre
    names: List[str]                # entries of the file the spectra came from
    edges: List[np.ndarray]         # each entry's wavelength bin edges, in Angstrom
    radiance: List[np.ndarray]      # each entry's (image rows, n_wavelength), erg / (s cm2 sr cm)


def _slit_columns(x_edges: u.Quantity, slit_position: u.Quantity, slit_width: u.Quantity,
                  psf_across_slit: Optional[u.Quantity], extend: bool):
    """
    Image columns the slit takes light from, as a slice, and the weight of each.

    Raises ValueError if the slit is off the image.
    """
    slit_position = u.Quantity(slit_position)
    if not slit_position.isscalar:
        raise ValueError(f"slit_position must be one position, got {slit_position}.")
    x = _angle(x_edges)
    centre = _angle(slit_position[np.newaxis])[0]
    width = u.Quantity(slit_width).to_value(u.arcsec)
    sigma = (0.0 if psf_across_slit is None
             else _fwhm_to_sigma(u.Quantity(psf_across_slit).to_value(u.arcsec)))
    slit = np.array([centre - width / 2, centre + width / 2])
    across = _scene_weights(x, slit, sigma, extend)[0]
    columns = np.flatnonzero(across > 0)
    if columns.size == 0:
        raise ValueError(f"The slit at {slit_position} is off the image, which runs from "
                         f"{x_edges[0]} to {x_edges[-1]}.")
    columns = slice(columns[0], columns[-1] + 1)
    return columns, across[columns]


def _centred(y_edges: u.Quantity) -> np.ndarray:
    """Edges along the slit in arcsec, from the middle of the image."""
    y = _angle(y_edges)
    return y - 0.5 * (y[0] + y[-1])


def read_slit_spectra(path: str | Path, slit_position: u.Quantity, slit_width: u.Quantity,
                      psf_across_slit: Optional[u.Quantity] = None,
                      extend: bool = False) -> _Scene:
    """
    Spectra under the slit from a synthesis file, one per entry.

    Parameters
    ----------
    path : str or Path
        Synthesis file, from ECLIPSE or another code.
    slit_position : u.Quantity
        Position of the slit centre across the slit, in the file's x
        coordinates.
    slit_width : u.Quantity
        Slit width, as an angle.
    psf_across_slit : u.Quantity, optional
        FWHM of the telescope's blur across the slit. Default None, no blur.
    extend : bool, optional
        Take the Sun beyond the file's x edges to look like the edge cells,
        and the slit's light only from its part inside them. Default False,
        dark beyond.
    """
    from .atmosphere import _check_format, _open_to_read
    from .synthesis_file import (FORMAT_NAME, FORMAT_VERSION, RADIANCE_UNIT, _synthesis_in,
                                 read_synthesis_layout)

    path = Path(path).expanduser()
    layout = read_synthesis_layout(path)
    columns, weights = _slit_columns(layout["x_edges"], slit_position, slit_width,
                                     psf_across_slit, extend)
    # One column at a time, as a band's spectrum is large.
    edges, radiance = {}, {}
    with _open_to_read(path, "synthesis") as f:
        _check_format(f, path, kind="synthesis", format_name=FORMAT_NAME,
                      format_version=FORMAT_VERSION)
        for column, weight in zip(range(columns.start, columns.stop), weights):
            synthesis = _synthesis_in(f, path, None, slice(column, column + 1))
            for name, line in synthesis.lines.items():
                # Skip converting spectra already in ECLIPSE's radiance unit,
                # which would copy a band's spectrum three times.
                values = (line.intensity.value if line.intensity.unit == RADIANCE_UNIT
                          else line.radiance().value)
                part = weight * values[:, 0, :]
                radiance[name] = radiance[name] + part if name in radiance else part
                edges[name] = _bin_edges(line.wavelength.to_value(u.Angstrom))
    return _Scene(y_edges=_centred(layout["y_edges"]), names=list(radiance),
                  edges=list(edges.values()), radiance=list(radiance.values()))


def _scene_light(scene: _Scene, telescope, solid_angle: float) -> list:
    """
    Part of :func:`_rates` both CCDs share.

    For each spectrum, its bins' lower and upper edges and middles, and the
    photons per second per pixel per unit radiance in each bin.
    """
    lights = []
    for edges in scene.edges:
        low, high = edges[:-1], edges[1:]
        middle = (low + high) / 2
        # Bin width in cm, through the pixel's solid angle.
        collected = solid_angle * (high - low) * 1e-8 * np.nan_to_num(
            photons_per_energy(telescope, middle * u.Angstrom).to_value(u.cm**2 / u.erg))
        lights.append((low, high, middle, collected))
    return lights


def _rates(scene: _Scene, lights: list, focal_plane: FocalPlane_SWC, ccd: str, telescope,
           det, slit_width: u.Quantity,
           blur: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Photons per second in each row of one CCD from each image row under the slit.

    Each wavelength sample covers the bin halfway to its neighbours. The
    bin's light is spread evenly over it, blurred by the spectral response
    and integrated over the rows, in photons at the bin's wavelength.
    Wavelengths outside the throughput tables give no light.

    Returns
    -------
    photons, per_wavelength, per_square : np.ndarray
        `(n_image_rows, n_rows)`: photons per second per pixel, and the same
        divided by each photon's wavelength in Angstrom and by its square,
        for the mean and root mean square photon energies.
    """
    row_edges, order = _sorted_rows(focal_plane, ccd)
    shape = (scene.y_edges.size - 1, focal_plane.n_rows)
    totals = tuple(np.zeros(shape) for _ in range(3))
    for (low, high, middle, collected), radiance in zip(lights, scene.radiance):
        if blur is None:
            sigma = box = 0.0
        else:
            sigma, box = _response(focal_plane, ccd, None, telescope, det, slit_width, blur,
                                   middle)
        onto_rows = light_onto_pixels(low, high, row_edges, sigma, box)   # (rows, bins)
        light = radiance * collected
        for total, power in zip(totals, (0, 1, 2)):
            total[:, order] += (onto_rows @ (light / middle**power).T).T
    first, last = focal_plane.lit_rows(ccd)
    for total in totals:
        total[:, :first] = 0.0
        total[:, last + 1:] = 0.0
    return totals


@dataclass
class FullFrame:
    """
    Full-CCD frame of both SW CCDs, at one slit position.

    The dict attributes are keyed by CCD, `"left"` and `"right"`.

    Attributes
    ----------
    rate : dict
        Photons per second reaching each image pixel, (n_rows, n_columns).
    photons : dict
        Photons in each pixel, including the parallel overscan rows. The
        mean if `noise` was off.
    dn : dict
        Frame as read out, in DN: the prescan, image and overscan of each
        output, (n_rows + parallel overscan, samples per row). NaN in rows
        outside the read-out windows.
    wavelength : dict
        Wavelength each image row records.
    photon_wavelength : dict
        Wavelength of the mean energy of the photons in each pixel,
        including the parallel overscan rows. Sets the electrons they free.
    photon_rms_wavelength : dict
        Wavelength of the root mean square energy of the same photons. Sets
        how far their electrons spread.
    exposure : u.Quantity
    slit_position, slit_width : u.Quantity
    settings : dict
        Options the frame was made with, the prescan and overscan sizes in
        `dn`, and the read-out `windows` and `dump_rows`.
    field_angle : dict
        Angle along the slit of each image column from the slit centre,
        positive toward solar north in nominal observations. Columns count
        from the left output, so on the left CCD the angle falls with column
        number. Default from a `FocalPlane_SWC` with the columns of `rate`.
    """

    rate: Dict[str, np.ndarray]
    photons: Dict[str, np.ndarray]
    dn: Dict[str, np.ndarray]
    wavelength: Dict[str, u.Quantity]
    photon_wavelength: Dict[str, u.Quantity]
    photon_rms_wavelength: Dict[str, u.Quantity]
    exposure: u.Quantity
    slit_position: u.Quantity
    slit_width: u.Quantity
    settings: dict = field(default_factory=dict)
    field_angle: Dict[str, u.Quantity] = field(default_factory=dict)

    def __post_init__(self):
        self.field_angle = dict(self.field_angle)
        for ccd, rate in self.rate.items():
            if ccd not in self.field_angle:
                focal_plane = FocalPlane_SWC(n_columns=np.shape(rate)[1])
                self.field_angle[ccd] = focal_plane.field_angle(
                    np.arange(focal_plane.n_columns), ccd)


def lay_out_register(image_dn: np.ndarray, det, noise: bool = True,
                     overscan_dn: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Frame samples as the two outputs read them, with prescan and overscan.

    Across the CCD, a row is: left prescan, first half of the image, left
    overscan, right overscan, second half of the image, right prescan
    (`ReadoutSequence`). The prescan samples read only read noise. So do
    the overscan samples unless `overscan_dn` is given: the light and dark
    charge the serial register carries past the image,
    `(n_rows, 2 * serial_overscan)` in that order. The serial register's
    own dark current is not modelled.
    """
    n_rows, n_columns = image_dn.shape
    half = (n_columns + 1) // 2
    prescan, overscan = det.serial_prescan, det.serial_overscan
    empty = np.zeros((n_rows, 2 * (prescan + overscan)))
    blank = digitise(detect(empty, np.full(n_rows, 200.0) * u.AA, 0.0 * u.s, det,
                            noise=noise), det)
    pieces = np.split(blank, np.cumsum([prescan, overscan, overscan]), axis=1)
    if overscan_dn is not None:
        if np.shape(overscan_dn) != (n_rows, 2 * overscan):
            raise ValueError(f"overscan_dn must be shaped {(n_rows, 2 * overscan)}, got "
                             f"{np.shape(overscan_dn)}.")
        pieces[1], pieces[2] = overscan_dn[:, :overscan], overscan_dn[:, overscan:]
    return np.concatenate([pieces[0], image_dn[:, :half], pieces[1], pieces[2],
                           image_dn[:, half:], pieces[3]], axis=1)


# Arguments of `full_ccd_frame` that a `Simulation` sets.
_FROM_SIMULATION = ("slit_width", "psf", "spectral_psf", "noise", "psf_boundary",
                    "slit_mask")

# `Simulation` fields a frame ignores. The `exposure` argument replaces
# `expos`, and the wavelengths come from the focal plane.
_IGNORED_SIMULATION_FIELDS = ("expos", "n_iter", "ncpu", "focal_plane_dispersion")

# Fields not checked one by one; the pinhole settings are checked through
# `enable_pinholes`.
_UNCHECKED_FIELDS = frozenset(_FROM_SIMULATION + _IGNORED_SIMULATION_FIELDS
                              + ("pinhole_sizes", "pinhole_positions",
                                 "pinhole_positions_spectral", "pinhole_visible_wavelength"))


def _same(a, b) -> bool:
    """Whether `a` and `b` are equal and of one shape, False if they cannot be compared."""
    try:
        return np.shape(a) == np.shape(b) and bool(np.all(a == b))
    except (TypeError, ValueError):
        return False


def _field_default(f: dataclasses.Field):
    """Default of a dataclass field, or `dataclasses.MISSING` if it has none."""
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def _settings_from_simulation(sim: Simulation, given: Dict[str, object],
                              exposure: u.Quantity) -> dict:
    """
    Slit, PSF and noise settings for `full_ccd_frame` from a `Simulation`.

    An argument in `given` at its default takes the value in `sim`. Raises
    TypeError if `sim` is not a `Simulation`, and one ValueError listing
    every argument that differs from both its default and `sim` and every
    setting in `sim` the frame does not model, including fields a subclass
    adds. Warns if `sim.expos` differs from both its 1 s default and
    `exposure`.
    """
    from .pinhole_diffraction import _pinholes_on

    if not isinstance(sim, Simulation):
        raise TypeError(f"sim must be a Simulation, got {type(sim).__name__}.")
    arguments = inspect.signature(full_ccd_frame).parameters
    problems = [f"{name} is {value!r} here and {getattr(sim, name)!r} in sim"
                for name, value in given.items()
                if not _same(value, arguments[name].default)
                and not _same(value, getattr(sim, name))]
    # Defaults of `Simulation` itself, so a subclass cannot hide a setting by
    # changing its default.
    base = {f.name: f for f in dataclasses.fields(Simulation)}
    for f in dataclasses.fields(sim):
        name, value = f.name, getattr(sim, f.name)
        if name in _UNCHECKED_FIELDS:
            continue
        if name not in base:
            problems.append(f"{name}: the frame does not know this setting")
        elif name == "instrument":
            if str(value).upper() != "SWC":
                problems.append(f"instrument {value!r}: a frame is of SWC")
        elif name == "enable_pinholes":
            if _pinholes_on(sim):
                problems.append("enable_pinholes: the frame does not model pinholes in the "
                                "filter")
        elif not _same(value, _field_default(base[name])):
            problems.append(f"{name} {value!r}: the frame does not model it")
    if problems:
        raise ValueError("full_ccd_frame cannot take these settings: " + "; ".join(problems)
                         + ".")
    if not _same(sim.expos, _field_default(base["expos"])) and not _same(sim.expos, exposure):
        warnings.warn(f"The frame's exposure is {exposure}, from exposure; sim.expos, "
                      f"{sim.expos}, is not used.", UserWarning, stacklevel=3)
    return {name: getattr(sim, name) for name in _FROM_SIMULATION}


def _readout_sequence(det, windows, dump_rows) -> ReadoutSequence:
    """Read-out of the frame, with `windows` and `dump_rows` checked against the CCD."""
    given = windows
    try:
        windows = [] if windows is None else [tuple(window) for window in windows]
    except TypeError:
        windows = None
    if windows is None or any(
            len(window) != 2 or not all(isinstance(row, (int, np.integer))
                                        and not isinstance(row, bool) for row in window)
            or not 0 <= window[0] <= window[1] < det.n_rows for window in windows):
        raise ValueError(f"windows are ranges of rows read out, each two rows from 0 to "
                         f"{det.n_rows - 1}, the first no later than the last, such as "
                         f"[(400, 520)]; got {given!r}.")
    if dump_rows is not None and (isinstance(dump_rows, bool)
                                  or not isinstance(dump_rows, (int, np.integer))
                                  or dump_rows < 0):
        raise ValueError(f"dump_rows is a number of row transfers, a whole number, zero or "
                         f"more; got {dump_rows!r}.")
    return ReadoutSequence.from_detector(det, windows=[(int(a), int(b)) for a, b in windows],
                                         dump_rows=None if dump_rows is None else int(dump_rows))


def full_ccd_frame(synthesis_path: str | Path, lines: Optional[BandLines] = None,
                   slit_position: Optional[u.Quantity] = None,
                   exposure: Optional[u.Quantity] = None, *,
                   slit_width: u.Quantity = 0.2 * u.arcsec,
                   det=None, telescope=None, focal_plane: Optional[FocalPlane_SWC] = None,
                   psf: bool = True, spectral_psf: str = "quadrature",
                   continuum: Optional[Tuple[u.Quantity, np.ndarray, np.ndarray]] = None,
                   noise: bool = True, n_workers: int = 0,
                   sim: Optional[Simulation] = None,
                   psf_boundary: str = "replicate", slit_mask: bool = False,
                   windows: Optional[Sequence[Tuple[int, int]]] = (),
                   dump_rows: Optional[int] = None,
                   common_random_numbers: bool = False) -> FullFrame:
    """
    Full-CCD frame of the SW camera from the spectra of a synthesis file.

    Every entry of the file is laid onto both CCDs, and the entries are
    added. Give `slit_position` and `exposure` by name.

    Parameters
    ----------
    synthesis_path : str or Path
        Synthesis file, from ECLIPSE or another code.
    lines : BandLines, optional
        Deprecated. Every line in the band, from
        `eclipse.band.band_contribution_functions`, to compute the band's
        spectrum from the emission measure an ECLIPSE synthesis file keeps,
        in place of the file's own spectra. Synthesise the band into the
        file instead, with `synthesise-spectra --band`.
    slit_position : u.Quantity
        Position of the slit centre across the slit, in the file's x
        coordinates. Along the slit, the image centre is at the slit centre.
    exposure : u.Quantity
        Exposure time.
    slit_width : u.Quantity, optional
        Default 0.2 arcsec.
    det, telescope : optional
        Default `Detector_SWC()` and `Telescope_EUVST()`.
    focal_plane : FocalPlane_SWC, optional
        Default `FocalPlane_SWC.from_detector(det)`. Its `dispersion`,
        `wavelength_offset`, `lit_band` and `slit_image_tilt` must be the
        detector's.
    psf : bool, optional
        Blur with the telescope's PSF along the slit and in wavelength, and
        across the slit if the telescope gives that. Default True.
    spectral_psf : str, optional
        `"quadrature"` (default) or `"convolution"`, as in the configuration.
    continuum : tuple, optional
        Deprecated, and only with `lines`. Continuum per unit emission
        measure: wavelengths, free-free and free-bound emission (nT,
        n_wavelength) and two-photon emission (nN, nT, n_wavelength), as
        returned by `eclipse.continuum.compute_continuum_fiasco` on the
        grids of `lines`. Synthesise the continuum into the file instead,
        with `synthesise-spectra --continuum`.
    noise : bool, optional
        If False, every random draw is replaced by its mean. Default True.
    n_workers : int, optional
        Only with `lines`: processes computing the band's spectrum. Default
        0 uses every CPU this process may use.
    sim : Simulation, optional
        Source of `slit_width`, `psf`, `spectral_psf`, `noise`,
        `psf_boundary` and `slit_mask`. An argument at its default, given or
        not, takes the `Simulation`'s value; any other value must agree with
        it. A `Simulation` has the PSF off unless made with `psf=True`. Its
        `expos`, `n_iter` and `ncpu` are not used, nor the pinhole settings
        with pinholes off. Default None.
    psf_boundary : str, optional
        What the PSF brings in from beyond the edges of the synthesis, as
        in a run. `"replicate"` (default) takes the Sun there to look like
        the edge cells. Each column and the slit then get light only over
        their part inside the scene, so a uniform scene gives every column
        inside it the same light and the columns beyond none. `"zero"` takes
        the Sun there as dark, so the end columns get up to about a third
        less light, which the blur carries to the columns just beyond.
        Keyword only.
    slit_mask : bool, optional
        Block the light beyond the ends of the slit and behind its
        fiducials, from `telescope.slit_openings`, as in a run. The slit
        centre is the middle column and the middle of the scene. The blur
        along the slit comes first, so a column partly behind a fiducial
        gets the light on its open part. Default False. Keyword only.
    windows : sequence of tuple of int, optional
        Rows read out, as inclusive ranges such as `[(400, 520)]`, on both
        CCDs. The other rows are dumped, which shortens the read-out and its
        smear, and their DN are NaN.
        :func:`eclipse.readout.windows_from_wavelengths` gives the rows of
        wavelength ranges. Default every row. Keyword only.
    dump_rows : int, optional
        Row transfers that clear the image area before the exposure.
        Default a whole CCD. Keyword only.
    common_random_numbers : bool, optional
        Use one random value per pixel for each noise draw. Frames that
        differ only in their light, such as with and without the shutter,
        then share their noise. Seed NumPy with the same value before each
        frame. Slower. Default False. Keyword only.

    Returns
    -------
    FullFrame
        The synthesis's y is taken to increase towards solar north. Each
        CCD's columns count from its left output: from the north end of the
        slit on the left CCD, and from the south end on the right CCD (see
        `FullFrame.field_angle`).

    Raises
    ------
    TypeError
        If `slit_position` or `exposure` is missing, or `sim` is not a
        `Simulation`.
    ValueError
        Among others: if `exposure` or `slit_width` is not above zero,
        `psf_boundary` is not `"replicate"` or `"zero"`, `slit_mask` is on
        and the telescope does not give the slit's openings for this slit
        width, the slit is off the synthesis, or `focal_plane` does not
        match `det` or has a tilted slit image. With `sim`, if an argument
        differs from both its default and the `Simulation`, or the
        `Simulation` asks for something the frame does not model: an
        instrument other than SWC, pinholes, visible stray light, or a
        field a subclass adds. One error lists every such setting. Also if
        a window is not two rows on the CCD, the first no later than the
        last, or `dump_rows` is not a whole number, zero or more.

    Warns
    -----
    FutureWarning
        If `lines` is given.
    UserWarning
        If `sim.expos` differs from both 1 s, its default, and `exposure`,
        which is used.
    """
    from .config import Detector_SWC, Telescope_EUVST
    from .continuum import is_continuum

    if lines is not None and not isinstance(lines, BandLines):
        raise TypeError("Give slit_position and exposure by name, as in "
                        "full_ccd_frame(path, slit_position=..., exposure=...).")
    if slit_position is None or exposure is None:
        raise TypeError("full_ccd_frame needs a slit_position and an exposure.")
    if continuum is not None and lines is None:
        raise ValueError("continuum goes with lines, and both are deprecated: a frame holds "
                         "the spectra in the synthesis file, so synthesise the continuum "
                         "into it with synthesise-spectra --continuum.")
    if lines is not None:
        warnings.warn(
            "The lines and continuum arguments of full_ccd_frame are deprecated and will be "
            "removed in a future release. A frame holds the spectra in the synthesis file: "
            "synthesise the band into it with synthesise-spectra --band, or make the frame "
            "from a run file, and leave these arguments out.", FutureWarning, stacklevel=2)
    if sim is not None:
        given = {"slit_width": slit_width, "psf": psf, "spectral_psf": spectral_psf,
                 "noise": noise, "psf_boundary": psf_boundary, "slit_mask": slit_mask}
        settings = _settings_from_simulation(sim, given, exposure)
        slit_width, psf = settings["slit_width"], settings["psf"]
        spectral_psf, noise = settings["spectral_psf"], settings["noise"]
        psf_boundary, slit_mask = settings["psf_boundary"], settings["slit_mask"]
    if not (isinstance(psf_boundary, str) and psf_boundary in ("replicate", "zero")):
        raise ValueError(f"psf_boundary must be 'replicate' or 'zero', got {psf_boundary!r}.")
    extend = psf_boundary == "replicate"
    if slit_mask not in (True, False):
        raise ValueError(f"slit_mask must be True or False, got {slit_mask!r}.")
    det = Detector_SWC() if det is None else det
    telescope = Telescope_EUVST() if telescope is None else telescope
    focal_plane = FocalPlane_SWC.from_detector(det) if focal_plane is None else focal_plane
    for name, value, unit in (("exposure", exposure, u.s), ("slit_width", slit_width, u.arcsec)):
        value = u.Quantity(value)
        if not (value.isscalar and value.unit.is_equivalent(unit) and np.isfinite(value.value)
                and value.value > 0):
            raise ValueError(f"{name} must be a finite {unit.physical_type} above zero, got "
                             f"{value}.")
    openings = _openings_along_the_slit(telescope, slit_width) if slit_mask else None
    sequence = _readout_sequence(det, windows, dump_rows)
    if (focal_plane.n_rows, focal_plane.n_columns) != (det.n_rows, det.n_columns):
        raise ValueError(
            f"The focal plane has {focal_plane.n_rows} rows and {focal_plane.n_columns} "
            f"columns and the detector {det.n_rows} and {det.n_columns}; they must match.")
    if any(value != 0.0 for value in focal_plane.slit_image_tilt):
        raise ValueError("The slit's image is tilted, so each column's rows have their own "
                         "wavelengths, which this frame does not yet work out.")
    differ = [name for name in _FOCAL_PLANE_SETTINGS
              if not _same_setting(getattr(det, name), getattr(focal_plane, name))]
    if differ:
        raise ValueError(
            f"The detector and the focal plane give different {', '.join(differ)}. Set them on "
            f"the detector, Detector_SWC, and leave out focal_plane, or make it with "
            f"FocalPlane_SWC.from_detector(det).")
    across = across_slit_fwhm(telescope) if psf else None
    if lines is None:
        scene = read_slit_spectra(synthesis_path, slit_position, slit_width, across, extend)
    else:
        scene = _band_scene(synthesis_path, lines, continuum, slit_position, slit_width,
                            across, extend, n_workers)

    # Along the slit: weight of each image row in each detector column. The
    # middle column, the middle of the scene and the slit's centre are at 0.
    columns = np.arange(focal_plane.n_columns + 1) - 0.5
    column_edges = focal_plane.field_angle(columns).to_value(u.arcsec)
    sigma = (_fwhm_to_sigma(spatial_psf_fwhm(telescope, det))
             * focal_plane.plate_scale.to_value(u.arcsec) if psf else 0.0)
    along = _scene_weights(scene.y_edges, column_edges, sigma, extend,
                           openings)                                # (columns, image rows)

    blur = spectral_psf if psf else None
    solid_angle = pixel_solid_angle(focal_plane, slit_width).to_value(u.sr)
    totals = tuple({} for _ in range(3))
    lights = _scene_light(scene, telescope, solid_angle)
    for ccd in CCDS:
        parts = _rates(scene, lights, focal_plane, ccd, telescope, det, u.Quantity(slit_width),
                       blur)
        for total, part in zip(totals, parts):
            total[ccd] = along @ part                               # (columns, rows)
    rate, per_wavelength, per_square = totals
    # Count each CCD's columns from its left output, as the camera reads them.
    for ccd in CCDS:
        if focal_plane.field_angle(1, ccd) < focal_plane.field_angle(0, ccd):
            for total in totals:
                total[ccd] = total[ccd][::-1].copy()

    dark = dark_current_time(exposure, sequence, focal_plane.n_rows)
    # Each sample's share of its row's dark charge along the serial register.
    dark_image, dark_overscan = serial_dark_shares(sequence, focal_plane.n_columns)
    dark_time = dark[:, np.newaxis] * dark_image
    rows = np.arange(focal_plane.n_rows)
    unread = ~sequence.is_read(focal_plane.n_rows)
    draw = _poisson_inverse_transform if common_random_numbers else np.random.poisson
    frame = {"rate": {}, "photons": {}, "dn": {}, "wavelength": {}, "photon_wavelength": {},
             "photon_rms_wavelength": {}, "field_angle": {}}
    for ccd in CCDS:
        image_rate = rate[ccd].T                                          # (rows, columns)
        wavelength = focal_plane.wavelength(rows, ccd)
        # Wavelengths of the mean and root mean square photon energies in
        # each pixel; an unlit pixel keeps its row's.
        own = np.broadcast_to(wavelength.to_value(u.Angstrom)[:, np.newaxis], image_rate.shape)
        lit = image_rate > 0
        reaching = np.divide(image_rate, per_wavelength[ccd].T, out=np.array(own, dtype=float),
                             where=lit & (per_wavelength[ccd].T > 0)) * u.Angstrom
        reaching_rms = np.sqrt(np.divide(image_rate, per_square[ccd].T,
                                         out=np.array(own, dtype=float) ** 2,
                                         where=lit & (per_square[ccd].T > 0))) * u.Angstrom
        overscan = None
        if sequence.cte_serial != 1:
            # Charge left in the serial register reaches the overscan; one
            # transfer gives both.
            (photons, mean_wavelength, rms_wavelength), overscan = (
                expose_with_serial_overscan_with_wavelengths(
                    image_rate, reaching, exposure, sequence, rms_wavelength=reaching_rms))
        else:
            photons, mean_wavelength, rms_wavelength = expose_with_wavelengths(
                image_rate, reaching, exposure, sequence, rms_wavelength=reaching_rms)
        drawn = draw(photons).astype(float) if noise else photons
        electrons = detect(drawn, mean_wavelength, dark_time, det, noise=noise,
                           rms_wavelength=rms_wavelength,
                           common_random_numbers=common_random_numbers)
        overscan_dn = None
        if overscan is not None:
            trail, trail_wavelength, trail_rms = overscan
            trail = draw(trail).astype(float) if noise else trail
            overscan_dn = digitise(detect(trail, trail_wavelength,
                                          dark[:, np.newaxis] * dark_overscan, det,
                                          noise=noise, rms_wavelength=trail_rms,
                                          common_random_numbers=common_random_numbers), det)
        frame["rate"][ccd] = image_rate
        frame["photons"][ccd] = drawn
        dn = lay_out_register(digitise(electrons, det), det, noise=noise,
                              overscan_dn=overscan_dn)
        # Rows outside the windows are dumped, not read.
        dn[unread] = np.nan
        frame["dn"][ccd] = dn
        frame["wavelength"][ccd] = wavelength
        frame["photon_wavelength"][ccd] = mean_wavelength
        frame["photon_rms_wavelength"][ccd] = rms_wavelength
        frame["field_angle"][ccd] = focal_plane.field_angle(np.arange(focal_plane.n_columns),
                                                            ccd)
    return FullFrame(**frame, exposure=u.Quantity(exposure),
                     slit_position=u.Quantity(slit_position), slit_width=u.Quantity(slit_width),
                     settings={"psf": psf, "spectral_psf": spectral_psf, "noise": noise,
                               "psf_boundary": psf_boundary, "slit_mask": bool(slit_mask),
                               "continuum": any(is_continuum(name) or name == "continuum"
                                                for name in scene.names),
                               "spectra": list(scene.names),
                               "serial_prescan": det.serial_prescan,
                               "serial_overscan": det.serial_overscan,
                               "parallel_overscan_rows": det.parallel_overscan_rows,
                               "windows": [list(window) for window in sequence.windows],
                               "dump_rows": sequence.dump_rows,
                               "common_random_numbers": bool(common_random_numbers)})


FRAME_FORMAT = "eclipse-full-ccd-frame"


def write_full_frame(frame: FullFrame, path: str | Path) -> Path:
    """
    Write a full-CCD frame to an HDF5 file.

    The root attributes hold the settings, including the prescan and
    overscan sizes in `dn` (`serial_prescan`, `serial_overscan` and
    `parallel_overscan_rows`), the rows read out and the clear (`windows`,
    an array of first and last rows, empty for every row, and `dump_rows`),
    and the ECLIPSE version and git commit that wrote the file
    (`software_version` and `git_commit_id`). The dataset
    `spectra` names the synthesis file entries in the frame. Groups `left`
    and `right` hold these datasets, each with its unit:

    - `dn`: frame as read out, with its prescan and overscan.
    - `photons` and `rate`.
    - `wavelength`: wavelength each image row records.
    - `field_angle`: angle along the slit of each image column, positive
      towards solar north in nominal observations.
    - `photon_wavelength` and `photon_rms_wavelength`: wavelengths of the
      mean and root mean square energies of each pixel's photons.

    Parameters
    ----------
    frame : FullFrame
        Frame to write.
    path : str or Path
        File to write. An existing file is replaced.

    Returns
    -------
    Path
        File written.
    """
    from .atmosphere import _replacing

    path = Path(path).expanduser()
    with _replacing(path) as partial, h5py.File(partial, "w") as f:
        f.attrs["format"] = FRAME_FORMAT
        f.attrs["software_version"] = _get_software_version()
        f.attrs["git_commit_id"] = get_git_commit_id()
        for name in ("exposure", "slit_position", "slit_width"):
            value = getattr(frame, name)
            f.attrs[name] = value.value
            f.attrs[f"{name}_unit"] = str(value.unit)
        for key, value in frame.settings.items():
            if key == "windows":
                value = np.asarray(value, dtype=np.int64).reshape(-1, 2)
            if key != "spectra":
                f.attrs[key] = value
        # A file can hold more entries than an attribute has room for.
        f.create_dataset("spectra", data=list(frame.settings.get("spectra", [])),
                         dtype=h5py.string_dtype())
        for ccd in CCDS:
            group = f.create_group(ccd)
            for name, data, unit in (("dn", frame.dn[ccd], "DN"),
                                     ("photons", frame.photons[ccd], "photon"),
                                     ("rate", frame.rate[ccd], "photon / s"),
                                     ("wavelength", frame.wavelength[ccd].to_value(u.AA),
                                      "Angstrom"),
                                     ("field_angle", frame.field_angle[ccd].to_value(u.arcsec),
                                      "arcsec"),
                                     ("photon_wavelength",
                                      frame.photon_wavelength[ccd].to_value(u.AA), "Angstrom"),
                                     ("photon_rms_wavelength",
                                      frame.photon_rms_wavelength[ccd].to_value(u.AA),
                                      "Angstrom")):
                group.create_dataset(name, data=data, compression="gzip")
                group[name].attrs["unit"] = unit
    return path
