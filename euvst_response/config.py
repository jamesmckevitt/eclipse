"""
Configuration classes for instruments, detectors, and simulation parameters.
"""

from __future__ import annotations
import dataclasses
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import numpy as np
import astropy.units as u
import scipy.interpolate
from . import eis_calibration
from .utils import angle_to_distance
from importlib.resources import files


# ------------------------------------------------------------------
#  Detector materials and shared properties
# ------------------------------------------------------------------

# Material properties (Fano factors)
DETECTOR_MATERIALS = {
    "silicon": {
        "fano_factor": 0.115,
    }
}

def _check_settings(obj, section: str, positive: tuple = (), non_negative: tuple = (),
                    fractions: tuple = (), optional: tuple = ()) -> None:
    """
    Refuse any setting of `obj` not of its default's kind, or out of range.

    A quantity needs a unit of the same kind as its default. A number with
    no unit, or one of another kind, would be read in whatever unit the code
    works in: a `D_ap` of 0.28, meant in metres, would give 1e4 times too
    few photons. A switch must be true or false, and a number a number.
    `positive`, `non_negative` and `fractions` name the fields that must be
    above zero, at or above zero, or from 0 to 1; a value outside would run
    to completion and save meaningless results. `optional` names those that
    may also be None.
    """
    for f in dataclasses.fields(obj):
        if not f.init:
            continue
        name, value = f"{section}.{f.name.lstrip('_')}", getattr(obj, f.name)
        if value is None and f.name in optional:
            continue
        default = None if f.default is dataclasses.MISSING else f.default
        if isinstance(default, u.Quantity):
            if not isinstance(value, u.Quantity):
                raise ValueError(f"{name} needs a unit, of {default.unit.physical_type}, such as "
                                 f"{default}; got {value!r}.")
            if not value.unit.is_equivalent(default.unit, equivalencies=u.temperature()):
                raise ValueError(f"{name} must be in a unit of {default.unit.physical_type}, "
                                 f"such as {default.unit}; got {value.unit}.")
            if not np.all(np.isfinite(value.value)):
                raise ValueError(f"{name} must be finite, got {value}.")
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, got {value!r}.")
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {value!r}.")
        magnitude = getattr(value, "value", value)
        if f.name in positive and not np.all(magnitude > 0):
            raise ValueError(f"{name} must be more than zero, got {value}.")
        if f.name in non_negative and not np.all(magnitude >= 0):
            raise ValueError(f"{name} cannot be negative, got {value}.")
        if f.name in fractions and not (0 <= magnitude <= 1):
            raise ValueError(f"{name} is a fraction, from 0 to 1, got {value}.")


def _check_psf_params(tel, section: str = "telescope") -> None:
    """
    Check the PSF settings.

    The PSF must be a Gaussian, `psf_params` two widths above zero in
    pixels, along the slit and along the dispersion, and `psf_across_slit`
    None or an angle above zero. Otherwise they would only be read part way
    through a run, when the scene is laid onto the detector.
    """
    if not isinstance(tel.psf_type, str) or tel.psf_type.lower() != "gaussian":
        raise ValueError(f"{section}.psf_type must be 'gaussian', the only PSF there is; got "
                         f"{tel.psf_type!r}.")
    value = tel.psf_params
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(q, u.Quantity) and q.unit.is_equivalent(u.pix)
                       and np.isfinite(q.value) and q.value > 0 for q in value)):
        raise ValueError(f"{section}.psf_params must be two FWHMs in pixels, along the slit "
                         f"and along the dispersion, such as [2.66 pix, 2.54 pix]; got {value!r}.")
    across = tel.psf_across_slit
    if across is not None and not (isinstance(across, u.Quantity) and across.isscalar
                                   and across.unit.is_equivalent(u.arcsec)
                                   and np.isfinite(across.value) and across.value > 0):
        raise ValueError(f"{section}.psf_across_slit must be a FWHM in an angle above zero, "
                         f"such as 1 arcsec, or left out for none; got {across!r}.")


def _check_fiducials(tel, section: str = "telescope") -> None:
    """
    Check the fiducials are angles along the slit, with one length for each
    of the four slits, and each inside the slit and clear of the others at
    the longest of those lengths.
    """
    def angles(value):
        return (isinstance(value, (list, tuple))
                and all(isinstance(q, u.Quantity) and q.isscalar and q.unit.is_equivalent(u.arcsec)
                        and np.isfinite(q.value) for q in value))

    if not angles(tel.fiducial_centres):
        raise ValueError(f"{section}.fiducial_centres must be a list of angles along the slit "
                         f"from its centre, such as [-25 arcsec, 70 arcsec]; got "
                         f"{tel.fiducial_centres!r}.")
    lengths = tel.fiducial_lengths
    if not angles(lengths) or len(lengths) != 4 or not all(q.value > 0 for q in lengths):
        raise ValueError(f"{section}.fiducial_lengths must be four lengths above zero, one for "
                         f"each of the 0.2, 0.4, 0.8 and 1.6 arcsec slits, such as [0.8 arcsec, "
                         f"0.8 arcsec, 0.8 arcsec, 1.6 arcsec]; got {lengths!r}.")
    longest = max(q.to_value(u.arcsec) for q in lengths)
    centres = np.sort([q.to_value(u.arcsec) for q in tel.fiducial_centres])
    half = tel.slit_length.to_value(u.arcsec) / 2
    if centres.size and (centres[0] - longest / 2 <= -half or centres[-1] + longest / 2 >= half):
        raise ValueError(f"{section}.fiducial_centres must keep every fiducial inside the "
                         f"{tel.slit_length} slit; got {tel.fiducial_centres!r}.")
    if np.any(np.diff(centres) <= longest):
        raise ValueError(f"{section}.fiducial_centres must keep the fiducials apart; got "
                         f"{tel.fiducial_centres!r} for fiducials up to {longest} arcsec long.")


def _check_tables(obj, names: tuple, section: str) -> None:
    """
    Check each of `names` is a table path; text, as from a configuration, becomes a Path.

    Whether the table exists is only checked when it is read, since a
    results file made on another machine rebuilds these with that machine's
    paths.
    """
    for name in names:
        value = getattr(obj, name)
        if isinstance(value, (str, os.PathLike)):
            value = Path(value).expanduser()
            setattr(obj, name, value)
        if not hasattr(value, "is_file"):
            raise ValueError(f"{section}.{name} must be the path of a table, got {value!r}.")


def _check_detector(det, section: str = "detector") -> None:
    """Checks both detectors share: kinds, ranges, temperature above 0 K and a known material."""
    _check_settings(det, section,
                    positive=("gain_e_per_dn", "max_dn", "full_well", "pix_size", "wvl_res",
                              "plate_scale_angle", "filter_distance", "ccd_gap",
                              "row_transfer_time", "pixel_period"),
                    non_negative=("read_noise_rms", "_dark_current_293k"),
                    fractions=("qe_euv", "qe_vis"))
    if det.ccd_temperature.to_value(u.K, equivalencies=u.temperature()) <= 0:
        raise ValueError(f"{section}.ccd_temperature must be above absolute zero, got "
                         f"{det.ccd_temperature}; -60 K was perhaps meant as -60 C.")
    if det.material not in DETECTOR_MATERIALS:
        raise ValueError(f"{section}.material must be one of {sorted(DETECTOR_MATERIALS)}, got "
                         f"{det.material!r}.")


def calculate_dark_current(temp: u.Quantity, q_d0_293k: u.Quantity, ccd_type: str = "NIMO") -> u.Quantity:
    """
    Calculate dark current based on CCD temperature and type.
    
    Parameters
    ----------
    temp : u.Quantity
        CCD temperature with units (e.g., -60 * u.deg_C)
    q_d0_293k : u.Quantity
        Dark current at 293K in electrons per pixel per second
    ccd_type : str
        "NIMO" (non-inverted mode), "AIMO" (advanced inverted mode)
        
    Returns
    -------
    dark_current : u.Quantity
        Dark current in electrons per pixel per second
        
    Raises
    ------
    ValueError
        If temperature is above 300K (27 deg C) or unknown CCD type
    """
    temp_kelvin = temp.to(u.Kelvin, equivalencies=u.temperature())
    max_temp = 300 * u.K
    min_temp = 198 * u.K
    
    # Check temperature limits
    if temp_kelvin > max_temp:
        raise ValueError(f"Cannot calculate dark current at {temp_kelvin}. "
                       f"Maximum temperature is {max_temp}")
    
    # Apply minimum temperature limit (clamp to 198K; based on MSSL test results)
    if temp_kelvin < min_temp:
        temp_kelvin = min_temp
    
    Q_d0 = q_d0_293k.to_value(u.electron / (u.pixel * u.s))
    T = temp_kelvin.value
    
    if ccd_type.upper() == "NIMO":
        # Q_d = Q_d0 * 122 * T^3 * exp(-6400/T)
        dark_current = Q_d0 * 122 * T**3 * np.exp(-6400/T)
    elif ccd_type.upper() == "AIMO":
        # Q_d = Qd0 * 1.14e6 * T^3 * exp(-9080/T)
        dark_current = Q_d0 * 1.14e6 * T**3 * np.exp(-9080/T)
    else:
        raise ValueError(f"Unknown CCD type: {ccd_type}. Must be 'NIMO' or 'AIMO'.")

    return dark_current * u.electron / (u.pixel * u.s)


# ------------------------------------------------------------------
#  Throughput helpers & AluminiumFilter
# ------------------------------------------------------------------
# Throughput tables already read, by path.
_THROUGHPUT_TABLES: dict = {}


def _starts_with_a_number(text: str) -> bool:
    try:
        float(text.split()[0])
    except ValueError:
        return False
    return True


def _load_throughput_table(path) -> tuple[u.Quantity, np.ndarray]:
    """
    Wavelength (nm) and throughput arrays from a two-column text table.

    Comment lines are skipped. Each table is read once and cached: the
    effective area is evaluated at every wavelength of a spectrum, and
    rereading five tables each time would make a spectrum take minutes. The
    arrays are shared, so they are read-only.
    """
    key = str(path)
    if key not in _THROUGHPUT_TABLES:
        # Text becomes a Path; a packaged table may be a resource inside an
        # archive, which can read itself but is not a path.
        content = (path if hasattr(path, "read_text") else Path(path)).read_text()
        # The header is every line before the data that is not two numbers,
        # however many there are, rather than a fixed count, which would drop
        # data rows from a table with a shorter header. A line that is not
        # two numbers once the data has begun is an error in the table, which
        # would otherwise change the curve.
        data = []
        for number, line in enumerate(content.splitlines(), start=1):
            text = line.strip()
            if not text or text.startswith('#'):
                continue
            # A data line is all numbers, the first two being wavelength and
            # throughput.
            try:
                row = [float(x) for x in text.split()]
            except ValueError:
                row = []
            if len(row) >= 2:
                # A throughput in per cent, or a NaN, would make every value
                # of the effective area meaningless.
                wavelength, throughput = row[:2]
                if not (np.isfinite(wavelength) and wavelength > 0
                        and np.isfinite(throughput) and 0 <= throughput <= 1):
                    raise ValueError(f"{path}, line {number}: {text!r} needs a wavelength "
                                     f"above zero, in nm, and a throughput from 0 to 1.")
                data.append([wavelength, throughput])
            elif data:
                raise ValueError(f"{path}, line {number}: {text!r} is not a wavelength and a "
                                 f"throughput, and the table's data had begun.")
            elif _starts_with_a_number(text):
                # A line before the data is a header, as in the packaged
                # tables, but one starting with a number may be a first data
                # line with an error in it, which would change the curve's end.
                warnings.warn(f"{path}, line {number}: {text!r} is taken as a header, but it "
                              f"begins with a number. If it is a line of data, it has a slip "
                              f"in it; if it is a header, a # in front of it says so.",
                              stacklevel=2)
        if not data:
            raise ValueError(f"{path} has no lines of a wavelength and a throughput.")
        if len(data) < 2:
            raise ValueError(f"{path} has one line of a wavelength and a throughput, and "
                             f"needs two or more to interpolate between.")
        arr = np.array(data)
        wl = arr[:, 0] * u.nm
        tr = arr[:, 1]
        wl.flags.writeable = False
        tr.flags.writeable = False
        _THROUGHPUT_TABLES[key] = (wl, tr)
    return _THROUGHPUT_TABLES[key]


def _interp_tr(wavelength_nm, wl_tab: np.ndarray, tr_tab: np.ndarray) -> float | np.ndarray:
    """Linear interpolation: a float for one wavelength, an array for several."""
    f = scipy.interpolate.interp1d(wl_tab, tr_tab, bounds_error=False, fill_value=np.nan)
    # Snap to the table's ends: a change of unit can put an end wavelength a
    # part in 1e16 beyond it, as 214 Angstrom becomes 21.400000000000002 nm.
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    ends = np.asarray(wl_tab.value if hasattr(wl_tab, "value") else wl_tab, dtype=float)
    for end in (ends[0], ends[-1]):
        wavelength_nm = np.where(np.isclose(wavelength_nm, end, rtol=1e-12, atol=0.0),
                                 end, wavelength_nm)
    out = f(wavelength_nm)
    return float(out) if np.ndim(out) == 0 else out


def check_pinhole_lists(sizes: list, positions: list, spectral: list) -> tuple[list, list]:
    """
    Validate the paired pinhole lists and return the positions as floats.

    Used for both configuration files and a directly built `Simulation`, so
    they agree on what a valid pinhole is.

    Parameters
    ----------
    sizes, positions : list
        Pinhole diameters and positions along the slit, one entry per pinhole.
    spectral : list
        Positions along the spectral axis, one per pinhole, or empty to put
        every pinhole at the centre of the spectral window.

    Returns
    -------
    tuple of list
        `(positions, spectral)`, each entry converted to float.
    """
    # Always compared: guarding this on the sizes would let positions without
    # sizes through, and the run would silently produce no pinholes.
    if len(sizes) != len(positions):
        raise ValueError(
            f"pinhole_sizes and pinhole_positions are a paired list, one entry "
            f"per pinhole, so they must have the same length. Got "
            f"{len(sizes)} size(s) and {len(positions)} position(s)."
        )
    if spectral and len(spectral) != len(sizes):
        raise ValueError(
            f"pinhole_positions_spectral, when given, must have the same "
            f"length as pinhole_sizes. Got {len(spectral)} spectral "
            f"position(s) and {len(sizes)} size(s)."
        )
    return (_fractions(positions, "pinhole_positions"),
            _fractions(spectral, "pinhole_positions_spectral"))


def _fractions(values: list, name: str) -> list:
    """Return `values` as floats, rejecting any outside [0, 1].

    Both position lists are fractions of the way across the detector. A
    value out of range puts the pinhole off the detector, where it looks like
    a working pinhole whose light is missing. Converting also stops a quoted
    YAML value such as '0.3' reaching the diffraction code as a string.
    """
    out = []
    for idx, value in enumerate(values):
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name}[{idx}] is {value!r}. Positions are a plain fraction "
                f"of the detector, so they carry no units."
            ) from None
        if not 0.0 <= as_float <= 1.0:
            raise ValueError(
                f"{name}[{idx}] is {as_float}. Positions are a fraction of the "
                f"way across the detector and must lie in [0, 1]."
            )
        out.append(as_float)
    return out


@dataclass
class AluminiumFilter:
    """
    EUVST-SW filter: aluminium, aluminium oxide and carbon layers on a mesh.

    Each layer's transmission is the tabulated one for a thickness of
    `table_thickness`, raised to the power of the layer's thickness over
    `table_thickness`.

    Parameters
    ----------
    al_thickness, oxide_thickness, c_thickness : u.Quantity
        Thickness of each layer. Default 1485, 95 and 0 angstrom.
    mesh_throughput : float
        Fraction of light the supporting mesh lets through. Default 0.8.
    al_table, oxide_table, c_table : path
        Transmission table for each layer, two columns: wavelength in nm and
        transmission. Default the packaged tables.
    table_thickness : u.Quantity
        Thickness the tables are for. Default 1000 angstrom.
    """
    al_thickness: u.Quantity = 1485 * u.angstrom
    oxide_thickness: u.Quantity = 95 * u.angstrom
    c_thickness: u.Quantity = 0 * u.angstrom
    mesh_throughput: float = 0.8
    al_table: Path = field(default_factory=lambda: files('euvst_response') / 'data' / 'throughput' / 'throughput_aluminium_1000_angstrom.dat')
    oxide_table: Path = field(default_factory=lambda: files('euvst_response') / 'data' / 'throughput' / 'throughput_aluminium_oxide_1000_angstrom.dat')
    c_table: Path = field(default_factory=lambda: files('euvst_response') / 'data' / 'throughput' / 'throughput_carbon_1000_angstrom.dat')
    table_thickness: u.Quantity = 1000 * u.angstrom

    def __post_init__(self):
        _check_settings(self, "filter", positive=("table_thickness",),
                        non_negative=("al_thickness", "oxide_thickness", "c_thickness"),
                        fractions=("mesh_throughput",))
        _check_tables(self, ("al_table", "oxide_table", "c_table"), "filter")

    def total_throughput(self, wl0: u.Quantity) -> u.Quantity:
        """
        EUV transmission of the filter, including the mesh.

        Parameters
        ----------
        wl0 : u.Quantity
            Wavelength, or an array of them.

        Returns
        -------
        u.Quantity
            Dimensionless transmission, one value or one per wavelength. NaN
            outside the tables' wavelengths.
        """
        wl_nm = wl0.to_value(u.nm)
        wl_al, tr_al = _load_throughput_table(self.al_table)
        wl_ox, tr_ox = _load_throughput_table(self.oxide_table)
        wl_c,  tr_c  = _load_throughput_table(self.c_table)
        t_al = _interp_tr(wl_nm, wl_al, tr_al) ** (self.al_thickness.cgs / self.table_thickness.cgs)
        t_ox = _interp_tr(wl_nm, wl_ox, tr_ox) ** (self.oxide_thickness.cgs / self.table_thickness.cgs)
        t_c  = _interp_tr(wl_nm, wl_c,  tr_c)  ** (self.c_thickness.cgs / self.table_thickness.cgs)
        return t_al * t_ox * t_c * self.mesh_throughput

    def visible_light_throughput(self) -> float:
        """
        Visible-light transmission of the filter, including the mesh.

        It falls by a factor of 10 per 170 angstrom of aluminium.
        """
        thickness_aa = self.al_thickness.to(u.angstrom).value
        layers = thickness_aa / 170.0
        return 10.0 ** (-layers) * self.mesh_throughput


# -----------------------------------------------------------------------------
# Configuration objects
# -----------------------------------------------------------------------------
@dataclass
class Detector_SWC:
    """
    EUVST-SW detector.

    The dark current is computed from the CCD temperature when the detector
    is made, and stored as `dark_current`.

    Parameters
    ----------
    ccd_temperature : u.Quantity
        CCD temperature. Default -60 Celsius.
    qe_vis, qe_euv : float
        Quantum efficiency for visible stray light and for EUV. Default 1.0
        and 0.76.
    read_noise_rms : u.Quantity
        Read noise. Default 10 electron per pixel.
    gain_e_per_dn : u.Quantity
        Electrons per DN. Default 2.78.
    max_dn : u.Quantity
        Digitiser maximum, where DN are clipped. Default 65535 DN per pixel.
    full_well : u.Quantity
        CCD full well, for comparison only: nothing is clipped at it.
        Default 150000 electron per pixel. Keyword only.
    pix_size : u.Quantity
        Pixel size. Default 13.5 um.
    wvl_res : u.Quantity
        Wavelength step from one pixel to the next. Default 16.9 mA.
    plate_scale_angle : u.Quantity
        Angle on the sky of one pixel along the slit. Default 0.159 arcsec.
    material : str
        Detector material. Only `"silicon"` is modelled.
    filter_distance : u.Quantity
        Distance from the filter to the detector, for pinhole diffraction.
        Default 250 mm.
    shutter : bool
        Whether frames are taken with the mechanical shutter, which keeps the
        CCDs dark while a frame is cleared and read. Default True. An
        instrument run models frames taken with it and refuses False; the
        read-out model, :mod:`euvst_response.readout`, takes either. Keyword
        only, as are the settings below, which describe the SW camera's CCDs
        and their read-out.
    n_rows, n_columns : int
        Pixels of each CCD along the dispersion and along the slit. Default
        2048 by 2048 (SOLC-EUVST-MSSL-SP-0001 v1.4).
    ccd_gap : u.Quantity
        Gap between the two CCDs' imaging areas, along the dispersion.
        Default 1 mm.
    row_transfer_time : u.Quantity
        Time to move every row one step towards the serial register.
        Default 15 us, as in the SWC modes table and the read-out times of
        SOLC-EUVST-MSSL-RP-0007 v1.2 Table 5-11. It can be changed in
        flight.
    pixel_period : u.Quantity
        Time to move one pixel along the serial register and digitise it.
        Default 500 ns (SOLC-EUVST-MSSL-RS-0002 v2.0).
    serial_prescan, serial_overscan : int
        Samples each output digitises before and after the image pixels of
        each row it reads. Default 50 and 20. Each can be set in flight,
        from 0 to 200.
    parallel_overscan_rows : int
        Rows clocked and read after the last image row. Default 20.
    """
    ccd_temperature: u.Quantity = -60 * u.deg_C
    qe_vis: float = 1.0
    qe_euv: float = 0.76
    read_noise_rms: u.Quantity = 10.0 * u.electron / u.pixel
    dark_current: u.Quantity = field(init=False)
    _dark_current_293k: u.Quantity = 20000.0 * u.electron / (u.pixel * u.s)  # Q_d0 at 293 K
    gain_e_per_dn: u.Quantity = 2.78 * u.electron / u.DN  # MSSL EM test results
    max_dn: u.Quantity = 65535 * u.DN / u.pixel
    # Peak charge storage of the CCD42-40 in non-inverted mode, where
    # resolution begins to degrade: 150 ke- typical, 80 ke- minimum (Teledyne
    # e2v CCD42-40 BSI datasheet, 1B300000-A1A version 1, January 2024). It is
    # below the 182 ke- the FEE accepts, so a pixel fills before the digitiser
    # does. Nothing is clipped or spilled at it; it only shows which pixels a
    # frame would saturate. Keyword-only, so the fields after it keep their
    # places in the constructor.
    full_well: u.Quantity = field(default=150000 * u.electron / u.pixel, kw_only=True)
    pix_size: u.Quantity = (13.5 * u.um).cgs / u.pixel
    wvl_res: u.Quantity = (16.9 * u.mAA).cgs / u.pixel
    plate_scale_angle: u.Quantity = 0.159 * u.arcsec / u.pixel
    material: str = "silicon"
    filter_distance: u.Quantity = 250 * u.mm  # Distance from filter to detector for pinhole diffraction
    # Settings of the camera's CCDs and their read-out, which
    # euvst_response.readout models. An instrument run only checks and
    # records them.
    shutter: bool = field(default=True, kw_only=True)
    n_rows: int = field(default=2048, kw_only=True)
    n_columns: int = field(default=2048, kw_only=True)
    ccd_gap: u.Quantity = field(default=1.0 * u.mm, kw_only=True)
    row_transfer_time: u.Quantity = field(default=15.0 * u.us, kw_only=True)
    pixel_period: u.Quantity = field(default=500.0 * u.ns, kw_only=True)
    serial_prescan: int = field(default=50, kw_only=True)
    serial_overscan: int = field(default=20, kw_only=True)
    parallel_overscan_rows: int = field(default=20, kw_only=True)

    def __post_init__(self):
        _check_detector(self)
        # Scan limits are the FEE's settable range (SOLC-EUVST-MSSL-RS-0002 v2.0).
        for name, least, most in (("n_rows", 1, None), ("n_columns", 1, None),
                                  ("serial_prescan", 0, 200), ("serial_overscan", 0, 200),
                                  ("parallel_overscan_rows", 0, None)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < least:
                raise ValueError(f"detector.{name} must be a whole number, {least} or more, "
                                 f"got {value!r}.")
            if most is not None and value > most:
                raise ValueError(f"detector.{name} can be set between {least} and {most}, "
                                 f"got {value}.")
        self.dark_current = self.calculate_dark_current(self.ccd_temperature,
                                                        self._dark_current_293k)

    @property
    def si_fano(self) -> float:
        """Fano factor of the detector material, which sets the spread in electrons per photon."""
        return DETECTOR_MATERIALS[self.material]["fano_factor"]

    @staticmethod
    def calculate_dark_current(temp: u.Quantity,
                               dark_current_293k: u.Quantity | None = None) -> u.Quantity:
        """
        Dark current of the EUVST-SW CCD at a given temperature.

        Parameters
        ----------
        temp : u.Quantity
            CCD temperature. Below 198 K the dark current at 198 K is used;
            above 300 K the temperature is refused.
        dark_current_293k : u.Quantity, optional
            Dark current at 293 K. Default 20000 electron per pixel per
            second.

        Returns
        -------
        u.Quantity
            Dark current in electron per pixel per second.
        """
        rate = Detector_SWC._dark_current_293k if dark_current_293k is None else dark_current_293k
        return calculate_dark_current(temp, rate, ccd_type="NIMO")

    @property
    def plate_scale_length(self) -> u.Quantity:
        """Length on the Sun, seen from 1 AU, of one pixel along the slit."""
        return angle_to_distance(self.plate_scale_angle * 1*u.pix) / u.pixel


@dataclass
class Detector_EIS:
    """
    Hinode/EIS detector.

    The pixel, noise and gain settings of `Detector_SWC`, with EIS values:
    quantum efficiency 0.64 for EUV and 0.65 for visible light, read noise 5
    electron per pixel, 6.3 electron per DN, 22.3 mA per pixel and 1 arcsec
    per pixel along the slit. It has none of SWC's full-well, filter,
    shutter or CCD read-out settings.
    """
    ccd_temperature: u.Quantity = -60 * u.deg_C
    qe_euv: float = 0.64  # EIS SW Note 2
    qe_vis: float = 0.65  # MSSL engineering test report
    read_noise_rms: u.Quantity = 5.0 * u.electron / u.pixel
    dark_current: u.Quantity = field(init=False)
    _dark_current_293k: u.Quantity = 250.0 * u.electron / (u.pixel * u.s)  # Q_d0 at 293K for EIS
    gain_e_per_dn: u.Quantity = 6.3 * u.electron / u.DN
    max_dn: u.Quantity = 65535 * u.DN / u.pixel
    pix_size: u.Quantity = (13.5 * u.um).cgs / u.pixel
    wvl_res: u.Quantity = (22.3 * u.mAA).cgs / u.pixel
    plate_scale_angle: u.Quantity = 1 * u.arcsec / u.pixel
    material: str = "silicon"

    @property
    def si_fano(self) -> float:
        """Fano factor of the detector material, which sets the spread in electrons per photon."""
        return DETECTOR_MATERIALS[self.material]["fano_factor"]

    def __post_init__(self):
        _check_detector(self)
        self.dark_current = self.calculate_dark_current(self.ccd_temperature,
                                                        self._dark_current_293k)

    @property
    def plate_scale_length(self) -> u.Quantity:
        """Length on the Sun, seen from 1 AU, of one pixel along the slit."""
        return angle_to_distance(self.plate_scale_angle * 1*u.pix) / u.pixel

    @staticmethod
    def calculate_dark_current(temp: u.Quantity,
                               dark_current_293k: u.Quantity | None = None) -> u.Quantity:
        """
        Dark current of the EIS CCD at a given temperature.

        Parameters
        ----------
        temp : u.Quantity
            CCD temperature. Below 198 K the dark current at 198 K is used;
            above 300 K the temperature is refused.
        dark_current_293k : u.Quantity, optional
            Dark current at 293 K. Default 250 electron per pixel per second.

        Returns
        -------
        u.Quantity
            Dark current in electron per pixel per second.
        """
        rate = Detector_EIS._dark_current_293k if dark_current_293k is None else dark_current_293k
        return calculate_dark_current(temp, rate, ccd_type="AIMO")


@dataclass
class Telescope_EUVST:
    """
    EUVST telescope, with the EUVST-SW grating and filter.

    Parameters
    ----------
    D_ap : u.Quantity
        Aperture diameter. Half of the aperture area feeds the SW channel.
        Default 0.28 m.
    microroughness_sigma : u.Quantity
        RMS roughness of the primary mirror. Default 0.3 nm.
    filter : AluminiumFilter
        Filter. Default `AluminiumFilter()`.
    psf_type : str
        PSF shape. Only `"gaussian"` is modelled.
    psf_params : list of u.Quantity
        PSF FWHMs in pixels, along the slit and in wavelength. Default 2.66
        and 2.54 pixels.
    psf_slit_width : u.Quantity or None
        Slit the FWHM in wavelength was measured with. The FWHM for other
        slits is derived from it. Default 0.2 arcsec.
    psf_across_slit : u.Quantity or None
        FWHM of the telescope's blur across the slit, which brings in light
        from either side of it. Default None, for no blur. Keyword only.
    slit_length : u.Quantity
        Length of the slits. Default 300 arcsec. Keyword only.
    fiducial_centres : list of u.Quantity
        Middle of each fiducial along the slit, from the slit's centre. A
        fiducial is a short gap in the slit, used to co-align observations
        made with different slits. Default -25 and 70 arcsec. Keyword only.
    fiducial_lengths : list of u.Quantity
        Fiducial length for the 0.2, 0.4, 0.8 and 1.6 arcsec slits, in that
        order. Default 0.8, 0.8, 0.8 and 1.6 arcsec. Keyword only.
    pm_table, grating_table : path
        Tables of primary mirror reflectance and grating efficiency, two
        columns: wavelength in nm and value. Default the packaged tables.
    """
    D_ap: u.Quantity = 0.28 * u.m
    microroughness_sigma: u.Quantity = 0.3 * u.nm  # RMS microroughness for primary mirror
    filter: AluminiumFilter = field(default_factory=AluminiumFilter)
    psf_type: str = "gaussian"
    # psf_params: list = field(default_factory=lambda: [1.26 * u.pixel, 1.95 * u.pixel])  # [spatial_fwhm, spectral_fwhm] in pixels. From 0.200 arcsec (w/ slit-scan; FOV2) and 33.00 mA in RSC-2022021 (Oct 2023) and RSC-2022021B (Feb 2024).
    psf_params: list = field(default_factory=lambda: [2.66 * u.pixel, 2.54 * u.pixel])  # [spatial_fwhm, spectral_fwhm] in pixels. From 0.423 arcsec (w/ slit-scan; FOV2) and 43.00 mA in RSC-2022021C (Mar 2025).
    # Slit width the spectral FWHM in psf_params is for. RSC-2022021C quotes
    # the spectral resolution for the 0.2 arcsec slit as the optics FWHM after
    # the slit (0.352 arcsec at 212.3 A) added in quadrature to the slit width
    # (giving 0.405 arcsec, 43.00 mA). The spectral PSF for any other slit is
    # derived from it; see radiometric.spectral_psf_fwhm.
    psf_slit_width: u.Quantity = 0.2 * u.arcsec
    # Telescope blur across the slit: the FWHM of its image at the slit, which
    # sets how much light from beside the slit falls into it. None leaves it
    # out, so the slit takes in only the light it covers. This is not
    # psf_params[0], the blur along the slit measured at the detector, which
    # includes the spectrograph after the slit.
    psf_across_slit: u.Quantity | None = field(default=None, kw_only=True)
    # The slits are 4.07 mm long, 300 arcsec at the slit's 13.57 um per arcsec,
    # and each has two fiducials, gaps 10.9 um (0.8 arcsec) long in the three
    # narrow slits and 21.8 um (1.6 arcsec) in the widest, centred -340 um
    # (-25 arcsec) and +950 um (+70 arcsec) from its centre along the slit
    # (TASI-SCE-SSS-0001 issue 1, R-SA-SYS-REQ-1160 and 1170). The documents
    # do not say which end of the slit is positive on the detector.
    slit_length: u.Quantity = field(default=300.0 * u.arcsec, kw_only=True)
    fiducial_centres: list = field(
        default_factory=lambda: [-25.0 * u.arcsec, 70.0 * u.arcsec], kw_only=True)
    fiducial_lengths: list = field(
        default_factory=lambda: [0.8 * u.arcsec, 0.8 * u.arcsec, 0.8 * u.arcsec, 1.6 * u.arcsec],
        kw_only=True)

    # Wavelength-dependent efficiency tables
    pm_table: Path = field(default_factory=lambda: files('euvst_response') / 'data' / 'throughput' / 'primary_mirror_coating_reflectance.dat')
    grating_table: Path = field(default_factory=lambda: files('euvst_response') / 'data' / 'throughput' / 'grating_reflection_efficiency.dat')

    def __post_init__(self):
        _check_settings(self, "telescope", positive=("D_ap", "psf_slit_width", "slit_length"),
                        non_negative=("microroughness_sigma",), optional=("psf_slit_width",))
        _check_psf_params(self)
        _check_fiducials(self)
        _check_tables(self, ("pm_table", "grating_table"), "telescope")

    def slit_openings(self, slit_width: u.Quantity) -> u.Quantity:
        """
        Stretches of the slit that let light through: its length less its fiducials.

        Parameters
        ----------
        slit_width : u.Quantity
            Slit, 0.2, 0.4, 0.8 or 1.6 arcsec, which sets the fiducial length.

        Returns
        -------
        u.Quantity
            `(n, 2)` start and end of each open stretch, along the slit from
            its centre, in increasing order.
        """
        widths = [0.2, 0.4, 0.8, 1.6]
        width = u.Quantity(slit_width).to_value(u.arcsec)
        if width not in widths:
            raise ValueError(f"The fiducials are known for the 0.2, 0.4, 0.8 and 1.6 arcsec "
                             f"slits, not {slit_width}.")
        length = self.fiducial_lengths[widths.index(width)].to_value(u.arcsec)
        centres = np.sort([c.to_value(u.arcsec) for c in self.fiducial_centres])
        half = self.slit_length.to_value(u.arcsec) / 2
        edges = np.concatenate([[-half], np.ravel(np.column_stack(
            [centres - length / 2, centres + length / 2])), [half]])
        return edges.reshape(-1, 2) * u.arcsec

    @property
    def collecting_area(self) -> u.Quantity:
        """Aperture area feeding the SW channel: half of the total."""
        return 0.5 * np.pi * (self.D_ap / 2) ** 2  # Accounting for 50% loss due to beam division between SW and LW channels.

    def primary_mirror_efficiency(self, wl0: u.Quantity) -> float | np.ndarray:
        """
        Primary mirror reflectance from its table, without roughness.

        Parameters
        ----------
        wl0 : u.Quantity
            Wavelength, or an array of them.

        Returns
        -------
        float or np.ndarray
            Reflectance, one value or one per wavelength. NaN outside the
            table's wavelengths.
        """
        wl_nm = wl0.to_value(u.nm)
        wl_pm, eff_pm = _load_throughput_table(self.pm_table)
        return _interp_tr(wl_nm, wl_pm, eff_pm)

    def grating_efficiency(self, wl0: u.Quantity) -> float | np.ndarray:
        """
        Grating efficiency from its table.

        Parameters
        ----------
        wl0 : u.Quantity
            Wavelength, or an array of them.

        Returns
        -------
        float or np.ndarray
            Efficiency, one value or one per wavelength. NaN outside the
            table's wavelengths.
        """
        wl_nm = wl0.to_value(u.nm)
        wl_grat, eff_grat = _load_throughput_table(self.grating_table)
        return _interp_tr(wl_nm, wl_grat, eff_grat)

    def microroughness_efficiency(self, wl0: u.Quantity) -> float | np.ndarray:
        """
        Fraction of light the primary mirror's roughness leaves in the image.

        This is `exp(-(4 pi sigma / lambda)**2)` for RMS roughness sigma
        (`microroughness_sigma`) at wavelength lambda.

        Parameters
        ----------
        wl0 : u.Quantity
            Wavelength, or an array of them.

        Returns
        -------
        float or np.ndarray
            Fraction, one value or one per wavelength.
        """
        # Convert both wavelength and sigma to the same units (nm for convenience)
        wl_nm = wl0.to(u.nm)
        sigma_nm = self.microroughness_sigma.to(u.nm)
        
        # Calculate (4*pi*sigma/lambda)^2
        roughness_term = (4 * np.pi * sigma_nm / wl_nm) ** 2
        
        # Return exp(-(4*pi*sigma/lambda)^2)  [Debye-Waller specular efficiency]
        return np.exp(-roughness_term.value)

    def throughput(self, wl0: u.Quantity) -> u.Quantity:
        """
        Throughput of the primary mirror (with roughness), grating and filter.

        Parameters
        ----------
        wl0 : u.Quantity
            Wavelength, or an array of them.

        Returns
        -------
        u.Quantity
            Dimensionless throughput, one value or one per wavelength.
        """
        # Get wavelength-dependent efficiencies
        pm_eff_wl = self.primary_mirror_efficiency(wl0)
        grat_eff_wl = self.grating_efficiency(wl0)
        
        # Apply microroughness efficiency to primary mirror efficiency
        pm_eff_with_roughness = pm_eff_wl * self.microroughness_efficiency(wl0)
        
        # Calculate total throughput
        return pm_eff_with_roughness * grat_eff_wl * self.filter.total_throughput(wl0)

    def ea_and_throughput(self, wl0: u.Quantity) -> u.Quantity:
        """
        Collecting area times throughput: the effective area before the detector.

        Multiplied by the detector's quantum efficiency, `qe_euv`, this gives
        the instrument's effective area.

        Parameters
        ----------
        wl0 : u.Quantity
            Wavelength, or an array of them.

        Returns
        -------
        u.Quantity
            Area, one value or one per wavelength.
        """
        return self.collecting_area * self.throughput(wl0)


@dataclass
class Telescope_EIS:
    """
    Hinode/EIS telescope.

    The effective area comes from the EIS calibration tables, so it varies
    with wavelength and, for the in-flight calibrations, with the date of the
    observation.

    Parameters
    ----------
    psf_type : str
        PSF shape. Only `"gaussian"` is modelled.
    psf_params : list of u.Quantity
        PSF FWHMs in pixels, along the slit and in wavelength. Default 3 and 3
        pixels.
    psf_slit_width : u.Quantity or None
        Slit the FWHM in wavelength was measured with. Default None, which
        gives every slit the same FWHM.
    psf_across_slit : u.Quantity or None
        FWHM of the telescope's blur across the slit. Default None, for no
        blur. Keyword only.
    calibration : str
        `"ground"` (default), `"dz2013"`, `"warren2014"` or `"dz2025"`.
        `"ground"` is the pre-flight calibration, which needs no date.
        `"dz2025"` is the one to use for a real observation.
    date : str, optional
        Observation date, such as `"2012-06-03"`, needed by every calibration
        except `"ground"`. A `datetime` or an `astropy.time.Time` also works.

    Examples
    --------
    >>> tel = Telescope_EIS()                                  # pre-flight
    >>> tel = Telescope_EIS(calibration="dz2025", date="2012-06-03")
    """
    psf_type: str = "gaussian"
    psf_params: list = field(default_factory=lambda: [3.0 * u.pixel, 3.0 * u.pixel])  # [spatial_fwhm, spectral_fwhm] in pixels
    # The EIS PSF is not tied to a slit width, so its spectral FWHM is the
    # same for every slit. Setting this gives the slit psf_params was measured
    # with, and the spectral PSF then follows the slit as for SWC.
    psf_slit_width: u.Quantity | None = None
    # Telescope blur across the slit, the FWHM of its image at the slit, as
    # for Telescope_EUVST. None leaves it out.
    psf_across_slit: u.Quantity | None = field(default=None, kw_only=True)
    calibration: str = "ground"
    date: str | None = None

    def __post_init__(self):
        _check_psf_params(self)
        if self.calibration not in eis_calibration.CALIBRATIONS:
            raise ValueError(
                f"Unknown EIS calibration {self.calibration!r}. Choose from: "
                f"{', '.join(eis_calibration.CALIBRATIONS)}."
            )
        if self.psf_slit_width is not None and not (
                isinstance(self.psf_slit_width, u.Quantity) and self.psf_slit_width.isscalar
                and self.psf_slit_width.unit.is_equivalent(u.arcsec)
                and np.isfinite(self.psf_slit_width.value) and self.psf_slit_width.value > 0):
            raise ValueError(f"telescope.psf_slit_width must be a finite angle above zero, got "
                             f"{self.psf_slit_width!r}.")
        if self.date is not None:
            self.date = eis_calibration.normalise_date(self.date)
            # Parse the date now, rather than part-way through a run when the
            # effective area is first needed.
            try:
                eis_calibration._parse_date(self.date)
            except ValueError:
                raise ValueError(f"telescope.date {self.date!r} is not a date ECLIPSE can "
                                 f"read; give it in ISO form, such as 2012-06-03.") from None
        elif self.calibration in eis_calibration.TIME_DEPENDENT_CALIBRATIONS:
            raise ValueError(
                f"The {self.calibration!r} EIS calibration is time-dependent, "
                f"so Telescope_EIS needs a date, for example "
                f"Telescope_EIS(calibration='{self.calibration}', "
                f"date='2012-06-03'). Use calibration='ground' for the "
                f"epoch-independent pre-flight calibration."
            )

    def effective_area(self, wl0: u.Quantity) -> u.Quantity:
        """
        EIS effective area, including the detector quantum efficiency.

        This is the effective area as published by the EIS calibrations.
        Wavelengths outside the two EIS bands are refused.

        Parameters
        ----------
        wl0 : u.Quantity
            Wavelength, or an array of them.

        Returns
        -------
        u.Quantity
            Area, one value or one per wavelength.
        """
        wl_aa = u.Quantity(wl0).to_value(u.AA)
        area = eis_calibration.effective_area(
            wl_aa, date=self.date, method=self.calibration
        )

        if np.any(~np.isfinite(area)):
            bad = np.atleast_1d(wl_aa)[~np.isfinite(area)]
            raise ValueError(
                f"EIS has no effective area at {bad[0]:.3f} Angstrom "
                f"({bad.size} wavelength(s) affected). Its bands are "
                f"{eis_calibration.SW_BAND} and {eis_calibration.LW_BAND} "
                f"Angstrom."
            )

        area = area * u.cm**2
        return area[0] if np.ndim(wl_aa) == 0 else area

    def ea_and_throughput(self, wl0: u.Quantity) -> u.Quantity:
        """
        EIS effective area without the detector quantum efficiency.

        ECLIPSE applies the quantum efficiency, `Detector_EIS.qe_euv`, in the
        detector, so this is the area the rest of the simulation uses.

        Parameters
        ----------
        wl0 : u.Quantity
            Wavelength, or an array of them.

        Returns
        -------
        u.Quantity
            Area, one value or one per wavelength.
        """
        # The tabulated areas include the QE the EIS calibration is quoted
        # against. ECLIPSE applies the QE separately, so it is divided out
        # here. This is the table constant on purpose, not a setting: dividing
        # by a settable value while to_electrons applies Detector_EIS.qe_euv
        # would silently scale the whole response.
        # https://hinode.nao.ac.jp/en/for-researchers/instruments/eis/fact-sheet/
        # https://solarb.mssl.ucl.ac.uk/SolarB/eis_docs/eis_notes/02_RADIOMETRIC_CALIBRATION/eis_swnote_02.pdf
        return self.effective_area(wl0) / eis_calibration.QE_IN_TABLES


@dataclass
class Simulation:
    """
    Settings of one simulation: exposure, slit, PSF and noise.

    Unlike in the configuration file, each setting takes one value here.

    Parameters
    ----------
    expos : u.Quantity
        Exposure time. Default 1 s.
    n_iter : int
        Number of Monte Carlo iterations. Default 10.
    slit_width : u.Quantity
        Slit width: 0.2, 0.4, 0.8 or 1.6 arcsec for SWC, 1 or 2 arcsec for
        EIS. Default 0.2 arcsec.
    ncpu : int
        Number of CPU cores to use; -1 for all of them. Default -1.
    instrument : str
        `"SWC"` (default) or `"EIS"`.
    vis_sl : u.Quantity
        Visible stray light, in photons per second per cm2: before the filter
        for SWC, and at the CCD for EIS. Default 0.
    psf : bool
        Whether to blur the spectra with the PSF. Default False.
    psf_boundary : str
        What the PSF brings in from beyond the edges of the atmosphere:
        `"replicate"` (default) or `"zero"`.
    spectral_psf : str
        How the slit is combined with the optics blur: `"quadrature"`
        (default) or `"convolution"`.
    slit_mask : bool
        Whether the slit's ends and fiducials block the light (SWC only).
        Default False. Keyword only.
    noise : bool
        If False, every random draw in the detector is replaced by its mean.
        Default True.
    enable_pinholes : bool
        Whether to model pinholes in the filter (SWC only). Default False.
    pinhole_sizes : list of u.Quantity
        Pinhole diameters.
    pinhole_positions : list of float
        Position of each pinhole along the slit, as a fraction from 0 to 1.
    pinhole_positions_spectral : list of float
        Position of each pinhole along the spectral axis, as a fraction from
        0 to 1. Default empty, for the middle.
    """
    expos: u.Quantity = 1.0 * u.s  # Single exposure time
    n_iter: int = 10
    slit_width: u.Quantity = 0.2 * u.arcsec
    ncpu: int = -1
    instrument: str = "SWC"
    vis_sl: u.Quantity = 0 * u.photon / (u.s * u.cm**2)  # Visible stray light flux before SWC's filter; at EIS's CCD
    psf: bool = False
    # What the spatial PSF convolution assumes beyond the ends of the slit.
    # "replicate" extends the edge rows outward, assuming the Sun looks much
    # as it does at the edge of the field. "zero" treats everything outside
    # as dark, which removes real signal from the outermost rows. The
    # spectral direction is zero-filled either way: the wavelength grid runs
    # several sigma past the line, so nothing is lost at its ends.
    psf_boundary: str = "replicate"
    # How the slit enters the spectral PSF. "quadrature" keeps the PSF a
    # Gaussian and adds the slit width to the optics FWHM in quadrature, as
    # RSC-2022021C quotes the spectral resolution. "convolution" convolves
    # the optics Gaussian with the slit's rectangular image, as the same
    # document defines the line profile; it gives the flat-topped profile of
    # a wide slit, and a narrower profile than quadrature for the 0.2 arcsec
    # slit. See radiometric.spectral_line_spread.
    spectral_psf: str = "quadrature"
    # With slit_mask True, no light gets through beyond the slit's ends or in
    # its fiducials, as set in the telescope. The slit's centre is the middle
    # of the scene along the slit. The blur along the slit comes before the
    # slit, so a fiducial's edges are as sharp as the pixels sampling them.
    slit_mask: bool = field(default=False, kw_only=True)
    # With noise False, every random draw in the detector chain is replaced by
    # its mean, so the run gives the signal the instrument would measure on
    # average. Quantisation stays: DN are still rounded and still clipped at
    # the digitiser maximum, max_dn.
    noise: bool = True
    enable_pinholes: bool = False
    pinhole_sizes: List[u.Quantity] = field(default_factory=list)
    pinhole_positions: List[float] = field(default_factory=list)
    # Position along the spectral axis, as a fraction (0.0 to 1.0) of the
    # detector width, one per pinhole. Empty (the default) puts every pinhole
    # at the centre of the spectral window. On a slit-scan spectrograph this
    # fraction decides which emission lines a pinhole contaminates.
    pinhole_positions_spectral: List[float] = field(default_factory=list)

    @property
    def slit_scan_step(self) -> u.Quantity:
        return self.slit_width

    def __post_init__(self):
        _check_settings(self, "simulation", positive=("expos", "slit_width"),
                        non_negative=("vis_sl",))
        if isinstance(self.n_iter, bool) or not isinstance(self.n_iter, int) or self.n_iter < 1:
            raise ValueError(f"n_iter must be a whole number of iterations, 1 or more, got "
                             f"{self.n_iter!r}.")
        for index, size in enumerate(self.pinhole_sizes):
            if not (isinstance(size, u.Quantity) and size.unit.is_equivalent(u.um) and size > 0):
                raise ValueError(f"pinhole_sizes[{index}] must be a diameter, a positive "
                                 f"length, such as 5 um; got {size!r}.")
        # EIS has 1 and 2 arcsec slits, and 40 and 266 arcsec slots, which
        # ECLIPSE does not model.
        allowed_slits = {
            "EIS": [1, 2],
            "SWC": [0.2, 0.4, 0.8, 1.6],
        }
        inst = self.instrument.upper()
        slit_val = self.slit_width.to_value(u.arcsec)
        if inst == "EIS":
            if slit_val not in allowed_slits["EIS"]:
                raise ValueError("For EIS, slit_width must be 1 or 2 arcsec, its two slits.")
        elif inst in ("SWC"):
            if slit_val not in allowed_slits["SWC"]:
                raise ValueError("For SWC, slit_width must be 0.2, 0.4, 0.8, or 1.6 arcsec.")

        if self.psf_boundary not in ("replicate", "zero"):
            raise ValueError(
                f"psf_boundary must be 'replicate' or 'zero', got "
                f"{self.psf_boundary!r}."
            )
        if self.spectral_psf not in ("quadrature", "convolution"):
            raise ValueError(
                f"spectral_psf must be 'quadrature' or 'convolution', got "
                f"{self.spectral_psf!r}."
            )
        if self.slit_mask and inst == "EIS":
            raise ValueError("slit_mask is for SWC, whose slits' fiducials ECLIPSE knows; "
                             "EIS's slits are taken to be open along their length.")

        # Both pipelines zip the paired pinhole lists, and zip stops at the
        # shortest, so a mismatch would silently drop pinholes. main() runs
        # the same check on configuration files, but Simulation is also
        # constructed directly.
        self.pinhole_positions, self.pinhole_positions_spectral = check_pinhole_lists(
            self.pinhole_sizes, self.pinhole_positions,
            self.pinhole_positions_spectral)
