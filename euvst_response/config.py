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
    Refuse any setting of *obj* not of its default's kind, or out of its range.

    A quantity needs a unit of the same kind as its default: a number with
    none, or with one of another kind, was read in whatever unit the code
    happened to work in, so that a D_ap of 0.28, meant in metres, gave 1e4
    times too few photons. A switch must be true or false, and a number a
    number. *positive*, *non_negative* and *fractions* name the fields that
    must be above zero, at or above it, or from 0 to 1, where a value outside
    ran to completion and saved results that meant nothing; *optional* those
    that may also be None.
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
    The PSF settings: a Gaussian, psf_params two widths in pixels, along the
    slit and along the dispersion, above zero, and psf_across_slit None or an
    angle above zero. Otherwise they are only read once the scene is being
    laid onto the detector, part way through a run.
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


def _check_tables(obj, names: tuple, section: str) -> None:
    """
    Each of *names* the path of a table; one named in a configuration comes as text.

    Whether the table is there is left until it is read, since a results
    file made on another machine rebuilds these with that machine's paths.
    """
    for name in names:
        value = getattr(obj, name)
        if isinstance(value, (str, os.PathLike)):
            value = Path(value).expanduser()
            setattr(obj, name, value)
        if not hasattr(value, "is_file"):
            raise ValueError(f"{section}.{name} must be the path of a table, got {value!r}.")


def _check_detector(det, section: str = "detector") -> None:
    """The checks both detectors share: kinds, ranges, a temperature above absolute zero and a known material."""
    _check_settings(det, section,
                    positive=("gain_e_per_dn", "max_dn", "full_well", "pix_size", "wvl_res",
                              "plate_scale_angle", "filter_distance"),
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
    Return (lambda, T) arrays from a 2-col ASCII table (skip comments). lambda is in nm.

    Each table is read once: the effective area is asked for at every
    wavelength of a spectrum, and reading five tables again for each one made
    a spectrum take minutes.  The arrays are shared, so they are read-only.
    """
    key = str(path)
    if key not in _THROUGHPUT_TABLES:
        # A path given as text is made one; a package's own table may be a
        # resource inside an archive, which reads itself but is not a path.
        content = (path if hasattr(path, "read_text") else Path(path)).read_text()
        # Headers are skipped as the lines before the data that are not two
        # numbers, however many there are: skipping the first two, as the
        # packaged tables have, lost the data rows of a table with fewer. A
        # line that is not two numbers once the data has begun is a slip in
        # the table, which would otherwise change the curve.
        data = []
        for number, line in enumerate(content.splitlines(), start=1):
            text = line.strip()
            if not text or text.startswith('#'):
                continue
            # Every value of a line of data is a number, the first two its
            # wavelength and throughput.
            try:
                row = [float(x) for x in text.split()]
            except ValueError:
                row = []
            if len(row) >= 2:
                # A throughput in per cent, or a nan, reached the effective
                # area and made its every value meaningless.
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
                # Before the data, a header, as the packaged tables' are, but
                # one that begins with a number may be a first line of data
                # with a slip in it, which would change the curve's end.
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
    # A table's own end, reached through a change of unit, can land a part in
    # 1e16 beyond it, as 214 Angstrom does at 21.400000000000002 nm.
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

    Configuration files and a directly built ``Simulation`` both go through
    this, so they cannot disagree about what a valid pinhole is.

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
        ``(positions, spectral)``, each entry converted to float.
    """
    # Compared unconditionally. Guarding this on the sizes let positions alone
    # through, and the run then produced no pinholes and said nothing about it.
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
    """Return *values* as floats, rejecting any that do not lie in [0, 1].

    Both position lists are a fraction of the way across the detector. Out of
    range puts the pinhole off it, where it looks like a working pinhole whose
    light merely happens to be missing. Converting also stops a quoted YAML
    value such as '0.3' reaching the diffraction code as a string.
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
    EUVST-SW's filter: layers of aluminium, aluminium oxide and carbon on a mesh.

    Each layer's transmission comes from a table for a layer ``table_thickness``
    thick, raised to the power of its own thickness over that one.

    Parameters
    ----------
    al_thickness, oxide_thickness, c_thickness : u.Quantity
        The thickness of each layer. Default 1485, 95 and 0 angstrom.
    mesh_throughput : float
        The fraction of light the supporting mesh lets through. Default 0.8.
    al_table, oxide_table, c_table : path
        Tables of each layer's transmission, two columns: the wavelength in nm
        and the transmission. Default the packaged tables.
    table_thickness : u.Quantity
        The thickness the tables are for. Default 1000 angstrom.
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
        The filter's EUV transmission, mesh included.

        Parameters
        ----------
        wl0 : u.Quantity
            A wavelength, or an array of them.

        Returns
        -------
        u.Quantity
            The transmission, dimensionless: one value, or one per wavelength.
            It is NaN outside the tables' wavelengths.
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
        The filter's transmission for visible light, mesh included.

        It falls by a factor of 10 for every 170 angstrom of aluminium.
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
    EUVST-SW's detector.

    The dark current is worked out from the CCD's temperature when the
    detector is made, and is held as ``dark_current``.

    Parameters
    ----------
    ccd_temperature : u.Quantity
        The CCD's temperature. Default -60 Celsius.
    qe_vis, qe_euv : float
        The quantum efficiency for visible stray light and for EUV. Default
        1.0 and 0.76.
    read_noise_rms : u.Quantity
        The read noise. Default 10 electron per pixel.
    gain_e_per_dn : u.Quantity
        Electrons per DN. Default 2.78.
    max_dn : u.Quantity
        The digitiser's maximum, where DN are clipped. Default 65535 DN per
        pixel.
    full_well : u.Quantity
        The CCD's full well, for comparison only: nothing is clipped at it.
        Default 150000 electron per pixel. Keyword only.
    pix_size : u.Quantity
        The size of a pixel. Default 13.5 um.
    wvl_res : u.Quantity
        The wavelength step from one pixel to the next. Default 16.9 mA.
    plate_scale_angle : u.Quantity
        The angle on the sky of one pixel along the slit. Default 0.159
        arcsec.
    material : str
        The detector's material. Only ``"silicon"`` is modelled.
    filter_distance : u.Quantity
        The distance from the filter to the detector, for the pinholes'
        diffraction. Default 250 mm.
    """
    ccd_temperature: u.Quantity = -60 * u.deg_C
    qe_vis: float = 1.0
    qe_euv: float = 0.76
    read_noise_rms: u.Quantity = 10.0 * u.electron / u.pixel
    dark_current: u.Quantity = field(init=False)
    _dark_current_293k: u.Quantity = 20000.0 * u.electron / (u.pixel * u.s)  # Q_d0 at 293 K
    gain_e_per_dn: u.Quantity = 2.78 * u.electron / u.DN  # MSSL EM test results
    max_dn: u.Quantity = 65535 * u.DN / u.pixel
    # Peak charge storage of the CCD42-40 in non-inverted mode, the signal at
    # which resolution begins to degrade: 150 ke- typical, 80 ke- minimum
    # (Teledyne e2v CCD42-40 BSI datasheet, 1B300000-A1A version 1, January
    # 2024). It is below the 182 ke- the FEE accepts, so a pixel fills before
    # the digitiser does. Nothing is clipped or spilled at it; it says which
    # pixels a frame would saturate. Keyword-only, so that the fields after it
    # keep their places in the constructor.
    full_well: u.Quantity = field(default=150000 * u.electron / u.pixel, kw_only=True)
    pix_size: u.Quantity = (13.5 * u.um).cgs / u.pixel
    wvl_res: u.Quantity = (16.9 * u.mAA).cgs / u.pixel
    plate_scale_angle: u.Quantity = 0.159 * u.arcsec / u.pixel
    material: str = "silicon"
    filter_distance: u.Quantity = 250 * u.mm  # Distance from filter to detector for pinhole diffraction

    def __post_init__(self):
        _check_detector(self)
        self.dark_current = self.calculate_dark_current(self.ccd_temperature,
                                                        self._dark_current_293k)

    @property
    def si_fano(self) -> float:
        """The Fano factor of the detector's material, which sets the spread in electrons per photon."""
        return DETECTOR_MATERIALS[self.material]["fano_factor"]

    @staticmethod
    def calculate_dark_current(temp: u.Quantity,
                               dark_current_293k: u.Quantity | None = None) -> u.Quantity:
        """
        The dark current of EUVST-SW's CCD at a temperature.

        Parameters
        ----------
        temp : u.Quantity
            The CCD's temperature. Below 198 K the dark current is taken as it
            is at 198 K, and above 300 K the temperature is refused.
        dark_current_293k : u.Quantity, optional
            The dark current at 293 K. Default 20000 electron per pixel per
            second.

        Returns
        -------
        u.Quantity
            The dark current, in electron per pixel per second.
        """
        rate = Detector_SWC._dark_current_293k if dark_current_293k is None else dark_current_293k
        return calculate_dark_current(temp, rate, ccd_type="NIMO")

    @property
    def plate_scale_length(self) -> u.Quantity:
        """The length on the Sun, seen from 1 AU, of one pixel along the slit."""
        return angle_to_distance(self.plate_scale_angle * 1*u.pix) / u.pixel


@dataclass
class Detector_EIS:
    """
    Hinode/EIS's detector.

    It takes the same settings as `Detector_SWC`, except ``full_well`` and
    ``filter_distance``, with EIS's values: a quantum efficiency of 0.64 for
    EUV and 0.65 for visible light, a read noise of 5 electron per pixel, 6.3
    electron per DN, 22.3 mA per pixel and 1 arcsec per pixel along the slit.
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
        """The Fano factor of the detector's material, which sets the spread in electrons per photon."""
        return DETECTOR_MATERIALS[self.material]["fano_factor"]

    def __post_init__(self):
        _check_detector(self)
        self.dark_current = self.calculate_dark_current(self.ccd_temperature,
                                                        self._dark_current_293k)

    @property
    def plate_scale_length(self) -> u.Quantity:
        """The length on the Sun, seen from 1 AU, of one pixel along the slit."""
        return angle_to_distance(self.plate_scale_angle * 1*u.pix) / u.pixel

    @staticmethod
    def calculate_dark_current(temp: u.Quantity,
                               dark_current_293k: u.Quantity | None = None) -> u.Quantity:
        """
        The dark current of EIS's CCD at a temperature.

        Parameters
        ----------
        temp : u.Quantity
            The CCD's temperature. Below 198 K the dark current is taken as it
            is at 198 K, and above 300 K the temperature is refused.
        dark_current_293k : u.Quantity, optional
            The dark current at 293 K. Default 250 electron per pixel per
            second.

        Returns
        -------
        u.Quantity
            The dark current, in electron per pixel per second.
        """
        rate = Detector_EIS._dark_current_293k if dark_current_293k is None else dark_current_293k
        return calculate_dark_current(temp, rate, ccd_type="AIMO")


@dataclass
class Telescope_EUVST:
    """
    EUVST's telescope, with EUVST-SW's grating and filter.

    Parameters
    ----------
    D_ap : u.Quantity
        The aperture's diameter. Half of its area feeds the SW channel.
        Default 0.28 m.
    microroughness_sigma : u.Quantity
        The RMS roughness of the primary mirror. Default 0.3 nm.
    filter : AluminiumFilter
        The filter. Default `AluminiumFilter()`.
    psf_type : str
        The shape of the PSF. Only ``"gaussian"`` is modelled.
    psf_params : list of u.Quantity
        The PSF's FWHMs in pixels, along the slit and in wavelength. Default
        2.66 and 2.54 pixels.
    psf_slit_width : u.Quantity or None
        The slit the FWHM in wavelength was measured with. The FWHM for other
        slits is worked out from it. Default 0.2 arcsec.
    psf_across_slit : u.Quantity or None
        The FWHM of the telescope's blur across the slit, which brings in
        light from either side of it. Default None, for no blur. Keyword only.
    pm_table, grating_table : path
        Tables of the primary mirror's reflectance and the grating's
        efficiency, two columns: the wavelength in nm and the value. Default
        the packaged tables.
    """
    D_ap: u.Quantity = 0.28 * u.m
    microroughness_sigma: u.Quantity = 0.3 * u.nm  # RMS microroughness for primary mirror
    filter: AluminiumFilter = field(default_factory=AluminiumFilter)
    psf_type: str = "gaussian"
    # psf_params: list = field(default_factory=lambda: [1.26 * u.pixel, 1.95 * u.pixel])  # [spatial_fwhm, spectral_fwhm] in pixels. From 0.200 arcsec (w/ slit-scan; FOV2) and 33.00 mA in RSC-2022021 (Oct 2023) and RSC-2022021B (Feb 2024).
    psf_params: list = field(default_factory=lambda: [2.66 * u.pixel, 2.54 * u.pixel])  # [spatial_fwhm, spectral_fwhm] in pixels. From 0.423 arcsec (w/ slit-scan; FOV2) and 43.00 mA in RSC-2022021C (Mar 2025).
    # The slit width the spectral FWHM in psf_params is for. RSC-2022021C
    # quotes the spectral resolution with the 0.2 arcsec slit, as the optics
    # FWHM after the slit (0.352 arcsec at 212.3 A) added in quadrature to the
    # slit width (giving 0.405 arcsec, 43.00 mA), so the spectral PSF of any
    # other slit is worked out from it; see radiometric.spectral_psf_fwhm.
    psf_slit_width: u.Quantity = 0.2 * u.arcsec
    # The telescope's blur across the slit: the FWHM of the image it forms at
    # the slit, which decides how much light from beside the slit falls into
    # it. None leaves it out, so that the slit takes in only the light it
    # covers. It is not psf_params[0], the blur along the slit measured at the
    # detector, which includes the spectrograph after the slit.
    psf_across_slit: u.Quantity | None = field(default=None, kw_only=True)

    # Wavelength-dependent efficiency tables
    pm_table: Path = field(default_factory=lambda: files('euvst_response') / 'data' / 'throughput' / 'primary_mirror_coating_reflectance.dat')
    grating_table: Path = field(default_factory=lambda: files('euvst_response') / 'data' / 'throughput' / 'grating_reflection_efficiency.dat')

    def __post_init__(self):
        _check_settings(self, "telescope", positive=("D_ap", "psf_slit_width"),
                        non_negative=("microroughness_sigma",), optional=("psf_slit_width",))
        _check_psf_params(self)
        _check_tables(self, ("pm_table", "grating_table"), "telescope")

    @property
    def collecting_area(self) -> u.Quantity:
        """The area of the aperture that feeds the SW channel: half of it."""
        return 0.5 * np.pi * (self.D_ap / 2) ** 2  # Accounting for 50% loss due to beam division between SW and LW channels.

    def primary_mirror_efficiency(self, wl0: u.Quantity) -> float | np.ndarray:
        """
        The primary mirror's reflectance, from its table, without its roughness.

        Parameters
        ----------
        wl0 : u.Quantity
            A wavelength, or an array of them.

        Returns
        -------
        float or np.ndarray
            The reflectance: one value, or one per wavelength. It is NaN
            outside the table's wavelengths.
        """
        wl_nm = wl0.to_value(u.nm)
        wl_pm, eff_pm = _load_throughput_table(self.pm_table)
        return _interp_tr(wl_nm, wl_pm, eff_pm)

    def grating_efficiency(self, wl0: u.Quantity) -> float | np.ndarray:
        """
        The grating's efficiency, from its table.

        Parameters
        ----------
        wl0 : u.Quantity
            A wavelength, or an array of them.

        Returns
        -------
        float or np.ndarray
            The efficiency: one value, or one per wavelength. It is NaN outside
            the table's wavelengths.
        """
        wl_nm = wl0.to_value(u.nm)
        wl_grat, eff_grat = _load_throughput_table(self.grating_table)
        return _interp_tr(wl_nm, wl_grat, eff_grat)

    def microroughness_efficiency(self, wl0: u.Quantity) -> float | np.ndarray:
        """
        The fraction of light the primary mirror's roughness leaves in the image.

        This is ``exp(-(4 pi sigma / lambda)**2)``, for an RMS roughness sigma,
        ``microroughness_sigma``, at a wavelength lambda.

        Parameters
        ----------
        wl0 : u.Quantity
            A wavelength, or an array of them.

        Returns
        -------
        float or np.ndarray
            The fraction: one value, or one per wavelength.
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
        The throughput of the mirror, with its roughness, the grating and the filter.

        Parameters
        ----------
        wl0 : u.Quantity
            A wavelength, or an array of them.

        Returns
        -------
        u.Quantity
            The throughput, dimensionless: one value, or one per wavelength.
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
        The collecting area times the throughput: the effective area before the detector.

        Multiplied by the detector's quantum efficiency, ``qe_euv``, this is
        the instrument's effective area.

        Parameters
        ----------
        wl0 : u.Quantity
            A wavelength, or an array of them.

        Returns
        -------
        u.Quantity
            The area: one value, or one per wavelength.
        """
        return self.collecting_area * self.throughput(wl0)


@dataclass
class Telescope_EIS:
    """
    Hinode/EIS's telescope.

    Its effective area comes from EIS's own calibration tables, so it varies
    with wavelength and, for the in-flight calibrations, with the date of the
    observation.

    Parameters
    ----------
    psf_type : str
        The shape of the PSF. Only ``"gaussian"`` is modelled.
    psf_params : list of u.Quantity
        The PSF's FWHMs in pixels, along the slit and in wavelength. Default 3
        and 3 pixels.
    psf_slit_width : u.Quantity or None
        The slit the FWHM in wavelength was measured with. Default None, which
        gives every slit the same FWHM.
    psf_across_slit : u.Quantity or None
        The FWHM of the telescope's blur across the slit. Default None, for no
        blur. Keyword only.
    calibration : str
        ``"ground"`` (default), ``"dz2013"``, ``"warren2014"`` or ``"dz2025"``.
        ``"ground"`` is the pre-flight calibration, which needs no date.
        ``"dz2025"`` is the one to use for a real observation.
    date : str, optional
        The date of the observation, such as ``"2012-06-03"``, which every
        calibration but ``"ground"`` needs. A ``datetime`` or an
        ``astropy.time.Time`` also works.

    Examples
    --------
    >>> tel = Telescope_EIS()                                  # pre-flight
    >>> tel = Telescope_EIS(calibration="dz2025", date="2012-06-03")
    """
    psf_type: str = "gaussian"
    psf_params: list = field(default_factory=lambda: [3.0 * u.pixel, 3.0 * u.pixel])  # [spatial_fwhm, spectral_fwhm] in pixels
    # The EIS PSF is not tied to a slit width, so its spectral FWHM stays the
    # same whichever slit is used. Setting this says which slit psf_params was
    # measured with, and the spectral PSF then follows the slit as for SWC.
    psf_slit_width: u.Quantity | None = None
    # The telescope's blur across the slit, the FWHM of its image at the
    # slit, as for Telescope_EUVST. None leaves it out.
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
            # Read now, rather than when the effective area is first wanted,
            # part-way through a run.
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
        EIS's effective area, including its detector's quantum efficiency.

        This is the effective area as EIS's calibrations publish it. It
        refuses wavelengths outside EIS's two bands.

        Parameters
        ----------
        wl0 : u.Quantity
            A wavelength, or an array of them.

        Returns
        -------
        u.Quantity
            The area: one value, or one per wavelength.
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
        EIS's effective area without its detector's quantum efficiency.

        ECLIPSE applies the quantum efficiency, ``Detector_EIS.qe_euv``, in
        the detector, so this is the area the rest of the simulation uses.

        Parameters
        ----------
        wl0 : u.Quantity
            A wavelength, or an array of them.

        Returns
        -------
        u.Quantity
            The area: one value, or one per wavelength.
        """
        # The tabulated areas include the QE the EIS calibration is quoted
        # against, which ECLIPSE applies separately, so it is divided back out
        # here. This is deliberately the table constant and not a configurable
        # field: were it settable on the telescope, dividing by one value
        # while to_electrons applied Detector_EIS.qe_euv would silently scale
        # the whole response.
        # https://hinode.nao.ac.jp/en/for-researchers/instruments/eis/fact-sheet/
        # https://solarb.mssl.ucl.ac.uk/SolarB/eis_docs/eis_notes/02_RADIOMETRIC_CALIBRATION/eis_swnote_02.pdf
        return self.effective_area(wl0) / eis_calibration.QE_IN_TABLES


@dataclass
class Simulation:
    """
    The settings of one simulation: its exposure, slit, PSF and noise.

    Unlike the configuration file, each setting takes one value here.

    Parameters
    ----------
    expos : u.Quantity
        The exposure time. Default 1 s.
    n_iter : int
        How many Monte Carlo iterations to run. Default 10.
    slit_width : u.Quantity
        The slit: 0.2, 0.4, 0.8 or 1.6 arcsec for SWC, 1 or 2 arcsec for EIS.
        Default 0.2 arcsec.
    ncpu : int
        How many CPU cores to use; -1 for all of them. Default -1.
    instrument : str
        ``"SWC"`` (default) or ``"EIS"``.
    vis_sl : u.Quantity
        Visible stray light, in photons per second per cm2: before the filter
        for SWC, and at the CCD for EIS. Default 0.
    psf : bool
        Whether to blur the spectra with the PSF. Default False.
    psf_boundary : str
        What the PSF brings in from beyond the edges of the atmosphere:
        ``"replicate"`` (default) or ``"zero"``.
    spectral_psf : str
        How the slit is added to the optics' blur: ``"quadrature"`` (default)
        or ``"convolution"``.
    noise : bool
        With False, every random draw in the detector is replaced by its mean.
        Default True.
    enable_pinholes : bool
        Whether to model pinholes in the filter (SWC only). Default False.
    pinhole_sizes : list of u.Quantity
        The pinholes' diameters.
    pinhole_positions : list of float
        Where each pinhole is along the slit, as a fraction from 0 to 1.
    pinhole_positions_spectral : list of float
        Where each pinhole is along the spectral axis, as a fraction from 0 to
        1. Default empty, for the middle.
    """
    expos: u.Quantity = 1.0 * u.s  # Single exposure time
    n_iter: int = 10
    slit_width: u.Quantity = 0.2 * u.arcsec
    ncpu: int = -1
    instrument: str = "SWC"
    vis_sl: u.Quantity = 0 * u.photon / (u.s * u.cm**2)  # Visible stray light flux before SWC's filter; at EIS's CCD
    psf: bool = False
    # What the spatial PSF convolution assumes lies beyond the ends of the
    # slit. "replicate" continues the edge rows outward, which says the Sun
    # goes on looking much as it does at the edge of the field. "zero" treats
    # everything outside as dark, which is what ECLIPSE did before and which
    # removes real signal from the outermost rows. The spectral direction is
    # zero-filled either way: the wavelength grid runs several sigma past the
    # line, so there is nothing at its ends to lose.
    psf_boundary: str = "replicate"
    # How the slit enters the spectral PSF. "quadrature" keeps the PSF a
    # Gaussian and adds the slit's width to the optics FWHM in quadrature,
    # which is how RSC-2022021C quotes the spectral resolution. "convolution"
    # convolves the optics Gaussian with the slit's rectangular image, which
    # is how the same document defines the line profile; it gives the
    # flat-topped profile of a wide slit, and a narrower one than quadrature
    # for the 0.2 arcsec slit. See radiometric.spectral_line_spread.
    spectral_psf: str = "quadrature"
    # With noise False every random draw in the detector chain is replaced by
    # its own mean, so the run returns the signal the instrument would measure
    # on average. Deterministic quantisation stays: DN are still rounded and
    # still clip at the digitiser's maximum, max_dn.
    noise: bool = True
    enable_pinholes: bool = False
    pinhole_sizes: List[u.Quantity] = field(default_factory=list)
    pinhole_positions: List[float] = field(default_factory=list)
    # Position along the spectral axis, as a fraction (0.0 to 1.0) of the
    # detector width, one per pinhole.  Empty (the default) projects every
    # pinhole to the centre of the spectral window, as before.  On a slit-scan
    # spectrograph this fraction is what decides which emission lines a
    # pinhole contaminates.
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

        # The pinhole lists are paired, and both pipelines zip them together.
        # zip stops at the shortest, so a mismatch would drop pinholes from
        # the run without saying anything. main() runs the same check on
        # configuration files, but Simulation is also constructed directly.
        self.pinhole_positions, self.pinhole_positions_spectral = check_pinhole_lists(
            self.pinhole_sizes, self.pinhole_positions,
            self.pinhole_positions_spectral)
