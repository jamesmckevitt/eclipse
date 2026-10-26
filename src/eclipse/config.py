"""
Configuration classes for instruments, detectors, and simulation parameters.
"""

from __future__ import annotations
import dataclasses
import functools
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import numpy as np
import astropy.units as u
import scipy.interpolate
from . import eis_calibration
from .utils import _warn_outside, angle_to_distance
from importlib.resources import files


# ------------------------------------------------------------------
#  Detector materials and shared properties
# ------------------------------------------------------------------

# Properties of each detector material. A detector's `fano_factor`,
# `pair_energy_300k` and `pair_energy_slope` default to its material's.
# The energy to free one electron-hole pair in silicon is 3.71 eV at 300 K
# and rises by 0.0006 eV per kelvin of cooling.
DETECTOR_MATERIALS = {
    "silicon": {
        "fano_factor": 0.115,
        "pair_energy_300k": 3.71 * u.eV,
        "pair_energy_slope": -0.0006 * u.eV / u.K,
    }
}

# Wavelength in Angstrom against position along the dispersion, in mm from the
# SW focal plane centre (the middle of the gap), positive toward longer
# wavelength. Fitted to the RSC-2022021C positions of the two detector edges,
# the focal plane centre, the chief ray and the dispersion there, and to the
# ray-traced line positions on its SW footprint figure. A linear scale is not
# good enough: the design runs from 17.0 mA per row at the short-wavelength
# end to 16.8 at the long end, so the nominal 16.9 misplaces a line by up to
# about seven rows.
DISPERSION_COEFFICIENTS = (198.885944, 1.2516577, -1.361114e-4)

# A spectral line is an image of the slit, and the optics do not put it on
# exactly one row: it drifts along the dispersion from one end of the slit to
# the other. These are the row offsets at 140 arcsec from the field centre,
# linear term then quadratic, measured from the ray-traced line positions on
# the SW footprint figure of RSC-2022021C. All eleven lines drawn there agree
# on them to a few tenths of a row. Set them as `Detector_SWC.slit_image_tilt`
# to model the drift; by default a line lands on one row along the whole slit.
MEASURED_SLIT_IMAGE_TILT = (0.991, 0.605)

# The field angle those offsets are quoted at.
TILT_REFERENCE_ANGLE = 140.0 * u.arcsec


def _material_value(det, name: str):
    """Detector setting `name`, or its material's when the setting is None or missing."""
    value = getattr(det, name, None)
    material = getattr(det, "material", "silicon")
    return DETECTOR_MATERIALS[material][name] if value is None else value


def _pair_energy(det) -> u.Quantity:
    """Energy to free one electron-hole pair at the detector's CCD temperature."""
    temp_kelvin = det.ccd_temperature.to_value(u.K, equivalencies=u.temperature())
    at_300k = _material_value(det, "pair_energy_300k").to_value(u.eV)
    slope = _material_value(det, "pair_energy_slope").to_value(u.eV / u.K)
    return (at_300k + slope * (temp_kelvin - 300.0)) * u.eV


class _RenamedSetting:
    """Old name of a setting, which reads and sets the new one with a FutureWarning."""

    def __init__(self, old: str, new: str):
        self.old, self.new = old, new

    def warn(self, section: str = "") -> None:
        """Warn that the old name is used; `section`, such as "detector.", prefixes it."""
        _warn_outside(f"{section}{self.old} is renamed {self.new}; the old name will be "
                      f"removed in a future release.")

    def __get__(self, obj, owner=None):
        self.warn()
        return getattr(owner if obj is None else obj, self.new)

    def __set__(self, obj, value):
        self.warn()
        setattr(obj, self.new, value)


def _old_names_used(cls, given, section: str = "", replacing: bool = False,
                    error=TypeError) -> dict:
    """
    Old names of `cls`'s settings among the keys of `given`, mapped to their new names.

    Each warns with a FutureWarning. Both names of one setting are refused
    with `error`, unless `replacing`, when the old name's value is meant.
    """
    used = {}
    for old, new in cls._OLD_NAMES.items():
        if old in given:
            if new in given and not replacing:
                raise error(f"Give {section}{new} only, not {section}{old} as well.")
            _RenamedSetting(old, new).warn(section)
            used[old] = new
    return used


def _keep_old_names(cls):
    """
    Accept the names in `cls._OLD_NAMES` for their new ones, with a FutureWarning.

    They work as constructor arguments and attributes, and an object
    pickled with an old name reads back with the new one.
    `dataclasses.replace` with an old name passes both names, the new one
    with the object's current value, so there the old name's value is used.
    """
    init = cls.__init__

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        replacing = sys._getframe(1).f_globals.get("__name__") == "dataclasses"
        for old, new in _old_names_used(cls, kwargs, replacing=replacing).items():
            kwargs.pop(new, None)
            kwargs[new] = kwargs.pop(old)
        init(self, *args, **kwargs)

    def __setstate__(self, state):
        state = dict(state)
        for old, new in cls._OLD_NAMES.items():
            if old in state:
                state.setdefault(new, state.pop(old))
        self.__dict__.update(state)

    cls.__init__ = __init__
    cls.__setstate__ = __setstate__
    for old, new in cls._OLD_NAMES.items():
        setattr(cls, old, _RenamedSetting(old, new))
    return cls


def _check_detector_material(det, section: str) -> None:
    """Check the settings that default to the material's, and fill those left as None."""
    fano = det.fano_factor
    if fano is not None and (isinstance(fano, bool)
                             or not isinstance(fano, (int, float, np.integer, np.floating))
                             or not np.isfinite(fano) or fano < 0):
        raise ValueError(f"{section}.fano_factor must be a number, zero or more, or left out "
                         f"for the material's; got {fano!r}.")
    for name, unit, example in (("pair_energy_300k", u.eV, "3.71 eV"),
                                ("pair_energy_slope", u.eV / u.K, "-0.0006 eV / K")):
        value = getattr(det, name)
        if value is not None and not (isinstance(value, u.Quantity) and value.isscalar
                                      and value.unit.is_equivalent(unit)
                                      and np.isfinite(value.value)):
            raise ValueError(f"{section}.{name} must be in a unit of {unit.physical_type}, such "
                             f"as {example}, or left out for the material's; got {value!r}.")
    energy = _pair_energy(det)
    if not energy.value > 0:
        raise ValueError(f"{section}.pair_energy_300k and pair_energy_slope give {energy} per "
                         f"electron-hole pair at {det.ccd_temperature}; it must be above zero.")
    # Filled in, so a run records the values it used.
    for name in ("fano_factor", "pair_energy_300k", "pair_energy_slope"):
        setattr(det, name, _material_value(det, name))


def _numbers(value, count, section: str, name: str, example: str) -> tuple:
    """
    `value` as a tuple of finite floats.

    Refused unless `count`, a test of how many there are, is true of them.
    """
    def number(item):
        if isinstance(item, u.Quantity) and item.unit == u.dimensionless_unscaled:
            item = item.value
        if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)):
            raise TypeError
        return float(item)

    try:
        numbers = tuple(number(item) for item in value)
    except TypeError:
        numbers = None
    if numbers is None or not count(len(numbers)) or not np.all(np.isfinite(numbers)):
        raise ValueError(f"{section}.{name} must be {example}; got {value!r}.")
    return numbers


# The SW focal plane settings, which a detector and a focal plane both hold.
_FOCAL_PLANE_SETTINGS = ("dispersion", "wavelength_offset", "lit_band", "slit_image_tilt")


def _same_setting(first, second) -> bool:
    """Whether two values of a focal plane setting are the same, to a part in 1e12."""
    first, second = u.Quantity(first), u.Quantity(second)
    if first.shape != second.shape or not second.unit.is_equivalent(first.unit):
        return False
    return bool(np.allclose(first.value, second.to_value(first.unit), rtol=1e-12, atol=0.0))


def _one_length(value, zero: bool = False):
    """
    `value` as one length, read with `u.Quantity`, or None if it is not one.

    With `zero`, a plain zero is taken as 0 Angstrom, as it adds to a wavelength.
    """
    if isinstance(value, str) and not value.strip():
        return None
    try:
        items = ([] if isinstance(value, (str, u.Quantity))
                 else np.ravel(np.asarray(value, dtype=object)))
        if any(isinstance(item, (bool, np.bool_)) for item in items):
            return None
        value = u.Quantity(value)
    except (TypeError, ValueError):
        return None
    if value.size != 1:
        return None
    value = value.reshape(())
    if zero and value.unit == u.dimensionless_unscaled and value.value == 0:
        value = value.value * u.Angstrom
    if not (value.unit.is_equivalent(u.AA) and np.isfinite(value.value)):
        return None
    return value


def _check_focal_plane(det, section: str = "detector") -> None:
    """Check the SW focal plane settings of a detector or focal plane; store the lists as tuples."""
    offset = _one_length(det.wavelength_offset, zero=True)
    if offset is None:
        raise ValueError(f"{section}.wavelength_offset must be one wavelength, such as "
                         f"0.1 Angstrom; got {det.wavelength_offset!r}.")
    det.wavelength_offset = offset
    det.dispersion = _numbers(det.dispersion, lambda n: n >= 2, section, "dispersion",
                              "two or more coefficients, lowest order first, such as "
                              "[198.9, 1.25]")
    det.slit_image_tilt = _numbers(det.slit_image_tilt, lambda n: n == 2, section,
                                   "slit_image_tilt", "two numbers of rows, such as [0.991, 0.605]")
    band = det.lit_band
    try:
        limits = None if isinstance(band, str) else tuple(_one_length(item) for item in band)
    except TypeError:
        limits = None
    if limits is None or len(limits) != 2 or any(limit is None for limit in limits):
        raise ValueError(f"{section}.lit_band must be two wavelengths, such as [170.0 Angstrom, "
                         f"212.3 Angstrom]; got {band!r}.")
    det.lit_band = tuple(sorted(limits, key=lambda limit: limit.to_value(u.AA)))


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


def _same_widths(given, widths) -> bool:
    try:
        return len(given) == len(widths) and all(
            u.Quantity(a).unit.is_equivalent(u.pix) and u.Quantity(a).to_value(u.pix) == b.value
            for a, b in zip(given, widths))
    except (TypeError, ValueError):
        return False


def _check_resolution_budget(tel, section: str = "telescope") -> None:
    """Check `resolution_budget` is one the budget has, and `pointing_jitter` None or an angle."""
    from .resolution_budget import BUDGETS

    if not isinstance(tel.resolution_budget, str) or tel.resolution_budget not in BUDGETS:
        raise ValueError(f"{section}.resolution_budget must be one of {', '.join(BUDGETS)}, "
                         f"got {tel.resolution_budget!r}.")
    jitter = tel.pointing_jitter
    if jitter is not None and not (isinstance(jitter, u.Quantity) and jitter.isscalar
                                   and jitter.unit.is_equivalent(u.arcsec)
                                   and np.isfinite(jitter.value) and jitter.value >= 0):
        raise ValueError(f"{section}.pointing_jitter must be a FWHM in an angle, zero or more, "
                         f"such as 0.2 arcsec, or left out for the budget's; got {jitter!r}.")


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
                    non_negative=("read_noise_rms", "dark_current_293k"),
                    fractions=("qe_euv", "qe_vis"))
    if det.ccd_temperature.to_value(u.K, equivalencies=u.temperature()) <= 0:
        raise ValueError(f"{section}.ccd_temperature must be above absolute zero, got "
                         f"{det.ccd_temperature}; -60 K was perhaps meant as -60 C.")
    if det.material not in DETECTOR_MATERIALS:
        raise ValueError(f"{section}.material must be one of {sorted(DETECTOR_MATERIALS)}, got "
                         f"{det.material!r}.")
    _check_detector_material(det, section)


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


# Each instrument setting gives where its default comes from, as `source` in
# its field's metadata, and a quantity also the `unit` to show it in.
# tools/update_instrument_sources.py writes them to docs/instrument-sources.md.
# These sources are shared by several settings.
_FILTER_THICKNESS_SOURCE = ("Not recorded. SOLC-EUVST-MSSL-RP-0007 v1.2, Table 6-2, gives a "
                            "1500 Angstrom aluminium foil")
_HENKE = "Henke filter transmission calculator (henke.lbl.gov)"
_SENSITIVITY_BUDGET = "RSC-2022035A, Sensitivity Budget (13 Dec 2024)"
_SLIT_REQUIREMENTS = "TASI-SCE-SSS-0001 issue 1, R-SA-SYS-REQ-1160 and 1170"
_SW_CCD_SPEC = "SOLC-EUVST-MSSL-SP-0001 v1.4"
_SW_QE_SOURCE = ("Not recorded. SOLC-EUVST-MSSL-RP-0007 v1.2, Table 6-1, expects an EUV QE of "
                 "0.75; SOLC-EUVST-MSSL-SP-0001 v1.4 CCD-4.5.3 requires more than 0.71")
_SCAN_SOURCE = ("SOLC-EUVST-MSSL-RS-0002 v2.0 for the range. The read-out times of "
                "SOLC-EUVST-MSSL-RP-0007 v1.2, Table 5-11, follow from 50 prescan and 16 "
                "overscan")
_CTE_SOURCE = ("Not recorded. SOLC-EUVST-MSSL-SP-0001 v1.4 CCD-4.5.1 requires more than "
               "0.999993; SOLC-EUVST-MSSL-RP-0007 v1.2, Table 6-1, gives 0.99999 as typical")
_NOT_RECORDED = "Not recorded"


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
    visible_decade_thickness : u.Quantity
        Thickness of aluminium that cuts visible light by a factor of 10.
        Default 170 angstrom. Keyword only.
    """
    al_thickness: u.Quantity = field(
        default=1485 * u.angstrom,
        metadata={"source": _FILTER_THICKNESS_SOURCE, "unit": "Angstrom"})
    oxide_thickness: u.Quantity = field(
        default=95 * u.angstrom,
        metadata={"source": _FILTER_THICKNESS_SOURCE, "unit": "Angstrom"})
    c_thickness: u.Quantity = field(
        default=0 * u.angstrom,
        metadata={"source": _FILTER_THICKNESS_SOURCE, "unit": "Angstrom"})
    mesh_throughput: float = field(
        default=0.8,
        metadata={"source": "SOLC-EUVST-MSSL-RP-0007 v1.2, Table 6-2: 80% transmission, to be "
                            "confirmed. Its effective area in Sec. 7.1.3 uses 0.81"})
    al_table: Path = field(default_factory=lambda: files('eclipse') / 'data' / 'throughput' / 'throughput_aluminium_1000_angstrom.dat',
                           metadata={"source": _HENKE})
    oxide_table: Path = field(default_factory=lambda: files('eclipse') / 'data' / 'throughput' / 'throughput_aluminium_oxide_1000_angstrom.dat',
                              metadata={"source": _HENKE})
    c_table: Path = field(default_factory=lambda: files('eclipse') / 'data' / 'throughput' / 'throughput_carbon_1000_angstrom.dat',
                          metadata={"source": _HENKE})
    table_thickness: u.Quantity = field(default=1000 * u.angstrom,
                                        metadata={"source": _HENKE, "unit": "Angstrom"})
    visible_decade_thickness: u.Quantity = field(
        default=170.0 * u.angstrom, kw_only=True,
        metadata={"source": _NOT_RECORDED, "unit": "Angstrom"})

    def __post_init__(self):
        _check_settings(self, "filter", positive=("table_thickness", "visible_decade_thickness"),
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

        It falls by a factor of 10 per `visible_decade_thickness` of
        aluminium.
        """
        thickness_aa = self.al_thickness.to(u.angstrom).value
        layers = thickness_aa / self.visible_decade_thickness.to_value(u.angstrom)
        return 10.0 ** (-layers) * self.mesh_throughput


# -----------------------------------------------------------------------------
# Configuration objects
# -----------------------------------------------------------------------------
@_keep_old_names
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
    dark_current_293k : u.Quantity
        Dark current at 293 K, scaled to `ccd_temperature` by the
        non-inverted mode law of :func:`calculate_dark_current`. Default
        20000 electron per pixel per second. `_dark_current_293k` is an old
        name for it, which still works with a FutureWarning.
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
        read-out model, :mod:`eclipse.readout`, takes either. Keyword
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
    cte_parallel, cte_serial : float
        Charge transfer efficiency along the columns and along the serial
        register: the fraction of a packet's charge one transfer moves, the
        rest being left to the next packet. Default 1, a perfect transfer.
        SOLC-EUVST-MSSL-SP-0001 v1.4 CCD-4.5.1 requires more than 0.999993
        at -60 Celsius at the start of the mission, and
        SOLC-EUVST-MSSL-RP-0007 v1.2 Table 6.1 gives 0.99999 as typical for
        both.
    fano_factor : float or None
        Fano factor, which sets the spread in electrons per photon. Default
        None, for the material's in :data:`DETECTOR_MATERIALS` (0.115 for
        silicon).
    pair_energy_300k, pair_energy_slope : u.Quantity or None
        Energy to free one electron-hole pair at 300 K, and its change per
        kelvin; it sets the electrons each photon frees. Default None, for
        the material's in :data:`DETECTOR_MATERIALS` (3.71 eV and -0.0006 eV
        per kelvin for silicon).
    dispersion : tuple of float
        Wavelength in Angstrom against position along the dispersion in mm
        from the focal plane centre: polynomial coefficients, lowest order
        first. Default :data:`DISPERSION_COEFFICIENTS`.
    wavelength_offset : u.Quantity
        Added to every row's wavelength, for a wavelength calibration of the
        camera as built or in flight. Default 0 Angstrom.
    lit_band : tuple of u.Quantity
        Wavelengths between which light reaches the CCDs; outside them a
        baffle blocks the beam. Default 170.0 and 212.3 Angstrom.
    slit_image_tilt : tuple of float
        Drift of a line along the dispersion, in rows, between the field
        centre and 140 arcsec from it: a linear and a quadratic term.
        Default (0, 0), a line on one row along the whole slit.
        :data:`MEASURED_SLIT_IMAGE_TILT` is measured from the ray-traced
        spots in the SW footprint figure of RSC-2022021C.

    The last four describe the SW focal plane, which
    :class:`eclipse.readout.FocalPlane_SWC` models. An instrument run
    takes its wavelengths from `wvl_res`, not from the focal plane. It uses
    the focal plane only to find the row recording each wavelength, for the
    dark current, so it refuses values other than the defaults, which would
    make the two disagree.
    """
    _OLD_NAMES = {"_dark_current_293k": "dark_current_293k"}

    ccd_temperature: u.Quantity = field(
        default=-60 * u.deg_C,
        metadata={"source": "Not recorded. SOLC-EUVST-MSSL-RP-0007 v1.2, Sec. 5.5.2.1, gives a "
                            "goal of -50 to -60 Celsius", "unit": "Celsius"})
    qe_vis: float = field(default=1.0, metadata={"source": _SW_QE_SOURCE})
    qe_euv: float = field(default=0.76, metadata={"source": _SW_QE_SOURCE})
    read_noise_rms: u.Quantity = field(
        default=10.0 * u.electron / u.pixel,
        metadata={"source": "Not recorded. SOLC-EUVST-MSSL-SP-0001 v1.4 CCD-4.7.1 requires 6 "
                            "electrons rms; SOLC-EUVST-MSSL-RP-0007 v1.2, Table 6-1, predicts "
                            "11.0 electrons at most", "unit": "electrons"})
    dark_current: u.Quantity = field(init=False)
    dark_current_293k: u.Quantity = field(
        default=20000.0 * u.electron / (u.pixel * u.s),
        metadata={"source": "SOLC-EUVST-MSSL-SP-0001 v1.4 CCD-4.7.3 (requirement)",
                  "unit": "electrons per pixel per second"})
    gain_e_per_dn: u.Quantity = field(
        default=2.78 * u.electron / u.DN,
        metadata={"source": "MSSL engineering model test results; SOLC-EUVST-MSSL-RP-0012 v2.0, "
                            "Table 5-5, gives the same design value",
                  "unit": "electrons per DN"})
    max_dn: u.Quantity = field(
        default=65535 * u.DN / u.pixel,
        metadata={"source": "SOLC-EUVST-MSSL-RP-0012 v2.0, Table 5-5: a 16-bit converter",
                  "unit": "DN"})
    # Peak charge storage of the CCD42-40 in non-inverted mode, where
    # resolution begins to degrade: 150 ke- typical, 80 ke- minimum (Teledyne
    # e2v CCD42-40 BSI datasheet, 1B300000-A1A version 1, January 2024). It is
    # below the 182 ke- the FEE accepts, so a pixel fills before the digitiser
    # does. Nothing is clipped or spilled at it; it only shows which pixels a
    # frame would saturate. Keyword-only, so the fields after it keep their
    # places in the constructor.
    full_well: u.Quantity = field(
        default=150000 * u.electron / u.pixel, kw_only=True,
        metadata={"source": "Teledyne e2v CCD42-40 BSI datasheet, 1B300000-A1A version 1 "
                            "(January 2024); SOLC-EUVST-MSSL-SP-0001 v1.4 CCD-4.6.1 requires "
                            "more than 100000, typically 150000. SOLC-EUVST-MSSL-RP-0007 v1.2, "
                            "Table 6-1, gives 75000 typical for the image area",
                  "unit": "electrons"})
    pix_size: u.Quantity = field(default=(13.5 * u.um).cgs / u.pixel,
                                 metadata={"source": _SW_CCD_SPEC, "unit": "micron"})
    wvl_res: u.Quantity = field(
        default=(16.9 * u.mAA).cgs / u.pixel,
        metadata={"source": "RSC-2022021C: 1.253 mA per micron at the chief ray",
                  "unit": "mA per pixel"})
    plate_scale_angle: u.Quantity = field(
        default=0.159 * u.arcsec / u.pixel,
        metadata={"source": "RSC-2022021C: 0.0118 arcsec per micron", "unit": "arcsec per pixel"})
    material: str = "silicon"
    filter_distance: u.Quantity = field(default=250 * u.mm,
                                        metadata={"source": _NOT_RECORDED, "unit": "mm"})
    # Settings of the camera's CCDs and their read-out, which
    # eclipse.readout models. An instrument run uses only their timing, for
    # the dark current collected during the clear and the read-out.
    shutter: bool = field(
        default=True, kw_only=True,
        metadata={"source": "SOLC-EUVST-MSSL-RP-0007 v1.2, Sec. 6.1.2: the SW CCD uses a "
                            "mechanical shutter"})
    n_rows: int = field(default=2048, kw_only=True, metadata={"source": _SW_CCD_SPEC})
    n_columns: int = field(default=2048, kw_only=True, metadata={"source": _SW_CCD_SPEC})
    ccd_gap: u.Quantity = field(
        default=1.0 * u.mm, kw_only=True,
        metadata={"source": "SOLC-EUVST-MSSL-RP-0007 v1.2, Sec. 7.1; RSC-2022021C", "unit": "mm"})
    row_transfer_time: u.Quantity = field(
        default=15.0 * u.us, kw_only=True,
        metadata={"source": "SWC modes table; it gives the read-out times of "
                            "SOLC-EUVST-MSSL-RP-0007 v1.2, Table 5-11", "unit": "us"})
    pixel_period: u.Quantity = field(
        default=500.0 * u.ns, kw_only=True,
        metadata={"source": "SOLC-EUVST-MSSL-RS-0002 v2.0", "unit": "ns"})
    serial_prescan: int = field(
        default=50, kw_only=True,
        metadata={"source": _SCAN_SOURCE})
    serial_overscan: int = field(
        default=20, kw_only=True,
        metadata={"source": _SCAN_SOURCE})
    parallel_overscan_rows: int = field(default=20, kw_only=True,
                                        metadata={"source": _NOT_RECORDED})
    cte_parallel: float = field(default=1.0, kw_only=True,
                                metadata={"source": _CTE_SOURCE})
    cte_serial: float = field(default=1.0, kw_only=True,
                              metadata={"source": _CTE_SOURCE})
    # Properties of the material, None for those in DETECTOR_MATERIALS.
    fano_factor: float | None = field(
        default=None, kw_only=True,
        metadata={"source": _NOT_RECORDED})
    pair_energy_300k: u.Quantity | None = field(
        default=None, kw_only=True,
        metadata={"source": _NOT_RECORDED, "unit": "eV"})
    pair_energy_slope: u.Quantity | None = field(
        default=None, kw_only=True,
        metadata={"source": _NOT_RECORDED, "unit": "eV per K"})
    # The SW focal plane, which eclipse.readout models.
    dispersion: tuple = field(
        default=DISPERSION_COEFFICIENTS, kw_only=True,
        metadata={"source": "Fitted to RSC-2022021C: the detector edges, the focal plane "
                            "centre, the chief ray and the ray-traced line positions"})
    wavelength_offset: u.Quantity = field(
        default=0.0 * u.Angstrom, kw_only=True,
        metadata={"source": "SOLC-EUVST-MSSL-RP-0007 v1.2, page 45", "unit": "Angstrom"})
    lit_band: tuple = field(
        default=(170.0 * u.Angstrom, 212.3 * u.Angstrom), kw_only=True,
        metadata={"source": "The required range, RSC-2022021C REQ-103 and "
                            "SOLC-EUVST-MSSL-RP-0007 v1.2, Table 7-2. No document gives the "
                            "baffle's edges, so where the band stops to within a row or two is "
                            "a modelling choice", "unit": "Angstrom"})
    slit_image_tilt: tuple = field(
        default=(0.0, 0.0), kw_only=True,
        metadata={"source": "No document gives it. Measured from the ray-traced spots in the SW "
                            "footprint figure of RSC-2022021C, where a row is about a quarter "
                            "of a pixel of the figure"})

    def __post_init__(self):
        _check_detector(self)
        _check_focal_plane(self)
        for name in ("cte_parallel", "cte_serial"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float, np.floating))
                    or not 0 < value <= 1):
                raise ValueError(f"detector.{name} is the fraction of a packet's charge one "
                                 f"transfer moves, above 0 and at most 1, got {value!r}.")
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
                                                        self.dark_current_293k)

    @property
    def si_fano(self) -> float:
        """Fano factor: `fano_factor`, or the material's."""
        return _material_value(self, "fano_factor")

    @property
    def pair_energy(self) -> u.Quantity:
        """Energy to free one electron-hole pair at `ccd_temperature`."""
        return _pair_energy(self)

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
        rate = Detector_SWC.dark_current_293k if dark_current_293k is None else dark_current_293k
        return calculate_dark_current(temp, rate, ccd_type="NIMO")

    @property
    def plate_scale_length(self) -> u.Quantity:
        """Length on the Sun, seen from 1 AU, of one pixel along the slit."""
        return angle_to_distance(self.plate_scale_angle * 1*u.pix) / u.pixel


@_keep_old_names
@dataclass
class Detector_EIS:
    """
    Hinode/EIS detector.

    The pixel, noise and gain settings of `Detector_SWC`, with EIS values:
    quantum efficiency 0.64 for EUV and 0.65 for visible light, read noise 5
    electron per pixel, 6.3 electron per DN, 22.3 mA per pixel and 1 arcsec
    per pixel along the slit. Its `dark_current_293k` is 250 electron per
    pixel per second, scaled by the advanced inverted mode law of
    :func:`calculate_dark_current`. `fano_factor`, `pair_energy_300k` and
    `pair_energy_slope` are as for `Detector_SWC`. It has none of SWC's
    full-well, filter, shutter, focal plane or CCD read-out settings.
    """
    _OLD_NAMES = {"_dark_current_293k": "dark_current_293k"}

    ccd_temperature: u.Quantity = field(default=-60 * u.deg_C,
                                        metadata={"source": _NOT_RECORDED, "unit": "Celsius"})
    qe_euv: float = field(default=0.64, metadata={"source": "EIS Software Note 2"})
    qe_vis: float = field(default=0.65, metadata={"source": "MSSL engineering test report"})
    read_noise_rms: u.Quantity = field(default=5.0 * u.electron / u.pixel,
                                       metadata={"source": _NOT_RECORDED, "unit": "electrons"})
    dark_current: u.Quantity = field(init=False)
    dark_current_293k: u.Quantity = field(
        default=250.0 * u.electron / (u.pixel * u.s),
        metadata={"source": _NOT_RECORDED, "unit": "electrons per pixel per second"})
    gain_e_per_dn: u.Quantity = field(
        default=6.3 * u.electron / u.DN,
        metadata={"source": _NOT_RECORDED, "unit": "electrons per DN"})
    max_dn: u.Quantity = field(default=65535 * u.DN / u.pixel,
                               metadata={"source": _NOT_RECORDED, "unit": "DN"})
    pix_size: u.Quantity = field(default=(13.5 * u.um).cgs / u.pixel,
                                 metadata={"source": _NOT_RECORDED, "unit": "micron"})
    wvl_res: u.Quantity = field(default=(22.3 * u.mAA).cgs / u.pixel,
                                metadata={"source": _NOT_RECORDED, "unit": "mA per pixel"})
    plate_scale_angle: u.Quantity = field(
        default=1 * u.arcsec / u.pixel,
        metadata={"source": _NOT_RECORDED, "unit": "arcsec per pixel"})
    material: str = "silicon"
    # Properties of the material, None for those in DETECTOR_MATERIALS.
    fano_factor: float | None = field(default=None, kw_only=True)
    pair_energy_300k: u.Quantity | None = field(default=None, kw_only=True)
    pair_energy_slope: u.Quantity | None = field(default=None, kw_only=True)

    @property
    def si_fano(self) -> float:
        """Fano factor: `fano_factor`, or the material's."""
        return _material_value(self, "fano_factor")

    @property
    def pair_energy(self) -> u.Quantity:
        """Energy to free one electron-hole pair at `ccd_temperature`."""
        return _pair_energy(self)

    def __post_init__(self):
        _check_detector(self)
        self.dark_current = self.calculate_dark_current(self.ccd_temperature,
                                                        self.dark_current_293k)

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
        rate = Detector_EIS.dark_current_293k if dark_current_293k is None else dark_current_293k
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
    psf_params : list of u.Quantity or None
        FWHMs in pixels of the optical design's PSF, along the slit and in
        wavelength. Default None, which takes them from `field_of_view`.
    psf_slit_width : u.Quantity or None
        Slit the FWHM in wavelength was measured with. The FWHM for other
        slits is derived from it. Default 0.2 arcsec.
    field_of_view : str or None
        Field the optical design's PSF is the worst case over, in arcsec:
        `"280x280"`, `"100x100"` or `"70x70"`. It sets `psf_params` when
        that is not given. Default None, which is `"280x280"`, the full
        scan. Keyword only.
    resolution_budget : str
        Blur added in quadrature to the optical design's PSF, from the
        resolution budget RSC-2022033B: `"current_estimate"`, `"allocation"`
        or `"none"`. Default `"current_estimate"`. Keyword only.
    pointing_jitter : u.Quantity or None
        FWHM of the spacecraft's pointing jitter, in place of the budget's
        0.110 arcsec. It blurs along and across the slit, and the grating's
        micro-vibration is taken as 2/3 of it, as in the budget. It is added
        even with `resolution_budget="none"`. Default None, the budget's
        own. Keyword only.
    psf_across_slit : u.Quantity or None
        FWHM of the telescope optics' blur across the slit, which brings in
        light from either side of it. The budget's blur before the slit is
        added to it in quadrature. Default None, for none. Keyword only.
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
    D_ap: u.Quantity = field(
        default=0.28 * u.m,
        metadata={"source": "RSC-2022021C, Optical Design Summary: entrance pupil 280 mm",
                  "unit": "m"})
    microroughness_sigma: u.Quantity = field(default=0.3 * u.nm,
                                             metadata={"source": _NOT_RECORDED, "unit": "nm"})
    filter: AluminiumFilter = field(default_factory=AluminiumFilter)
    psf_type: str = "gaussian"
    # [spatial_fwhm, spectral_fwhm] in pixels, or None to take them from
    # field_of_view (FIELDS_OF_VIEW).
    psf_params: list | None = field(default=None,
                                    metadata={"source": "RSC-2022021C"})
    # Slit width the spectral FWHM in psf_params is for. RSC-2022021C quotes
    # the spectral resolution for the 0.2 arcsec slit as the optics FWHM after
    # the slit (0.352 arcsec at 212.3 A) added in quadrature to the slit width
    # (giving 0.405 arcsec, 43.00 mA). The spectral PSF for any other slit is
    # derived from it; see radiometric.spectral_psf_fwhm.
    psf_slit_width: u.Quantity = field(
        default=0.2 * u.arcsec,
        metadata={"source": "RSC-2022021C, Appendix A.2", "unit": "arcsec"})
    # Telescope optics' blur across the slit: the FWHM of their image at the
    # slit, which sets how much light from beside the slit falls into it. None
    # leaves it out; the resolution budget's blur before the slit still
    # applies (radiometric.across_slit_fwhm). This is not psf_params[0], the
    # blur along the slit measured at the detector, which includes the
    # spectrograph after the slit.
    psf_across_slit: u.Quantity | None = field(
        default=None, kw_only=True,
        metadata={"source": _NOT_RECORDED, "unit": "arcsec"})
    field_of_view: str | None = field(default=None, kw_only=True,
                                      metadata={"source": "RSC-2022021C"})
    resolution_budget: str = field(
        default="current_estimate", kw_only=True,
        metadata={"source": "RSC-2022033B rev. B (28 Oct 2025), Tables 3-1 and 4-1"})
    pointing_jitter: u.Quantity | None = field(
        default=None, kw_only=True,
        metadata={"source": "RSC-2022033B rev. B, Table 3-1 row 20", "unit": "arcsec"})
    # The slits are 4.07 mm long, 300 arcsec at the slit's 13.57 um per arcsec,
    # and each has two fiducials, gaps 10.9 um (0.8 arcsec) long in the three
    # narrow slits and 21.8 um (1.6 arcsec) in the widest, centred -340 um
    # (-25 arcsec) and +950 um (+70 arcsec) from its centre along the slit
    # (TASI-SCE-SSS-0001 issue 1, R-SA-SYS-REQ-1160 and 1170). The documents
    # do not say which end of the slit is positive on the detector.
    slit_length: u.Quantity = field(default=300.0 * u.arcsec, kw_only=True,
                                    metadata={"source": _SLIT_REQUIREMENTS, "unit": "arcsec"})
    fiducial_centres: list = field(
        default_factory=lambda: [-25.0 * u.arcsec, 70.0 * u.arcsec], kw_only=True,
        metadata={"source": _SLIT_REQUIREMENTS, "unit": "arcsec"})
    fiducial_lengths: list = field(
        default_factory=lambda: [0.8 * u.arcsec, 0.8 * u.arcsec, 0.8 * u.arcsec, 1.6 * u.arcsec],
        kw_only=True,
        metadata={"source": _SLIT_REQUIREMENTS, "unit": "arcsec"})

    # Wavelength-dependent efficiency tables
    pm_table: Path = field(default_factory=lambda: files('eclipse') / 'data' / 'throughput' / 'primary_mirror_coating_reflectance.dat',
                           metadata={"source": _SENSITIVITY_BUDGET})
    grating_table: Path = field(default_factory=lambda: files('eclipse') / 'data' / 'throughput' / 'grating_reflection_efficiency.dat',
                                metadata={"source": _SENSITIVITY_BUDGET})

    # Worst-case FWHMs of the optical design's PSF in each field, in pixels
    # at 0.159 arcsec per pixel, from RSC-2022021C (Mar 2025). Along the slit,
    # 0.423, 0.195 and 0.161 arcsec, with the slit scanned across the field
    # (FOV2, FOV1 and FOV0). In wavelength, 43.00 mA with the 0.2 arcsec slit,
    # given for the 70 x 70 arcsec field only.
    FIELDS_OF_VIEW = {"280x280": [2.66 * u.pixel, 2.54 * u.pixel],
                      "100x100": [1.23 * u.pixel, 2.54 * u.pixel],
                      "70x70": [1.01 * u.pixel, 2.54 * u.pixel]}

    def __post_init__(self):
        _check_settings(self, "telescope", positive=("D_ap", "psf_slit_width", "slit_length"),
                        non_negative=("microroughness_sigma",), optional=("psf_slit_width",))
        if self.field_of_view is not None and (not isinstance(self.field_of_view, str)
                                               or self.field_of_view not in self.FIELDS_OF_VIEW):
            raise ValueError(f"telescope.field_of_view must be one of "
                             f"{', '.join(self.FIELDS_OF_VIEW)}, got {self.field_of_view!r}.")
        if self.psf_params is None:
            self.field_of_view = self.field_of_view or "280x280"
            self.psf_params = [q.copy() for q in self.FIELDS_OF_VIEW[self.field_of_view]]
        elif (self.field_of_view is not None
              and not _same_widths(self.psf_params, self.FIELDS_OF_VIEW[self.field_of_view])):
            raise ValueError(f"telescope.field_of_view sets psf_params, so give one or the "
                             f"other; got {self.field_of_view!r} and {self.psf_params!r}.")
        _check_psf_params(self)
        _check_resolution_budget(self)
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
    psf_params: list = field(default_factory=lambda: [3.0 * u.pixel, 3.0 * u.pixel],  # [spatial_fwhm, spectral_fwhm] in pixels
                             metadata={"source": "Ugarte-Urra (2016), EIS Software Note 2",
                                       "unit": "pixels"})
    # The EIS PSF is not tied to a slit width, so its spectral FWHM is the
    # same for every slit. Setting this gives the slit psf_params was measured
    # with, and the spectral PSF then follows the slit as for SWC.
    psf_slit_width: u.Quantity | None = None
    # Telescope blur across the slit, the FWHM of its image at the slit, as
    # for Telescope_EUVST. None leaves it out.
    psf_across_slit: u.Quantity | None = field(default=None, kw_only=True)
    calibration: str = field(
        default="ground",
        metadata={"source": "Pre-flight MSSL calibration, April 2007 (Lang et al. 2006, Applied "
                            "Optics, 45, 8689); `dz2013`, `warren2014` and `dz2025` from Del "
                            "Zanna (2013), Warren, Ugarte-Urra & Landi (2014) and Del Zanna et "
                            "al. (2025)"})
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
    readout_dark_current : bool
        Whether each pixel also collects dark current while the CCD is
        cleared before the exposure and while it waits to be read out
        afterwards, as in a full-CCD frame (SWC only). Without it, dark
        current is collected over the exposure only. Default True. Keyword
        only.
    enable_pinholes : bool
        Whether to model pinholes in the filter (SWC only). Default False.
    pinhole_sizes : list of u.Quantity
        Pinhole diameters.
    pinhole_positions : list of float
        Position of each pinhole along the slit, as a fraction from 0 to 1.
    pinhole_positions_spectral : list of float
        Position of each pinhole along the spectral axis, as a fraction from
        0 to 1. Default empty, for the middle.
    pinhole_visible_wavelength : u.Quantity
        Wavelength of the visible light through the pinholes, which sets its
        diffraction. Default 600 nm. Keyword only.
    """
    expos: u.Quantity = 1.0 * u.s  # Single exposure time
    n_iter: int = 10
    slit_width: u.Quantity = field(
        default=0.2 * u.arcsec,
        metadata={"source": "RSC-2022021C", "unit": "arcsec"})
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
    # A run takes only the rows of its own wavelengths to be read. The clear
    # and the read-out then add 0.03 to 0.05 s for 32 rows with the default
    # timing.
    readout_dark_current: bool = field(default=True, kw_only=True)
    enable_pinholes: bool = False
    pinhole_sizes: List[u.Quantity] = field(default_factory=list)
    pinhole_positions: List[float] = field(default_factory=list)
    # Position along the spectral axis, as a fraction (0.0 to 1.0) of the
    # detector width, one per pinhole. Empty (the default) puts every pinhole
    # at the centre of the spectral window. On a slit-scan spectrograph this
    # fraction decides which emission lines a pinhole contaminates.
    pinhole_positions_spectral: List[float] = field(default_factory=list)
    pinhole_visible_wavelength: u.Quantity = field(
        default=600.0 * u.nm, kw_only=True, metadata={"source": _NOT_RECORDED, "unit": "nm"})

    @property
    def slit_scan_step(self) -> u.Quantity:
        return self.slit_width

    def __post_init__(self):
        _check_settings(self, "simulation",
                        positive=("expos", "slit_width", "pinhole_visible_wavelength"),
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
