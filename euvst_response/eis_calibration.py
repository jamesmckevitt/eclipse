"""
Hinode/EIS effective area, as a function of wavelength and observation date.

EIS is included in ECLIPSE as a reference instrument, so its throughput has to
be as honest as the EUVST one. The effective area is strongly
wavelength-dependent - roughly 0.33 cm^2 at Fe XII 195 against 0.013 cm^2 at
Ca XV 181.9, a factor of 25 across the short-wavelength channel alone - and it
has fallen substantially since launch, by more than an order of magnitude in
parts of the long-wavelength channel. Any study comparing photon statistics
between lines, or between epochs, depends on both.

Four calibrations are available, matching the ``calib=`` keywords of
``eis_get_fitdata.pro``:

``ground``
    Pre-flight MSSL calibration, April 2007. No time dependence.
    Port of ``eis_ea.pro``.
``dz2013``
    Del Zanna (2013). Port of ``eis_ltds.pro``. The long-wavelength channel
    carries a polynomial degradation, clamped after 2012-09-14 where the
    original fit ends; the short-wavelength channel is time-independent.
``warren2014``
    Warren, Ugarte-Urra & Landi (2014). Exponential decay per spline knot,
    from ``nrl_ea_coeff_v1.3.genx``.
``dz2025``
    Del Zanna et al. (2025), the current recommendation. Time- and
    wavelength-dependent areas fitted to 2007-2022 EIS spectra. Port of
    ``interpol_eis_ea.pro``.

All four return the effective area **including** the CCD quantum efficiency
(0.64), because that is the convention the tables are quoted in and the one
the EIS radiometric calibration formula uses. Callers that apply the quantum
efficiency separately must divide it back out; ``config.Telescope_EIS`` does
exactly that, since ECLIPSE draws detected photons binomially further down the
chain.

Only the effective area lives here. Converting measured DN back to physical
units is a data-analysis step with its own conventions (per spectral pixel,
per arcsec^2, exposure time excluded) and is deliberately out of scope for a
forward-modelling package.

References
----------
- Lang, J. et al. 2006, Applied Optics, 45, 8689
- Del Zanna, G. 2013, A&A, 555, A47
- Warren, H. P., Ugarte-Urra, I. & Landi, E. 2014, ApJS, 213, 11
- Del Zanna, G. et al. 2025, doi:10.48550/arXiv.2308.06609
"""

from __future__ import annotations

from datetime import date as _date, datetime as _datetime, timezone as _tz
from functools import lru_cache
from importlib.resources import as_file, files

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = [
    "CALIBRATIONS",
    "TIME_DEPENDENT_CALIBRATIONS",
    "SW_BAND",
    "LW_BAND",
    "band_of",
    "effective_area",
    "normalise_date",
]

# EIS observes two disjoint bands; there is no effective area between them.
# Edges as in eis_get_band (eis_ea_nrl.pro). The ground tables are sampled
# 165-212 and 245-292 Angstrom, so the last Angstrom of the short-wavelength
# band is extrapolated, as it is in eis_ea.pro.
SW_BAND = (165.0, 213.0)
LW_BAND = (245.0, 292.0)

CALIBRATIONS = ("ground", "dz2013", "warren2014", "dz2025")

# 'ground' is the pre-flight calibration and has no epoch; the others do.
TIME_DEPENDENT_CALIBRATIONS = ("dz2013", "warren2014", "dz2025")

_DATA_DIR = files("euvst_response") / "data" / "eis_response"
_DZ2025_SAV = _DATA_DIR / "fit_eis_ea_2023-05-04_smooth.sav"

# The tabulated areas include this CCD quantum efficiency (EIS SW Note 2).
# It is a property of how the tables are quoted, not a free parameter: change
# it and the areas no longer mean what their source says they mean. The
# detector QE that ECLIPSE applies is a separate, configurable quantity on
# Detector_EIS.
QE_IN_TABLES = 0.64


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def normalise_date(date) -> str:
    """Coerce a date to an ISO string.

    Accepts a string, a ``datetime.date``, a ``datetime.datetime`` or anything
    with an ``isot`` attribute (``astropy.time.Time``). YAML parses an unquoted
    ``2012-06-03`` into a ``datetime.date``, so this is not a theoretical case.
    """
    if isinstance(date, str):
        return date
    if hasattr(date, "isot"):            # astropy.time.Time
        return str(date.isot)
    if isinstance(date, (_datetime, _date)):
        return date.isoformat()
    raise TypeError(
        f"Cannot interpret {date!r} as an observation date. Give an ISO "
        f"string such as '2012-06-03', a datetime, or an astropy Time."
    )


def _parse_date(date_str: str) -> _datetime:
    """Parse an ISO date string to a UTC datetime.

    A string carrying an offset is *converted* to UTC rather than relabelled.
    ``replace(tzinfo=utc)`` would keep the wall clock and move the instant, so
    ``2012-06-03T00:00:00-05:00`` would enter the degradation calculation five
    hours early.
    """
    text = normalise_date(date_str).replace("Z", "+00:00")
    try:
        parsed = _datetime.fromisoformat(text)
    except ValueError:
        parsed = _datetime.fromisoformat(text.split("+")[0])
    if parsed.tzinfo is None:
        # Naive input is taken to be UTC, which is what EIS dates are.
        return parsed.replace(tzinfo=_tz.utc)
    return parsed.astimezone(_tz.utc)


def _date_to_year_fraction(date_str: str) -> float:
    """Convert an ISO date string to a decimal year."""
    dt = _parse_date(date_str)
    start = _datetime(dt.year, 1, 1, tzinfo=_tz.utc)
    end = _datetime(dt.year + 1, 1, 1, tzinfo=_tz.utc)
    return dt.year + (dt - start).total_seconds() / (end - start).total_seconds()


# ---------------------------------------------------------------------------
# Pre-flight ground calibration
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _ground_table(band: str) -> tuple[np.ndarray, np.ndarray]:
    """Wavelength (Angstrom) and effective area (cm^2) from the SSW tables."""
    fname = "EIS_EffArea_B.004" if band == "SW" else "EIS_EffArea_A.004"
    wave, area = [], []
    for line in (_DATA_DIR / fname).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        wave.append(float(parts[0]))
        area.append(float(parts[1]))
    return np.array(wave), np.array(area)


# ---------------------------------------------------------------------------
# Del Zanna (2013) - eis_ltds.pro
# ---------------------------------------------------------------------------

# Spline control points. The divisors are part of the published correction,
# not a normalisation applied here: they are how eis_ltds.pro scales the
# pre-flight curve onto the 2013 calibration, point by point.
_DZ2013_SW_WAVE = np.array([
    165, 171, 174.5, 177.2, 178.1, 180.4, 182.2,
    184.5, 185.2, 186.9, 188.3, 190,
    192.4, 192.8, 193.5, 194.7, 195.1,
    196.6, 197.4, 200, 201.1, 202.,
    202.7, 204.9, 208., 209.9, 211.3,
])
_DZ2013_SW_EA = np.array([
    0.000174973/1.5, 0.000255772/1.5, 0.00158207/1.5, 0.00476608/1.55,
    0.00705735/1.5, 0.0168637/1.45, 0.0316499/1.4,
    0.0647319/1.35, 0.0779082/1.35, 0.115240/1.4, 0.150199/1.45, 0.194897/1.25,
    0.255993/1.13, 0.264945/1.1, 0.279607/1.05, 0.298884/1.02, 0.302737*1.,
    0.301859/1.05, 0.287675/1.15, 0.174608*1.05, 0.119586/1.0, 0.0838537/1.,
    0.0635698/1., 0.0332376/1.0, 0.0189209/1., 0.0133581/1., 0.0105513/1.,
])
_DZ2013_LW_WAVE = np.array([
    245., 252., 255, 257., 259,
    263., 265., 268., 270.,
    272., 274., 277., 281., 286., 292,
])
_DZ2013_LW_EA = np.array([
    0.022673*0.8, 0.03908*0.75, 0.05065*0.78, 0.0588*0.8, 0.06738*0.85,
    0.0861*0.9, 0.09551*0.95, 0.106984*1.0, 0.110764*1.02,
    0.10944*1.03, 0.1026*1.03, 0.084775*0.9, 0.05718*0.87, 0.0333*0.85,
    0.01679*0.85,
]) / 1.1

# Long-wavelength degradation polynomial, in seconds since the reference time.
_DZ2013_LW_COEFF = np.array([1.0326230, -5.2495791e-09, 1.2055185e-17])

# 22-Sept-2006 21:36 UTC, and the 14-Sept-2012 end of the fitted range.
# datetime rather than IDL anytim2tai, for portability.
_DZ2013_REF_DT = _datetime(2006, 9, 22, 21, 36, 0, tzinfo=_tz.utc)
_DZ2013_LAST_DT = _datetime(2012, 9, 14, 0, 0, 0, tzinfo=_tz.utc)


def _dz2013_reference(date_str: str, band: str) -> tuple[np.ndarray, np.ndarray]:
    """DZ2013 areas on the ground-calibration wavelength grid."""
    grid_wave, _ = _ground_table(band)

    if band == "SW":
        area = np.interp(grid_wave, _DZ2013_SW_WAVE, _DZ2013_SW_EA)
        return grid_wave, area

    area = np.interp(grid_wave, _DZ2013_LW_WAVE, _DZ2013_LW_EA)

    elapsed = (_parse_date(date_str) - _DZ2013_REF_DT).total_seconds()
    elapsed = min(max(elapsed, 0.0),
                  (_DZ2013_LAST_DT - _DZ2013_REF_DT).total_seconds())
    return grid_wave, np.polyval(_DZ2013_LW_COEFF[::-1], elapsed) * area


# ---------------------------------------------------------------------------
# Warren, Ugarte-Urra & Landi (2014) - nrl_ea_coeff_v1.3.genx
# ---------------------------------------------------------------------------

_W14_T0 = _datetime(2006, 9, 22, 0, 0, 0, tzinfo=_tz.utc)   # Hinode launch

_W14_KNOTS = {
    "SW": np.array([165., 173., 181., 189., 197., 205., 213.]),
    "LW": np.array([245., 252.833, 260.667, 268.500, 276.333, 284.167, 292.]),
}
# Effective area at t0 (cm^2), fitted.
_W14_A0 = {
    "SW": np.array([0.000142335, 0.000894009, 0.0194364, 0.207179,
                    0.384759, 0.0502141, 0.0130786]),
    "LW": np.array([0.0253011, 0.0504537, 0.124469, 0.194087,
                    0.147056, 0.0674649, 0.0175320]),
}
# Decay time constants (years). The two negative short-wavelength values are
# in the published coefficient set: those knots brighten rather than decay.
_W14_TAU = {
    "SW": np.array([96.6543, 96.6547, 41.4198, 78.6531, 53.0495,
                    -39.2594, -15.6689]),
    "LW": np.array([11.2182, 13.9551, 10.4183, 7.54961, 8.58022,
                    8.01669, 14.8060]),
}


def _warren2014_reference(date_str: str, band: str) -> tuple[np.ndarray, np.ndarray]:
    """Warren (2014) areas at the seven spline knots for this band."""
    years = (_parse_date(date_str) - _W14_T0).total_seconds() / (86400.0 * 365.25)
    return _W14_KNOTS[band], _W14_A0[band] * np.exp(-years / _W14_TAU[band])


# ---------------------------------------------------------------------------
# Del Zanna et al. (2025) - interpol_eis_ea.pro
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _dz2025_table() -> dict:
    """Load the fitted effective-area save file."""
    import scipy.io

    # readsav needs a filesystem path, so the resource is materialised first.
    # For a normal install as_file just hands back the path unchanged.
    with as_file(_DZ2025_SAV) as path:
        fit_ea = scipy.io.readsav(str(path))["fit_ea"]
    return {
        "year_fr": fit_ea["YEAR_FR"][0].astype(np.float64),
        "SW_ea": fit_ea["SW_EA"][0].astype(np.float64),
        "SW_wave": fit_ea["SW_WAVE"][0].astype(np.float64),
        "LW_ea": fit_ea["LW_EA"][0].astype(np.float64),
        "LW_wave": fit_ea["LW_WAVE"][0].astype(np.float64),
    }


def _dz2025_reference(date_str: str, band: str) -> tuple[np.ndarray, np.ndarray]:
    """DZ2025 areas at the fitted reference wavelengths, for this date."""
    table = _dz2025_table()
    ref_wave = table[f"{band}_wave"]
    ref_area = table[f"{band}_ea"]
    year_fr = table["year_fr"]

    # The fit covers 2007-2022; outside it the endpoint value is held, as
    # interpol_eis_ea.pro does. Extrapolating a degradation curve would be
    # worse than saying so.
    year = np.clip(_date_to_year_fraction(date_str), year_fr[0], year_fr[-1])

    area = np.array([np.interp(year, year_fr, ref_area[i, :])
                     for i in range(len(ref_wave))])
    return ref_wave, area


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Each calibration is (reference-point builder, interpolation convention).
# 'log_spline' is a natural cubic spline through log(EA), which is what the
# IDL routines do for the ground and Warren curves; 'spline' and 'linear' are
# the conventions of interpol_eis_ea.pro and eis_ltds.pro respectively. The
# choice matters: on the ground tables, a linear interpolation differs from
# the log spline by up to 1.3 percent where the curve turns over sharply,
# around Fe XIII 203.8.
_METHODS = {
    "ground": (lambda date, band: _ground_table(band), "log_spline"),
    "dz2013": (_dz2013_reference, "linear"),
    "warren2014": (_warren2014_reference, "log_spline"),
    "dz2025": (_dz2025_reference, "spline"),
}


def _check_method(method: str) -> str:
    if method not in _METHODS:
        raise ValueError(
            f"Unknown EIS calibration {method!r}. Choose from: "
            f"{', '.join(CALIBRATIONS)}."
        )
    return method


@lru_cache(maxsize=256)
def _interpolator(method: str, date_str: str, band: str):
    """Build, once per calibration and epoch, a callable EA(wavelength_aa).

    Cached because ``radiometric.add_telescope_throughput`` evaluates the
    telescope response one wavelength sample at a time, in a Python loop, on
    every Monte Carlo iteration. Rebuilding a 49-point spline per sample would
    dominate the run time. The cache holds interpolators, not results, so it
    is exact.
    """
    build, kind = _METHODS[_check_method(method)]
    wave, area = build(date_str, band)

    if kind == "linear":
        return lambda wl: np.interp(wl, wave, area)
    if kind == "spline":
        return CubicSpline(wave, area, extrapolate=True)
    if kind == "log_spline":
        spline = CubicSpline(wave, np.log(area), bc_type="natural")
        return lambda wl: np.exp(spline(wl))
    raise AssertionError(f"Unhandled interpolation convention {kind!r}.")


def band_of(wavelength_aa: float) -> str | None:
    """``'SW'``, ``'LW'``, or ``None`` for a wavelength EIS does not observe."""
    if SW_BAND[0] <= wavelength_aa <= SW_BAND[1]:
        return "SW"
    if LW_BAND[0] <= wavelength_aa <= LW_BAND[1]:
        return "LW"
    return None


def effective_area(wavelengths_aa, date=None, method="ground") -> np.ndarray:
    """EIS effective area in cm^2, including the CCD quantum efficiency.

    Parameters
    ----------
    wavelengths_aa : array_like
        Wavelengths in Angstrom. May span both EIS bands.
    date : str or datetime or astropy.time.Time, optional
        Observation date. Required for every calibration except ``ground``,
        which is the pre-flight measurement and has no epoch.
    method : str
        One of ``ground``, ``dz2013``, ``warren2014``, ``dz2025``.

    Returns
    -------
    numpy.ndarray
        Effective area in cm^2, NaN outside the two EIS bands.

    Notes
    -----
    The bands are handled separately rather than by one interpolation across
    the gap: EIS has two detectors with independent calibrations, and a curve
    fitted through both would put a meaningless area in the 213-245 Angstrom
    gap and distort the band edges.
    """
    _check_method(method)
    if date is None:
        if method in TIME_DEPENDENT_CALIBRATIONS:
            raise ValueError(
                f"The {method!r} EIS calibration is time-dependent, so it "
                f"needs an observation date, for example "
                f"date='2012-06-03'. Use method='ground' for the "
                f"epoch-independent pre-flight calibration."
            )
        date_str = ""       # unused, but keeps the cache key hashable
    else:
        date_str = normalise_date(date)

    wavelengths = np.atleast_1d(np.asarray(wavelengths_aa, dtype=np.float64))
    out = np.full(wavelengths.shape, np.nan, dtype=np.float64)

    for band, (low, high) in (("SW", SW_BAND), ("LW", LW_BAND)):
        mask = (wavelengths >= low) & (wavelengths <= high)
        if np.any(mask):
            out[mask] = _interpolator(method, date_str, band)(wavelengths[mask])

    return out
