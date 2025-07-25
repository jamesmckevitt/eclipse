"""
Configuration classes for instruments, detectors, and simulation parameters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import astropy.units as u
import scipy.interpolate
from .utils import angle_to_distance
import numpy as np


# ------------------------------------------------------------------
#  Throughput helpers & AluminiumFilter
# ------------------------------------------------------------------
def _load_throughput_table(path: str | Path) -> tuple[u.Quantity, np.ndarray]:
    """Return (λ, T) arrays from a 2-col ASCII table (skip comments). λ is in nm."""
    arr = np.loadtxt(path, skiprows=2)
    wl = arr[:, 0] * u.nm
    tr = arr[:, 1]
    return wl, tr


def _interp_tr(wavelength_nm: float, wl_tab: np.ndarray, tr_tab: np.ndarray) -> float:
    """Linear interpolation."""
    f = scipy.interpolate.interp1d(wl_tab, tr_tab, bounds_error=False, fill_value=np.nan)
    return float(f(wavelength_nm))


@dataclass
class AluminiumFilter:
    """Multi-layer EUV filter (Al + Al₂O₃ + C) in front of SWC detector."""
    al_thickness: u.Quantity = 1485 * u.angstrom
    oxide_thickness: u.Quantity = 95 * u.angstrom
    c_thickness: u.Quantity = 0 * u.angstrom
    mesh_throughput: float = 0.8
    al_table: Path = Path("data/throughput/throughput_aluminium_1000_angstrom.dat")
    oxide_table: Path = Path("data/throughput/throughput_aluminium_oxide_1000_angstrom.dat")
    c_table: Path = Path("data/throughput/throughput_carbon_1000_angstrom.dat")
    table_thickness: u.Quantity = 1000 * u.angstrom

    def total_throughput(self, wl0: u.Quantity) -> float:
        """Calculate throughput at a given central wavelength (wl0, astropy Quantity)."""
        wl_nm = wl0.to_value(u.nm)
        wl_al, tr_al = _load_throughput_table(self.al_table)
        wl_ox, tr_ox = _load_throughput_table(self.oxide_table)
        wl_c,  tr_c  = _load_throughput_table(self.c_table)
        t_al = _interp_tr(wl_nm, wl_al, tr_al) ** (self.al_thickness.cgs / self.table_thickness.cgs)
        t_ox = _interp_tr(wl_nm, wl_ox, tr_ox) ** (self.oxide_thickness.cgs / self.table_thickness.cgs)
        t_c  = _interp_tr(wl_nm, wl_c,  tr_c)  ** (self.c_thickness.cgs / self.table_thickness.cgs)
        return t_al * t_ox * t_c * self.mesh_throughput


# -----------------------------------------------------------------------------
# Configuration objects
# -----------------------------------------------------------------------------
@dataclass
class Detector_SWC:
    """Solar-C/EUVST SWC detector configuration."""
    qe_vis: float = 1.0
    qe_euv: float = 0.76
    e_per_ph_euv: u.Quantity = 18.0 * u.electron / u.photon
    e_per_ph_vis: u.Quantity = 2.0 * u.electron / u.photon
    read_noise_rms: u.Quantity = 10.0 * u.electron / u.pixel
    dark_current: u.Quantity = 21.0 * u.electron / (u.pixel * u.s)  # Default value, will be overridden
    _dark_current_293k: u.Quantity = 20000.0 * u.electron / (u.pixel * u.s)  # Q_d0 at 293K
    gain_e_per_dn: u.Quantity = 2.0 * u.electron / u.DN
    max_dn: u.Quantity = 65535 * u.DN / u.pixel
    pix_size: u.Quantity = (13.5 * u.um).cgs / u.pixel
    wvl_res: u.Quantity = (16.9 * u.mAA).cgs / u.pixel
    plate_scale_angle: u.Quantity = 0.159 * u.arcsec / u.pixel
    si_fano: float = 0.115

    @staticmethod
    def calculate_dark_current(temp: u.Quantity) -> u.Quantity:
        """
        Calculate dark current based on CCD temperature.
        
        Uses the formula: Q_d = Q_d0 * 122 * T^3 * e^(-6400/T)
        where Q_d0 = 20000 e-/pix/s at 293K
        
        Parameters
        ----------
        temp : u.Quantity
            CCD temperature with units (e.g., -60 * u.deg_C)
            
        Returns
        -------
        dark_current : u.Quantity
            Dark current in electrons per pixel per second
            
        Raises
        ------
        ValueError
            If temperature is above 300K (27°C)
        """
        
        temp_kelvin = temp.to(u.Kelvin, equivalencies=u.temperature())

        max_temp = 300 * u.K
        min_temp = 230 * u.K

        # Check temperature limits
        if temp_kelvin > max_temp:
            raise ValueError(f"Cannot calculate dark current at {temp_kelvin}. "
                           f"Maximum temperature is {max_temp}")
        
        # Apply minimum temperature limit (clamp to 230K)
        if temp_kelvin < min_temp:
            temp_kelvin = min_temp
            
        # Calculate dark current using the provided formula
        # Q_d = Q_d0 * 122 * T^3 * e^(-6400/T)
        Q_d0 = Detector_SWC._dark_current_293k.to_value(u.electron / (u.pixel * u.s))
        dark_current = Q_d0 * 122 * (temp_kelvin.value**3) * np.exp(-6400/temp_kelvin.value)

        return dark_current * u.electron / (u.pixel * u.s)

    @classmethod
    def with_temperature(cls, temp: u.Quantity):
        """
        Create a detector instance with dark current calculated from temperature.
        
        Parameters
        ----------
        temp : u.Quantity
            CCD temperature with units (e.g., -60 * u.deg_C)
            
        Returns
        -------
        detector : Detector_SWC
            Detector instance with calculated dark current and stored temperature
        """
        from dataclasses import replace
        dark_current = cls.calculate_dark_current(temp)
        
        detector = replace(cls(), dark_current=dark_current)
        detector._ccd_temperature = temp  # Store original temperature with units
        return detector

    @property
    def plate_scale_length(self) -> u.Quantity:
        return angle_to_distance(self.plate_scale_angle * 1*u.pix) / u.pixel


@dataclass
class Detector_EIS:
    """Hinode/EIS detector configuration for comparison."""
    qe_euv: float = 0.64  # EIS SW Note 2
    qe_vis: float = 1
    pix_size: u.Quantity = (13.5 * u.um).cgs / u.pixel
    wvl_res: u.Quantity = (22.3 * u.mAA).cgs / u.pixel
    plate_scale_angle: u.Quantity = 1 * u.arcsec / u.pixel

    @property
    def plate_scale_length(self) -> u.Quantity:
        return angle_to_distance(self.plate_scale_angle * 1*u.pix) / u.pixel
    
    @property
    def e_per_ph_euv(self) -> u.Quantity:
        return Detector_SWC.e_per_ph_euv
    @property
    def e_per_ph_vis(self) -> u.Quantity:
        return Detector_SWC.e_per_ph_vis
    @property
    def read_noise_rms(self) -> u.Quantity:
        return Detector_SWC.read_noise_rms
    @staticmethod
    def calculate_dark_current(temp: u.Quantity) -> u.Quantity:
        """Calculate dark current - uses same formula as SWC detector."""
        return Detector_SWC.calculate_dark_current(temp)
    
    @classmethod
    def with_temperature(cls, temp: u.Quantity):
        """
        Create a detector instance with dark current calculated from temperature.
        
        Parameters
        ----------
        temp : u.Quantity
            CCD temperature with units (e.g., -60 * u.deg_C)
            
        Returns
        -------
        detector : Detector_EIS
            Detector instance with calculated dark current and stored temperature
        """
        # For EIS, we need to create a modified instance
        # Since it inherits properties from SWC, we can't use dataclass replace
        detector = cls()
        detector._ccd_temperature = temp  # Store original temperature with units
        detector._calculated_dark_current = cls.calculate_dark_current(temp)
        
        return detector
    
    @property
    def dark_current(self) -> u.Quantity:
        """Return calculated dark current if available, otherwise default."""
        if hasattr(self, '_calculated_dark_current'):
            return self._calculated_dark_current
        return Detector_SWC.dark_current
    
    @property
    def gain_e_per_dn(self) -> u.Quantity:
        return Detector_SWC.gain_e_per_dn
    @property
    def max_dn(self) -> u.Quantity:
        return Detector_SWC.max_dn
    @property
    def si_fano(self) -> float:
        return Detector_SWC.si_fano


@dataclass
class Telescope_EUVST:
    """Solar-C/EUVST telescope configuration."""
    D_ap: u.Quantity = 0.28 * u.m
    pm_eff: float = 0.161
    grat_eff: float = 0.0623
    psf_focus_res: u.Quantity = 0.5 * u.um / u.pixel
    psf_mesh_res: u.Quantity = 6.12e-4 * u.mm / u.pixel
    psf_focus_file: Path = Path("data/psf/psf_euvst_v20230909_195119_focus.txt")
    psf_mesh_file: Path = Path("data/psf/psf_euvst_v20230909_derived_195119_mesh.txt")
    psf: np.ndarray | None = field(default=None, init=False)
    filter: AluminiumFilter = field(default_factory=AluminiumFilter)

    @property
    def collecting_area(self) -> u.Quantity:
        return 0.5 * np.pi * (self.D_ap / 2) ** 2

    def throughput(self, wl0: u.Quantity) -> float:
        return self.pm_eff * self.grat_eff * self.filter.total_throughput(wl0)

    def ea_and_throughput(self, wl0: u.Quantity) -> u.Quantity:
        return self.collecting_area * self.throughput(wl0)


class Telescope_EIS:
    """Hinode/EIS telescope configuration for comparison."""
    def ea_and_throughput(self, wl0: u.Quantity) -> u.Quantity:
        # Effective area including detector QE is 0.23 cm2
            # https://hinode.nao.ac.jp/en/for-researchers/instruments/eis/fact-sheet/
            # https://solarb.mssl.ucl.ac.uk/SolarB/eis_docs/eis_notes/02_RADIOMETRIC_CALIBRATION/eis_swnote_02.pdf
        # The QE value taken from software note two is 0.64. Therefore, returning the throughput:
        return (0.23 * u.cm**2) / 0.64


@dataclass
class Simulation:
    """
    Simulation configuration and parameters.
    
    The expos parameter is a single exposure time for this simulation.
    """
    expos: u.Quantity = 1.0 * u.s  # Single exposure time
    n_iter: int = 25
    slit_width: u.Quantity = 0.2 * u.arcsec
    ncpu: int = -1
    instrument: str = "SWC"
    vis_sl: u.Quantity = 0 * u.photon / (u.s * u.pixel)

    @property
    def slit_scan_step(self) -> u.Quantity:
        return self.slit_width

    def __post_init__(self):
        allowed_slits = {
            "EIS": [1, 2, 4],
            "SWC": [0.2, 0.4, 1],
        }
        inst = self.instrument.upper()
        slit_val = self.slit_width.to_value(u.arcsec)
        if inst == "EIS":
            if slit_val not in allowed_slits["EIS"]:
                raise ValueError("For EIS, slit_width must be 1, 2, or 4 arcsec.")
        elif inst in ("SWC"):
            if slit_val not in allowed_slits["SWC"]:
                raise ValueError("For SWC, slit_width must be 0.2, 0.4, or 1 arcsec.")
