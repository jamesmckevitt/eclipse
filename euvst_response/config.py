"""
Configuration classes for instruments, detectors, and simulation parameters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import numpy as np
import astropy.units as u
import scipy.interpolate
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
def _load_throughput_table(path) -> tuple[u.Quantity, np.ndarray]:
    """Return (lambda, T) arrays from a 2-col ASCII table (skip comments). lambda is in nm."""
    content = path.read_text()
    lines = content.strip().split('\n')[2:]  # Skip first 2 lines
    data = []
    for line in lines:
        if line.strip() and not line.strip().startswith('#'):
            data.append([float(x) for x in line.split()])
    arr = np.array(data)
    wl = arr[:, 0] * u.nm
    tr = arr[:, 1]
    return wl, tr


def _interp_tr(wavelength_nm: float, wl_tab: np.ndarray, tr_tab: np.ndarray) -> float:
    """Linear interpolation."""
    f = scipy.interpolate.interp1d(wl_tab, tr_tab, bounds_error=False, fill_value=np.nan)
    return float(f(wavelength_nm))


@dataclass
class AluminiumFilter:
    """Multi-layer EUV filter (Al + Al2O3 + C) in front of SWC detector."""
    al_thickness: u.Quantity = 1485 * u.angstrom
    oxide_thickness: u.Quantity = 95 * u.angstrom
    c_thickness: u.Quantity = 0 * u.angstrom
    mesh_throughput: float = 0.8
    al_table: Path = field(default_factory=lambda: files('euvst_response') / 'data' / 'throughput' / 'throughput_aluminium_1000_angstrom.dat')
    oxide_table: Path = field(default_factory=lambda: files('euvst_response') / 'data' / 'throughput' / 'throughput_aluminium_oxide_1000_angstrom.dat')
    c_table: Path = field(default_factory=lambda: files('euvst_response') / 'data' / 'throughput' / 'throughput_carbon_1000_angstrom.dat')
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

    def visible_light_throughput(self) -> float:
        """Calculate visible light throughput reduction due to aluminum filter (For every 170 Angstrom of aluminum, throughput is reduced by factor of 10)."""
        thickness_aa = self.al_thickness.to(u.angstrom).value
        layers = thickness_aa / 170.0
        return 10.0 ** (-layers) * self.mesh_throughput


# -----------------------------------------------------------------------------
# Configuration objects
# -----------------------------------------------------------------------------
@dataclass
class Detector_SWC:
    """Solar-C/EUVST SWC detector configuration."""
    qe_vis: float = 1.0
    qe_euv: float = 0.76
    read_noise_rms: u.Quantity = 10.0 * u.electron / u.pixel
    dark_current: u.Quantity = 21.0 * u.electron / (u.pixel * u.s)  # Default value, will be overridden
    _dark_current_293k: u.Quantity = 20000.0 * u.electron / (u.pixel * u.s)  # Q_d0 at 293 K
    gain_e_per_dn: u.Quantity = 2.78 * u.electron / u.DN  # MSSL EM test results
    max_dn: u.Quantity = 65535 * u.DN / u.pixel
    pix_size: u.Quantity = (13.5 * u.um).cgs / u.pixel
    wvl_res: u.Quantity = (16.9 * u.mAA).cgs / u.pixel
    plate_scale_angle: u.Quantity = 0.159 * u.arcsec / u.pixel
    material: str = "silicon"
    filter_distance: u.Quantity = 250 * u.mm  # Distance from filter to detector for pinhole diffraction

    @property
    def si_fano(self) -> float:
        """Get Fano factor for the detector material."""
        return DETECTOR_MATERIALS[self.material]["fano_factor"]

    @staticmethod
    def calculate_dark_current(temp: u.Quantity) -> u.Quantity:
        """Calculate dark current for SWC (NIMO) CCD."""
        return calculate_dark_current(temp, Detector_SWC._dark_current_293k, ccd_type="NIMO")

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
    qe_vis: float = 0.65  # MSSL engineering test report
    read_noise_rms: u.Quantity = 5.0 * u.electron / u.pixel
    dark_current: u.Quantity = 21.0 * u.electron / (u.pixel * u.s)  # Default value, will be overridden
    _dark_current_293k: u.Quantity = 250.0 * u.electron / (u.pixel * u.s)  # Q_d0 at 293K for EIS
    gain_e_per_dn: u.Quantity = 6.3 * u.electron / u.DN
    max_dn: u.Quantity = 65535 * u.DN / u.pixel
    pix_size: u.Quantity = (13.5 * u.um).cgs / u.pixel
    wvl_res: u.Quantity = (22.3 * u.mAA).cgs / u.pixel
    plate_scale_angle: u.Quantity = 1 * u.arcsec / u.pixel
    material: str = "silicon"

    @property
    def si_fano(self) -> float:
        """Get Fano factor for the detector material."""
        return DETECTOR_MATERIALS[self.material]["fano_factor"]

    @property
    def plate_scale_length(self) -> u.Quantity:
        return angle_to_distance(self.plate_scale_angle * 1*u.pix) / u.pixel
    
    @staticmethod
    def calculate_dark_current(temp: u.Quantity) -> u.Quantity:
        """Calculate dark current for EIS (AIMO) CCD."""
        return calculate_dark_current(temp, Detector_EIS._dark_current_293k, ccd_type="AIMO")
    
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
        from dataclasses import replace
        dark_current = cls.calculate_dark_current(temp)
        
        detector = replace(cls(), dark_current=dark_current)
        detector._ccd_temperature = temp  # Store original temperature with units
        return detector


@dataclass
class Telescope_EUVST:
    """Solar-C/EUVST telescope configuration."""
    D_ap: u.Quantity = 0.28 * u.m
    microroughness_sigma: u.Quantity = 0.3 * u.nm  # RMS microroughness for primary mirror
    filter: AluminiumFilter = field(default_factory=AluminiumFilter)
    psf_type: str = "gaussian"
    psf_params: list = field(default_factory=lambda: [0.343 * u.pixel])  # FWHM of 0.805 pix from 0.128 arcsec from optical design RSC-2022021B in sigma
    
    # Wavelength-dependent efficiency tables
    pm_table: Path = field(default_factory=lambda: files('euvst_response') / 'data' / 'throughput' / 'primary_mirror_coating_reflectance.dat')
    grating_table: Path = field(default_factory=lambda: files('euvst_response') / 'data' / 'throughput' / 'grating_reflection_efficiency.dat')

    @property
    def collecting_area(self) -> u.Quantity:
        return 0.5 * np.pi * (self.D_ap / 2) ** 2

    def primary_mirror_efficiency(self, wl0: u.Quantity) -> float:
        """
        Calculate wavelength-dependent primary mirror efficiency.
        
        Parameters
        ----------
        wl0 : u.Quantity
            Wavelength
            
        Returns
        -------
        float
            Primary mirror efficiency (dimensionless)
        """
        wl_nm = wl0.to_value(u.nm)
        wl_pm, eff_pm = _load_throughput_table(self.pm_table)
        return _interp_tr(wl_nm, wl_pm, eff_pm)

    def grating_efficiency(self, wl0: u.Quantity) -> float:
        """
        Calculate wavelength-dependent grating efficiency.
        
        Parameters
        ----------
        wl0 : u.Quantity
            Wavelength
            
        Returns
        -------
        float
            Grating efficiency (dimensionless)
        """
        wl_nm = wl0.to_value(u.nm)
        wl_grat, eff_grat = _load_throughput_table(self.grating_table)
        return _interp_tr(wl_nm, wl_grat, eff_grat)

    def microroughness_efficiency(self, wl0: u.Quantity) -> float:
        """
        Calculate the efficiency reduction due to primary mirror microroughness.
        
        Formula: 1 - (4*pi*sigma/lambda)^2
        where sigma is the RMS microroughness and lambda is the wavelength.
        
        Parameters
        ----------
        wl0 : u.Quantity
            Wavelength
            
        Returns
        -------
        float
            Microroughness efficiency factor (dimensionless)
        """
        # Convert both wavelength and sigma to the same units (nm for convenience)
        wl_nm = wl0.to(u.nm)
        sigma_nm = self.microroughness_sigma.to(u.nm)
        
        # Calculate (4*pi*sigma/lambda)^2
        roughness_term = (4 * np.pi * sigma_nm / wl_nm) ** 2
        
        # Return 1 - (4*pi*sigma/lambda)^2
        return 1.0 - roughness_term.value

    def throughput(self, wl0: u.Quantity) -> float:
        """
        Calculate total telescope throughput including wavelength-dependent efficiencies.
        
        Parameters
        ----------
        wl0 : u.Quantity
            Wavelength
            
        Returns
        -------
        float
            Total telescope throughput (dimensionless)
        """
        # Get wavelength-dependent efficiencies
        pm_eff_wl = self.primary_mirror_efficiency(wl0)
        grat_eff_wl = self.grating_efficiency(wl0)
        
        # Apply microroughness efficiency to primary mirror efficiency
        pm_eff_with_roughness = pm_eff_wl * self.microroughness_efficiency(wl0)
        
        # Calculate total throughput
        return pm_eff_with_roughness * grat_eff_wl * self.filter.total_throughput(wl0)

    def ea_and_throughput(self, wl0: u.Quantity) -> u.Quantity:
        return self.collecting_area * self.throughput(wl0)


@dataclass
class Telescope_EIS:
    """Hinode/EIS telescope configuration for comparison."""
    psf_type: str = "gaussian"
    psf_params: list = field(default_factory=lambda: [1.28 * u.pixel])  # FWHM of 3 in sigma
    
    def ea_and_throughput(self, wl0: u.Quantity) -> u.Quantity:
        # Effective area including detector QE is 0.23 cm2
            # https://hinode.nao.ac.jp/en/for-researchers/instruments/eis/fact-sheet/
            # https://solarb.mssl.ucl.ac.uk/SolarB/eis_docs/eis_notes/02_RADIOMETRIC_CALIBRATION/eis_swnote_02.pdf
        # Returning the throughput (without the QE):
        eis_detector = Detector_EIS()
        return (0.23 * u.cm**2) / eis_detector.qe_euv


@dataclass
class Simulation:
    """
    Simulation configuration and parameters.
    
    The expos parameter is a single exposure time for this simulation.
    """
    expos: u.Quantity = 1.0 * u.s  # Single exposure time
    n_iter: int = 10
    slit_width: u.Quantity = 0.2 * u.arcsec
    ncpu: int = -1
    instrument: str = "SWC"
    vis_sl: u.Quantity = 0 * u.photon / (u.s * u.cm**2)  # Visible stray light flux before filter
    psf: bool = False
    enable_pinholes: bool = False
    pinhole_sizes: List[u.Quantity] = field(default_factory=list)
    pinhole_positions: List[float] = field(default_factory=list)

    @property
    def slit_scan_step(self) -> u.Quantity:
        return self.slit_width

    def __post_init__(self):
        allowed_slits = {
            "EIS": [1, 2, 4],
            "SWC": [0.2, 0.4, 0.8, 1.6],
        }
        inst = self.instrument.upper()
        slit_val = self.slit_width.to_value(u.arcsec)
        if inst == "EIS":
            if slit_val not in allowed_slits["EIS"]:
                raise ValueError("For EIS, slit_width must be 1, 2, or 4 arcsec.")
        elif inst in ("SWC"):
            if slit_val not in allowed_slits["SWC"]:
                raise ValueError("For SWC, slit_width must be 0.2, 0.4, 0.8, or 1.6 arcsec.")
