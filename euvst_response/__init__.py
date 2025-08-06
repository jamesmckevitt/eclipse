"""
M-ECLIPSES: MSSL Emission Calculation and Line Intensity Prediction for SOLAR-C EUVST-SW

This package provides tools for modeling the performance of the ESA/MSSL short wavelength camera,
part of the EUV spectrograph EUVST, on SOLAR-C.
"""

__version__ = "0.3.2"
__author__ = "James McKevitt"
__email__ = "jm2@mssl.ucl.ac.uk"

# Import main classes and functions for easy access
from .config import Detector_SWC, Detector_EIS, Telescope_EUVST, Telescope_EIS, Simulation, AluminiumFilter
from .utils import wl_to_vel, vel_to_wl, angle_to_distance, distance_to_angle
from .radiometric import (
    intensity_to_photons, add_telescope_throughput, photons_to_pixel_counts,
    apply_exposure_and_poisson, add_poisson, apply_focusing_optics_psf, to_electrons, 
    to_dn, add_visible_stray_light, add_pinhole_visible_light
)
from .pinhole_diffraction import apply_euv_pinhole_diffraction, airy_disk_pattern
from .fitting import fit_cube_gauss, velocity_from_fit, width_from_fit, analyse
from .monte_carlo import simulate_once, monte_carlo
from .main import main
from .data_processing import load_atmosphere
from .analysis import (
    load_instrument_response_results,
    get_parameter_combinations,
    analyse_fit_statistics,
    get_results_for_combination,
    summary_table,
    create_sunpy_maps_from_combo,
    get_dem_data_from_results
)

__all__ = [
    "Detector_SWC", "Detector_EIS", "Telescope_EUVST", "Telescope_EIS", 
    "Simulation", "AluminiumFilter",
    "wl_to_vel", "vel_to_wl", "angle_to_distance", "distance_to_angle",
    "intensity_to_photons", "add_telescope_throughput", "photons_to_pixel_counts",
    "apply_exposure_and_poisson", "add_poisson", "apply_focusing_optics_psf", "to_electrons", 
    "to_dn", "add_visible_stray_light", "add_pinhole_visible_light",
    "fit_cube_gauss", "velocity_from_fit", "width_from_fit", "analyse",
    "simulate_once", "monte_carlo",
    "main",
    "load_atmosphere",
    "load_instrument_response_results",
    "get_parameter_combinations",
    "analyse_fit_statistics", 
    "get_results_for_combination",
    "summary_table",
    "create_sunpy_maps_from_combo",
    "get_dem_data_from_results"
]
