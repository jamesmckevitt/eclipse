"""
M-ECLIPSES: MSSL Emission Calculation and Line Intensity Prediction for SOLAR-C EUVST-SW

This package provides tools for modeling the performance of the ESA/MSSL short wavelength camera,
part of the EUV spectrograph EUVST, on SOLAR-C.
"""

__version__ = "1.0.0"
__author__ = "James McKevitt"
__email__ = "jm2@mssl.ucl.ac.uk"
__author__ = "James McKevitt"

# Import main classes and functions for easy access
from .config import Detector_SWC, Detector_EIS, Telescope_EUVST, Telescope_EIS, Simulation, AluminiumFilter
from .utils import wl_to_vel, vel_to_wl, angle_to_distance, distance_to_angle
from .radiometric import intensity_to_photons, add_effective_area, photons_to_pixel_rate
from .fitting import fit_cube_gauss, velocity_from_fit, width_from_fit, analyse
from .monte_carlo import simulate_once, monte_carlo
from .main import main

__all__ = [
    "Detector_SWC", "Detector_EIS", "Telescope_EUVST", "Telescope_EIS", 
    "Simulation", "AluminiumFilter",
    "wl_to_vel", "vel_to_wl", "angle_to_distance", "distance_to_angle",
    "intensity_to_photons", "add_effective_area", "photons_to_pixel_rate",
    "fit_cube_gauss", "velocity_from_fit", "width_from_fit", "analyse",
    "simulate_once", "monte_carlo",
    "main"
]
