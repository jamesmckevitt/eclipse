"""
ECLIPSE: Emission Calculation and Line Prediction for SOLAR-C EUVST

Tools for modeling the performance of EUVST, the EUV spectrograph on SOLAR-C.
"""

__version__ = "0.13.1"
__author__ = "James McKevitt"
__email__ = "jm2@mssl.ucl.ac.uk"

# Import main classes and functions for easy access
from .config import (Detector_SWC, Detector_EIS, Detector_LW, Telescope_EUVST, Telescope_EIS,
                     Telescope_EUVST_LW, Simulation, AluminiumFilter)
from .eis_calibration import effective_area as eis_effective_area
from .utils import wl_to_vel, vel_to_wl, angle_to_distance, distance_to_angle
from .radiometric import (
    intensity_to_photons, add_telescope_throughput, photons_to_pixel_counts,
    apply_exposure, sample_photon_arrivals, add_poisson, apply_focusing_optics_psf,
    spatial_psf_fwhm, spectral_psf_fwhm, spectral_line_spread, across_slit_fwhm, to_electrons,
    to_dn, add_visible_stray_light, add_pinhole_visible_light
)
from .pinhole_diffraction import apply_euv_pinhole_diffraction, airy_disk_pattern
from .fitting import fit_cube_gauss, velocity_from_fit, width_from_fit, analyse, FitConfig, FitComponent
from .monte_carlo import simulate_once, monte_carlo
from .main import main
from .data_processing import load_atmosphere, create_uniform_intensity_cube
from .atmosphere import (Atmosphere, describe_atmosphere_file, edges_from_centres,
                         read_atmosphere, write_atmosphere)
from .synthesis_file import (SpectralLine, Synthesis, convert_synthesis_pickle, load_synthesis,
                             read_synthesis, read_synthesis_products, write_line_cubes,
                             write_synthesis)
from .results_file import convert_results_pickle
from .band import (BandLines, IonLines, band_contribution_functions, read_band_lines,
                   write_band_lines)
from .readout import FocalPlane_SWC, ReadoutSequence
from .full_frame import FullFrame, full_ccd_frame, write_full_frame
from .analysis import (
    load_instrument_response_results,
    get_parameter_combinations,
    analyse_fit_statistics,
    list_fit_components,
    get_results_for_combination,
    summary_table,
    create_sunpy_maps_from_combo,
)
from .deprecated import get_dem_data_from_results

__all__ = [
    "Detector_SWC", "Detector_EIS", "Detector_LW", "Telescope_EUVST", "Telescope_EIS",
    "Telescope_EUVST_LW", "Simulation", "AluminiumFilter",
    "eis_effective_area",
    "wl_to_vel", "vel_to_wl", "angle_to_distance", "distance_to_angle",
    "intensity_to_photons", "add_telescope_throughput", "photons_to_pixel_counts",
    "apply_exposure", "sample_photon_arrivals", "add_poisson", "apply_focusing_optics_psf",
    "spatial_psf_fwhm", "spectral_psf_fwhm", "spectral_line_spread", "across_slit_fwhm",
    "to_electrons",
    "to_dn", "add_visible_stray_light", "add_pinhole_visible_light",
    "fit_cube_gauss", "velocity_from_fit", "width_from_fit", "analyse",
    "FitConfig", "FitComponent",
    "simulate_once", "monte_carlo",
    "main",
    "load_atmosphere",
    "create_uniform_intensity_cube",
    "Atmosphere",
    "describe_atmosphere_file",
    "edges_from_centres",
    "read_atmosphere",
    "write_atmosphere",
    "SpectralLine",
    "Synthesis",
    "convert_synthesis_pickle",
    "load_synthesis",
    "read_synthesis_products",
    "write_line_cubes",
    "read_synthesis",
    "write_synthesis",
    "convert_results_pickle",
    "BandLines",
    "IonLines",
    "band_contribution_functions",
    "read_band_lines",
    "write_band_lines",
    "FocalPlane_SWC",
    "ReadoutSequence",
    "FullFrame",
    "full_ccd_frame",
    "write_full_frame",
    "load_instrument_response_results",
    "get_parameter_combinations",
    "analyse_fit_statistics", 
    "list_fit_components",
    "get_results_for_combination",
    "summary_table",
    "create_sunpy_maps_from_combo",
    "get_dem_data_from_results"
]
