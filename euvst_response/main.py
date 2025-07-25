"""
Main execution script for instrument response simulations.
"""

from __future__ import annotations
import argparse
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path
import dill
import yaml
import astropy.units as u
from tqdm import tqdm

from .config import AluminiumFilter, Detector_SWC, Detector_EIS, Telescope_EUVST, Telescope_EIS, Simulation
from .data_processing import load_atmosphere, rebin_atmosphere
from .fitting import fit_cube_gauss
from .monte_carlo import monte_carlo
from .utils import parse_yaml_input, ensure_list
import numpy as np


def main() -> None:
    """Main function for running instrument response simulations."""
    
    # Suppress astropy warnings that clutter output
    warnings.filterwarnings(
        "ignore",
        message="target cannot be converted to ICRS",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore", 
        message="target cannot be converted to ICRS, so will not be set on SpectralCoord",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="No observer defined on WCS, SpectralCoord will be converted without any velocity frame change",
        category=UserWarning,
    )
    # Catch any astropy.wcs warnings about ICRS conversion
    warnings.filterwarnings(
        "ignore",
        module="astropy.wcs.wcsapi.fitswcs",
        category=UserWarning,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="YAML config file", required=True)
    args = parser.parse_args()

    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set up instrument, detector, telescope, simulation from config
    instrument = config.get("instrument", "SWC").upper()
    psf = config.get("psf", False)
    n_iter = config.get("n_iter", 25)
    ncpu = config.get("ncpu", -1)

    if psf:
        if instrument == "SWC":
            warnings.warn(
                "The SWC PSF is the modelled PSF including simulations and some microroughness measurements. Final PSF will be measured before launch.",
                UserWarning,
            )
        elif instrument == "EIS":
            warnings.warn(
                "The EIS PSF is not well understood. We use a symmetrical Voigt profile with a FWHM of 3 pixels from Ugarte-Urra (2016) EIS Software Note 2.",
                UserWarning,
            )

    # Parse configuration parameters - can be single values or lists
    # Each parameter combination will be run independently, including exposure times
    slit_widths = ensure_list(parse_yaml_input(config.get("slit_width", ['0.2 arcsec'])))
    
    # Handle instrument-specific parameters
    if instrument == "SWC":
        # SWC requires oxide, carbon, and aluminium thickness parameters
        oxide_thicknesses = ensure_list(parse_yaml_input(config.get("oxide_thickness", ['95 angstrom'])))
        c_thicknesses = ensure_list(parse_yaml_input(config.get("c_thickness", ['0 angstrom'])))
        aluminium_thicknesses = ensure_list(parse_yaml_input(config.get("aluminium_thickness", ['1485 angstrom'])))
    elif instrument == "EIS":
        # EIS doesn't use these parameters - check they weren't specified
        if "oxide_thickness" in config:
            raise ValueError("EIS does not support oxide thickness parameter. Remove 'oxide_thickness' from configuration.")
        if "c_thickness" in config:
            raise ValueError("EIS does not support carbon thickness parameter. Remove 'c_thickness' from configuration.")
        if "aluminium_thickness" in config:
            raise ValueError("EIS does not support custom aluminium thickness parameter. Remove 'aluminium_thickness' from configuration.")
        
        # Set defaults for EIS (these won't be used but are needed for parameter combination logic)
        oxide_thicknesses = [0 * u.nm]
        c_thicknesses = [0 * u.nm] 
        aluminium_thicknesses = [1500 * u.angstrom]
    
    ccd_temperatures = ensure_list(parse_yaml_input(config.get("ccd_temperature", ['-60 Celsius'])))  # Temperature in Celsius
    vis_sl_vals = ensure_list(parse_yaml_input(config.get("vis_sl", ['0 photon / (s * pixel)'])))
    exposures = ensure_list(parse_yaml_input(config.get("expos", ['1 s'])))

    # Load synthetic atmosphere cube
    print("Loading atmosphere...")
    cube_sim = load_atmosphere("./run/input/synthesised_spectra.pkl")

    # Set up base detector configuration (doesn't change with parameters)
    if instrument == "SWC":
        DET = Detector_SWC()
    elif instrument == "EIS":
        DET = Detector_EIS()
    else:
        raise ValueError(f"Unknown instrument: {instrument}")

    # Create results structure for all parameter combinations
    all_results = {}

    # Loop over all parameter combinations
    total_combinations = len(slit_widths) * len(oxide_thicknesses) * len(c_thicknesses) * len(aluminium_thicknesses) * len(ccd_temperatures) * len(vis_sl_vals) * len(exposures)
    print(f"Running {total_combinations} parameter combinations...")
    
    combination_idx = 0
    for slit_width in slit_widths:
        # Rebin atmosphere only when slit width changes (expensive operation)
        print(f"\nRebinning atmosphere cube for slit width {slit_width}...")
        SIM_temp = Simulation(
            expos=1.0 * u.s,  # Temporary value for rebinning
            n_iter=n_iter,
            slit_width=slit_width,
            ncpu=ncpu,
            instrument=instrument,
            psf=psf,
        )
        cube_reb = rebin_atmosphere(cube_sim, DET, SIM_temp)
        
        print("Fitting ground truth cube...")
        fit_truth = fit_cube_gauss(cube_reb, n_jobs=ncpu)
        
        for oxide_thickness in oxide_thicknesses:
            for c_thickness in c_thicknesses:
                for aluminium_thickness in aluminium_thicknesses:
                    for ccd_temperature in ccd_temperatures:
                        for vis_sl in vis_sl_vals:
                            for exposure in exposures:
                                combination_idx += 1
                                print(f"--- Combination {combination_idx}/{total_combinations} ---")
                                print(f"Slit width: {slit_width}")
                                print(f"Oxide thickness: {oxide_thickness}")
                                print(f"Carbon thickness: {c_thickness}")
                                print(f"Aluminium thickness: {aluminium_thickness}")
                                print(f"CCD temperature: {ccd_temperature}")
                                print(f"Visible stray light: {vis_sl}")
                                print(f"Exposure time: {exposure}")
                                
                                # Set up telescope configuration for this combination
                                if instrument == "SWC":
                                    filter_obj = AluminiumFilter(
                                        oxide_thickness=oxide_thickness,
                                        c_thickness=c_thickness,
                                        al_thickness=aluminium_thickness,
                                    )
                                    TEL = Telescope_EUVST(filter=filter_obj)
                                elif instrument == "EIS":
                                    TEL = Telescope_EIS()
                                    # EIS uses fixed filter configuration - no custom parameters needed
                                
                                # Set up detector configuration with calculated dark current
                                if instrument == "SWC":
                                    # Create a detector with calculated dark current for this temperature
                                    DET = Detector_SWC.with_temperature(ccd_temperature)
                                    print(f"Calculated dark current: {DET.dark_current:.2e}")
                                elif instrument == "EIS":
                                    DET = Detector_EIS.with_temperature(ccd_temperature)
                                    print(f"Calculated dark current: {DET.dark_current:.2e}")
                                else:
                                    raise ValueError(f"Unknown instrument: {instrument}")

                                # Create simulation object
                                SIM = Simulation(
                                    expos=exposure,  # Single exposure value
                                    n_iter=n_iter,
                                    slit_width=slit_width,
                                    ncpu=ncpu,
                                    instrument=instrument,
                                    vis_sl=vis_sl,
                                    psf=psf,
                                )

                                # Run Monte Carlo for this single parameter combination
                                dn_signals, dn_fits, photon_signals, photon_fits = monte_carlo(
                                    cube_reb, exposure, DET, TEL, SIM, n_iter=SIM.n_iter
                                )

                                def process_fit_results(fits):
                                    """
                                    Process fit results from Monte Carlo simulations.
                                    Returns a dict with first_fit, mean, std (with units if present).
                                    """
                                    import astropy.units as u

                                    # Check if fits have units (astropy Quantity)
                                    has_units = hasattr(fits[0, 0, 0, 0], "unit")
                                    fits_shape = fits.shape

                                    # Extract units for each parameter
                                    fits_units = np.empty(fits_shape[-1], dtype=object)
                                    for i in range(fits_shape[-1]):
                                        fits_units[i] = u.Unit(fits[0, 0, 0, i].unit)
                                    # Strip units for computation
                                    fits_values = np.empty(fits_shape, dtype=float)
                                    for i in tqdm(range(fits_shape[0]), desc="Processing fits", leave=False):
                                        for j in range(fits_shape[1]):
                                            for k in range(fits_shape[2]):
                                                for l in range(fits_shape[3]):
                                                    fits_values[i, j, k, l] = fits[i, j, k, l].value
                                    # Compute stats
                                    mean = fits_values.mean(axis=0)
                                    std = fits_values.std(axis=0)
                                    # Re-attach units
                                    mean_with_units = np.empty(fits_shape[1:], dtype=object)
                                    std_with_units = np.empty(fits_shape[1:], dtype=object)
                                    for j in tqdm(range(fits_shape[1]), desc="Reattaching units to fit stats", leave=False):
                                        for k in range(fits_shape[2]):
                                            for l in range(fits_shape[3]):
                                                mean_with_units[j, k, l] = mean[j, k, l] * fits_units[l]
                                                std_with_units[j, k, l] = std[j, k, l] * fits_units[l]
                                    return {
                                        "first_fit": fits[0],
                                        "mean": mean_with_units,
                                        "std": std_with_units,
                                    }
                                
                                dn_fit_stats = process_fit_results(dn_fits)
                                photon_fit_stats = process_fit_results(photon_fits)

                                # Store results for this parameter combination
                                sec = exposure.to_value(u.s)
                                param_key = (
                                    slit_width.to_value(u.arcsec),
                                    oxide_thickness.to_value(u.nm) if oxide_thickness.unit.is_equivalent(u.nm) else oxide_thickness.to_value(u.AA),
                                    c_thickness.to_value(u.nm) if c_thickness.unit.is_equivalent(u.nm) else c_thickness.to_value(u.AA),
                                    aluminium_thickness.to_value(u.AA),
                                    ccd_temperature.to_value(u.Celsius,equivalencies=u.temperature()),
                                    vis_sl.to_value(u.photon / (u.s * u.pixel)),
                                    sec
                                )
                                
                                all_results[param_key] = {
                                    "parameters": {
                                        "slit_width": slit_width,
                                        "oxide_thickness": oxide_thickness,
                                        "c_thickness": c_thickness,
                                        "aluminium_thickness": aluminium_thickness,
                                        "ccd_temperature": ccd_temperature,
                                        "vis_sl": vis_sl,
                                        "exposure": exposure,
                                    },
                                    "first_dn_signal": dn_signals[0],
                                    "first_photon_signal": photon_signals[0],
                                    "dn_fit_stats": dn_fit_stats,
                                    "photon_fit_stats": photon_fit_stats,
                                    "ground_truth": {
                                        "fit_truth": fit_truth,
                                    }
                                }
                                
                                # Clean up memory
                                del dn_signals, dn_fits, photon_signals, photon_fits, dn_fit_stats, photon_fit_stats

    # Prepare final results structure
    results = {
        "all_combinations": all_results,
        "parameter_ranges": {
            "slit_widths": slit_widths,
            "oxide_thicknesses": oxide_thicknesses,
            "c_thicknesses": c_thicknesses,
            "aluminium_thicknesses": aluminium_thicknesses,
            "ccd_temperatures": ccd_temperatures,
            "vis_sl_vals": vis_sl_vals,
            "exposures": exposures,
        }
    }

    # Generate output filename based on config file
    config_path = Path(args.config)
    config_base = config_path.stem
    output_file = Path(f"run/result/instrument_response_{config_base}.pkl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to {output_file}")
    with open(output_file, "wb") as f:
      dill.dump({
        "results": results,
        "config": config,
        "instrument": instrument,
        "cube_sim": cube_sim,
        "cube_reb": cube_reb,
      }, f)

    print(f"Saved results to {output_file} ({os.path.getsize(output_file) / 1e6:.1f} MB)")

    print(f"Instrument response simulation complete!")
    print(f"Total parameter combinations: {total_combinations}")


if __name__ == "__main__":
        main()
