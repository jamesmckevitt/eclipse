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
from .fitting import fit_cube_gauss, velocity_from_fit, analyse
from .monte_carlo import monte_carlo
from .utils import parse_yaml_input, ensure_list


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

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set up instrument, detector, telescope, simulation from config
    instrument = config.get("instrument", "SWC").upper()
    psf = config.get("psf", False)
    n_iter = config.get("n_iter", 25)
    ncpu = config.get("ncpu", -1)

    # Parse configuration parameters - can be single values or lists
    # Each parameter combination will be run independently, including exposure times
    slit_widths = ensure_list(parse_yaml_input(config.get("slit_width", ['0.2 arcsec'])))
    oxide_thicknesses = ensure_list(parse_yaml_input(config.get("oxide_thickness", ['95 nm'])))
    c_thicknesses = ensure_list(parse_yaml_input(config.get("c_thickness", ['0 nm'])))
    aluminium_thicknesses = ensure_list(parse_yaml_input(config.get("aluminium_thickness", ['1485 angstrom'])))
    ccd_temperatures = ensure_list(config.get("ccd_temperature", [-60]))  # Temperature in Celsius
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
            slit_width=slit_width,
            instrument=instrument,
        )
        cube_reb = rebin_atmosphere(cube_sim, DET, SIM_temp)
        
        print("Fitting ground truth cube...")
        fit_truth = fit_cube_gauss(cube_reb)
        v_true = velocity_from_fit(fit_truth, cube_reb.meta['rest_wav'])
        
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
                                print(f"CCD temperature: {ccd_temperature}°C")
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
                                    if oxide_thickness.value != 0 or c_thickness.value != 0:
                                        raise ValueError("EIS does not support oxide or C thicknesses.")
                                    if aluminium_thickness.value != 1485:
                                        raise ValueError("EIS does not support custom aluminium thicknesses.")
                                
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

                            if psf:
                                if instrument != "SWC":
                                    raise ValueError("PSF loading is only supported for SWC/EUVST instrument.")
                                if instrument == "SWC":
                                    raise NotImplementedError("PSF loading is not implemented yet, waiting for NAOJ measurements of mirror scattering.")

                            # Create simulation object
                            SIM = Simulation(
                                expos=exposure,  # Single exposure value
                                n_iter=n_iter,
                                slit_width=slit_width,
                                ncpu=ncpu,
                                instrument=instrument,
                                vis_sl=vis_sl,
                            )

                            # Run Monte Carlo for this single parameter combination
                            dn_signals, dn_fits, photon_signals, photon_fits = monte_carlo(
                                cube_reb, exposure, DET, TEL, SIM, n_iter=SIM.n_iter
                            )
                            
                            # Store results for this parameter combination
                            sec = exposure.to_value(u.s)
                            param_key = (
                                slit_width.to_value(u.arcsec),
                                oxide_thickness.to_value(u.nm) if oxide_thickness.unit.is_equivalent(u.nm) else oxide_thickness.to_value(u.AA),
                                c_thickness.to_value(u.nm) if c_thickness.unit.is_equivalent(u.nm) else c_thickness.to_value(u.AA),
                                aluminium_thickness.to_value(u.AA),
                                ccd_temperature,
                                vis_sl.to_value() if hasattr(vis_sl, 'to_value') else vis_sl,
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
                                "dn_signals": dn_signals,
                                "photon_signals": photon_signals,
                                "dn_fits": dn_fits,
                                "photon_fits": photon_fits,
                                "analysis": analyse(dn_fits, v_true, cube_reb.meta['rest_wav']),
                                "ground_truth": {
                                    "fit_truth": fit_truth,
                                    "v_true": v_true,
                                }
                            }
                            
                            # Clean up memory
                            del dn_signals, dn_fits, photon_signals, photon_fits

    # Prepare final results structure
    results = {
        "all_combinations": all_results,
        "parameter_ranges": {
            "slit_widths": slit_widths,
            "oxide_thicknesses": oxide_thicknesses,
            "c_thicknesses": c_thicknesses,
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
      }, f)

    print(f"Saved results to {output_file} ({os.path.getsize(output_file) / 1e6:.1f} MB)")

    print(f"Instrument response simulation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total parameter combinations: {total_combinations}")


if __name__ == "__main__":
        main()
