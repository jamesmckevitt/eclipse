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
    slit_widths = ensure_list(parse_yaml_input(config.get("slit_width", ['0.2 arcsec'])))
    oxide_thicknesses = ensure_list(parse_yaml_input(config.get("oxide_thickness", ['0 nm'])))
    c_thicknesses = ensure_list(parse_yaml_input(config.get("c_thickness", ['0 nm'])))
    vis_sl_vals = ensure_list(parse_yaml_input(config.get("vis_sl", ['0 photon / (s * pixel)'])))
    
    # Handle exposure times with optional units
    expos_config = config.get("expos", ["1 s"])
    if isinstance(expos_config, list):
        # Check if first element has units
        if isinstance(expos_config[0], str):
            # Parse as quantities with units
            exposures = ensure_list(parse_yaml_input(expos_config))
        else:
            # Legacy format - assume seconds
            exposures = [u.Quantity(exp, u.s) for exp in expos_config]
    else:
        # Single value
        if isinstance(expos_config, str):
            exposures = [parse_yaml_input(expos_config)]
        else:
            exposures = [u.Quantity(expos_config, u.s)]

    # Load synthetic atmosphere cube
    print("Loading atmosphere...")
    cube_sim = load_atmosphere("./run/input/synthesised_spectra.pkl")

    # Create results structure for all parameter combinations
    all_results = {}

    # Loop over all parameter combinations
    total_combinations = len(slit_widths) * len(oxide_thicknesses) * len(c_thicknesses) * len(vis_sl_vals)
    print(f"Running {total_combinations} parameter combinations...")
    
    combination_idx = 0
    for slit_width in slit_widths:
        for oxide_thickness in oxide_thicknesses:
            for c_thickness in c_thicknesses:
                for vis_sl in vis_sl_vals:
                    combination_idx += 1
                    print(f"\n--- Combination {combination_idx}/{total_combinations} ---")
                    print(f"Slit width: {slit_width}")
                    print(f"Oxide thickness: {oxide_thickness}")
                    print(f"Carbon thickness: {c_thickness}")
                    print(f"Visible stray light: {vis_sl}")
                    
                    # Set up instrument configuration for this combination
                    if instrument == "SWC":
                        filter_obj = AluminiumFilter(
                            oxide_thickness=oxide_thickness,
                            c_thickness=c_thickness,
                        )
                        DET = Detector_SWC()
                        TEL = Telescope_EUVST(filter=filter_obj)
                    elif instrument == "EIS":
                        DET = Detector_EIS()
                        TEL = Telescope_EIS()
                        if oxide_thickness.value != 0 or c_thickness.value != 0:
                            raise ValueError("EIS does not support oxide or C thicknesses.")
                    else:
                        raise ValueError(f"Unknown instrument: {instrument}")
                    
                    if psf:
                        if instrument != "SWC":
                            raise ValueError("PSF loading is only supported for SWC/EUVST instrument.")
                        if instrument == "SWC":
                            raise NotImplementedError("PSF loading is not implemented yet, waiting for NAOJ measurements of mirror scattering.")

                    # Create simulation object
                    SIM = Simulation(
                        expos=exposures,
                        n_iter=n_iter,
                        slit_width=slit_width,
                        ncpu=ncpu,
                        instrument=instrument,
                        vis_sl=vis_sl,
                    )

                    print("Rebinning atmosphere cube to instrument resolution for each slit position...")
                    cube_reb = rebin_atmosphere(cube_sim, DET, SIM)

                    print("Fitting ground truth cube...")
                    fit_truth = fit_cube_gauss(cube_reb)
                    v_true = velocity_from_fit(fit_truth, cube_reb.meta['rest_wav'])

                    # Storage for this parameter combination
                    first_dn_signal_per_exp = {}
                    first_photon_signal_per_exp = {}
                    first_fit_per_exp = {}
                    first_photon_fit_per_exp = {}
                    analysis_per_exp = {}

                    for t_exp in tqdm(exposures, desc="Exposure time", unit="exposure"):
                        dn_signals, dn_fits, photon_signals, photon_fits = monte_carlo(
                            cube_reb, t_exp, DET, TEL, SIM, n_iter=SIM.n_iter
                        )
                        sec = t_exp.to_value(u.s)
                        first_dn_signal_per_exp[sec] = dn_signals[0]
                        first_photon_signal_per_exp[sec] = photon_signals[0]
                        first_fit_per_exp[sec] = dn_fits[0]
                        first_photon_fit_per_exp[sec] = photon_fits[0]
                        analysis_per_exp[sec] = analyse(dn_fits, v_true, cube_reb.meta['rest_wav'])
                        del dn_signals, dn_fits, photon_signals, photon_fits

                    # Store results for this parameter combination
                    param_key = (
                        slit_width.to_value(u.arcsec),
                        oxide_thickness.to_value(u.nm) if oxide_thickness.unit.is_equivalent(u.nm) else oxide_thickness.to_value(u.AA),
                        c_thickness.to_value(u.nm) if c_thickness.unit.is_equivalent(u.nm) else c_thickness.to_value(u.AA),
                        vis_sl.to_value() if hasattr(vis_sl, 'to_value') else vis_sl
                    )
                    
                    all_results[param_key] = {
                        "parameters": {
                            "slit_width": slit_width,
                            "oxide_thickness": oxide_thickness,
                            "c_thickness": c_thickness,
                            "vis_sl": vis_sl,
                        },
                        "first_dn_signal_per_exp": first_dn_signal_per_exp,
                        "first_photon_signal_per_exp": first_photon_signal_per_exp,
                        "first_fit_per_exp": first_fit_per_exp,
                        "first_photon_fit_per_exp": first_photon_fit_per_exp,
                        "analysis_per_exp": analysis_per_exp,
                        "ground_truth": {
                            "fit_truth": fit_truth,
                            "v_true": v_true,
                        }
                    }

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

    # Generate descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    config_path = Path(args.config)
    config_base = config_path.stem
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Create descriptive filename based on parameter ranges
    def format_range(vals, unit_str=""):
        if len(vals) == 1:
            val = vals[0]
            if hasattr(val, 'value'):
                return f"{val.value:.3g}{unit_str}"
            else:
                return f"{val:.3g}{unit_str}"
        else:
            min_val = min(vals)
            max_val = max(vals)
            if hasattr(min_val, 'value'):
                return f"{min_val.value:.3g}-{max_val.value:.3g}{unit_str}"
            else:
                return f"{min_val:.3g}-{max_val:.3g}{unit_str}"

    slit_str = format_range(slit_widths, "as")
    oxide_str = format_range(oxide_thicknesses, "nm") 
    carbon_str = format_range(c_thicknesses, "nm")
    
    filename_parts = [
        f"instrument_response_{instrument.lower()}",
        f"slit{slit_str}",
        f"oxide{oxide_str}",
        f"carbon{carbon_str}",
        timestamp
    ]
    output_file = scratch_dir / f"{'_'.join(filename_parts)}.pkl"

    print(f"\nSaving results to {output_file}")
    with open(output_file, "wb") as f:
        dill.dump({
            "results": results,
            "config": config,
            "instrument": instrument,
            "cube_sim": cube_sim,
        }, f)

    print(f"Saved results to {output_file} ({os.path.getsize(output_file) / 1e6:.1f} MB)")

    dest_path = Path(f"run/result/instrument_response_{config_base}.pkl")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_file, dest_path)
    print(f"Copied {output_file} to {dest_path}")

    print(f"Instrument response simulation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total parameter combinations: {total_combinations}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="target cannot be converted to ICRS",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="No observer defined on WCS, SpectralCoord will be converted without any velocity frame change",
            category=UserWarning,
        )

        main()
