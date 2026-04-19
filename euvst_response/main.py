"""
Main execution script for instrument response simulations.
"""

from __future__ import annotations
import argparse
import os
import sys
import shutil
import warnings
from datetime import datetime
from pathlib import Path
import dill
import yaml
import astropy.units as u
from tqdm import tqdm
import gzip
import h5py

from .config import AluminiumFilter, Detector_SWC, Detector_EIS, Telescope_EUVST, Telescope_EIS, Simulation
from .data_processing import load_atmosphere, rebin_atmosphere
from .fitting import fit_cube_gauss, FitConfig, FitComponent
from .monte_carlo import monte_carlo
from .utils import parse_yaml_input, ensure_list, deduplicate_list, set_debug_mode, debug_break, debug_on_error, rebin_slit_offchip
import numpy as np


@debug_on_error
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
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Set debug mode globally
    set_debug_mode(args.debug)
    if args.debug:
        print("Debug mode enabled - will break to IPython on errors")

    # MPI auto-detection: when launched via srun/mpirun with multiple tasks,
    # Monte Carlo iterations are distributed across ranks automatically.
    from .utils import _get_mpi_info
    _comm, _mpi_rank, _mpi_size = _get_mpi_info()
    if _mpi_size > 1:
        # Intel MPI pins each rank to cores, breaking joblib/loky.
        # Reset affinity to the full SLURM allocation.
        _slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if _slurm_cpus is not None:
            os.sched_setaffinity(0, range(int(_slurm_cpus)))

        if _mpi_rank == 0:
            print(f"MPI distributed mode: {_mpi_size} processes "
                  f"(MC iterations will be split across ranks)")
        else:
            # Silence stdout and stderr on non-root ranks to avoid duplicated
            # output. Redirect at the OS file-descriptor level, not
            # just the Python objects, because SLURM captures fd 1/2 directly
            # and tqdm can bypass the Python sys.stderr object.
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_devnull_fd, 1)  # redirect fd 1 (stdout)
            os.dup2(_devnull_fd, 2)  # redirect fd 2 (stderr)
            os.close(_devnull_fd)
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set up instrument, detector, telescope, simulation from config
    instrument = config.get("instrument", "SWC").upper()
    psf_settings = ensure_list(config.get("psf", [False]))  # Handle PSF as a list
    
    # Synthesis file path - allow user to specify where the synthesised_spectra.pkl file is located
    synthesis_file = config.get("synthesis_file", "./run/input/synthesised_spectra.pkl")
    
    # Reference line for wavelength grid and metadata
    reference_line = config.get("reference_line", "Fe12_195.1190")
    
    # Check if synthesis file exists
    synthesis_path = Path(synthesis_file)
    if not synthesis_path.is_file():
        raise FileNotFoundError(f"Synthesis file not found: {synthesis_file}. "
                              f"Please check the 'synthesis_file' path in your config file.")
    psf_settings = deduplicate_list(psf_settings, "psf")  # Remove duplicates
    n_iter = config.get("n_iter", 25)
    ncpu = config.get("ncpu", -1)

    # In MPI mode, if a ncpu value specified, cap it to
    # the CPUs available to this rank so joblib doesn't oversubscribe.
    # Leave ncpu=-1 alone - joblib handles it natively via the OS affinity
    # mask (which I_MPI_PIN_DOMAIN=auto sets correctly).
    if _mpi_size > 1:
        if ncpu != -1:
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm_cpus is not None:
                available_cpus = int(slurm_cpus)
            else:
                available_cpus = os.cpu_count() or 1
            if ncpu > available_cpus:
                ncpu = available_cpus
        if _mpi_rank == 0:
            print(f"MPI: ncpu={ncpu} per rank")

    # Print PSF warnings for any True values in the list
    if any(psf_settings):
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
    slit_widths = deduplicate_list(slit_widths, "slit_width")
    
    # Handle instrument-specific parameters
    if instrument == "SWC":
        # SWC requires oxide, carbon, and aluminium thickness parameters
        oxide_thicknesses = ensure_list(parse_yaml_input(config.get("oxide_thickness", ['95 angstrom'])))
        oxide_thicknesses = deduplicate_list(oxide_thicknesses, "oxide_thickness")
        c_thicknesses = ensure_list(parse_yaml_input(config.get("c_thickness", ['0 angstrom'])))
        c_thicknesses = deduplicate_list(c_thicknesses, "c_thickness")
        aluminium_thicknesses = ensure_list(parse_yaml_input(config.get("aluminium_thickness", ['1485 angstrom'])))
        aluminium_thicknesses = deduplicate_list(aluminium_thicknesses, "aluminium_thickness")
        microroughness_sigma = parse_yaml_input(config.get("microroughness_sigma", "0.3 nm"))
    elif instrument == "EIS":
        # EIS doesn't use these parameters - check they weren't specified
        if "oxide_thickness" in config:
            raise ValueError("EIS does not support oxide thickness parameter. Remove 'oxide_thickness' from configuration.")
        if "c_thickness" in config:
            raise ValueError("EIS does not support carbon thickness parameter. Remove 'c_thickness' from configuration.")
        if "aluminium_thickness" in config:
            raise ValueError("EIS does not support custom aluminium thickness parameter. Remove 'aluminium_thickness' from configuration.")
        if "microroughness_sigma" in config:
            raise ValueError("EIS does not support microroughness_sigma parameter. Remove 'microroughness_sigma' from configuration.")
        
        # Set defaults for EIS (these won't be used but are needed for parameter combination logic)
        oxide_thicknesses = [0 * u.nm]
        c_thicknesses = [0 * u.nm] 
        aluminium_thicknesses = [1500 * u.angstrom]
    
    ccd_temperatures = ensure_list(parse_yaml_input(config.get("ccd_temperature", ['-60 Celsius'])))  # Temperature in Celsius
    ccd_temperatures = deduplicate_list(ccd_temperatures, "ccd_temperature")
    vis_sl_vals = ensure_list(parse_yaml_input(config.get("vis_sl", ['0 photon / (s * cm^2)'])))
    vis_sl_vals = deduplicate_list(vis_sl_vals, "vis_sl")
    exposures = ensure_list(parse_yaml_input(config.get("expos", ['1 s'])))
    exposures = deduplicate_list(exposures, "expos")

    # Parse pinhole parameters
    enable_pinholes_vals = ensure_list(config.get("enable_pinholes", [False]))
    enable_pinholes_vals = deduplicate_list(enable_pinholes_vals, "enable_pinholes")

    # if pinholes are enabled, raise warning that this is only intended to be used by the instrument team
    if any(enable_pinholes_vals):
        warnings.warn(
            "Pinhole effects are only intended for use by the instrument team. "
            "Please contact MSSL for more information.",
            UserWarning
        )
    
    # Parse pinhole sizes and positions (these are not swept - used together for multiple pinholes per simulation)
    pinhole_sizes = []
    pinhole_positions = []
    
    if "pinhole_sizes" in config:
        pinhole_sizes = ensure_list(parse_yaml_input(config["pinhole_sizes"]))
    if "pinhole_positions" in config:
        pinhole_positions = ensure_list(config["pinhole_positions"])
    
    # Validate pinhole configuration
    if len(pinhole_sizes) != len(pinhole_positions) and len(pinhole_sizes) > 0:
        raise ValueError("pinhole_sizes and pinhole_positions must have the same length")
    
    # Check if pinholes are enabled with EIS instrument
    if instrument == "EIS" and any(enable_pinholes_vals):
        raise ValueError("Pinhole effects are not supported for EIS instrument. Pinholes are only available for SWC.")
    
    # Check if pinhole parameters are specified for EIS
    if instrument == "EIS":
        if "pinhole_sizes" in config:
            raise ValueError("EIS does not support pinhole_sizes parameter. Remove 'pinhole_sizes' from configuration.")
        if "pinhole_positions" in config:
            raise ValueError("EIS does not support pinhole_positions parameter. Remove 'pinhole_positions' from configuration.")
        if "enable_pinholes" in config and any(config.get("enable_pinholes", [False])):
            raise ValueError("EIS does not support enable_pinholes parameter. Remove 'enable_pinholes' from configuration.")
    
    if any(enable_pinholes_vals) and len(pinhole_sizes) == 0:
        warnings.warn("enable_pinholes is True but no pinhole_sizes specified. Pinhole effects will be disabled.")
        enable_pinholes_vals = [False]

    # Parse fitting configuration (multi-component Gaussian)
    fit_config = None
    fitting_cfg = config.get("fitting", None)
    if fitting_cfg is not None:
        raw_components = fitting_cfg.get("components", [])
        if len(raw_components) >= 2:
            components = []
            for comp_dict in raw_components:
                wl = parse_yaml_input(comp_dict["wavelength"])
                tie_center = comp_dict.get("tie_center", None)
                tie_width = comp_dict.get("tie_width", None)
                amp_gt = comp_dict.get("amplitude_greater_than", None)
                components.append(FitComponent(wavelength=wl,
                                               tie_center=tie_center,
                                               tie_width=tie_width,
                                               amplitude_greater_than=amp_gt))
            primary = fitting_cfg.get("primary_component", 0)
            constrain_pos = fitting_cfg.get("constrain_positive_intensity", False)
            backend_override = fitting_cfg.get("backend", None)
            if backend_override is not None and backend_override not in ("scipy", "mpfit"):
                raise ValueError(
                    f"Unknown fitting backend '{backend_override}'. "
                    f"Supported values: 'scipy', 'mpfit', or omit for auto."
                )
            fit_config = FitConfig(components=components,
                                   primary_component=primary,
                                   constrain_positive_intensity=constrain_pos,
                                   backend=backend_override)
            if backend_override == "mpfit":
                backend_label = "mpfit (forced)"
            elif backend_override == "scipy":
                backend_label = "scipy (forced)"
            else:
                backend_label = "scipy (auto)"
            print(f"Multi-component fitting enabled: {fit_config.n_components} components "
                  f"(primary={primary}, {fit_config.n_full_params} params, "
                  f"backend={backend_label})")

    # Parse off-chip slit binning (ground-based spatial binning along the slit)
    offchip_bin_slits = ensure_list(config.get("offchip_bin_slit", [1]))

    # Parse which signals to fit (default: both DN and photon)
    fit_signals = config.get("fit_signals", "both")
    if fit_signals not in ("both", "dn", "photon"):
        raise ValueError(
            f"Unknown fit_signals value '{fit_signals}'. "
            f"Supported values: 'both', 'dn', 'photon'."
        )
    if fit_signals != "both":
        skipped = "photon" if fit_signals == "dn" else "dn"
        print(f"Fitting only '{fit_signals}' signal (skipping '{skipped}')")
    # ensure_list wraps scalars in a list; values are plain ints (no units)
    offchip_bin_slits = [int(v) for v in offchip_bin_slits]
    offchip_bin_slits = deduplicate_list(offchip_bin_slits, "offchip_bin_slit")
    if any(b > 1 for b in offchip_bin_slits):
        print(f"Off-chip slit binning values: {offchip_bin_slits} "
              f"(ground-based, all noise per pixel before summation)")

    # Build slit_width -> [offchip_bin_slit, ...] mapping
    # If slit_bin_pairs is provided, use explicit pairing instead of Cartesian product
    from collections import OrderedDict
    slit_bin_pairs_cfg = config.get("slit_bin_pairs", None)
    if slit_bin_pairs_cfg is not None:
        slit_bin_grouped = OrderedDict()
        for pair in slit_bin_pairs_cfg:
            sw = parse_yaml_input(pair["slit_width"])
            ob = int(pair.get("offchip_bin_slit", 1))
            slit_bin_grouped.setdefault(sw, []).append(ob)
        # Override slit_widths and offchip_bin_slits for parameter_ranges
        slit_widths = list(slit_bin_grouped.keys())
        offchip_bin_slits = sorted(set(ob for obs in slit_bin_grouped.values() for ob in obs))
        print(f"Using explicit slit/bin pairs: {[(sw, obs) for sw, obs in slit_bin_grouped.items()]}")
    else:
        slit_bin_grouped = OrderedDict((sw, list(offchip_bin_slits)) for sw in slit_widths)

    # Load synthetic atmosphere cube
    print("Loading atmosphere...")
    print(f"Using '{reference_line}' as reference line for wavelength grid and metadata...")
    cube_sim, dynamic_mode_info = load_atmosphere(synthesis_file, reference_line)
    
    # Check for dynamic mode and validate parameters
    is_dynamic_mode = dynamic_mode_info.get("enabled", False)
    if is_dynamic_mode:
        print("Synthesis was done in DYNAMIC MODE (time-varying atmosphere)")
        print(f"  Slit width: {dynamic_mode_info['slit_width']}")
        print(f"  Slit rest time: {dynamic_mode_info['slit_rest_time']}")
        print(f"  Timesteps used: {len(dynamic_mode_info['available_timesteps'])}")
        
        # Check that exactly one slit width is provided for dynamic mode
        synth_slit_width = dynamic_mode_info["slit_width"]
        if len(slit_widths) != 1:
            raise ValueError(
                f"Dynamic mode synthesis requires exactly one slit width. "
                f"Config specifies {len(slit_widths)} slit widths: {slit_widths}. "
                f"Please provide only the synthesis slit width: {synth_slit_width}"
            )
        
        # Validate that the single slit width matches synthesis
        if not np.isclose(slit_widths[0].to_value(u.arcsec), synth_slit_width.to_value(u.arcsec), rtol=1e-6):
            raise ValueError(
                f"Slit width mismatch: synthesis was done with {synth_slit_width}, "
                f"but config specifies {slit_widths[0]}. "
                f"For dynamic mode synthesis, the slit width must match exactly: {synth_slit_width}"
            )
        
        # Check that exactly one exposure time is provided for dynamic mode
        synth_rest_time = dynamic_mode_info["slit_rest_time"]
        if len(exposures) != 1:
            raise ValueError(
                f"Dynamic mode synthesis requires exactly one exposure time. "
                f"Config specifies {len(exposures)} exposure times: {exposures}. "
                f"Please provide only the synthesis slit rest time: {synth_rest_time}"
            )
        
        # Validate that the single exposure time matches synthesis
        if not np.isclose(exposures[0].to_value(u.s), synth_rest_time.to_value(u.s), rtol=1e-6):
            raise ValueError(
                f"Exposure time mismatch: synthesis was done with {synth_rest_time}, "
                f"but config specifies {exposures[0]}. "
                f"For dynamic mode synthesis, the exposure time must match exactly: {synth_rest_time}"
            )
        
        print("  Dynamic mode parameters validated successfully!")
    
    # Set up base detector configuration (doesn't change with parameters)
    if instrument == "SWC":
        DET = Detector_SWC()
    elif instrument == "EIS":
        DET = Detector_EIS()
    else:
        raise ValueError(f"Unknown instrument: {instrument}")

    # Create results structure for all parameter combinations
    all_results = {}
    
    # Dictionary to store rebinned cubes for each slit width
    cube_reb_dict = {}

    # Loop over all parameter combinations
    n_slit_bin_pairs = sum(len(obs) for obs in slit_bin_grouped.values())
    total_combinations = n_slit_bin_pairs * len(oxide_thicknesses) * len(c_thicknesses) * len(aluminium_thicknesses) * len(ccd_temperatures) * len(vis_sl_vals) * len(exposures) * len(psf_settings) * len(enable_pinholes_vals)
    print(f"Running {total_combinations} parameter combinations...")
    
    combination_idx = 0
    for slit_width, bin_list in slit_bin_grouped.items():
        # Rebin atmosphere only when slit width changes (expensive operation)
        print(f"\nRebinning atmosphere cube for slit width {slit_width}...")
        SIM_temp = Simulation(
            expos=1.0 * u.s,  # Temporary value for rebinning
            n_iter=n_iter,
            slit_width=slit_width,
            ncpu=ncpu,
            instrument=instrument,
            psf=False,  # Use False for rebinning
        )
        cube_reb = rebin_atmosphere(cube_sim, DET, SIM_temp)
        
        for offchip_bin_slit in bin_list:
            # Apply off-chip binning to the rebinned cube and store
            cube_reb_binned = rebin_slit_offchip(cube_reb, offchip_bin_slit)
            slit_width_key = slit_width.to_value(u.arcsec)
            cube_reb_dict[(slit_width_key, offchip_bin_slit)] = cube_reb_binned
            
            print(f"Fitting ground truth cube (offchip_bin_slit={offchip_bin_slit})...")
            fit_truth_data, fit_truth_units = fit_cube_gauss(cube_reb_binned, n_jobs=ncpu, fit_config=fit_config)
        
            for oxide_thickness in oxide_thicknesses:
                for c_thickness in c_thicknesses:
                    for aluminium_thickness in aluminium_thicknesses:
                        for ccd_temperature in ccd_temperatures:
                            for vis_sl in vis_sl_vals:
                                for exposure in exposures:
                                    for psf in psf_settings:
                                        for enable_pinholes in enable_pinholes_vals:
                                            combination_idx += 1
                                            print(f"--- Combination {combination_idx}/{total_combinations} ---")
                                            print(f"Slit width: {slit_width}")
                                            print(f"Off-chip slit binning: {offchip_bin_slit}")
                                            print(f"Oxide thickness: {oxide_thickness}")
                                            print(f"Carbon thickness: {c_thickness}")
                                            print(f"Aluminium thickness: {aluminium_thickness}")
                                            print(f"CCD temperature: {ccd_temperature}")
                                            print(f"Visible stray light (before filter): {vis_sl}")
                                            print(f"Exposure time: {exposure}")
                                            print(f"PSF enabled: {psf}")
                                            print(f"Pinhole effects enabled: {enable_pinholes}")
                                            if enable_pinholes and len(pinhole_sizes) > 0:
                                                print(f"Pinhole sizes: {pinhole_sizes}")
                                                print(f"Pinhole positions: {pinhole_positions}")
                                            
                                            # Set up telescope configuration for this combination
                                            if instrument == "SWC":
                                                filter_obj = AluminiumFilter(
                                                    oxide_thickness=oxide_thickness,
                                                    c_thickness=c_thickness,
                                                    al_thickness=aluminium_thickness,
                                                )
                                                print(f"Microroughness sigma: {microroughness_sigma}")
                                                TEL = Telescope_EUVST(filter=filter_obj, microroughness_sigma=microroughness_sigma)
                                            elif instrument == "EIS":
                                                TEL = Telescope_EIS()
                                                # EIS uses fixed filter configuration - no custom parameters needed
                                            else:
                                                raise ValueError(f"Unknown instrument: {instrument}")

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
                                                enable_pinholes=enable_pinholes,
                                                pinhole_sizes=pinhole_sizes if enable_pinholes else [],
                                                pinhole_positions=pinhole_positions if enable_pinholes else [],
                                            )

                                            # Run Monte Carlo for this single parameter combination
                                            first_dn_signal, dn_fit_stats, first_photon_signal, photon_fit_stats = monte_carlo(
                                                cube_reb, exposure, DET, TEL, SIM, n_iter=SIM.n_iter,
                                                fit_config=fit_config,
                                                offchip_bin_slit=offchip_bin_slit,
                                                fit_signals=fit_signals
                                            )

                                            # Store results for this parameter combination
                                            # (only rank 0 has gathered statistics in MPI mode)
                                            if first_dn_signal is not None:
                                                sec = exposure.to_value(u.s)
                                                param_key = (
                                                    slit_width.to_value(u.arcsec),
                                                    oxide_thickness.to_value(u.nm) if oxide_thickness.unit.is_equivalent(u.nm) else oxide_thickness.to_value(u.AA),
                                                    c_thickness.to_value(u.nm) if c_thickness.unit.is_equivalent(u.nm) else c_thickness.to_value(u.AA),
                                                    aluminium_thickness.to_value(u.AA),
                                                    ccd_temperature.to_value(u.Celsius,equivalencies=u.temperature()),
                                                    vis_sl.to_value(u.photon / (u.s * u.cm**2)),
                                                    sec,
                                                    psf,
                                                    enable_pinholes,
                                                    offchip_bin_slit,
                                                )
                                                
                                                # Store fit_truth data and units separately
                                                all_results[param_key] = {
                                                    "parameters": {
                                                        "slit_width": slit_width,
                                                        "oxide_thickness": oxide_thickness,
                                                        "c_thickness": c_thickness,
                                                        "aluminium_thickness": aluminium_thickness,
                                                        "ccd_temperature": ccd_temperature,
                                                        "vis_sl": vis_sl,
                                                        "exposure": exposure,
                                                        "psf": psf,
                                                        "enable_pinholes": enable_pinholes,
                                                        "pinhole_sizes": pinhole_sizes if enable_pinholes else [],
                                                        "pinhole_positions": pinhole_positions if enable_pinholes else [],
                                                        "offchip_bin_slit": offchip_bin_slit,
                                                    },
                                                    # Store signal data and units separately
                                                    "first_dn_signal_data": first_dn_signal.data,
                                                    "first_dn_signal_unit": first_dn_signal.unit,
                                                    "first_photon_signal_data": first_photon_signal.data,
                                                    "first_photon_signal_unit": first_photon_signal.unit,
                                                    "first_signal_wcs": first_dn_signal.wcs,
                                                    "dn_fit_stats": dn_fit_stats,
                                                    "photon_fit_stats": photon_fit_stats,
                                                    "fit_signals": fit_signals,
                                                    "ground_truth": {
                                                        "fit_truth_data": fit_truth_data,
                                                        "fit_truth_units": fit_truth_units,
                                                    }
                                                }
                                            
                                            # Clean up memory
                                            del first_dn_signal, first_photon_signal, dn_fit_stats, photon_fit_stats

    # Prepare final results structure and save (rank 0 only in MPI mode)
    if _mpi_rank == 0:
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
                "psf_settings": psf_settings,
                "enable_pinholes_vals": enable_pinholes_vals,
                "pinhole_sizes": pinhole_sizes,
                "pinhole_positions": pinhole_positions,
                "offchip_bin_slits": offchip_bin_slits,
            }
        }

        # Generate output filename based on config file
        config_path = Path(args.config)
        config_base = config_path.stem
        output_file = Path(f"run/result/{config_base}.pkl")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving results to {output_file}")
        
        # Prepare the data to save
        save_data = {
            "results": results,
            "config": config,
            "instrument": instrument,
            "cube_sim": cube_sim,
            "cube_reb_dict": cube_reb_dict,
            "fit_config": fit_config,
            "fit_signals": fit_signals,
        }
        
        with open(output_file, "wb") as f:
            dill.dump(save_data, f)

        print(f"Saved results to {output_file} ({os.path.getsize(output_file) / 1e6:.1f} MB)")

        print(f"Instrument response simulation complete!")
        print(f"Total parameter combinations: {total_combinations}")


if __name__ == "__main__":
        main()
