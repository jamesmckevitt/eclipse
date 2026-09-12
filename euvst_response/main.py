"""
Main execution script for instrument response simulations.
"""

from __future__ import annotations
import argparse
import os
import sys
import warnings
from itertools import product as itertools_product
from pathlib import Path
import dill
import yaml
import astropy.units as u
import gzip
import h5py

from .config import AluminiumFilter, Detector_SWC, Detector_EIS, Telescope_EUVST, Telescope_EIS, Simulation
from .data_processing import load_atmosphere, rebin_atmosphere, create_uniform_intensity_cube
from .fitting import fit_cube_gauss, FitConfig, FitComponent
from .monte_carlo import monte_carlo
from .utils import (
    parse_yaml_input, ensure_list, set_debug_mode, debug_break, debug_on_error,
    deduplicate_list, get_git_commit_id, _get_software_version,
    _parse_section, _params_to_key, _extract_config_params, _SECTION_LIST_FIELDS,
    rebin_slit_offchip,
)
import numpy as np


@debug_on_error
def main() -> None:
    """Main function for running instrument response simulations."""

    # Suppress noisy astropy WCS warnings
    for msg in [
        "target cannot be converted to ICRS",
        "target cannot be converted to ICRS, so will not be set on SpectralCoord",
        "No observer defined on WCS, SpectralCoord will be converted without any velocity frame change",
    ]:
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
    warnings.filterwarnings("ignore", module="astropy.wcs.wcsapi.fitswcs", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="YAML config file", required=True)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

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
            # Wrap the already-redirected fds rather than opening new devnull
            # handles; closefd=False prevents a ResourceWarning when these
            # objects are later garbage-collected.
            sys.stdout = open(1, "w", closefd=False)
            sys.stderr = open(2, "w", closefd=False)

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Top-level scalar settings
    instrument = config.get("instrument", "SWC").upper()
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

    # Simulation mode
    uniform_intensity_mode = "uniform_intensity" in config

    if uniform_intensity_mode:
        uniform_intensity = parse_yaml_input(config["uniform_intensity"])
        if not hasattr(uniform_intensity, "unit"):
            raise ValueError(
                "uniform_intensity must include units, e.g. '5000 erg / (s cm2 sr)'"
            )
        uniform_rest_wavelength = parse_yaml_input(config.get("rest_wavelength", "195.119 AA"))
        uniform_thermal_width = parse_yaml_input(config.get("thermal_width", "20 km/s"))
        print("UNIFORM INTENSITY MODE")
        print(f"  Intensity: {uniform_intensity}")
        print(f"  Rest wavelength: {uniform_rest_wavelength}")
        print(f"  Thermal width (1-sigma): {uniform_thermal_width}")
    else:
        synthesis_file = config.get("synthesis_file", "./run/input/synthesised_spectra.pkl")
        reference_line = config.get("reference_line", "Fe12_195.1190")
        if not Path(synthesis_file).is_file():
            raise FileNotFoundError(
                f"Synthesis file not found: {synthesis_file}. "
                "Please check the 'synthesis_file' path in your config file."
            )

    # Pinhole config (fixed paired lists, not swept)
    pinhole_sizes = []
    pinhole_positions = []
    if "pinhole_sizes" in config:
        pinhole_sizes = ensure_list(parse_yaml_input(config["pinhole_sizes"]))
    if "pinhole_positions" in config:
        pinhole_positions = ensure_list(config["pinhole_positions"])
    if pinhole_sizes and len(pinhole_sizes) != len(pinhole_positions):
        raise ValueError("pinhole_sizes and pinhole_positions must have the same length.")

    # Optional spectral positions, one per pinhole, as a fraction (0.0-1.0) of
    # the detector's spectral width.  Omit to project every pinhole to the
    # centre of the spectral window, which is what ECLIPSE always did.
    pinhole_positions_spectral = []
    if "pinhole_positions_spectral" in config:
        pinhole_positions_spectral = ensure_list(config["pinhole_positions_spectral"])
        if len(pinhole_positions_spectral) != len(pinhole_sizes):
            raise ValueError("pinhole_positions_spectral, when given, must have "
                             "the same length as pinhole_sizes.")

    # Parse config sections
    sim_fixed, sim_sweep = _parse_section(config.get("simulation", {}), "simulation")
    det_fixed, det_sweep = _parse_section(config.get("detector", {}), "detector")
    tel_fixed, tel_sweep = _parse_section(config.get("telescope", {}), "telescope")
    fil_fixed, fil_sweep = _parse_section(config.get("filter", {}), "filter")

    # Instrument-specific validation
    if instrument == "EIS":
        if config.get("filter"):
            warnings.warn(
                "EIS does not use an aluminium filter. The 'filter:' section will be ignored.",
                UserWarning,
            )
            fil_fixed, fil_sweep = {}, {}
        for key in ("microroughness_sigma",):
            if key in tel_fixed or key in tel_sweep:
                warnings.warn(
                    f"EIS does not support '{key}'. This parameter will be ignored.",
                    UserWarning,
                )
                tel_fixed.pop(key, None)
                tel_sweep.pop(key, None)
        if pinhole_sizes or sim_fixed.get("enable_pinholes") or sim_sweep.get("enable_pinholes"):
            raise ValueError("Pinhole effects are not supported for EIS.")

    # Parse fitting configuration (multi-component Gaussian)
    fit_config = None
    fitting_cfg = config.get("fitting", None)
    if fitting_cfg is not None:
        raw_components = fitting_cfg.get("components", [])
        if len(raw_components) >= 2:
            components = []
            for idx, comp_dict in enumerate(raw_components):
                if "wavelength" not in comp_dict:
                    raise ValueError(
                        f"fitting.components[{idx}] is missing required field "
                        f"'wavelength' (rest wavelength of this Gaussian component, "
                        f"e.g. 'wavelength: 195.119 angstrom')."
                    )
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
            max_iter = fitting_cfg.get("max_iter", FitConfig.max_iter)
            if not isinstance(max_iter, int) or max_iter < 1:
                raise ValueError(
                    f"fitting.max_iter must be a positive integer, got "
                    f"{max_iter!r}."
                )
            fit_config = FitConfig(components=components,
                                   primary_component=primary,
                                   constrain_positive_intensity=constrain_pos,
                                   backend=backend_override,
                                   max_iter=max_iter)
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

    # Apply defaults for required params not specified anywhere
    _sim_defaults = {
        "slit_width": 0.2 * u.arcsec,
        "expos": 1.0 * u.s,
        "vis_sl": 0.0 * u.photon / (u.s * u.cm**2),
        "psf": False,
        "enable_pinholes": False,
    }
    _det_defaults = {
        "ccd_temperature": -60 * u.Celsius,
    }
    for attr, default in _sim_defaults.items():
        if attr not in sim_fixed and attr not in sim_sweep:
            sim_fixed[attr] = default
    for attr, default in _det_defaults.items():
        if attr not in det_fixed and attr not in det_sweep:
            det_fixed[attr] = default

    # Warnings
    psf_vals = list(sim_sweep.get("psf", [sim_fixed.get("psf", False)]))
    if any(psf_vals):
        if instrument == "SWC":
            warnings.warn(
                "The SWC PSF is the modelled PSF including simulations and some microroughness "
                "measurements. Final PSF will be measured before launch.",
                UserWarning,
            )
        elif instrument == "EIS":
            warnings.warn(
                "The EIS PSF is not well understood. We use a symmetrical Gaussian kernel with "
                "a FWHM of 3 pixels from Ugarte-Urra (2016) EIS Software Note 2.",
                UserWarning,
            )

    enable_ph_vals = list(sim_sweep.get("enable_pinholes", [sim_fixed.get("enable_pinholes", False)]))
    if any(enable_ph_vals):
        warnings.warn(
            "Pinhole effects are only intended for use by the instrument team. "
            "Please contact MSSL for more information.",
            UserWarning,
        )
        if not pinhole_sizes:
            warnings.warn(
                "enable_pinholes is True but no pinhole_sizes specified. "
                "Pinhole effects will be disabled.",
                UserWarning,
            )
            if "enable_pinholes" in sim_sweep:
                sim_sweep["enable_pinholes"] = [False]
            else:
                sim_fixed["enable_pinholes"] = False

    # Build sweep dimensions: every list-valued entry in a section becomes a named sweep dimension.
    # Keys use "section.attribute" notation.
    sweep_dims = {}
    for attr, vals in sim_sweep.items():
        sweep_dims[f"simulation.{attr}"] = vals
    for attr, vals in det_sweep.items():
        sweep_dims[f"detector.{attr}"] = vals
    for attr, vals in tel_sweep.items():
        sweep_dims[f"telescope.{attr}"] = vals
    for attr, vals in fil_sweep.items():
        if instrument != "EIS":
            sweep_dims[f"filter.{attr}"] = vals

    dim_names = list(sweep_dims.keys())
    dim_values = [sweep_dims[n] for n in dim_names]
    total_combinations = 1
    for v in dim_values:
        total_combinations *= len(v)

    print(f"\nRunning {total_combinations} parameter combination(s).")
    if sweep_dims:
        print("Swept dimensions:")
        for dim, vals in sweep_dims.items():
            print(f"  {dim}: {vals}")

    # Load or create input cube
    if uniform_intensity_mode:
        cube_sim = None
        is_dynamic_mode = False
        print("\nSkipping atmosphere loading (uniform intensity mode).")
    else:
        print("\nLoading atmosphere...")
        print(f"Using '{reference_line}' as reference line for wavelength grid and metadata...")
        cube_sim, dynamic_mode_info = load_atmosphere(synthesis_file, reference_line)

        is_dynamic_mode = dynamic_mode_info.get("enabled", False)
        if is_dynamic_mode:
            print("Synthesis was done in DYNAMIC MODE (time-varying atmosphere)")
            print(f"  Slit width: {dynamic_mode_info['slit_width']}")
            print(f"  Slit rest time: {dynamic_mode_info['slit_rest_time']}")
            print(f"  Timesteps used: {len(dynamic_mode_info['available_timesteps'])}")

            synth_slit_width = dynamic_mode_info["slit_width"]
            synth_rest_time = dynamic_mode_info["slit_rest_time"]

            slit_width_vals = sweep_dims.get(
                "simulation.slit_width", [sim_fixed["slit_width"]]
            )
            expos_vals = sweep_dims.get(
                "simulation.expos", [sim_fixed["expos"]]
            )

            if len(slit_width_vals) != 1:
                raise ValueError(
                    f"Dynamic mode synthesis requires exactly one slit width. "
                    f"Config specifies {len(slit_width_vals)}: {slit_width_vals}. "
                    f"Please provide only the synthesis slit width: {synth_slit_width}"
                )
            if not np.isclose(
                slit_width_vals[0].to_value(u.arcsec),
                synth_slit_width.to_value(u.arcsec),
                rtol=1e-6,
            ):
                raise ValueError(
                    f"Slit width mismatch: synthesis was done with {synth_slit_width}, "
                    f"but config specifies {slit_width_vals[0]}."
                )
            if len(expos_vals) != 1:
                raise ValueError(
                    f"Dynamic mode synthesis requires exactly one exposure time. "
                    f"Config specifies {len(expos_vals)}: {expos_vals}. "
                    f"Please provide only the synthesis slit rest time: {synth_rest_time}"
                )
            if not np.isclose(
                expos_vals[0].to_value(u.s),
                synth_rest_time.to_value(u.s),
                rtol=1e-6,
            ):
                raise ValueError(
                    f"Exposure time mismatch: synthesis was done with {synth_rest_time}, "
                    f"but config specifies {expos_vals[0]}."
                )
            print("  Dynamic mode parameters validated successfully!")

    # Main sweep loop
    all_results = {}
    # cube_reb_cache: keyed by (slit_width_arcsec, plate_scale, wvl_res)
    cube_reb_cache = {}
    # rebin_cache: keyed by (slit_width_arcsec, plate_scale, wvl_res, offchip_bin_slit)
    rebin_cache = {}
    # Keyed by slit_width_arcsec (first match) for convenient downstream access
    cube_reb_dict = {}

    # Add offchip_bin_slit as a sweep dimension if needed
    if len(offchip_bin_slits) > 1:
        sweep_dims["offchip_bin_slit"] = offchip_bin_slits
        dim_names = list(sweep_dims.keys())
        dim_values = [sweep_dims[n] for n in dim_names]
        total_combinations = 1
        for v in dim_values:
            total_combinations *= len(v)
        print(f"\nUpdated to {total_combinations} parameter combination(s) (including offchip_bin_slit sweep).")

    product_iter = itertools_product(*dim_values) if dim_names else [()]

    for combination_idx, combo_values in enumerate(product_iter, start=1):
        combo = dict(zip(dim_names, combo_values)) if dim_names else {}

        # Extract offchip_bin_slit from combo if present
        offchip_bin_slit = combo.pop("offchip_bin_slit", offchip_bin_slits[0])

        # Merge sweep values with fixed values for this combination
        all_sim = {
            **sim_fixed,
            **{k[len("simulation."):]: v for k, v in combo.items() if k.startswith("simulation.")},
        }
        all_det = {
            **det_fixed,
            **{k[len("detector."):]: v for k, v in combo.items() if k.startswith("detector.")},
        }
        all_tel = {
            **tel_fixed,
            **{k[len("telescope."):]: v for k, v in combo.items() if k.startswith("telescope.")},
        }
        all_fil = {
            **fil_fixed,
            **{k[len("filter."):]: v for k, v in combo.items() if k.startswith("filter.")},
        }

        # Extract core simulation params
        slit_width = all_sim["slit_width"]
        expos = all_sim["expos"]
        vis_sl = all_sim.get("vis_sl", 0.0 * u.photon / (u.s * u.cm**2))
        psf = all_sim.get("psf", False)
        enable_pinholes = all_sim.get("enable_pinholes", False)

        # Build config objects
        if instrument == "SWC":
            filter_obj = AluminiumFilter(**all_fil) if all_fil else AluminiumFilter()
            tel_kwargs = {k: v for k, v in all_tel.items() if k != "filter"}
            tel_kwargs["filter"] = filter_obj
            TEL = Telescope_EUVST(**tel_kwargs)
            DET = Detector_SWC(**all_det) if all_det else Detector_SWC()
        else:
            filter_obj = None
            tel_kwargs = {k: v for k, v in all_tel.items() if k != "filter"}
            TEL = Telescope_EIS(**tel_kwargs) if tel_kwargs else Telescope_EIS()
            DET = Detector_EIS(**all_det) if all_det else Detector_EIS()

        # Two-level rebinning cache: rebin_atmosphere does not depend on offchip_bin_slit,
        # so cube_reb_cache is keyed by the sampling alone to avoid redundant rebin calls
        # when sweeping multiple binning values at fixed spatial/spectral sampling.
        sampling_key = (
            slit_width.to_value(u.arcsec),
            DET.plate_scale_angle.to_value(u.arcsec / u.pixel),
            DET.wvl_res.to_value(u.cm / u.pixel),
        )
        # In uniform-intensity mode the cube is built with one slit pixel per binning
        # factor, so that rebin_slit_offchip has independent noise realisations to sum.
        # The cube therefore does depend on offchip_bin_slit, and the key must say so.
        cube_reb_key = (*sampling_key, offchip_bin_slit) if uniform_intensity_mode else sampling_key
        rebin_cache_key = (*sampling_key, offchip_bin_slit)

        if cube_reb_key not in cube_reb_cache:
            print(
                f"\nRebinning atmosphere "
                f"(slit_width={slit_width}, "
                f"plate_scale={DET.plate_scale_angle}, "
                f"wvl_res={DET.wvl_res})..."
            )
            SIM_rebin = Simulation(
                expos=1.0 * u.s,
                n_iter=n_iter,
                slit_width=slit_width,
                ncpu=ncpu,
                instrument=instrument,
                psf=False,
            )
            if uniform_intensity_mode:
                cube_reb_cache[cube_reb_key] = create_uniform_intensity_cube(
                    total_intensity=uniform_intensity,
                    rest_wavelength=uniform_rest_wavelength,
                    thermal_width=uniform_thermal_width,
                    det=DET,
                    sim=SIM_rebin,
                    n_slit_pixels=offchip_bin_slit,
                    tel=TEL,
                )
            else:
                cube_reb_cache[cube_reb_key] = rebin_atmosphere(cube_sim, DET, SIM_rebin)

        cube_reb = cube_reb_cache[cube_reb_key]

        if rebin_cache_key not in rebin_cache:
            # Apply off-chip binning
            cube_reb_binned = rebin_slit_offchip(cube_reb, offchip_bin_slit)

            print(f"Fitting ground truth cube (offchip_bin_slit={offchip_bin_slit})...")
            fit_truth_data, fit_truth_units = fit_cube_gauss(cube_reb_binned, n_jobs=ncpu, fit_config=fit_config)
            rebin_cache[rebin_cache_key] = (cube_reb_binned, fit_truth_data, fit_truth_units)
            # Key by (slit_width_arcsec, offchip_bin_slit) so that sweeps over
            # multiple binning factors at fixed slit width all retain their cubes
            # (a single-key dict would silently keep only the first one).
            cube_reb_dict.setdefault((sampling_key[0], offchip_bin_slit), cube_reb_binned)

        cube_reb_binned, fit_truth_data, fit_truth_units = rebin_cache[rebin_cache_key]

        # Build Simulation object
        SIM = Simulation(
            expos=expos,
            n_iter=n_iter,
            slit_width=slit_width,
            ncpu=ncpu,
            instrument=instrument,
            vis_sl=vis_sl,
            psf=psf,
            enable_pinholes=enable_pinholes,
            pinhole_sizes=pinhole_sizes if enable_pinholes else [],
            pinhole_positions=pinhole_positions if enable_pinholes else [],
            pinhole_positions_spectral=(pinhole_positions_spectral
                                        if enable_pinholes else []),
        )

        # Progress output
        print(f"\n--- Combination {combination_idx}/{total_combinations} ---")
        for k, v in combo.items():
            print(f"  {k}: {v}")
        if offchip_bin_slit > 1:
            print(f"  offchip_bin_slit: {offchip_bin_slit}")
        if not combo and offchip_bin_slit == 1:
            print("  (single combination - all parameters fixed)")
        print(f"  Calculated dark current: {DET.dark_current:.2e}")
        if instrument == "SWC":
            print(f"  Microroughness sigma: {TEL.microroughness_sigma}")
        if enable_pinholes and pinhole_sizes:
            print(f"  Pinhole sizes: {pinhole_sizes}")
            print(f"  Pinhole positions: {pinhole_positions}")

        # Run Monte Carlo
        first_dn_signal, dn_fit_stats, first_photon_signal, photon_fit_stats = monte_carlo(
            cube_reb, expos, DET, TEL, SIM,
            n_iter=SIM.n_iter,
            fit_config=fit_config,
            offchip_bin_slit=offchip_bin_slit,
            fit_signals=fit_signals,
            uniform_mode=uniform_intensity_mode,
        )

        # Only store results on rank 0 (MPI-aware)
        if _mpi_rank == 0 and first_dn_signal is not None:
            # Build parameters dict from actual config objects so all fields
            # (including those using class defaults) are recorded.
            parameters = {}
            parameters.update(_extract_config_params(SIM, "simulation"))
            parameters.update(_extract_config_params(DET, "detector"))
            if instrument == "SWC":
                parameters.update(_extract_config_params(filter_obj, "filter"))
            parameters.update(_extract_config_params(TEL, "telescope"))
            # Add offchip_bin_slit to the parameters dict
            parameters["offchip_bin_slit"] = offchip_bin_slit

            param_key = _params_to_key(parameters)

            all_results[param_key] = {
                "parameters": parameters,
                "config_objects": {
                    "detector": DET,
                    "telescope": TEL,
                    "simulation": SIM,
                },
                "first_dn_signal_data": first_dn_signal.data,
                "first_dn_signal_unit": first_dn_signal.unit,
                "first_photon_signal_data": first_photon_signal.data,
                "first_photon_signal_unit": first_photon_signal.unit,
                "first_signal_wcs": first_dn_signal.wcs,
                "dn_fit_stats": dn_fit_stats,
                "photon_fit_stats": photon_fit_stats,
                "ground_truth": {
                    "fit_truth_data": fit_truth_data,
                    "fit_truth_units": fit_truth_units,
                },
            }

            del first_dn_signal, first_photon_signal, dn_fit_stats, photon_fit_stats

    # Package results (rank 0 only in MPI mode)
    if _mpi_rank == 0:
        results = {
            "all_combinations": all_results,
            # All sweep dimensions and their value lists
            "sweep_dimensions": sweep_dims,
            # All fixed (non-swept) parameter values
            "fixed_params": {
                **{f"simulation.{k}": v for k, v in sim_fixed.items()},
                **{f"detector.{k}": v for k, v in det_fixed.items()},
                **{f"telescope.{k}": v for k, v in tel_fixed.items()},
                **(
                    {f"filter.{k}": v for k, v in fil_fixed.items()}
                    if instrument == "SWC"
                    else {}
                ),
                "offchip_bin_slit": offchip_bin_slits[0] if len(offchip_bin_slits) == 1 else None,
            },
            "fit_config": fit_config,
            "fit_signals": fit_signals,
        }

        # Save
        git_commit_id = get_git_commit_id()
        software_version = _get_software_version()

        output_file = Path(f"run/result/{Path(args.config).stem}.pkl")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving results to {output_file}")
        save_data = {
            "results": results,
            "config": config,
            "instrument": instrument,
            "cube_sim": cube_sim,
            "cube_reb_dict": cube_reb_dict,
            "git_commit_id": git_commit_id,
            "software_version": software_version,
        }

        with open(output_file, "wb") as f:
            dill.dump(save_data, f)

        print(f"Saved results to {output_file} ({os.path.getsize(output_file) / 1e6:.1f} MB)")
        print(f"Software version: {software_version}  |  Git commit: {git_commit_id}")
        print(f"Instrument response simulation complete! Total combinations: {total_combinations}")


if __name__ == "__main__":
    main()
