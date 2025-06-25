import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import glob
import dill
from instr_response import angle_to_distance

# Import plotting functions from instr_response.py
from instr_response import (
    plot_maps,
    plot_radiometric_pipeline,
    plot_velocity_std_map,
    plot_intensity_vs_vstd,
    plot_spectra,
    plot_velocity_maps,
    plot_exposure_time_map,
    plot_vstd_vs_exposure,
)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_pkl_results(pkl_file):
    with open(pkl_file, "rb") as f:
        dat = dill.load(f)
    results = dat["results"]
    config = dat["config"]
    plotting = dat["plotting"]
    cube_reb = dat["cube_reb"]
    wl_axis = dat["wl_axis"]
    wl0 = dat["wl0"]
    spt_sim = dat["spt_sim"]
    DET = dat["DET"]
    SIM = dat["SIM"]
    return results, config, plotting, cube_reb, wl_axis, wl0, spt_sim, DET, SIM

def get_plot_dir(base, contamination):
    return os.path.join(base, f"contam_{contamination}")

def plot_all_for_simulation(
    sim_name,
    results,
    plotting,
    cube_reb,
    wl_axis,
    wl0,
    spt_sim,
    DET,
    SIM,
    output_dir,
):
    for contamination, res in results.items():
        plot_dir = get_plot_dir(output_dir, contamination)
        ensure_dir(plot_dir)
        first_signal_per_exp = res["first_signal_per_exp"]
        first_fit_per_exp = res["first_fit_per_exp"]
        analysis_per_exp = res["analysis_per_exp"]

        # Per-exposure plots
        for sec, first_signals in first_signal_per_exp.items():
            first_fits = first_fit_per_exp[sec]
            analysis_res = analysis_per_exp[sec]
            sec_str = f"{sec:.1f}s"

            plot_maps(
                cube_reb,
                first_fits,
                wl_axis,
                wl0,
                plotting["minus_idx"],
                plotting["mean_idx"],
                plotting["plus_idx"],
                photon_cube=first_signals[4],
                save=os.path.join(plot_dir, f"maps_{sec_str}.png"),
                previous=None,
                save_data_path=None,
            )

            plot_radiometric_pipeline(
                signals=first_signals,
                wl_axis=wl_axis,
                idx_sim_minus=plotting["minus_idx"],
                idx_sim_mean=plotting["mean_idx"],
                idx_sim_plus=plotting["plus_idx"],
                spt_pitch_sim=spt_sim,
                spt_pitch_instr=DET.plate_scale_length,
                save=os.path.join(plot_dir, f"radiometric_pipeline_{sec_str}.png"),
            )

            plot_velocity_std_map(
                v_std_map=analysis_res["v_std"],
                save=os.path.join(plot_dir, f"vstd_{sec_str}.png"),
                x_pix_size=SIM.slit_scan_step.to(u.arcsec).value,
                y_pix_size=DET.plate_scale_angle.to(u.arcsec / u.pix).value,
                idx_minus=plotting["minus_idx"],
                idx_mean=plotting["mean_idx"],
                idx_plus=plotting["plus_idx"],
            )

            wl_pitch = (wl_axis[1] - wl_axis[0]).cgs
            intensity = (first_signals[0].sum(axis=2) * wl_pitch).value
            plot_intensity_vs_vstd(
                intensity=intensity,
                v_std=analysis_res["v_std"],
                save=os.path.join(plot_dir, f"intensity_vs_vstd_{sec_str}.png"),
                vstd_max=1e9,
                fit_intensity_min=1e2,
                idx_minus=plotting.get("minus_idx"),
                idx_mean=plotting.get("mean_idx"),
                idx_plus=plotting.get("plus_idx"),
            )

            plot_spectra(
                dn_cube=first_signals[7],
                wl_axis=wl_axis,
                idx_sim_minus=plotting["minus_idx"],
                idx_sim_mean=plotting["mean_idx"],
                idx_sim_plus=plotting["plus_idx"],
                wl0=wl0,
                fit_cube=first_fits,
                sigma_factor=plotting["sigma_factor"],
                key_pixel_colors=("mediumseagreen", "black", "deeppink"),
                save=os.path.join(plot_dir, f"spectra_dn_{sec_str}.png"),
            )

            plot_velocity_maps(
                analysis_res["v_mean"],
                analysis_res["v_std"],
                save=os.path.join(plot_dir, f"velocity_maps_{sec_str}.png"),
                x_pix_size=SIM.slit_scan_step.to_value(u.arcsec),
                y_pix_size=DET.plate_scale_angle.to_value(u.arcsec / u.pix),
                idx_minus=plotting["minus_idx"],
                idx_mean=plotting["mean_idx"],
                idx_plus=plotting["plus_idx"],
            )

        plot_exposure_time_map(
            analysis_per_exp=analysis_per_exp,
            precision_requirement=2.0,
            x_pix_size=SIM.slit_scan_step.to_value(u.arcsec),
            y_pix_size=DET.plate_scale_angle.to_value(u.arcsec / u.pix),
            save=os.path.join(plot_dir, "exposure_time_map.png"),
            cmap="viridis",
        )

        plot_vstd_vs_exposure(
            analysis_per_exp=analysis_per_exp,
            idx_minus=plotting.get("minus_idx"),
            idx_mean=plotting.get("mean_idx"),
            idx_plus=plotting.get("plus_idx"),
            save=os.path.join(plot_dir, "vstd_vs_exposure.png"),
            log_x=True,
            vstd_max=60,
        )

def main():

    # Find all *_results.pkl files in the current directory (or specify a directory)
    npz_files = sorted(glob.glob("*_results.pkl"))
    if not npz_files:
        print("No *_results.pkl files found in current directory.")
        return

    for npz_file in npz_files:
        print(f"Processing {npz_file} ...")
        results, config, plotting, cube_reb, wl_axis, wl0, spt_sim, DET, SIM = load_pkl_results(npz_file)

        sim_name = os.path.splitext(os.path.basename(npz_file))[0]
        output_dir = f"{sim_name}_plots"
        ensure_dir(output_dir)
        plot_all_for_simulation(
            sim_name,
            results,
            plotting,
            cube_reb,
            wl_axis,
            wl0,
            spt_sim,
            DET,
            SIM,
            output_dir,
        )
        print(f"Plots for {sim_name} saved in {output_dir}/")

if __name__ == "__main__":
    main()
