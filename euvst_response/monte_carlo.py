"""
Monte Carlo simulation functions for instrument response analysis.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import astropy.units as u
from ndcube import NDCube
from tqdm import tqdm
from .radiometric import (
    apply_exposure, sample_photon_arrivals, intensity_to_photons, add_telescope_throughput, 
    photons_to_pixel_counts, apply_focusing_optics_psf, to_electrons, add_visible_stray_light, to_dn,
    add_pinhole_visible_light
)
from .pinhole_diffraction import apply_euv_pinhole_diffraction
from .fitting import fit_cube_gauss, spectral_pixel_width, summarise_fits
from .utils import angle_to_distance, rebin_slit_offchip, _get_mpi_info


def _fit_results(label: str, fit_data: np.ndarray, failed: np.ndarray,
                 units: list, signal: NDCube, rest_wavelength: u.Quantity,
                 fit_config) -> dict:
    """Statistics of one signal's fits, and a line saying how many failed."""
    results = summarise_fits(fit_data, failed, units,
                             spectral_pixel_width(signal), rest_wavelength,
                             fit_config)
    failed_fits = results["failed_fits"]
    n_failed = np.count_nonzero(failed)
    if n_failed:
        line = (f"  {label} fits: {n_failed} of {failed.size} failed, in "
                f"{np.count_nonzero(failed_fits)} of {failed_fits.size} pixels, "
                f"and are left out of the statistics")
        no_fit = np.count_nonzero(failed_fits == results["n_iterations"])
        if no_fit:
            line += f"; no fit succeeded in {no_fit} of them"
    else:
        line = f"  {label} fits: none of {failed.size} failed"
    print(line)
    return results


def simulate_once(
    I_cube: NDCube,
    t_exp: u.Quantity,
    det,
    tel,
    sim,
    *,
    uniform_mode: bool = False,
    photon_shot_inverse_transform: bool = False,
    dark_current_inverse_transform: bool = False,
) -> Tuple[NDCube, ...]:
    """
    Run a single Monte Carlo simulation of the instrument response.
    
    Parameters
    ----------
    I_cube : NDCube
        Input intensity cube
    t_exp : u.Quantity
        Exposure time
    det : Detector_SWC or Detector_EIS
        Detector configuration
    tel : Telescope_EUVST or Telescope_EIS
        Telescope configuration
    sim : Simulation
        Simulation configuration
    uniform_mode : bool, optional
        If True the input cube is uniform along the slit, so the PSF is
        convolved in the spectral direction only.  See
        :func:`~euvst_response.radiometric.apply_focusing_optics_psf`.
        Default False.
    photon_shot_inverse_transform : bool, optional
        Use inverse-transform Poisson sampling for photon shot noise, so that
        common random numbers survive a change in photon flux.  Default False.
    dark_current_inverse_transform : bool, optional
        Use inverse-transform Poisson sampling for dark-current shot noise, so
        that common random numbers survive a change in dark-current level.
        Default False.
        
    Returns
    -------
    tuple of NDCube
        Signal cubes at each step of the radiometric pipeline:
        (intensity_exp, photons_total, photons_throughput, photons_pixels, 
         photons_focused, photon_arrivals, electrons, electrons_stray, 
         electrons_pinholes, dn)
    """
    # Apply exposure time
    intensity_exp = apply_exposure(I_cube, t_exp)
    
    # Convert to total photons
    photons_total = intensity_to_photons(intensity_exp)
    
    # Apply telescope optical throughput
    photons_throughput = add_telescope_throughput(photons_total, tel)
    
    # Convert to pixel counts
    photons_pixels = photons_to_pixel_counts(photons_throughput, det.wvl_res, det.plate_scale_length, angle_to_distance(sim.slit_width))

    # Apply focusing optics PSF (primary mirror + diffraction grating)
    if sim.psf:
        photons_focused = apply_focusing_optics_psf(
            photons_pixels, tel, det, sim, convolve_spatial=not uniform_mode,
            boundary=getattr(sim, "psf_boundary", "replicate"),
        )
    else:
        photons_focused = photons_pixels
    
    # Apply EUV pinhole diffraction effects (after focusing optics, if enabled)
    if sim.enable_pinholes and len(sim.pinhole_sizes) > 0:
        photons_euv_pinholes = apply_euv_pinhole_diffraction(photons_focused, det, sim, tel)
    else:
        photons_euv_pinholes = photons_focused

    # Every random draw below is controlled by this one flag, so a run with
    # sim.noise False returns the signal the instrument would measure on
    # average rather than one realisation of it.
    noise = getattr(sim, "noise", True)

    # Sample discrete photon arrivals (photon shot noise)
    photon_arrivals = sample_photon_arrivals(
        photons_euv_pinholes,
        photon_shot_inverse_transform=photon_shot_inverse_transform,
        noise=noise,
    )

    # Convert to electrons (detector response: QE, Fano noise, dark current, read noise)
    electrons = to_electrons(
        photon_arrivals, t_exp, det,
        dark_current_inverse_transform=dark_current_inverse_transform,
        noise=noise,
    )

    # Add visible stray light (with filter throughput)
    electrons_stray = add_visible_stray_light(electrons, t_exp, det, sim, tel,
                                              noise=noise)

    # Add visible light pinhole effects (if enabled)
    if sim.enable_pinholes and len(sim.pinhole_sizes) > 0:
        electrons_pinholes = add_pinhole_visible_light(electrons_stray, t_exp,
                                                       det, sim, tel, noise=noise)
    else:
        electrons_pinholes = electrons_stray
    
    # Convert to digital numbers
    dn = to_dn(electrons_pinholes, det)

    return (intensity_exp, photons_total, photons_throughput, photons_pixels, 
            photons_focused, photon_arrivals, electrons, electrons_stray, 
            electrons_pinholes, dn)


def monte_carlo(I_cube: NDCube, t_exp: u.Quantity, det, tel, sim, n_iter: int = 5,
                fit_config=None, offchip_bin_slit: int = 1,
                fit_signals: str = "both", uniform_mode: bool = False,
                *,
                photon_shot_inverse_transform: bool = False,
                dark_current_inverse_transform: bool = False) -> Tuple[NDCube, dict | None, NDCube, dict | None]:
    """
    Run Monte Carlo simulations and fit results.
    
    Parameters
    ----------
    I_cube : NDCube
        Input intensity cube
    t_exp : u.Quantity
        Exposure time
    det : Detector_SWC or Detector_EIS
        Detector configuration
    tel : Telescope_EUVST or Telescope_EIS
        Telescope configuration
    sim : Simulation
        Simulation configuration
    n_iter : int
        Number of Monte Carlo iterations
    fit_config : FitConfig, optional
        Multi-component Gaussian fit configuration.
    offchip_bin_slit : int
        Number of slit pixels to sum (off-chip, ground-based binning).
        Each pixel is read out independently so all noise sources are
        present per pixel before summation.  Default 1 (no binning).
    fit_signals : str
        Which signals to fit: ``"both"`` (default), ``"dn"``, or
        ``"photon"``.  Fitting is the most expensive step, so
        selecting only the signal of interest roughly halves runtime.
    uniform_mode : bool, optional
        If True the input cube is taken to be uniform-intensity mode: one scan
        position and ``offchip_bin_slit`` identical slit pixels, which
        *offchip_bin_slit* then sums back down to a single spatial pixel.  All
        MC simulations are run first and the resulting spectra are stacked so
        that fitting is parallelised over the n_iter iterations rather than
        over the spatial dimension.  Default: False.
    photon_shot_inverse_transform : bool, optional
        Use inverse-transform Poisson sampling for photon shot noise, so that
        common random numbers survive a change in photon flux.  Default False.
    dark_current_inverse_transform : bool, optional
        Use inverse-transform Poisson sampling for dark-current shot noise, so
        that common random numbers survive a change in dark-current level.
        Default False.
        
    Returns
    -------
    tuple
        (first_dn_signal, dn_fit_results, first_photon_signal, photon_fit_results)
        - first_dn_signal: First iteration DN signal (NDCube)
        - dn_fit_results: Dict of fit statistics from
          :func:`~euvst_response.fitting.summarise_fits`, or None if skipped
        - first_photon_signal: First iteration photon signal (NDCube)  
        - photon_fit_results: The same for the photon signal, or None if skipped
    """
    if fit_signals not in ("both", "dn", "photon"):
        raise ValueError(f"fit_signals must be 'both', 'dn', or 'photon', got '{fit_signals}'")

    do_dn = fit_signals in ("both", "dn")
    do_photon = fit_signals in ("both", "photon")
    rest_wavelength = I_cube.meta["rest_wav"]

    # --- MPI distribution: split iterations across ranks -----------------
    comm, rank, world_size = _get_mpi_info()
    if world_size > 1:
        base, remainder = divmod(n_iter, world_size)
        local_n_iter = base + (1 if rank < remainder else 0)
        if rank == 0:
            print(f"MPI: distributing {n_iter} MC iterations across "
                  f"{world_size} ranks ({local_n_iter} per rank)")
    else:
        local_n_iter = n_iter

    if uniform_mode:
        # Uniform intensity mode: batch all MC simulations, then fit in parallel
        first_dn_signal, first_photon_signal = None, None
        dn_data_list, photon_data_list = [], []

        show_progress = (rank == 0)
        desc = "Monte-Carlo (simulate)" if world_size == 1 else f"MC sim (rank 0/{world_size})"

        for i in tqdm(range(local_n_iter), desc=desc, unit="iter", leave=False,
                      disable=not show_progress):
            (intensity_exp, photons_total, photons_throughput, photons_pixels,
             photons_focused, photon_arrivals, electrons, electrons_stray,
             electrons_pinholes, dn) = simulate_once(
                I_cube, t_exp, det, tel, sim,
                uniform_mode=uniform_mode,
                photon_shot_inverse_transform=photon_shot_inverse_transform,
                dark_current_inverse_transform=dark_current_inverse_transform,
            )

            if i == 0 and rank == 0:
                first_dn_signal = rebin_slit_offchip(dn, offchip_bin_slit)
                first_photon_signal = rebin_slit_offchip(photon_arrivals, offchip_bin_slit)

            # Off-chip binning before batching
            dn_binned = rebin_slit_offchip(dn, offchip_bin_slit)
            photon_binned = rebin_slit_offchip(photon_arrivals, offchip_bin_slit)

            # .data shape is (n_slit, n_scan, n_lam); after binning both spatial
            # axes are length 1, so [0] leaves the one spectrum as (1, n_lam)
            if do_dn:
                dn_data_list.append(dn_binned.data[0])
            if do_photon:
                photon_data_list.append(photon_binned.data[0])

        # --- MPI gather: collect data from all ranks on root -----------
        if world_size > 1:
            if do_dn:
                all_dn_lists = comm.gather(dn_data_list, root=0)
                if rank == 0:
                    dn_data_list = [v for sublist in all_dn_lists for v in sublist]
            if do_photon:
                all_photon_lists = comm.gather(photon_data_list, root=0)
                if rank == 0:
                    photon_data_list = [v for sublist in all_photon_lists for v in sublist]

        # --- Fit (only on rank 0) --------------------------------------
        dn_fit_results = None
        photon_fit_results = None

        if rank == 0:
            if do_dn and dn_data_list:
                # Stack: each element is (1, n_lam), result is (n_iter, 1, n_lam).
                dn_stacked = np.stack(dn_data_list, axis=0)
                dn_batch = NDCube(data=dn_stacked, wcs=first_dn_signal.wcs,
                                  unit=first_dn_signal.unit)

                print(f"  Fitting {len(dn_data_list)} DN MC spectra in parallel...")
                dn_fit_values, dn_fit_units, dn_failed = fit_cube_gauss(
                    dn_batch, n_jobs=sim.ncpu, fit_config=fit_config,
                    return_failed=True)
                # Reshape from (n_iter, 1, ...) to (n_iter, 1, 1, ...)
                dn_fit_results = _fit_results(
                    "DN", dn_fit_values[:, np.newaxis, :, :],
                    dn_failed[:, np.newaxis, :], dn_fit_units,
                    first_dn_signal, rest_wavelength, fit_config)

            if do_photon and photon_data_list:
                photon_stacked = np.stack(photon_data_list, axis=0)
                photon_batch = NDCube(data=photon_stacked, wcs=first_photon_signal.wcs,
                                      unit=first_photon_signal.unit)

                print(f"  Fitting {len(photon_data_list)} photon MC spectra in parallel...")
                photon_fit_values, photon_fit_units, photon_failed = fit_cube_gauss(
                    photon_batch, n_jobs=sim.ncpu, fit_config=fit_config,
                    return_failed=True)
                photon_fit_results = _fit_results(
                    "Photon", photon_fit_values[:, np.newaxis, :, :],
                    photon_failed[:, np.newaxis, :], photon_fit_units,
                    first_photon_signal, rest_wavelength, fit_config)

    else:
        # -- Normal mode: fit each MC iteration separately ---------------
        first_dn_signal, first_photon_signal = None, None
        dn_fit_values_list, photon_fit_values_list = [], []

        show_progress = (rank == 0)
        desc = "Monte-Carlo" if world_size == 1 else f"MC (rank 0/{world_size})"

        for i in tqdm(range(local_n_iter), desc=desc, unit="iter", leave=False,
                      disable=not show_progress):
            # Simulate one run
            (intensity_exp, photons_total, photons_throughput, photons_pixels,
             photons_focused, photon_arrivals, electrons, electrons_stray,
             electrons_pinholes, dn) = simulate_once(
                I_cube, t_exp, det, tel, sim,
                uniform_mode=uniform_mode,
                photon_shot_inverse_transform=photon_shot_inverse_transform,
                dark_current_inverse_transform=dark_current_inverse_transform,
            )

            # Store first iteration signals only on rank 0 (binned, to match fit shapes)
            if i == 0 and rank == 0:
                first_dn_signal = rebin_slit_offchip(dn, offchip_bin_slit)
                first_photon_signal = rebin_slit_offchip(photon_arrivals, offchip_bin_slit)

            # Off-chip slit binning (sum already noisy pixels)
            if do_dn:
                dn_binned = rebin_slit_offchip(dn, offchip_bin_slit)
                dn_fit_values, dn_fit_units, dn_failed = fit_cube_gauss(
                    dn_binned, n_jobs=sim.ncpu, fit_config=fit_config,
                    return_failed=True)
                dn_fit_values_list.append((dn_fit_values, dn_failed))

            if do_photon:
                photon_binned = rebin_slit_offchip(photon_arrivals, offchip_bin_slit)
                photon_fit_values, photon_fit_units, photon_failed = fit_cube_gauss(
                    photon_binned, n_jobs=sim.ncpu, fit_config=fit_config,
                    return_failed=True)
                photon_fit_values_list.append((photon_fit_values, photon_failed))

        # --- MPI gather: collect fit arrays from all ranks on root -----------
        if world_size > 1:
            if do_dn:
                all_dn_lists = comm.gather(dn_fit_values_list, root=0)
                if rank == 0:
                    dn_fit_values_list = [v for sublist in all_dn_lists for v in sublist]
            if do_photon:
                all_photon_lists = comm.gather(photon_fit_values_list, root=0)
                if rank == 0:
                    photon_fit_values_list = [v for sublist in all_photon_lists for v in sublist]

        # --- Compute statistics (only meaningful on rank 0) ------------------
        dn_fit_results = None
        photon_fit_results = None

        if rank == 0:
            if do_dn and dn_fit_values_list:
                dn_fit_results = _fit_results(
                    "DN", np.stack([v for v, _ in dn_fit_values_list]),
                    np.stack([f for _, f in dn_fit_values_list]),
                    dn_fit_units, first_dn_signal, rest_wavelength, fit_config)

            if do_photon and photon_fit_values_list:
                photon_fit_results = _fit_results(
                    "Photon", np.stack([v for v, _ in photon_fit_values_list]),
                    np.stack([f for _, f in photon_fit_values_list]),
                    photon_fit_units, first_photon_signal, rest_wavelength,
                    fit_config)
    
    return first_dn_signal, dn_fit_results, first_photon_signal, photon_fit_results
