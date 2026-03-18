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
    apply_exposure, intensity_to_photons, add_telescope_throughput, 
    photons_to_pixel_counts, apply_focusing_optics_psf, add_poisson,
    to_electrons, add_visible_stray_light, to_dn,
    add_pinhole_visible_light
)
from .pinhole_diffraction import apply_euv_pinhole_diffraction
from .fitting import fit_cube_gauss
from .utils import angle_to_distance


def simulate_once(I_cube: NDCube, t_exp: u.Quantity, det, tel, sim) -> Tuple[NDCube, ...]:
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
        
    Returns
    -------
    tuple of NDCube
        Signal cubes at each step of the radiometric pipeline:
        (intensity_exp, photons_total, photons_throughput, photons_pixels, 
         photons_focused, photons_euv_pinholes, photons_noisy, electrons,
         electrons_stray, electrons_pinholes, dn)
    """
    # Apply exposure time (no noise yet — Poisson is applied later
    # once we have photon counts per detector pixel)
    intensity_exp = apply_exposure(I_cube, t_exp)
    
    # Convert to total photons
    photons_total = intensity_to_photons(intensity_exp)
    
    # Apply telescope optical throughput
    photons_throughput = add_telescope_throughput(photons_total, tel)
    
    # Convert to pixel counts
    photons_pixels = photons_to_pixel_counts(photons_throughput, det.wvl_res, det.plate_scale_length, angle_to_distance(sim.slit_width))

    # Apply focusing optics PSF (primary mirror + diffraction grating)
    if sim.psf:
        photons_focused = apply_focusing_optics_psf(photons_pixels, tel)
    else:
        photons_focused = photons_pixels
    
    # Apply EUV pinhole diffraction effects (after focusing optics, if enabled)
    if sim.enable_pinholes and len(sim.pinhole_sizes) > 0:
        photons_euv_pinholes = apply_euv_pinhole_diffraction(photons_focused, det, sim, tel)
    else:
        photons_euv_pinholes = photons_focused
    
    # Apply Poisson shot noise — this is the physically correct stage:
    # the expected photon counts per pixel have been determined by the
    # optics, so we now sample the stochastic photon arrivals.
    photons_noisy = add_poisson(photons_euv_pinholes)

    # Convert to electrons
    electrons = to_electrons(photons_noisy, t_exp, det)
    
    # Add visible stray light (with filter throughput)
    electrons_stray = add_visible_stray_light(electrons, t_exp, det, sim, tel)
    
    # Add visible light pinhole effects (if enabled)
    if sim.enable_pinholes and len(sim.pinhole_sizes) > 0:
        electrons_pinholes = add_pinhole_visible_light(electrons_stray, t_exp, det, sim, tel)
    else:
        electrons_pinholes = electrons_stray
    
    # Convert to digital numbers
    dn = to_dn(electrons_pinholes, det)

    return (intensity_exp, photons_total, photons_throughput, photons_pixels, 
            photons_focused, photons_euv_pinholes, photons_noisy, electrons,
            electrons_stray, electrons_pinholes, dn)


def monte_carlo(I_cube: NDCube, t_exp: u.Quantity, det, tel, sim, n_iter: int = 5, uniform_mode: bool = False) -> Tuple[NDCube, dict, NDCube, dict]:
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
    uniform_mode : bool, optional
        If True the input cube is assumed to be a single 1x1 spatial pixel
        (uniform-intensity mode).  All MC simulations are run first and
        the resulting spectra are stacked so that fitting is parallelised
        over the n_iter iterations rather than over the spatial dimension.
        Default: False.
        
    Returns
    -------
    tuple
        (first_dn_signal, dn_fit_results, first_photon_signal, photon_fit_results)
        - first_dn_signal: First iteration DN signal (NDCube)
        - dn_fit_results: Dict with fit data and units stored separately
        - first_photon_signal: First iteration photon signal (NDCube)  
        - photon_fit_results: Dict with fit data and units stored separately
    """
    if uniform_mode:
        first_dn_signal, first_photon_signal = None, None
        dn_data_list, photon_data_list = [], []

        for i in tqdm(range(n_iter), desc="Monte-Carlo (simulate)", unit="iter", leave=False):
            (intensity_exp, photons_total, photons_throughput, photons_pixels,
             photons_focused, photons_euv_pinholes, photons_noisy, electrons,
             electrons_stray, electrons_pinholes, dn) = simulate_once(I_cube, t_exp, det, tel, sim)

            if i == 0:
                first_dn_signal = dn
                first_photon_signal = photons_noisy

            # .data shape is (1, 1, n_lam); take first scan row -> (1, n_lam)
            dn_data_list.append(dn.data[0])
            photon_data_list.append(photons_noisy.data[0])

        # Stack: each element is (1, n_lam), result is (n_iter, 1, n_lam).
        # n_iter now sits in the "scan" axis so fit_cube_gauss parallelises
        # over the MC iterations.
        dn_stacked = np.stack(dn_data_list, axis=0)
        photon_stacked = np.stack(photon_data_list, axis=0)

        # Build batch NDCubes -- reuse the WCS from the first cube (only
        # the wavelength axis matters for fitting; spatial axes are ignored).
        dn_batch = NDCube(data=dn_stacked, wcs=first_dn_signal.wcs,
                          unit=first_dn_signal.unit)
        photon_batch = NDCube(data=photon_stacked, wcs=first_photon_signal.wcs,
                              unit=first_photon_signal.unit)

        # Single parallel fit call -- parallelised over n_iter
        print(f"  Fitting {n_iter} MC spectra in parallel...")
        dn_fit_values, dn_fit_units = fit_cube_gauss(dn_batch, n_jobs=sim.ncpu)
        photon_fit_values, photon_fit_units = fit_cube_gauss(photon_batch, n_jobs=sim.ncpu)

        # fit_cube_gauss returns (n_iter, 1, 4); reshape to (n_iter, 1, 1, 4)
        # to match the expected (n_mc, n_scan, n_slit, 4) layout.
        dn_fits_values = dn_fit_values[:, np.newaxis, :, :]
        photon_fits_values = photon_fit_values[:, np.newaxis, :, :]

    else:
        # -- Normal mode ---------------------------------------------------
        first_dn_signal, first_photon_signal = None, None
        dn_fit_values_list, photon_fit_values_list = [], []

        for i in tqdm(range(n_iter), desc="Monte-Carlo", unit="iter", leave=False):
            # Simulate one run
            (intensity_exp, photons_total, photons_throughput, photons_pixels,
             photons_focused, photons_euv_pinholes, photons_noisy, electrons,
             electrons_stray, electrons_pinholes, dn) = simulate_once(I_cube, t_exp, det, tel, sim)

            # Store first iteration signals only
            if i == 0:
                first_dn_signal = dn
                first_photon_signal = photons_noisy

            # Fit DN signal
            dn_fit_values, dn_fit_units = fit_cube_gauss(dn, n_jobs=sim.ncpu)
            dn_fit_values_list.append(dn_fit_values)

            # Fit photon signal (noisy — includes shot noise)
            photon_fit_values, photon_fit_units = fit_cube_gauss(photons_noisy, n_jobs=sim.ncpu)
            photon_fit_values_list.append(photon_fit_values)

        # Stack fit results
        dn_fits_values = np.stack(dn_fit_values_list)
        photon_fits_values = np.stack(photon_fit_values_list)
    
    # Compute statistics on stripped data
    dn_fit_results = {
        "first_fit_data": dn_fits_values[0],
        "mean_data": dn_fits_values.mean(axis=0),
        "std_data": dn_fits_values.std(axis=0),
        "units": dn_fit_units,
    }
    
    photon_fit_results = {
        "first_fit_data": photon_fits_values[0],
        "mean_data": photon_fits_values.mean(axis=0),
        "std_data": photon_fits_values.std(axis=0),
        "units": photon_fit_units,
    }
    
    return first_dn_signal, dn_fit_results, first_photon_signal, photon_fit_results
