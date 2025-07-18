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
    add_poisson, intensity_to_photons, add_effective_area, 
    photons_to_pixel_rate, to_electrons, add_stray_light, to_dn
)
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
        Signal cubes at each step of the radiometric pipeline
    """
    signal0 = add_poisson(I_cube)
    signal1 = intensity_to_photons(signal0)
    signal2 = add_effective_area(signal1, tel)
    signal3 = photons_to_pixel_rate(signal2, det.wvl_res, det.plate_scale_length, angle_to_distance(sim.slit_width))
    signal4 = signal3  # signal4 = apply_psf(signal3, tel.psf)
    signal5 = to_electrons(signal4, t_exp, det)
    signal6 = add_stray_light(signal5, t_exp, det, sim)
    signal7 = to_dn(signal6, det)

    return (signal0, signal1, signal2, signal3, signal4, signal5, signal6, signal7)


def monte_carlo(I_cube: NDCube, t_exp: u.Quantity, det, tel, sim, n_iter: int = 5) -> Tuple[np.ndarray, ...]:
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
        
    Returns
    -------
    tuple of arrays
        (dn_signals, dn_fits, photon_signals, photon_fits)
    """
    dn_signals, dn_fits, photon_signals, photon_fits = [], [], [], []
    for _ in tqdm(range(n_iter), desc="Monte-Carlo", unit="iter", leave=False):
        # Simulate one run
        signal0, signal1, signal2, signal3, signal4, signal5, signal6, signal7 = simulate_once(I_cube, t_exp, det, tel, sim)
        # signal7 is DN, signal2 is photon-rate after effective area
        dn_signals.append(signal7)
        photon_signals.append(signal2)
        # Fit DN signal
        dn_fit = fit_cube_gauss(signal7)
        dn_fits.append(dn_fit)
        # Fit photon signal
        photon_fit = fit_cube_gauss(signal2)
        photon_fits.append(photon_fit)
    # Stack results
    dn_signals = np.stack([d.data for d in dn_signals])
    dn_fits = np.stack(dn_fits)
    photon_signals = np.stack([p.data for p in photon_signals])
    photon_fits = np.stack(photon_fits)
    return dn_signals, dn_fits, photon_signals, photon_fits
