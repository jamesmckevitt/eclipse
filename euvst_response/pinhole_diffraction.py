"""
Pinhole diffraction effects for aluminum filter modeling.

This module calculates the diffraction patterns from pinholes in the aluminum filter,
including both EUV and visible light contributions.
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.special import j1
from ndcube import NDCube
from typing import List, Tuple


def airy_disk_pattern(r: np.ndarray, wavelength: u.Quantity, pinhole_diameter: u.Quantity, 
                     distance: u.Quantity) -> np.ndarray:
    """
    Calculate the Airy disk diffraction pattern for a circular pinhole.
    
    Parameters
    ----------
    r : np.ndarray
        Radial distances from optical axis (in detector plane) in meters
    wavelength : u.Quantity
        Wavelength of light
    pinhole_diameter : u.Quantity
        Diameter of the pinhole
    distance : u.Quantity
        Distance from pinhole to detector
        
    Returns
    -------
    np.ndarray
        Normalized intensity pattern (peak = 1.0)
    """
    # Calculate the exact sine of the diffraction angle
    # sin(theta) = r / sqrt(r^2 + distance^2)
    distance_m = distance.to(u.m).value
    sin_theta = r / np.sqrt(r**2 + distance_m**2)
    
    # Airy disk parameter
    # beta = (pi * D * sin(theta)) / lambda
    beta = (np.pi * pinhole_diameter.to(u.m).value * sin_theta) / wavelength.to(u.m).value
    
    # Avoid division by zero at center
    beta = np.where(beta == 0, 1e-10, beta)
    
    # Airy disk intensity pattern: I(beta) = (2*J1(beta)/beta)^2
    # where J1 is the first-order Bessel function
    intensity = (2 * j1(beta) / beta) ** 2
    
    return intensity

def calculate_pinhole_diffraction_pattern(
    detector_shape: Tuple[int, int],
    pixel_size: u.Quantity,
    pinhole_diameter: u.Quantity,
    pinhole_position_slit: float,
    slit_width: u.Quantity,
    plate_scale: u.Quantity,
    distance: u.Quantity,
    wavelength: u.Quantity
) -> np.ndarray:
    """
    Calculate the diffraction pattern from a single pinhole on the detector.
    
    Parameters
    ----------
    detector_shape : tuple of int
        (n_slit, n_spectral) shape of detector
    pixel_size : u.Quantity
        Physical size of detector pixels
    pinhole_diameter : u.Quantity
        Diameter of the pinhole
    pinhole_position_slit : float
        Position along slit as fraction (0.0 to 1.0)
    slit_width : u.Quantity
        Width of the slit
    plate_scale : u.Quantity
        Angular plate scale (arcsec/pixel)
    distance : u.Quantity
        Distance from pinhole to detector
    wavelength : u.Quantity
        Wavelength of light
        
    Returns
    -------
    np.ndarray
        2D diffraction pattern normalized to peak intensity of 1.0
    """
    n_slit, n_spectral = detector_shape
    
    # Create coordinate grids for detector
    slit_pixels = np.arange(n_slit)
    spectral_pixels = np.arange(n_spectral)
    
    # Convert pinhole position from slit fraction to pixel coordinate
    pinhole_pixel_slit = pinhole_position_slit * (n_slit - 1)
    
    # Calculate distances from pinhole position on detector
    # Assuming pinhole projects to center of spectral direction
    pinhole_pixel_spectral = n_spectral // 2
    
    # Create 2D coordinate arrays
    slit_grid, spectral_grid = np.meshgrid(slit_pixels, spectral_pixels, indexing='ij')
    
    # Calculate distances from pinhole center in detector plane
    dy_pixels = slit_grid - pinhole_pixel_slit
    dx_pixels = spectral_grid - pinhole_pixel_spectral
    
    # Convert to physical distances
    dy_physical = dy_pixels * pixel_size.to(u.m).value
    dx_physical = dx_pixels * pixel_size.to(u.m).value
    
    # Radial distance from pinhole center
    r_physical = np.sqrt(dx_physical**2 + dy_physical**2)
    
    # Calculate Airy disk pattern
    pattern = airy_disk_pattern(r_physical, wavelength, pinhole_diameter, distance)
    
    return pattern


def apply_euv_pinhole_diffraction(
    photon_counts: NDCube,
    det,
    sim,
    tel
) -> NDCube:
    """
    Apply EUV pinhole diffraction effects to photon counts.
    
    This adds EUV light that bypasses the aluminum filter through pinholes
    and creates diffraction patterns. This should be applied after the 
    focusing optics PSF (primary mirror + grating) since the filter is 
    positioned after these optical elements.
    
    This function correctly handles the physics by:
    1. Subtracting the filtered EUV signal in pinhole regions 
    2. Adding the unattenuated EUV signal through pinholes
    
    Parameters
    ----------
    photon_counts : NDCube
        EUV photon counts per pixel (shape: n_scan, n_slit, n_spectral)
        These should already have filter throughput applied.
    det : Detector_SWC
        Detector configuration
    sim : Simulation
        Simulation configuration containing pinhole parameters
    tel : Telescope_EUVST
        Telescope configuration (needed to calculate filter throughput)
        
    Returns
    -------
    NDCube
        Modified photon counts with EUV pinhole contributions added
    """
    if not (sim.enable_pinholes and len(sim.pinhole_sizes) > 0):
        return photon_counts  # No pinholes enabled
    
    # Get detector and data properties
    data_shape = photon_counts.data.shape  # (n_scan, n_slit, n_spectral)
    n_scan, n_slit, n_spectral = data_shape
    
    # Get rest wavelength for EUV calculations
    rest_wavelength = photon_counts.meta['rest_wav']
    
    # Calculate pixel area
    pixel_area = (det.pix_size*1*u.pix)**2
    
    # Initialize additional photon contributions
    additional_photons = np.zeros_like(photon_counts.data)
    
    # Get the wavelength axis and calculate filter throughput for EUV
    wl_axis = photon_counts.axis_world_coords(2)[0]
    
    # Calculate filter throughput at each wavelength
    filter_throughput_spectrum = np.array([tel.filter.total_throughput(wl) for wl in wl_axis])
    
    for pinhole_diameter, pinhole_position in zip(sim.pinhole_sizes, sim.pinhole_positions):
        # Calculate pinhole area
        pinhole_area = np.pi * (pinhole_diameter / 2)**2
        
        # === Physics Correction for EUV ===
        # Current photon_counts already have filter attenuation applied
        # We need to:
        # 1. Back-calculate what the unfiltered signal would be
        # 2. Apply pinhole diffraction to that unfiltered signal  
        # 3. Subtract the over-counted filtered signal in pinhole regions

        area_ratio = (pinhole_area / pixel_area).to(u.dimensionless_unscaled).value
        
        # Calculate theoretical diffraction size for validation
        # First Airy minimum: r = 1.22 * lambda * distance / diameter
        theoretical_radius = (1.22 * rest_wavelength * det.filter_distance / pinhole_diameter).to(u.m)
        theoretical_radius_pixels = (theoretical_radius / (det.pix_size*1*u.pix)).to(u.dimensionless_unscaled).value
        
        # Calculate EUV diffraction pattern
        euv_pattern = calculate_pinhole_diffraction_pattern(
            detector_shape=(n_slit, n_spectral),
            pixel_size=det.pix_size*u.pix,
            pinhole_diameter=pinhole_diameter,
            pinhole_position_slit=pinhole_position,
            slit_width=sim.slit_width,
            plate_scale=det.plate_scale_angle,
            distance=det.filter_distance,
            wavelength=rest_wavelength
        )
        
        # For EUV, the diffraction pattern is much smaller than visible light
        # Normalize the pattern to ensure total integrated intensity equals 1.0
        # This is crucial for proper photon conservation
        pattern_total = np.sum(euv_pattern)
        if pattern_total > 0:
            euv_pattern_normalized = euv_pattern / pattern_total
        else:
            euv_pattern_normalized = euv_pattern
        
        # Process each scan position
        for i in range(n_scan):
            # Current filtered signal at this scan position
            filtered_signal = photon_counts.data[i, :, :]  # Shape: (n_slit, n_spectral)
            
            # Back-calculate unfiltered signal (before filter attenuation)
            # filtered_signal = unfiltered_signal × filter_throughput
            # So: unfiltered_signal = filtered_signal / filter_throughput
            unfiltered_signal = filtered_signal / filter_throughput_spectrum[np.newaxis, :]
            
            # Calculate what would come through pinhole (unattenuated)
            # Use normalized pattern to ensure proper photon conservation
            pinhole_signal = unfiltered_signal * area_ratio * euv_pattern_normalized
            
            # Calculate what we incorrectly have from filter in pinhole regions
            # (filtered signal weighted by diffraction pattern and area ratio)
            overcounted_filtered = filtered_signal * area_ratio * euv_pattern_normalized
            
            # Net correction: add unfiltered pinhole signal, subtract overcounted filtered signal
            # This simplifies to: filtered_signal × area_ratio × pattern × (1/filter_throughput - 1)
            # Physical meaning: 
            # - unfiltered × area_ratio × pattern = total light through pinhole
            # - filtered × area_ratio × pattern = incorrectly counted filtered light
            # - difference = net additional light from pinhole
            correction = (pinhole_signal - overcounted_filtered)
            
            # Equivalent simplified form (more efficient):
            # correction = filtered_signal * area_ratio * euv_pattern * (1/filter_throughput_spectrum[np.newaxis, :] - 1)
            additional_photons[i, :, :] += correction
    
    # Create new photon counts with EUV pinhole contributions
    new_data = photon_counts.data + additional_photons
    
    return NDCube(
        data=new_data,
        wcs=photon_counts.wcs.deepcopy(),
        unit=photon_counts.unit,
        meta=photon_counts.meta,
    )