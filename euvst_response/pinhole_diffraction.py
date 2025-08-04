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
    sim
) -> NDCube:
    """
    Apply EUV pinhole diffraction effects to photon counts.
    
    This adds EUV light that bypasses the aluminum filter through pinholes
    and creates diffraction patterns. This should be applied after the 
    focusing optics PSF (primary mirror + grating) since the filter is 
    positioned after these optical elements.
    
    Parameters
    ----------
    photon_counts : NDCube
        EUV photon counts per pixel (shape: n_scan, n_slit, n_spectral)
    det : Detector_SWC
        Detector configuration
    sim : Simulation
        Simulation configuration containing pinhole parameters
        
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
    pixel_area = det.pix_size**2
    
    # Initialize additional photon contributions
    additional_photons = np.zeros_like(photon_counts.data)
    
    for pinhole_diameter, pinhole_position in zip(sim.pinhole_sizes, sim.pinhole_positions):
        # Calculate pinhole area
        pinhole_area = np.pi * (pinhole_diameter / 2)**2
        
        # === EUV Contribution ===
        # For EUV: flux through pinhole bypasses filter completely (throughput = 1.0)
        # Scale existing EUV photon flux by ratio of pinhole area to pixel area
        area_ratio = (pinhole_area / pixel_area).to(u.dimensionless_unscaled).value
        
        # Calculate EUV diffraction pattern
        euv_pattern = calculate_pinhole_diffraction_pattern(
            detector_shape=(n_slit, n_spectral),
            pixel_size=det.pix_size,
            pinhole_diameter=pinhole_diameter,
            pinhole_position_slit=pinhole_position,
            slit_width=sim.slit_width,
            plate_scale=det.plate_scale_angle,
            distance=det.filter_distance,
            wavelength=rest_wavelength
        )
        
        # Scale EUV contribution by pinhole area and add diffraction pattern
        for i in range(n_scan):
            # Take the EUV flux at this scan position and scale by pinhole
            euv_base_flux = photon_counts.data[i, :, :] * area_ratio
            additional_photons[i, :, :] += euv_base_flux * euv_pattern
    
    # Create new photon counts with EUV pinhole contributions
    new_data = photon_counts.data + additional_photons
    
    return NDCube(
        data=new_data,
        wcs=photon_counts.wcs.deepcopy(),
        unit=photon_counts.unit,
        meta=photon_counts.meta,
    )


def apply_pinhole_effects(
    photon_counts: NDCube,
    t_exp: u.Quantity,
    det,
    sim,
    vis_sl_before_filter: u.Quantity,
    pinhole_sizes: List[u.Quantity],
    pinhole_positions: List[float],
    filter_distance: u.Quantity,
    aluminum_filter
) -> NDCube:
    """
    Apply pinhole diffraction effects to photon counts, including both EUV and visible light.
    
    Parameters
    ----------
    photon_counts : NDCube
        EUV photon counts per pixel (shape: n_scan, n_slit, n_spectral)
    t_exp : u.Quantity
        Exposure time
    det : Detector
        Detector configuration
    sim : Simulation
        Simulation configuration
    vis_sl_before_filter : u.Quantity
        Visible light flux before filter (photon/s/cm²)
    pinhole_sizes : list of u.Quantity
        Diameters of pinholes
    pinhole_positions : list of float
        Positions along slit as fractions (0.0 to 1.0)
    filter_distance : u.Quantity
        Distance from filter to detector
    aluminum_filter : AluminiumFilter
        Aluminum filter object for throughput calculations
        
    Returns
    -------
    NDCube
        Modified photon counts with pinhole contributions added
    """
    if len(pinhole_sizes) != len(pinhole_positions):
        raise ValueError("pinhole_sizes and pinhole_positions must have the same length")
    
    if len(pinhole_sizes) == 0:
        return photon_counts  # No pinholes, return unchanged
    
    # Get detector and data properties
    data_shape = photon_counts.data.shape  # (n_scan, n_slit, n_spectral)
    n_scan, n_slit, n_spectral = data_shape
    
    # Get rest wavelength for EUV calculations
    rest_wavelength = photon_counts.meta['rest_wav']
    
    # Visible light wavelength (typical)
    visible_wavelength = 600 * u.nm
    
    # Calculate pixel area and pinhole areas
    pixel_area = det.pix_size**2
    
    # Initialize additional photon contributions
    additional_photons = np.zeros_like(photon_counts.data)
    
    # Calculate visible light throughput reduction through aluminum filter
    vis_throughput_filter = aluminum_filter.visible_light_throughput()
    
    for pinhole_diameter, pinhole_position in zip(pinhole_sizes, pinhole_positions):
        # Calculate pinhole area
        pinhole_area = np.pi * (pinhole_diameter / 2)**2
        
        # === EUV Contribution ===
        # For EUV: flux through pinhole bypasses filter completely (throughput = 1.0)
        # Scale existing EUV photon flux by ratio of pinhole area to pixel area
        area_ratio = (pinhole_area / pixel_area).to(u.dimensionless_unscaled).value
        
        # Calculate EUV diffraction pattern
        euv_pattern = calculate_pinhole_diffraction_pattern(
            detector_shape=(n_slit, n_spectral),
            pixel_size=det.pix_size,
            pinhole_diameter=pinhole_diameter,
            pinhole_position_slit=pinhole_position,
            slit_width=sim.slit_width,
            plate_scale=det.plate_scale_angle,
            distance=filter_distance,
            wavelength=rest_wavelength
        )
        
        # Scale EUV contribution by pinhole area and add diffraction pattern
        for i in range(n_scan):
            # Take the EUV flux at this scan position and scale by pinhole
            euv_base_flux = photon_counts.data[i, :, :] * area_ratio
            additional_photons[i, :, :] += euv_base_flux * euv_pattern
        
        # === Visible Light Contribution ===
        # Convert visible flux from per cm² to per pixel
        vis_flux_per_pixel_before = vis_sl_before_filter * pixel_area * t_exp  # Total photons per pixel before filter
        vis_flux_through_pinhole = vis_flux_per_pixel_before * area_ratio  # Scale by pinhole area
        
        # Visible light through pinhole bypasses filter (no attenuation)
        # For comparison, normal visible flux through filter would be:
        # vis_flux_through_filter = vis_flux_per_pixel_before * vis_throughput_filter
        
        # Calculate visible diffraction pattern
        vis_pattern = calculate_pinhole_diffraction_pattern(
            detector_shape=(n_slit, n_spectral),
            pixel_size=det.pix_size,
            pinhole_diameter=pinhole_diameter,
            pinhole_position_slit=pinhole_position,
            slit_width=sim.slit_width,
            plate_scale=det.plate_scale_angle,
            distance=filter_distance,
            wavelength=visible_wavelength
        )
        
        # Add visible light contribution to all scan positions
        vis_contribution = vis_flux_through_pinhole.to(u.photon / u.pixel).value * vis_pattern
        additional_photons += vis_contribution
    
    # Create new photon counts with pinhole contributions
    new_data = photon_counts.data + additional_photons
    
    return NDCube(
        data=new_data,
        wcs=photon_counts.wcs.deepcopy(),
        unit=photon_counts.unit,
        meta=photon_counts.meta,
    )
