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


def airy_peak_fraction_per_pixel(
    pinhole_diameter: u.Quantity,
    distance: u.Quantity,
    wavelength: u.Quantity,
    pixel_size: u.Quantity,
) -> float:
    """
    Fraction of a pinhole's transmitted photons that land in the single
    brightest detector pixel.

    This is the ABSOLUTE normalisation of the Airy pattern, which
    ``calculate_pinhole_diffraction_pattern`` deliberately does not carry (it
    returns a pattern normalised to a peak of 1.0).  For a circular aperture
    of area A at distance L, the on-axis irradiance is

        E_0 = P_total * A / (lambda^2 L^2)

    (integrating (2*J1(u)/u)^2 over the plane gives 4*lambda^2*L^2/(pi*D^2),
    which recovers P_total), so the fraction of the transmitted power falling
    on one pixel of area a is ``A * a / (lambda L)^2``.

    Why this matters: a pattern normalised by its sum over the detector array
    implicitly forces every pinhole photon onto the detector.  For a small
    pinhole the Airy disc is far larger than the detector - a 1 micron hole at
    250 mm has its first minimum at 183 mm, against a detector tens of mm
    across - so most of the light misses the detector entirely and must not be
    redistributed onto it.  Use this function when absolute photon numbers
    matter, e.g. deriving a pinhole budget.

    Returns
    -------
    float
        Fraction of transmitted photons in the brightest pixel, capped at 1.0
        (the cap only binds for holes so large that the geometric image is
        smaller than a pixel, where Fraunhofer diffraction no longer applies).
    """
    area = np.pi * (pinhole_diameter.to(u.m).value / 2.0) ** 2
    pix = pixel_size.to(u.m).value
    lam = wavelength.to(u.m).value
    dist = distance.to(u.m).value
    return float(min(1.0, area * pix ** 2 / (lam * dist) ** 2))

def calculate_pinhole_diffraction_pattern(
    detector_shape: Tuple[int, int],
    pixel_size: u.Quantity,
    pinhole_diameter: u.Quantity,
    pinhole_position_slit: float,
    slit_width: u.Quantity,
    plate_scale: u.Quantity,
    distance: u.Quantity,
    wavelength: u.Quantity,
    pinhole_position_spectral: float | None = None,
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
    pinhole_position_spectral : float, optional
        Position along the SPECTRAL axis as a fraction (0.0 to 1.0) of the
        detector width.  ``None`` (the default) reproduces the previous
        behaviour of projecting every pinhole to the centre of the spectral
        window.  On a slit-scan spectrograph the spectral axis is wavelength,
        so this fraction decides which emission lines a given pinhole
        contaminates - the centre-only assumption cannot answer that.

    Returns
    -------
    np.ndarray
        2D diffraction pattern normalized to peak intensity of 1.0.  This
        carries no absolute normalisation; see ``airy_peak_fraction_per_pixel``
        when photon numbers matter.
    """
    n_slit, n_spectral = detector_shape
    
    # Create coordinate grids for detector
    slit_pixels = np.arange(n_slit)
    spectral_pixels = np.arange(n_spectral)
    
    # Convert pinhole position from slit fraction to pixel coordinate
    pinhole_pixel_slit = pinhole_position_slit * (n_slit - 1)
    
    # Calculate distances from pinhole position on detector.  Without an
    # explicit spectral position, fall back to the centre of the spectral
    # window (the historical assumption).
    if pinhole_position_spectral is None:
        pinhole_pixel_spectral = n_spectral // 2
    else:
        if not 0.0 <= pinhole_position_spectral <= 1.0:
            raise ValueError(
                "pinhole_position_spectral is a fraction of the detector "
                f"width and must lie in [0, 1], got "
                f"{pinhole_position_spectral}. Out of range it would place "
                "the pinhole off the detector, where it looks like a valid "
                "pinhole whose light merely happens to be missing."
            )
        pinhole_pixel_spectral = pinhole_position_spectral * (n_spectral - 1)
    
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
    
    # Spectral positions are optional here for the same reason as in the
    # visible path: without them every pinhole projects to the centre of the
    # spectral window, as it always did.
    spectral_positions = (list(sim.pinhole_positions_spectral)
                          or [None] * len(sim.pinhole_sizes))

    for pinhole_diameter, pinhole_position, pinhole_spectral in zip(
            sim.pinhole_sizes, sim.pinhole_positions, spectral_positions):
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
            wavelength=rest_wavelength,
            pinhole_position_spectral=pinhole_spectral,
        )
        
        # Scale the pattern by its absolute normalisation, exactly as the
        # visible path does.  Dividing by the sum over the detector array would
        # force every photon through the pinhole onto the detector: the array
        # would sum to 1 by construction whatever the geometry.  The EUV Airy
        # pattern is about thirty times smaller than the visible one at the
        # same diameter, but not small enough for that to be harmless - a
        # 1 micron hole at 195 Angstrom still has its first minimum 5.95 mm out
        # against a detector a few mm across, so most of its light misses and
        # must be lost rather than redistributed.  euv_pattern peaks at 1.0, so
        # multiplying by the peak per-pixel fraction gives the correct fraction
        # everywhere and the array sums to less than 1 when light falls off the
        # detector.
        peak_fraction = airy_peak_fraction_per_pixel(
            pinhole_diameter=pinhole_diameter,
            distance=det.filter_distance,
            wavelength=rest_wavelength,
            pixel_size=det.pix_size * u.pix,
        )
        euv_pattern_normalized = euv_pattern * peak_fraction


        # Process each scan position
        for i in range(n_scan):
            # Current filtered signal at this scan position
            filtered_signal = photon_counts.data[i, :, :]  # Shape: (n_slit, n_spectral)
            
            # Back-calculate unfiltered signal (before filter attenuation)
            # filtered_signal = unfiltered_signal * filter_throughput
            # So: unfiltered_signal = filtered_signal / filter_throughput
            unfiltered_signal = filtered_signal / filter_throughput_spectrum[np.newaxis, :]
            
            # Calculate what would come through pinhole (unattenuated).
            # unfiltered_signal * area_ratio is the light collected over the
            # pinhole's area; the scaled pattern says what fraction of it
            # reaches each pixel, and does not have to add up to all of it.
            pinhole_signal = unfiltered_signal * area_ratio * euv_pattern_normalized
            
            # Calculate what we incorrectly have from filter in pinhole regions
            # (filtered signal weighted by diffraction pattern and area ratio)
            overcounted_filtered = filtered_signal * area_ratio * euv_pattern_normalized
            
            # Net correction: add unfiltered pinhole signal, subtract overcounted filtered signal
            # This simplifies to: filtered_signal * area_ratio * pattern * (1/filter_throughput - 1)
            # Physical meaning: 
            # - unfiltered * area_ratio * pattern = total light through pinhole
            # - filtered * area_ratio * pattern = incorrectly counted filtered light
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