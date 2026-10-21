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
    Airy pattern of a circular pinhole on a screen some distance behind it.

    Parameters
    ----------
    r : np.ndarray
        Distances from the pattern centre on the screen, in metres.
    wavelength : u.Quantity
        Wavelength of the light.
    pinhole_diameter : u.Quantity
        Pinhole diameter.
    distance : u.Quantity
        Distance from the pinhole to the screen.

    Returns
    -------
    np.ndarray
        Brightness at each distance, 1 at the centre.
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
    Fraction of a pinhole's transmitted photons in the brightest detector pixel.

    This is the absolute normalisation of the Airy pattern, which
    `calculate_pinhole_diffraction_pattern` leaves out (its pattern peaks at
    1.0). For a circular aperture of area A at distance L, the on-axis
    irradiance is

        E_0 = P_total * A / (lambda^2 L^2)

    (integrating (2*J1(u)/u)^2 over the plane gives 4*lambda^2*L^2/(pi*D^2),
    which recovers P_total), so the fraction of the transmitted power on one
    pixel of area a is `A * a / (lambda L)^2`.

    Normalising the pattern by its sum over the detector would force every
    pinhole photon onto the detector. For a small pinhole the Airy disc is
    far larger than the detector: a 1 micron hole at 250 mm has its first
    minimum at 183 mm, against a detector tens of mm across, so most of the
    light misses the detector and must not be redistributed onto it. Use
    this function when absolute photon numbers matter, e.g. for a pinhole
    budget.

    Returns
    -------
    float
        Fraction of transmitted photons in the brightest pixel, capped at
        1.0. The cap only binds for holes so large that the geometric image
        is smaller than a pixel, where Fraunhofer diffraction no longer
        applies.
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
        Position along the spectral axis as a fraction (0.0 to 1.0) of the
        detector width. The default, None, puts every pinhole at the centre
        of the spectral window. On a slit-scan spectrograph the spectral
        axis is wavelength, so this sets which emission lines a pinhole
        contaminates.

    Returns
    -------
    np.ndarray
        2D diffraction pattern normalized to a peak of 1.0, with no absolute
        normalisation; see `airy_peak_fraction_per_pixel` when photon
        numbers matter.
    """
    n_slit, n_spectral = detector_shape
    
    # Create coordinate grids for detector
    slit_pixels = np.arange(n_slit)
    spectral_pixels = np.arange(n_spectral)
    
    # Convert pinhole position from slit fraction to pixel coordinate
    pinhole_pixel_slit = pinhole_position_slit * (n_slit - 1)
    
    # Pinhole position on the detector. Without a spectral position, use the
    # centre of the spectral window.
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


def _pinholes_on(sim) -> bool:
    """Whether `sim` has pinholes enabled and at least one given."""
    return bool(sim.enable_pinholes and len(sim.pinhole_sizes) > 0)


def _filter_transmission(tel, wavelength: u.Quantity) -> np.ndarray:
    """
    Share of the EUV the filter passes at each `wavelength`, for the pinholes.

    The pinholes work out the light the filter blocks from the light it
    passes, which cannot be done where it passes none. A filter that passes
    no EUV at a wavelength its tables cover is therefore refused.
    """
    wavelength = u.Quantity(wavelength)
    passed = tel.filter.total_throughput(wavelength).to_value(u.dimensionless_unscaled)
    opaque = np.isfinite(passed) & (passed <= 0)
    if opaque.any():
        where = np.broadcast_to(wavelength, passed.shape, subok=True)[opaque].to(u.AA)
        raise ValueError(
            f"The filter passes no EUV from {where.min():.3f} to {where.max():.3f}, so the "
            f"light the pinholes let through there cannot be worked out from the light it "
            f"passes. Model the pinholes with a filter that passes some EUV at every "
            f"wavelength.")
    return passed


def _with_filter_transmission(photon_counts: NDCube, tel) -> NDCube:
    """
    `photon_counts` with the `meta` that :func:`apply_euv_pinhole_diffraction`
    needs, for photons counted at each pixel's own wavelength: the share of
    their light the filter passed there, and the light it blocked, at the
    same wavelength.
    """
    meta = dict(photon_counts.meta or {})
    own = photon_counts.axis_world_coords_values(2)[0]
    shape = photon_counts.data.shape
    meta["photon_filter_transmission"] = np.broadcast_to(_filter_transmission(tel, own), shape)
    meta["blocked_photon_wavelength"] = meta.get("photon_wavelength",
                                                 np.broadcast_to(own, shape, subok=True))
    meta["blocked_photon_rms_wavelength"] = meta.get("photon_rms_wavelength",
                                                     meta["blocked_photon_wavelength"])
    return NDCube(photon_counts.data, wcs=photon_counts.wcs, unit=photon_counts.unit, meta=meta)


def apply_euv_pinhole_diffraction(
    photon_counts: NDCube,
    det,
    sim,
    tel
) -> NDCube:
    """
    Add the unattenuated EUV light that passes through pinholes in the filter.

    At each pinhole the filtered light is replaced by the unattenuated light
    through the pinhole, spread in its diffraction pattern. The filter is
    behind the mirror and the grating, so this comes after the PSF. It does
    nothing unless `sim.enable_pinholes` is set and there are pinholes.

    The light the filter blocked is worked out at each photon's own
    wavelength, from `photon_filter_transmission` in the cube's `meta`, the
    share of the photons' light the filter passed, which `rebin_atmosphere`
    and `rebin_spectra` set with the pinholes on. Without it, each pixel's
    photons are taken to have passed the filter at the pixel's own
    wavelength, as photons counted there have. The pinholes' photons have
    the energies of the blocked light, and the wavelengths of the mean and
    RMS photon energies in `meta` include them.

    Parameters
    ----------
    photon_counts : NDCube
        EUV photons in each pixel, after the filter and the PSF.
    det : Detector_SWC
        EUVST-SW detector, whose `filter_distance` sets the diffraction.
    sim : Simulation
        Simulation, with the pinholes.
    tel : Telescope_EUVST
        Telescope, whose filter transmission is taken out where the pinholes
        are.

    Returns
    -------
    NDCube
        Photons in each pixel, including the pinholes' light.
    """
    if not (sim.enable_pinholes and len(sim.pinhole_sizes) > 0):
        return photon_counts  # No pinholes enabled
    
    # Get detector and data properties
    data_shape = photon_counts.data.shape  # (n_slit, n_scan, n_spectral)
    n_slit, n_scan, n_spectral = data_shape
    
    # Get rest wavelength for EUV calculations
    rest_wavelength = photon_counts.meta['rest_wav']
    
    # Calculate pixel area
    pixel_area = (det.pix_size*1*u.pix)**2
    
    # Initialize additional photon contributions
    additional_photons = np.zeros_like(photon_counts.data)
    
    # Light the filter blocked from each pixel's photons, at each photon's
    # own wavelength.
    counts = (photon_counts if "photon_filter_transmission" in (photon_counts.meta or {})
              else _with_filter_transmission(photon_counts, tel))
    data = np.asarray(photon_counts.data, dtype=float)
    passed = np.asarray(counts.meta["photon_filter_transmission"], dtype=float)
    blocked = data * np.divide(1.0 - passed, passed, out=np.zeros(data.shape), where=passed > 0)

    # Spectral positions are optional, as in the visible path: without them
    # every pinhole projects to the centre of the spectral window.
    spectral_positions = (list(sim.pinhole_positions_spectral)
                          or [None] * len(sim.pinhole_sizes))

    for pinhole_diameter, pinhole_position, pinhole_spectral in zip(
            sim.pinhole_sizes, sim.pinhole_positions, spectral_positions):
        # Calculate pinhole area
        pinhole_area = np.pi * (pinhole_diameter / 2)**2

        # photon_counts already have the filter attenuation, so:
        # 1. Work out the unfiltered signal.
        # 2. Apply the pinhole diffraction to it.
        # 3. Subtract the filtered signal already counted in the pinhole regions.

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
        
        # Scale the pattern by its absolute normalisation, as the visible path
        # does. Dividing by its sum over the detector would force every photon
        # through the pinhole onto the detector, whatever the geometry. The EUV
        # Airy pattern is about thirty times smaller than the visible one at
        # the same diameter, but a 1 micron hole at 195 Angstrom still has its
        # first minimum 5.95 mm out, against a detector a few mm across, so
        # most of its light misses and must be lost, not redistributed.
        # euv_pattern peaks at 1.0, so multiplying by the peak per-pixel
        # fraction gives the right fraction everywhere, summing to less than 1
        # when light falls off the detector.
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
            filtered_signal = photon_counts.data[:, i, :]  # Shape: (n_slit, n_spectral)
            
            # Unfiltered signal: the filtered signal plus the light the filter
            # blocked from it.
            unfiltered_signal = filtered_signal + blocked[:, i, :]
            
            # Unattenuated light through the pinhole. unfiltered_signal *
            # area_ratio is the light collected over the pinhole's area, and
            # the scaled pattern gives the fraction of it reaching each pixel,
            # which need not sum to 1.
            pinhole_signal = unfiltered_signal * area_ratio * euv_pattern_normalized
            
            # The filtered light already counted in the pinhole regions
            # (filtered signal weighted by diffraction pattern and area ratio)
            overcounted_filtered = filtered_signal * area_ratio * euv_pattern_normalized
            
            # Net extra light from the pinhole: its total light less the
            # filtered light already counted there, which is
            # blocked * area_ratio * pattern.
            correction = (pinhole_signal - overcounted_filtered)
            additional_photons[:, i, :] += correction

    # Create new photon counts with EUV pinhole contributions
    new_data = photon_counts.data + additional_photons

    # The pinholes' photons have the energies of the blocked light. The
    # filter entries in meta no longer hold once they are added, so drop them.
    meta = {key: value for key, value in counts.meta.items() if key not in _FILTER_KEYS}
    if "photon_wavelength" in meta:
        mean = u.Quantity(meta["photon_wavelength"])
        unit = mean.unit
        rms = u.Quantity(meta.get("photon_rms_wavelength", mean)).to_value(unit)
        blocked_mean = u.Quantity(counts.meta["blocked_photon_wavelength"]).to_value(unit)
        blocked_rms = u.Quantity(counts.meta["blocked_photon_rms_wavelength"]).to_value(unit)

        def over(photons, wavelength):
            return np.divide(photons, wavelength, out=np.zeros(data.shape), where=wavelength > 0)

        energy = over(data, mean.value) + over(additional_photons, blocked_mean)
        square = over(data, rms**2) + over(additional_photons, blocked_rms**2)
        meta["photon_wavelength"] = np.divide(new_data, energy, out=np.array(mean.value, dtype=float),
                                              where=energy > 0) * unit
        meta["photon_rms_wavelength"] = np.sqrt(np.divide(
            new_data, square, out=np.array(rms, dtype=float) ** 2, where=square > 0)) * unit

    return NDCube(
        data=new_data,
        wcs=photon_counts.wcs.deepcopy(),
        unit=photon_counts.unit,
        meta=meta,
    )


# Filter entries in a cube's meta, which the pinholes need.
_FILTER_KEYS = ("photon_filter_transmission", "blocked_photon_wavelength",
                "blocked_photon_rms_wavelength")