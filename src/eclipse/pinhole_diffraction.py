"""
Diffraction through pinholes in the aluminium filter.

Each pinhole passes unfiltered EUV and visible light, spread over the
detector in the pinhole's Airy pattern.
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
    # Exact sine of the diffraction angle, not the small-angle one.
    distance_m = distance.to(u.m).value
    sin_theta = r / np.sqrt(r**2 + distance_m**2)
    
    beta = (np.pi * pinhole_diameter.to(u.m).value * sin_theta) / wavelength.to(u.m).value
    
    # Avoid dividing by zero at the centre.
    beta = np.where(beta == 0, 1e-10, beta)
    
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

    The absolute normalisation of the Airy pattern, which
    `calculate_pinhole_diffraction_pattern` leaves out (its pattern peaks at
    1). For a circular aperture of area A at distance L, the on-axis
    irradiance is

        E_0 = P_total * A / (lambda^2 L^2)

    as (2*J1(u)/u)^2 integrates over the plane to 4*lambda^2*L^2/(pi*D^2).
    The fraction on one pixel of area a is then `A * a / (lambda L)^2`.

    Normalising the pattern by its sum over the detector would put every
    pinhole photon on the detector. A small pinhole's Airy disc is far
    larger than that: a 1 micron hole at 250 mm has its first minimum at
    183 mm, against a detector tens of mm across, so most of the light
    misses. Use this when absolute photon numbers matter, such as for a
    pinhole budget.

    Returns
    -------
    float
        Fraction in the brightest pixel, capped at 1. The cap applies only
        to holes so large that the geometric image is smaller than a pixel,
        where Fraunhofer diffraction does not hold.
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
    Diffraction pattern of one pinhole on the detector.

    Parameters
    ----------
    detector_shape : tuple of int
        Detector shape, (n_slit, n_spectral).
    pixel_size : u.Quantity
        Pixel size.
    pinhole_diameter : u.Quantity
        Pinhole diameter.
    pinhole_position_slit : float
        Position along the slit, as a fraction from 0 to 1.
    slit_width : u.Quantity
        Slit width.
    plate_scale : u.Quantity
        Plate scale, in arcsec per pixel.
    distance : u.Quantity
        Distance from the pinhole to the detector.
    wavelength : u.Quantity
        Wavelength.
    pinhole_position_spectral : float, optional
        Position along the spectral axis, as a fraction from 0 to 1 of the
        detector width, which sets the lines a pinhole contaminates. Default
        None, for the centre of the spectral window.

    Returns
    -------
    np.ndarray
        Pattern, 1 at its centre. For photon numbers, scale it by
        `airy_peak_fraction_per_pixel`.
    """
    n_slit, n_spectral = detector_shape
    
    slit_pixels = np.arange(n_slit)
    spectral_pixels = np.arange(n_spectral)
    
    pinhole_pixel_slit = pinhole_position_slit * (n_slit - 1)
    
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
    
    slit_grid, spectral_grid = np.meshgrid(slit_pixels, spectral_pixels, indexing='ij')
    
    # Distance of each pixel from the pinhole's centre, in metres.
    dy_pixels = slit_grid - pinhole_pixel_slit
    dx_pixels = spectral_grid - pinhole_pixel_spectral
    
    dy_physical = dy_pixels * pixel_size.to(u.m).value
    dx_physical = dx_pixels * pixel_size.to(u.m).value
    
    r_physical = np.sqrt(dx_physical**2 + dy_physical**2)
    
    pattern = airy_disk_pattern(r_physical, wavelength, pinhole_diameter, distance)
    
    return pattern


def _pinholes_on(sim) -> bool:
    """Whether `sim` has pinholes enabled and at least one given."""
    return bool(sim.enable_pinholes and len(sim.pinhole_sizes) > 0)


def _filter_transmission(tel, wavelength: u.Quantity) -> np.ndarray:
    """
    Fraction of the EUV the filter passes at each `wavelength`.

    The pinholes' light is worked out from the light the filter passes, so a
    filter that passes no EUV at a wavelength its tables cover raises a
    ValueError.
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
    `photon_counts` with the `meta` entries :func:`apply_euv_pinhole_diffraction` needs.

    For photons counted at each pixel's own wavelength: the fraction of
    their light the filter passed there, and the wavelengths of the blocked
    light, the same as the photons'.
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
    Add the unfiltered EUV light through pinholes in the filter.

    At each pinhole the filtered light is replaced by the unfiltered light,
    spread in the pinhole's diffraction pattern. The filter is behind the
    mirror and grating, so this comes after the PSF. Does nothing unless
    `sim.enable_pinholes` is set and there are pinholes.

    The blocked light is worked out at each photon's own wavelength from
    `meta["photon_filter_transmission"]`, the fraction the filter passed,
    which `rebin_atmosphere` and `rebin_spectra` set with the pinholes on.
    Without it, each pixel's photons are taken to have passed the filter at
    the pixel's wavelength. The pinholes' photons have the energies of the
    blocked light, and are included in the mean and RMS photon wavelengths
    in `meta`.

    Parameters
    ----------
    photon_counts : NDCube
        EUV photons in each pixel, after the filter and the PSF.
    det : Detector_SWC
        EUVST-SW detector, whose `filter_distance` sets the diffraction.
    sim : Simulation
        Simulation, with the pinholes.
    tel : Telescope_EUVST
        Telescope, whose filter the pinholes bypass.

    Returns
    -------
    NDCube
        Photons in each pixel, including the pinholes' light.
    """
    if not (sim.enable_pinholes and len(sim.pinhole_sizes) > 0):
        return photon_counts
    
    data_shape = photon_counts.data.shape  # (n_slit, n_scan, n_spectral)
    n_slit, n_scan, n_spectral = data_shape
    
    # The diffraction pattern is taken at the line's rest wavelength.
    rest_wavelength = photon_counts.meta['rest_wav']
    
    pixel_area = (det.pix_size*1*u.pix)**2
    
    additional_photons = np.zeros_like(photon_counts.data)
    
    # Light the filter blocked from each pixel's photons, at their own
    # wavelengths.
    counts = (photon_counts if "photon_filter_transmission" in (photon_counts.meta or {})
              else _with_filter_transmission(photon_counts, tel))
    data = np.asarray(photon_counts.data, dtype=float)
    passed = np.asarray(counts.meta["photon_filter_transmission"], dtype=float)
    blocked = data * np.divide(1.0 - passed, passed, out=np.zeros(data.shape), where=passed > 0)

    # Without spectral positions, every pinhole is at the centre of the
    # spectral window.
    spectral_positions = (list(sim.pinhole_positions_spectral)
                          or [None] * len(sim.pinhole_sizes))

    for pinhole_diameter, pinhole_position, pinhole_spectral in zip(
            sim.pinhole_sizes, sim.pinhole_positions, spectral_positions):
        pinhole_area = np.pi * (pinhole_diameter / 2)**2

        # The photons are already filtered, so add the unfiltered light
        # through the pinhole less the filtered light already counted there.

        area_ratio = (pinhole_area / pixel_area).to(u.dimensionless_unscaled).value
        
        # First Airy minimum, r = 1.22 lambda L / D.
        theoretical_radius = (1.22 * rest_wavelength * det.filter_distance / pinhole_diameter).to(u.m)
        theoretical_radius_pixels = (theoretical_radius / (det.pix_size*1*u.pix)).to(u.dimensionless_unscaled).value
        
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
        
        # Scale the pattern by its absolute normalisation, not its sum over the
        # detector, so light falling off the detector is lost. A 1 micron hole
        # at 195 Angstrom has its first minimum 5.95 mm out, against a detector
        # a few mm across. The pattern peaks at 1, so this gives the fraction
        # in every pixel.
        peak_fraction = airy_peak_fraction_per_pixel(
            pinhole_diameter=pinhole_diameter,
            distance=det.filter_distance,
            wavelength=rest_wavelength,
            pixel_size=det.pix_size * u.pix,
        )
        euv_pattern_normalized = euv_pattern * peak_fraction


        for i in range(n_scan):
            filtered_signal = photon_counts.data[:, i, :]  # (n_slit, n_spectral)
            
            unfiltered_signal = filtered_signal + blocked[:, i, :]
            
            # Unfiltered light over the pinhole's area, spread by the pattern,
            # which need not sum to 1.
            pinhole_signal = unfiltered_signal * area_ratio * euv_pattern_normalized
            
            # Filtered light already counted where the pinhole is.
            overcounted_filtered = filtered_signal * area_ratio * euv_pattern_normalized
            
            # Net extra light, blocked * area_ratio * pattern.
            correction = (pinhole_signal - overcounted_filtered)
            additional_photons[:, i, :] += correction

    new_data = photon_counts.data + additional_photons

    # The pinholes' photons have the energies of the blocked light. The
    # filter entries in meta do not hold once they are added, so drop them.
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