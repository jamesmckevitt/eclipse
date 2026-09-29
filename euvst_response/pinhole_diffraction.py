"""
Pinholes in the SW aluminium filter.

The filter sits ``Detector_SWC.filter_distance`` in front of the detector, in
the beam converging on it. A pinhole passes light the foil would have
attenuated, EUV and visible alike, and this module works out where it lands.

EUV. The light at the filter is the image on its way to focus. The light for
each point of the detector crosses the filter as a cone, whose section there
is the SW pupil, the half of the 280 mm primary on one side of a cut along the
slit, scaled to :func:`beam_footprint_radius`. A pinhole passes, for every
point whose cone covers it, the share (hole area / cone area) of that point's
light that crosses it, without the foil. The points it serves make a half disc
of the image beside it, on the long-wavelength side. That light is part of the
same wave as the light through the foil around the hole, so the two
interfere: with t the filter's amplitude transmission and T = |t|^2, the hole
adds 1 - T of the light through it, of which |1 - t|^2 is diffracted by the
hole, and 2 Re[conj(t) (1 - t)] stays in the image, in the pixel it was
already heading for. A hole diffracts a wave converging on a point into
exactly the Airy pattern of the hole, centred on that point, whatever its
size. The light is taken from the image as the focusing optics leave it.

Visible. Where the visible stray light comes from is not known, so it is taken
to reach the filter head-on as a plane wave, and the hole's light is its
near-field diffraction pattern, centred under the hole, rather than the
far-field limit a large hole is not in. It is the Rayleigh-Sommerfeld
integral with the path from each point of the hole taken to second order in
its distance from the centre, and the obliquity and distance of the centre
ray, which agrees with the full integral to 5e-4 for a 400 micron hole. The
foil passes some 1e-9 of the visible, so its interference with the light
through the hole, at most 2 |t| of it, some 1e-4, is left out.

A pinhole's position is a fraction of the simulated window along each axis,
not of the whole detector.
"""

from __future__ import annotations
import functools
import hashlib
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import CubicSpline
from scipy.special import j1, jv
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




# ---------------------------------------------------------------------------
# Where a pinhole is, and the beam it sits in
# ---------------------------------------------------------------------------
def pinhole_centre(position_slit: float, position_spectral: float | None,
                   n_slit: int, n_spectral: int) -> Tuple[float, float]:
    """
    The pixel under a pinhole's centre, as (row along the slit, column along
    the dispersion), from its positions as fractions of the simulated window.
    Without a spectral position it is under the middle column.
    """
    if position_spectral is None:
        column = float(n_spectral // 2)
    else:
        if not 0.0 <= position_spectral <= 1.0:
            raise ValueError(
                "pinhole_position_spectral is a fraction of the window along "
                f"the dispersion and must lie in [0, 1], got {position_spectral}.")
        column = position_spectral * (n_spectral - 1)
    return position_slit * (n_slit - 1), column


def beam_footprint_radius(det, tel) -> u.Quantity:
    """
    Radius at the filter of the cone of light converging on one point of the detector.

    The beam's f-number is its focal length, from the pixel size and the plate
    scale, over the diameter of the entrance pupil, and ``det.filter_distance``
    before focus the cone is that distance over twice the f-number in radius.
    The optical design has the same plate scale along the dispersion as along
    the slit (RSC-2022021C), so the cone is round before SW takes its half.
    """
    focal_length = (det.pix_size / det.plate_scale_angle.to(u.rad / u.pix)).to(
        u.m, equivalencies=u.dimensionless_angles())
    f_number = (focal_length / tel.D_ap).to_value(u.dimensionless_unscaled)
    return (det.filter_distance / (2 * f_number)).to(u.mm)


def _disc_corner_area(x, y, radius):
    """
    Area of the disc of *radius* about the origin inside the rectangle with
    corners at the origin and at (x, y), negative when one of x and y is.
    """
    sign = np.sign(x) * np.sign(y)
    x = np.minimum(np.abs(x), radius)
    y = np.minimum(np.abs(y), radius)
    # The disc's edge is above y up to turn, and below it after.
    turn = np.sqrt(np.maximum(radius**2 - y**2, 0.0))

    def under_edge(t):
        # The area under the disc's edge from 0 to t.
        return 0.5 * (t * np.sqrt(np.maximum(radius**2 - t**2, 0.0))
                      + radius**2 * np.arcsin(np.clip(t / radius, -1.0, 1.0)))

    return sign * np.where(x <= turn, x * y, turn * y + under_edge(x) - under_edge(turn))


def half_disc_fractions(shape: Tuple[int, int], centre: Tuple[float, float], radius: float,
                        toward_higher_columns: bool) -> np.ndarray:
    """
    The share of each pixel's area inside a half disc: *radius* pixels about
    *centre* (row, column), on the side of higher columns or of lower ones.
    """
    n_rows, n_columns = shape
    rows = np.arange(n_rows)[:, np.newaxis] - centre[0]
    columns = np.arange(n_columns)[np.newaxis, :] - centre[1]
    low, high = columns - 0.5, columns + 0.5
    if toward_higher_columns:
        low, high = np.maximum(low, 0.0), np.maximum(high, 0.0)
    else:
        low, high = np.minimum(low, 0.0), np.minimum(high, 0.0)

    def corner(x, y):
        return _disc_corner_area(x, y, radius)

    # The four corners' areas cancel to rounding error, of either sign, where
    # a pixel is wholly outside or inside.
    return np.clip(corner(high, rows + 0.5) - corner(low, rows + 0.5)
                   - corner(high, rows - 0.5) + corner(low, rows - 0.5), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Diffraction, integrated over the pixels
# ---------------------------------------------------------------------------
def _gauss_nodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0, 1]."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (nodes + 1) / 2, weights / 2


def _nodes_per_pixel(diameter: float, wavelength: float, distance: float, pixel: float) -> int:
    """Gauss-Legendre nodes across a pixel for fringes wavelength * distance / diameter apart."""
    return 4 + int(np.ceil(3 * pixel * diameter / (wavelength * distance)))


def _airy_irradiance(r, radius: float, wavelength: float, distance: float):
    """
    Irradiance, per unit of the light a hole of *radius* passes toward a point
    *distance* beyond it, at *r* from that point, all in metres.

    The Airy pattern of the hole at the angle r subtends, with the
    Rayleigh-Sommerfeld obliquity and the slant of the detector to the light.
    """
    return _airy_of_squared(np.square(r), radius, wavelength, distance)


def _airy_of_squared(r2, radius: float, wavelength: float, distance: float):
    """:func:`_airy_irradiance` at the squared distances *r2*, which is quicker to have."""
    cos2 = distance**2 / (r2 + distance**2)
    v = (2 * np.pi * radius / wavelength) * np.sqrt(1 - cos2)
    # 2 J1(v) / v is 1 at v = 0, where the quotient cannot be taken.
    centre = v == 0
    pattern = np.square(2 * j1(v) / np.where(centre, 1.0, v))
    pattern[centre] = 1.0
    return (np.pi * radius**2 / (wavelength * distance) ** 2) * cos2 * cos2 * np.sqrt(cos2) * pattern


def _converging_kernel(n_rows: int, n_columns: int, radius: float, wavelength: float,
                       distance: float, pixel: float) -> np.ndarray:
    """
    ``K[m, n]``: of the light a hole of *radius* passes toward the points of
    one pixel, spread evenly over it, the share landing on the pixel *m* rows
    and *n* columns away. All lengths in metres. Read-only. Not cached: each
    column has its own wavelength and so its own kernel, as large as the
    window, and apply_euv_pinhole_diffraction keeps what they add up to.
    """
    s, w = _gauss_nodes(_nodes_per_pixel(2 * radius, wavelength, distance, pixel))
    # Two pixels u apart overlap by 1 - |u| of one along each axis, a function
    # with a kink at 0, so each side of it has its own nodes.
    offsets = np.concatenate([-s, s])
    weights = np.concatenate([w * (1 - s), w * (1 - s)])
    rows2 = np.square(pixel * (np.arange(n_rows)[:, np.newaxis] + offsets))
    columns2 = np.square(pixel * (np.arange(n_columns)[:, np.newaxis] + offsets))
    kernel = np.empty((n_rows, n_columns))
    chunk = max(1, int(4e6) // (n_columns * offsets.size ** 2))
    for start in range(0, n_rows, chunk):
        r2 = rows2[start:start + chunk, np.newaxis, :, np.newaxis] + columns2[np.newaxis, :, np.newaxis, :]
        kernel[start:start + chunk] = pixel**2 * np.einsum(
            "abij,i,j->ab", _airy_of_squared(r2, radius, wavelength, distance), weights, weights)
    kernel.flags.writeable = False
    return kernel


def _head_on_irradiance(r, radius: float, wavelength: float, distance: float) -> np.ndarray:
    """
    Irradiance, per unit of the light through a hole of *radius* lit head-on
    by a plane wave, at *r* from the point under its centre *distance* below,
    all in metres.

    The path s from a point (rho, phi) of the hole to the detector is taken to
    second order in rho about the path from the centre, S = sqrt(distance^2 +
    r^2), at the angle theta from the axis::

        s - S = -rho sin(theta) cos(phi) + rho^2 (1 - sin^2(theta) cos^2(phi)) / (2 S)

    which is good to a phase of k rho^3 sin(theta) / (2 S^2), 2e-4 rad for a
    500 micron hole in the visible at the corner of the detector. The integral
    over phi is then a series of Bessel functions,

        Phi(rho) = 2 pi exp(-i c) sum_n (-1)^n (-i)^n J_2n(k rho sin(theta)) J_n(c),
        c = k rho^2 sin^2(theta) / (4 S),

    the field is U = cos(theta) / (i lambda S) int_0^R exp(i k rho^2 / (2 S))
    Phi(rho) rho d rho, with the Rayleigh-Sommerfeld obliquity, and the
    detector receives |U|^2 cos(theta) per unit area.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    k = 2 * np.pi / wavelength
    slant = np.hypot(distance, r)
    sin_theta, cos_theta = r / slant, distance / slant
    widest = k * radius * sin_theta.max()
    n_rho = 32 + int(np.ceil(widest + k * radius**2 / (2 * distance)))
    x, w = _gauss_nodes(n_rho)
    rho, w = radius * x, radius * w
    top = int(np.ceil(k * radius**2 * sin_theta.max() ** 2 / (4 * distance))) + 8
    orders = np.arange(-top, top + 1)
    out = np.empty(r.size)
    chunk = max(1, int(2e6) // (n_rho * orders.size))
    for start in range(0, r.size, chunk):
        part = slice(start, start + chunk)
        a = k * rho * sin_theta[part, np.newaxis]
        c = k * rho**2 * sin_theta[part, np.newaxis] ** 2 / (4 * slant[part, np.newaxis])
        series = np.zeros(a.shape, dtype=complex)
        for n in orders:
            series += (-1.0) ** n * (-1j) ** n * jv(2 * n, a) * jv(n, c)
        integrand = (np.exp(1j * k * rho**2 / (2 * slant[part, np.newaxis]))
                     * 2 * np.pi * np.exp(-1j * c) * series * rho)
        field = np.abs(integrand @ w) ** 2
        out[part] = (cos_theta[part] ** 3 / (wavelength * slant[part]) ** 2 * field
                     / (np.pi * radius**2))
    return out


@functools.lru_cache(maxsize=64)
def head_on_pinhole_fractions(shape: Tuple[int, int], centre: Tuple[float, float],
                              diameter: float, wavelength: float, distance: float,
                              pixel: float) -> np.ndarray:
    """
    The share of the light through a pinhole lit head-on that lands in each pixel.

    Parameters
    ----------
    shape : tuple of int
        The window, (rows along the slit, columns along the dispersion).
    centre : tuple of float
        The pixel under the hole's centre, (row, column).
    diameter, wavelength, distance, pixel : float
        The hole's diameter, the light's wavelength, the filter's distance
        from the detector and the pixel size, in metres.

    Returns
    -------
    np.ndarray
        The shares, of *shape*, integrated over each pixel. They add up to
        less than one by the light that misses the window. Read-only.
    """
    n_rows, n_columns = shape
    s, w = _gauss_nodes(_nodes_per_pixel(diameter, wavelength, distance, pixel))
    s = s - 0.5
    rows = (np.arange(n_rows) - centre[0])[:, np.newaxis] + s
    columns = (np.arange(n_columns) - centre[1])[:, np.newaxis] + s
    r = pixel * np.hypot(rows[:, np.newaxis, :, np.newaxis], columns[np.newaxis, :, np.newaxis, :])
    # The pattern is worked out on radii finely enough spaced that a cubic
    # spline through them is good to about 1e-9 of it, and read off at the
    # nodes.
    step = min(pixel / 8, wavelength * distance / (256 * diameter))
    radii = np.arange(0.0, r.max() + 2 * step, step)
    spline = CubicSpline(radii, _head_on_irradiance(radii, diameter / 2, wavelength, distance),
                         bc_type=((1, 0.0), "not-a-knot"))
    fractions = pixel**2 * np.einsum("abij,i,j->ab", spline(r), w, w)
    fractions.flags.writeable = False
    return fractions


def _spread(sources: np.ndarray, kernel_of_column) -> np.ndarray:
    """
    The light of *sources* ``(rows, scans, columns)`` as each source column's
    kernel ``(rows, columns)``, from :func:`_converging_kernel`, spreads it.
    """
    n_rows, n_scans, n_columns = sources.shape
    # Long enough that the convolution along the slit does not wrap round.
    size = 1 << int(np.ceil(np.log2(3 * n_rows)))
    scans_at_once = max(1, int(1e7) // (n_columns * size))
    spread = np.zeros((n_scans, n_columns, n_rows))
    along = np.moveaxis(sources, 0, -1)  # (scans, columns, rows)
    for column in np.flatnonzero(np.any(sources != 0, axis=(0, 1))):
        kernel = kernel_of_column(column)
        # Offsets from -(n_rows - 1) to n_rows - 1 along the slit, and the
        # offset of every column from this one.
        both_ways = np.concatenate([kernel[:0:-1], kernel])[:, np.abs(np.arange(n_columns) - column)]
        kernel_spectrum = np.fft.rfft(both_ways.T, size)[np.newaxis, :, :]
        for first in range(0, n_scans, scans_at_once):
            scans = slice(first, first + scans_at_once)
            product = np.fft.rfft(along[scans, column, :], size)[:, np.newaxis, :] * kernel_spectrum
            spread[scans] += np.fft.irfft(product, size)[..., n_rows - 1:2 * n_rows - 1]
    return np.moveaxis(spread, -1, 0)


_LAST_ADDED: dict = {}


def pinhole_light_weights(tel) -> tuple:
    """
    What a pinhole adds, per photon the filter passes, at each wavelength.

    Two functions of wavelength: the light that stays in the image,
    2 Re[conj(t) (1 - t)] / T, and the light that is diffracted,
    |1 - t|^2 / T, with t the filter's amplitude transmission and T = |t|^2.
    Given as ``weight`` to :func:`~euvst_response.data_processing.rebin_atmosphere`
    or :func:`~euvst_response.data_processing.rebin_spectra`, they weigh the
    light at its own wavelength before the PSF mixes wavelengths the filter
    passes in different shares.
    """
    def share(part):
        def weight(wavelength):
            t = np.asarray(tel.filter.amplitude_transmission(u.Quantity(wavelength)),
                           dtype=complex)
            throughput = np.abs(t) ** 2
            return np.divide(part(t), throughput, out=np.zeros(throughput.shape),
                             where=throughput > 0)
        return weight

    return (share(lambda t: 2 * np.real(np.conj(t) * (1 - t))),
            share(lambda t: np.abs(1 - t) ** 2))


def apply_euv_pinhole_diffraction(
    photon_counts: NDCube,
    det,
    sim,
    tel,
    *,
    unfocused: NDCube | None = None,
    focus=None,
    staying: np.ndarray | None = None,
    diffracting: np.ndarray | None = None,
    uniform: bool = False,
) -> NDCube:
    """
    Add the EUV light the filter's pinholes let through.

    Each pinhole passes, for every pixel whose cone of light covers it, the
    share (hole area / cone area) of that pixel's light that crosses it: the
    pixels of a half disc :func:`beam_footprint_radius` in radius, beside the
    hole on the long-wavelength side, each counted by the share of its area
    inside. Without the foil that light is 1 / T times brighter. Of what the
    hole adds, 1 - T of the light through it, |1 - t|^2 is diffracted into the
    hole's Airy pattern about the pixel it was heading for, integrated over
    the pixels it lands on, and 2 Re[conj(t) (1 - t)] stays in that pixel,
    where t is the filter's amplitude transmission,
    :meth:`~euvst_response.config.AluminiumFilter.amplitude_transmission`,
    at the light's own wavelength.

    Where the focusing optics blur the image, a pixel holds light of the
    wavelengths around its own, which the filter passed in different shares,
    so the light without the filter, and the shares of it that stay and are
    diffracted, are worked out from the photons before the blur, where each
    column holds its own wavelength, and then blurred by *focus* as the image
    is, or come as *staying* and *diffracting*, where the blur was in the
    rebinning (:func:`pinhole_light_weights`). The Airy pattern the
    diffracted light spreads into is taken at the wavelength of the pixel it
    was heading for, which the light in it differs from by the width of the
    spectral blur, a part in some 1e4.

    Parameters
    ----------
    photon_counts : NDCube
        EUV photons per pixel after the filter and the focusing optics,
        ``(n_slit, n_scan, n_spectral)``.
    det : Detector_SWC
    sim : Simulation
        The pinholes: their diameters, and their positions as fractions of
        the window.
    tel : Telescope_EUVST
        The filter, and the entrance pupil the beam's f-number comes from.
    unfocused : NDCube, optional
        The same photons before the focusing optics blurred them, where
        *focus* did. Without it, *photon_counts* are taken to be unblurred.
    focus : callable, optional
        The blur of the focusing optics, taking a cube like *unfocused* to
        one like *photon_counts*.
    staying, diffracting : np.ndarray, optional
        The photons per pixel, as the focusing optics leave them, of the
        light each pixel's pinholes add that stays and that is diffracted,
        before the share of the hole's area in each: the photons after the
        filter, weighted at their own wavelengths by
        :func:`pinhole_light_weights` before the blur. In place of
        *unfocused* and *focus*.
    uniform : bool, optional
        The photons are a uniform intensity, standing for a scene the same all
        along the slit, so the rows beyond the window send the pinholes' light
        into it too. Otherwise the scene is dark beyond the window.

    Returns
    -------
    NDCube
        The photons with the pinholes' light added.
    """
    if not (sim.enable_pinholes and len(sim.pinhole_sizes) > 0):
        return photon_counts  # No pinholes enabled

    if (unfocused is None) != (focus is None):
        raise ValueError("unfocused and focus go together: the photons before the focusing "
                         "optics' blur, and the blur.")
    given = staying is not None or diffracting is not None
    if given and (staying is None or diffracting is None or unfocused is not None):
        raise ValueError("staying and diffracting go together, in place of unfocused and focus.")
    source = photon_counts if unfocused is None else unfocused
    for light in (source.data, staying, diffracting):
        if light is not None and np.shape(light) != photon_counts.data.shape:
            raise ValueError(f"The pinholes' light is {np.shape(light)}, and the photons "
                             f"{photon_counts.data.shape}.")
    n_slit, n_scan, n_spectral = photon_counts.data.shape
    wavelength = photon_counts.axis_world_coords(2)[0].to(u.m)

    pixel = (det.pix_size * u.pix).to_value(u.m)
    distance = det.filter_distance.to_value(u.m)
    footprint = beam_footprint_radius(det, tel).to_value(u.m)
    footprint_area = np.pi * footprint**2 / 2
    longer_is_higher = bool(wavelength[-1] > wavelength[0])
    spectral_positions = (list(sim.pinhole_positions_spectral)
                          or [None] * len(sim.pinhole_sizes))
    pinholes = [((diameter / 2).to_value(u.m), float(along_slit), along_spectral)
                for diameter, along_slit, along_spectral in zip(
                    sim.pinhole_sizes, sim.pinhole_positions, spectral_positions)]

    if given:
        staying, diffracting = np.asarray(staying, float), np.asarray(diffracting, float)
    else:
        # The shares of the light that stay and are diffracted, at its own
        # wavelength, then as the focusing optics leave it.
        staying, diffracting = (source.data * share(wavelength)
                                for share in pinhole_light_weights(tel))
        if focus is not None:
            staying, diffracting = (np.asarray(focus(NDCube(light, wcs=source.wcs,
                                                            unit=source.unit,
                                                            meta=source.meta)).data, float)
                                    for light in (staying, diffracting))

    # The light is the same in every Monte Carlo iteration, which works it out
    # again before its noise is drawn, so the last answer is kept, found by
    # the light it spreads, as the optics left it, and where it goes.
    digest = hashlib.blake2b(digest_size=16)
    for light in (staying, diffracting):
        digest.update(np.ascontiguousarray(light).tobytes())
    digest.update(repr((photon_counts.data.shape, bool(uniform))).encode())
    digest.update(np.ascontiguousarray(wavelength.value).tobytes())
    digest.update(repr((pinholes, pixel, distance, footprint)).encode())
    key = digest.hexdigest()
    if key not in _LAST_ADDED:
        # A uniform intensity stands for a scene that is the same all along
        # the slit, beyond the rows it is worked out on, and a pinhole passes
        # the light of those rows too, as far as its half disc reaches, whose
        # diffracted light lands on the rows worked out. The rows are carried
        # on, alike, that far.
        margin = int(np.ceil(footprint / pixel)) + 1 if uniform else 0
        if margin:
            staying, diffracting = (np.pad(light, ((margin, margin), (0, 0), (0, 0)), mode="edge")
                                    for light in (staying, diffracting))
        n_rows = n_slit + 2 * margin
        added = np.zeros((n_rows, n_scan, n_spectral))
        for radius, along_slit, along_spectral in pinholes:
            row, column = pinhole_centre(along_slit, along_spectral, n_slit, n_spectral)
            share = ((np.pi * radius**2 / footprint_area) * half_disc_fractions(
                (n_rows, n_spectral), (margin + row, column), footprint / pixel,
                longer_is_higher))[:, np.newaxis, :]
            added += staying * share
            added += _spread(diffracting * share, lambda source: _converging_kernel(
                n_rows, n_spectral, radius, float(wavelength[source].to_value(u.m)), distance, pixel))
        added = added[margin:margin + n_slit].copy()
        added.flags.writeable = False
        _LAST_ADDED.clear()
        _LAST_ADDED[key] = added
    added = _LAST_ADDED[key]

    return NDCube(
        data=photon_counts.data + added,
        wcs=photon_counts.wcs.deepcopy(),
        unit=photon_counts.unit,
        meta=photon_counts.meta,
    )
