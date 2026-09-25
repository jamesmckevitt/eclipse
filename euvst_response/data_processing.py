"""
Data processing functions for atmosphere cubes and spectral resampling.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import astropy.units as u
import astropy.constants as const
import dill
from ndcube import NDCube
from astropy.wcs import WCS
from scipy.special import erf
from tqdm import tqdm
from .radiometric import spectral_psf_fwhm
from .utils import (_bin_edges, distance_to_angle, _fwhm_to_sigma, has_wrong_velocity_sign,
                    onto_wavelength_bins,
                    VELOCITY_CONVENTION)


def load_atmosphere(pkl_file: str, metadata_line: str = None) -> tuple:
    """
    Load synthetic atmosphere cube from pickle file.
    
    Creates a summed cube from all line cubes in the synthesis results.
    All line cubes are put onto the wavelength grid of the metadata_line,
    keeping their flux, before summing, as :func:`sum_line_cubes` does.
    
    Parameters
    ----------
    pkl_file : str
        Path to the synthesized spectra pickle file.
    metadata_line : str, optional
        Name of the line to use for metadata and wavelength grid reference. 
        If None, uses the first line.
        
    Returns
    -------
    tuple
        (summed_cube, dynamic_mode_info) where:
        - summed_cube: NDCube with summed line intensities
        - dynamic_mode_info: dict with dynamic mode metadata (or None if static)
    """
    with open(pkl_file, "rb") as f:
        tmp = dill.load(f)
    
    # Handle new synthesis format
    if "line_cubes" not in tmp:
        raise ValueError("File does not contain synthesis results with line_cubes")
        
    line_cubes = tmp["line_cubes"]
    if not line_cubes:
        raise ValueError("No line cubes found in synthesis results")
    
    # Get dynamic mode info if present
    dynamic_mode_info = tmp.get("dynamic_mode", {"enabled": False})
    
    # Get the line names
    line_names = list(line_cubes.keys())

    # Choose metadata source line
    if metadata_line is None:
        metadata_line = line_names[0]
    elif metadata_line not in line_names:
        raise ValueError(f"Metadata line '{metadata_line}' not found. Available lines: {line_names}")

    check_old_line_cubes({metadata_line: line_cubes[metadata_line]}, pkl_file)

    summed_cube = sum_line_cubes(line_cubes, metadata_line)
    summed_cube.meta["dynamic_mode"] = dynamic_mode_info
    return summed_cube, dynamic_mode_info


def check_old_line_cubes(line_cubes: dict, path) -> None:
    """Refuse line cubes that older versions of ECLIPSE wrote in a form it no longer reads right."""
    for cube in line_cubes.values():
        meta = cube.meta or {}
        # Refuse files written before the cube axis order was fixed (issue
        # #12). Those store data as (x, y, wavelength); everything downstream
        # now expects (y, x, wavelength), so an old file would come out
        # transposed. The WCS axis order tells the two apart.
        old_first_spatial = {"z": "SOLY", "x": "SOLZ", "y": "SOLZ"}
        axis = meta.get("integration_axis")
        if axis in old_first_spatial and cube.wcs.wcs.ctype[1] == old_first_spatial[axis]:
            raise ValueError(
                f"{path} was written by an older ECLIPSE that stored cubes "
                "as (x, y, wavelength). Cubes are now (y, x, wavelength). "
                "Re-run the synthesis with this version to regenerate the file."
            )

        # Refuse files written before the Doppler sign was fixed. Those used
        # the simulation velocity along the line of sight as it was, which for
        # views along x and z gives every velocity the wrong sign.
        if has_wrong_velocity_sign(meta):
            axis = meta.get("integration_axis", "z")
            raise ValueError(
                f"{path} was written by an older ECLIPSE that used the "
                "simulation velocity along the line of sight without turning it "
                "into a velocity away from the observer. For this view along "
                f"{axis}, every Doppler shift in it has the wrong sign: flows "
                "towards the observer are redshifted. Re-run the synthesis with "
                "this version to regenerate the file."
            )


def sum_line_cubes(line_cubes: dict, reference_line: str) -> NDCube:
    """
    Every line's cube summed onto the wavelength grid of *reference_line*.

    Each cube is averaged over the bins of the reference line's grid,
    keeping its flux, with zero outside its own, so lines outside the
    reference window contribute nothing. The summed cube keeps the reference
    cube's WCS, unit and metadata, plus the names of the lines it holds.
    """
    ref_cube = line_cubes[reference_line]
    ref_wavelengths = ref_cube.axis_world_coords(-1)[0]
    line_names = list(line_cubes.keys())

    # Get spatial dimensions from reference cube
    ny, nx, nw = ref_cube.data.shape

    # Initialize summed data with the reference wavelength grid
    summed_data = np.zeros((ny, nx, nw))

    for line_name, cube in tqdm(line_cubes.items(), desc="Summing line cubes", unit="line", leave=False):
        # Get wavelength grid for this cube
        cube_wavelengths = cube.axis_world_coords(-1)[0]

        # Check spatial dimensions match
        ny_cube, nx_cube, _ = cube.data.shape
        if ny_cube != ny or nx_cube != nx:
            raise ValueError(f"Spatial dimensions mismatch for {line_name}: expected ({ny}, {nx}), got ({ny_cube}, {nx_cube})")

        summed_data += onto_wavelength_bins(cube.data,
                                            cube_wavelengths.to_value(ref_wavelengths.unit),
                                            ref_wavelengths.value)

    # Create new metadata combining info from all lines
    combined_meta = ref_cube.meta.copy()
    combined_meta.update({
        "combined_lines": line_names,
        "n_lines": len(line_names),
        "metadata_source": reference_line,
        "summed_intensity": True,
    })

    # Create the summed cube using the reference cube's WCS
    return NDCube(
        summed_data,
        wcs=ref_cube.wcs,
        unit=ref_cube.unit,
        meta=combined_meta
    )


def resample_ndcube_spectral_axis(ndcube, spectral_axis, output_resolution, ncpu=-1):
    """
    Resample the spectral axis of an NDCube conserving flux, as :func:`resample_spectra` does.

    Parameters
    ----------
    ndcube : NDCube
        The input NDCube.
    spectral_axis : int
        The index of the spectral axis (e.g., 0, 1, or 2).
    output_resolution : astropy.units.Quantity
        The desired output spectral resolution (e.g., 0.01 * u.nm).
    ncpu : int, optional
        Kept so that existing calls still work; the resampling is one matrix
        product, which uses the threads numpy is given.

    Returns
    -------
    NDCube
        A new NDCube with the spectral axis resampled.
    """
    # Get the world coordinates of the spectral axis
    spectral_world = ndcube.axis_world_coords(spectral_axis)[0]

    # Move spectral axis to last for easier iteration
    data = np.moveaxis(ndcube.data, spectral_axis, -1)
    resampled, new_spec_grid = resample_spectra(data, spectral_world, output_resolution)

    # Move spectral axis back to original position
    resampled = np.moveaxis(resampled, -1, spectral_axis)

    # Update WCS for new spectral axis
    new_wcs = ndcube.wcs.deepcopy()

    wcs_axis = new_wcs.wcs.naxis - 1 - spectral_axis  # Reverse axis order for WCS
    unit = new_wcs.wcs.cunit[wcs_axis]
    (new_wcs.wcs.crpix[wcs_axis], new_wcs.wcs.crval[wcs_axis],
     new_wcs.wcs.cdelt[wcs_axis]) = _even_grid_wcs(new_spec_grid, unit)

    return NDCube(resampled, wcs=new_wcs, unit=ndcube.unit, meta=ndcube.meta)


def _even_grid_wcs(grid: u.Quantity, unit) -> tuple:
    """The CRPIX, CRVAL and CDELT, in *unit*, that describe the evenly spaced *grid*."""
    cdelt = (grid[1] - grid[0]).to_value(unit)
    # The reference pixel is the centre of the axis, which falls between two
    # pixels when there is an even number of them. The reference value has
    # to be the wavelength at that point, not at the pixel below it, or the
    # whole axis is labelled half a pixel low and every fitted velocity comes
    # out half a pixel blue.
    center_pixel = (len(grid) + 1) / 2  # 1-based index (FITS convention)
    return center_pixel, grid[0].to_value(unit) + (center_pixel - 1) * cdelt, cdelt


def resample_spectra(data: np.ndarray, spectral_world: u.Quantity,
                     output_resolution: u.Quantity) -> tuple:
    """
    Spectra resampled onto an evenly spaced wavelength grid, conserving flux.

    Each input wavelength stands for the interval halfway to its neighbours,
    and each pixel of the new grid gets the mean over it of whatever
    overlaps it, so the integral over wavelength is kept exactly, out to the
    outermost intervals.

    Parameters
    ----------
    data : np.ndarray
        Any number of spectra, with the wavelength on the last axis.
    spectral_world : astropy.units.Quantity
        The wavelength of each pixel along the last axis, increasing. The
        pixels need not be evenly spaced.
    output_resolution : astropy.units.Quantity
        The spacing of the new grid, which starts at the first wavelength and
        steps in whole pixels until it passes the last, as it always has, so
        its last pixel can lie beyond the outermost interval and stay empty.
        Where a coarse grid's outermost intervals reach past either end,
        whole pixels are added there too.

    Returns
    -------
    tuple
        ``(resampled, new_grid)``: the spectra on the new grid, with the
        wavelength last, and the new grid in the unit of *output_resolution*.
    """
    centres = spectral_world.to_value(output_resolution.unit)
    step = output_resolution.value
    grid = np.arange(centres.min(), centres.max() + step, step)

    # The outermost intervals of a grid coarser than the pixels at its ends
    # reach past the pixels at the first and last wavelength, and would lose
    # what they hold, so pixels are added either side, keeping the grid where
    # it is. A reach within a part in 1e9 of a pixel is rounding.
    edges = _bin_edges(centres)
    below = max(0, int(np.ceil((grid[0] - step / 2 - edges[0]) / step - 1e-9)))
    above = max(0, int(np.ceil((edges[-1] - (grid[-1] + step / 2)) / step - 1e-9)))
    grid = np.concatenate([grid[0] - step * np.arange(below, 0, -1), grid,
                           grid[-1] + step * np.arange(1, above + 1)])

    resampled = onto_wavelength_bins(data.reshape(-1, data.shape[-1]), centres, grid)
    return resampled.reshape(data.shape[:-1] + grid.shape), grid * output_resolution.unit


def _whole_pixels(extent: u.Quantity, pitch: u.Quantity) -> int:
    """
    How many pixels of *pitch* fit in *extent*.

    A field of view that is a whole number of pixels, as round angles often
    are, comes out a rounding error either side of it once converted to a
    length and back, so a count within a part in 1e9 of the next whole
    number is taken as reaching it rather than losing the last pixel.
    """
    ratio = (extent / pitch).decompose().value
    return int(np.floor(ratio * (1 + 1e-9)))


def reproject_ndcube_heliocentric_to_helioprojective(new_cube_spec, sim, det, ncpu=-1):
    """ Reproject an NDCube from heliocentric to helioprojective coordinates.
    
    Parameters
    ----------
    new_cube_spec : NDCube
        Input NDCube in heliocentric coordinates
    sim : Simulation
        Simulation configuration object
    det : Detector
        Detector configuration object
    ncpu : int, optional
        Number of CPU cores for parallel reprojection. -1 uses all cores,
        positive integers specify exact count. Default is -1.
    """

    ny, nx, _ = new_cube_spec.shape
    wcs_hc = new_cube_spec.wcs

    dx = wcs_hc.wcs.cdelt[1] * wcs_hc.wcs.cunit[1]
    dy = wcs_hc.wcs.cdelt[2] * wcs_hc.wcs.cunit[2]
    x_angle = distance_to_angle(dx)
    y_angle = distance_to_angle(dy)

    # The reference pixel goes to the middle of each spatial axis below, so
    # the reference value has to be the input's coordinate there, wherever
    # the input kept its own reference pixel.
    crval_x_hc = (wcs_hc.wcs.crval[1] + ((nx + 1) / 2 - wcs_hc.wcs.crpix[1])
                  * wcs_hc.wcs.cdelt[1]) * u.Unit(wcs_hc.wcs.cunit[1])
    crval_y_hc = (wcs_hc.wcs.crval[2] + ((ny + 1) / 2 - wcs_hc.wcs.crpix[2])
                  * wcs_hc.wcs.cdelt[2]) * u.Unit(wcs_hc.wcs.cunit[2])
    crval_x_hp = distance_to_angle(crval_x_hc).to_value(u.arcsec)
    crval_y_hp = distance_to_angle(crval_y_hc).to_value(u.arcsec)

    wcs_hp = WCS(naxis=3)
    wcs_hp.wcs.ctype = [wcs_hc.wcs.ctype[0], 'HPLN-TAN', 'HPLT-TAN']
    wcs_hp.wcs.cunit = [wcs_hc.wcs.cunit[0], 'arcsec', 'arcsec']
    wcs_hp.wcs.crpix = [wcs_hc.wcs.crpix[0],
                        (nx + 1) / 2,
                        (ny + 1) / 2]
    wcs_hp.wcs.crval = [wcs_hc.wcs.crval[0], crval_x_hp, crval_y_hp]
    wcs_hp.wcs.cdelt = [wcs_hc.wcs.cdelt[0], x_angle.to_value(u.arcsec), y_angle.to_value(u.arcsec)]
    new_cube_spec_hp = NDCube(new_cube_spec.data, wcs=wcs_hp, unit=new_cube_spec.unit, meta=new_cube_spec.meta)

    ny_in, nx_in, nl_in = new_cube_spec_hp.shape
    fov_x = nx_in * x_angle
    fov_y = ny_in * y_angle
    # A raster cube already has one column per slit position, so its scan
    # axis is kept as it is and only the slit axis is put on the plate scale.
    raster = bool((new_cube_spec.meta or {}).get("raster"))
    pitch_x = x_angle if raster else sim.slit_width
    pitch_y = det.plate_scale_angle
    nx_out = nx_in if raster else _whole_pixels(fov_x, pitch_x)
    ny_out = _whole_pixels(fov_y, pitch_y)
    if nx_out < 1 or ny_out < 1:
        raise ValueError(
            f"The field of view, {fov_x.to(u.arcsec):.3f} by {fov_y.to(u.arcsec):.3f}, "
            f"is smaller than one detector pixel ({pitch_x.to(u.arcsec):.3f} along the "
            f"scan, {(pitch_y * u.pix).to(u.arcsec):.3f} along the slit), so nothing "
            f"would be left after rebinning.")
    shape_out = [ny_out, nx_out, nl_in]

    crpix_spec = (nl_in + 1) / 2
    crpix_y = (ny_out + 1) / 2
    crpix_x = (nx_out + 1) / 2

    wcs_tgt = WCS(naxis=3)
    wcs_tgt.wcs.ctype = [wcs_hc.wcs.ctype[0], 'HPLN-TAN', 'HPLT-TAN']
    wcs_tgt.wcs.cunit = [wcs_hc.wcs.cunit[0], 'arcsec', 'arcsec']
    wcs_tgt.wcs.crpix = [crpix_spec, crpix_x, crpix_y]
    wcs_tgt.wcs.crval = [wcs_hc.wcs.crval[0], crval_x_hp, crval_y_hp]
    wcs_tgt.wcs.cdelt = [wcs_hc.wcs.cdelt[0],
                        pitch_x.to_value(u.arcsec),
                        (det.plate_scale_angle * u.pix).to_value(u.arcsec)]

    # Determine parallelization setting:
    # - If ncpu=-1, use True (all available cores)
    # - If ncpu is a positive integer, pass it directly to control thread count
    parallel_setting = True if ncpu == -1 else ncpu
    
    new_cube_spec_hp_spat = new_cube_spec_hp.reproject_to(
        wcs_tgt,
        shape_out=shape_out,
        algorithm='interpolation',
        parallel=parallel_setting,
        order='bilinear',
    ) * new_cube_spec_hp.unit

    return new_cube_spec_hp_spat


def rebin_atmosphere(cube_sim, det, sim, use_dask=False):
    """
    Rebin synthetic atmosphere cube to instrument resolution and spatial sampling.
    
    Parameters
    ----------
    cube_sim : NDCube
        Input synthetic atmosphere cube
    det : Detector_SWC or Detector_EIS
        Detector configuration
    sim : Simulation
        Simulation configuration
    use_dask : bool, optional
        Whether to use Dask for automatic parallelization (default: False)
        
    Returns
    -------
    NDCube
        Rebinned cube at instrument resolution
    """
    print("  Spectral rebinning to instrument resolution (ny,nx,*nl*)...")

    cube_spec = resample_ndcube_spectral_axis(cube_sim, spectral_axis=2, output_resolution=det.wvl_res*u.pix, ncpu=sim.ncpu)

    print("  Spatially rebinning to plate scale (*ny*,nx,nl) and slit width (ny,*nx*,nl)...")
    cube_det = reproject_ndcube_heliocentric_to_helioprojective(
        cube_spec,
        sim,
        det,
        ncpu=sim.ncpu
    )

    return cube_det


def rebin_spectra(synthesis, reference_line: str, det, sim, summed=None, meta=None) -> NDCube:
    """
    A synthesis file's spectra at instrument resolution and spatial sampling.

    The lines that reach the window of *reference_line* are added up on its
    wavelengths, as :func:`sum_line_cubes` adds up line cubes, and then go
    through the same two steps as :func:`rebin_atmosphere`, with the
    wavelengths taken from the file rather than from a WCS: they are
    resampled conserving flux straight from their own grid, which need not
    be evenly spaced, and the pixels are then laid onto the plate scale and
    the slit.

    Parameters
    ----------
    synthesis : euvst_response.synthesis_file.Synthesis
        The spectra, as read from a synthesis file.
    reference_line : str
        The line whose window is observed and whose velocity is measured.
    det : Detector_SWC or Detector_EIS
        Detector configuration
    sim : Simulation
        Simulation configuration
    summed : u.Quantity, optional
        ``synthesis.summed(reference_line)``, if it has been worked out
        already.
    meta : dict, optional
        More metadata for the cube, such as a time series' ``raster``
        entries, whose columns are its exposures and are kept as they are.

    Returns
    -------
    NDCube
        Rebinned cube at instrument resolution
    """
    print("  Spectral rebinning to instrument resolution (ny,nx,*nl*)...")
    reference = synthesis.lines[reference_line]
    radiance = synthesis.summed(reference_line) if summed is None else summed
    data, grid = resample_spectra(radiance.value, reference.wavelength, det.wvl_res * u.pix)

    # The cube a synthesis gives: wavelength in cm, then x and y on the Sun
    # in Mm, referenced to the middle of each axis.
    ny, nx, _ = data.shape
    crpix_wave, crval_wave, cdelt_wave = _even_grid_wcs(grid, u.cm)
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "SOLX", "SOLY"]
    wcs.wcs.cunit = ["cm", "Mm", "Mm"]
    wcs.wcs.crpix = [crpix_wave, (nx + 1) / 2, (ny + 1) / 2]
    wcs.wcs.crval = [crval_wave, synthesis.centre("x").to_value(u.Mm),
                     synthesis.centre("y").to_value(u.Mm)]
    wcs.wcs.cdelt = [cdelt_wave, synthesis.pixel_size("x").to_value(u.Mm),
                     synthesis.pixel_size("y").to_value(u.Mm)]
    # A synthesis file holds the spectra as the observer sees them, so the
    # Doppler shifts have the sign the fits expect.
    cube_spec = NDCube(data, wcs=wcs, unit=radiance.unit,
                       meta={"rest_wav": reference.rest_wavelength, "line_name": reference_line,
                             "source": synthesis.source,
                             "velocity_convention": VELOCITY_CONVENTION, **(meta or {})})

    print("  Spatially rebinning to plate scale (*ny*,nx,nl) and slit width (ny,*nx*,nl)...")
    return reproject_ndcube_heliocentric_to_helioprojective(cube_spec, sim, det, ncpu=sim.ncpu)


def pad_spectral_axis(cube: NDCube, n: int) -> NDCube:
    """
    *cube* with *n* empty pixels added at each end of its wavelength axis.

    The wavelength axis is the last data axis, as in every detector-grid
    cube, which is the first WCS axis, since the WCS lists its axes the other
    way round. The WCS moves its reference pixel with the data, so the pixels
    already there keep their wavelengths. Used to widen a synthesis window
    for a spectral PSF that reaches further than its margin
    (:func:`~euvst_response.radiometric.spectral_psf_margin`).
    """
    if n < 0:
        raise ValueError(f"Cannot pad by a negative number of pixels, got {n}.")
    if n == 0:
        return cube
    wcs = cube.wcs.deepcopy()
    if not wcs.wcs.ctype[0].startswith("WAVE"):
        raise ValueError(
            f"Expected wavelength on the last data axis, which is the first WCS "
            f"axis, but the WCS axes are {list(wcs.wcs.ctype)}.")
    wcs.wcs.crpix[0] += n
    data = np.pad(cube.data, [(0, 0)] * (cube.data.ndim - 1) + [(n, n)])
    return NDCube(data, wcs=wcs, unit=cube.unit, meta=cube.meta)


def create_uniform_intensity_cube(
    total_intensity: u.Quantity,
    rest_wavelength: u.Quantity,
    thermal_width: u.Quantity,
    det,
    sim,
    n_sigma_extent: float = 8.0,
    n_slit_pixels: int = 1,
    tel=None,
) -> NDCube:
    """
    Create an ``n_slit_pixels`` x 1 pixel NDCube containing a Gaussian emission line.

    The cube is built directly at the detector's spectral resolution and
    assigned a helioprojective WCS consistent with the output of
    ``rebin_atmosphere``, so it can be fed straight into ``monte_carlo``.
    Each wavelength pixel holds the line integrated across that pixel, so
    the cube holds exactly the part of the line on its wavelength grid
    however narrow the line is.  With the default ``n_sigma_extent`` that is
    ``total_intensity`` to a part in 1e15.

    Parameters
    ----------
    total_intensity : u.Quantity
        Spectrally-integrated line intensity, e.g. in ``erg / (s cm2 sr)``.
    rest_wavelength : u.Quantity
        Rest wavelength of the line, e.g. ``195.119 * u.AA``.
    thermal_width : u.Quantity
        Thermal / non-thermal line width expressed as a velocity
        (1-sigma Gaussian width), e.g. ``20 * u.km / u.s``.
    det : Detector_SWC or Detector_EIS
        Detector configuration (provides ``wvl_res``, ``plate_scale_angle``).
    sim : Simulation
        Simulation configuration (provides ``slit_width``).
    n_sigma_extent : float, optional
        Number of sigma either side of line centre to include in the
        wavelength grid (default: 8).  Measured on the width the line will have
        once the spectral PSF has been applied, if *tel* is given.  The part
        of the line beyond the grid is left out of the cube.
    n_slit_pixels : int, optional
        Number of (uniform) slit pixels to generate.  Set to the
        ``offchip_bin_slit`` value so that subsequent ``rebin_slit_offchip``
        sums ``n_slit_pixels`` independent noise realisations into a single
        binned pixel (default: 1).
    tel : Telescope_EUVST or Telescope_EIS, optional
        Telescope configuration.  When given, the grid is widened to hold the
        line after spectral PSF broadening, adding the PSF width to the thermal
        width in quadrature.  Without this a narrow line gets a grid only a
        couple of pixels wide, and convolving it with a PSF wider than the line
        pushes flux off the ends of the grid.  Default None, which sizes the
        grid on the thermal width alone.

    Returns
    -------
    NDCube
        Shape ``(n_slit_pixels, 1, n_lambda)`` with unit ``erg / (s cm2 sr cm)`` and a
        helioprojective + wavelength WCS.
    """
    if n_slit_pixels < 1:
        raise ValueError(f"n_slit_pixels must be >= 1, got {n_slit_pixels}")

    # --- Spectral grid --------------------------------------------------
    lam0 = rest_wavelength.to(u.cm)

    # Convert velocity width to wavelength width: sigma_lam = lam0 * v / c
    sigma_lam = (lam0 * thermal_width / const.c).to(u.cm)

    # Detector pixel pitch in cm
    dlam = det.wvl_res.to(u.cm / u.pix) * u.pix  # strip per-pixel to cm

    # The grid has to hold the line as it will be *measured*, not as it leaves
    # the Sun, so add the spectral PSF to the thermal width in quadrature.
    # Widths add that way for Gaussians, and the PSF is often the broader of
    # the two: at the default 20 km/s the line is 0.77 pixels against a PSF of
    # 1.08.  Always widening, rather than only when psf is set, keeps the grid
    # independent of a value that is swept and is not known when the cube is
    # built and cached.  The PSF is the one for this slit, with the slit added
    # in quadrature: that is at least as broad as the slit convolved with the
    # optics, so the grid holds the line under either spectral_psf.
    sigma_total = sigma_lam
    if tel is not None:
        sigma_psf = _fwhm_to_sigma(spectral_psf_fwhm(tel, det, sim.slit_width)) * dlam
        sigma_total = np.sqrt(sigma_lam**2 + sigma_psf**2)

    half_range = n_sigma_extent * sigma_total
    n_pix_half = int(np.ceil((half_range / dlam).decompose().value))
    n_lam = 2 * n_pix_half + 1  # always odd, centred on rest wavelength

    # --- Gaussian profile -----------------------------------------------
    # Each pixel holds the line integrated between its edges and divided by
    # its width, which is what resample_spectra gives a synthesised
    # spectrum, so the pixels add up to all of the line on the grid whatever
    # its width: total_intensity, less the tails beyond n_sigma_extent.
    # The Gaussian sampled at pixel centres only does that for a line more
    # than about half a pixel wide (sigma): centred on a pixel, a line of 0.3
    # pixels, such as Fe VIII 185.21 at its formation temperature, would come
    # out 35 per cent too bright.  The edges are counted in pixels from the
    # line centre, as absolute wavelengths would lose a part in 1e12 of a
    # pixel to rounding, and each pixel's upper edge is the next one's lower
    # edge, so nothing between two pixels is counted twice or missed.
    edges = ((np.arange(n_lam + 1) - n_pix_half - 0.5) * dlam
             / (np.sqrt(2.0) * sigma_lam)).decompose().value
    profile = (total_intensity * 0.5 * np.diff(erf(edges)) / dlam).to(
        u.erg / (u.s * u.cm**2 * u.sr * u.cm)
    )
    # Tile the profile along the slit axis.  Every slit pixel holds the same
    # intensity, but each is noised independently downstream, which is what
    # rebin_slit_offchip needs in order to sum them.
    data = np.tile(profile.value, (n_slit_pixels, 1, 1))  # shape (n_slit_pixels, 1, n_lam)

    # --- WCS (matches reproject_ndcube output format) --------------------
    # Axes: WAVE (cm), HPLN-TAN (arcsec), HPLT-TAN (arcsec)
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.crpix = [(n_lam + 1) / 2, 1.0, (n_slit_pixels + 1) / 2]
    wcs.wcs.crval = [lam0.value, 0.0, 0.0]
    wcs.wcs.cdelt = [
        dlam.to_value(u.cm),
        sim.slit_width.to_value(u.arcsec),
        (det.plate_scale_angle * u.pix).to_value(u.arcsec),
    ]

    unit = u.erg / (u.s * u.cm**2 * u.sr * u.cm)

    return NDCube(
        data=data,
        wcs=wcs,
        unit=unit,
        meta={
            "rest_wav": rest_wavelength,
            "uniform_intensity": total_intensity,
            "thermal_width": thermal_width,
            "line_name": "uniform",
            "uniform_mode": True,
        },
    )