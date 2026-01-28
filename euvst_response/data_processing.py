"""
Data processing functions for atmosphere cubes and spectral resampling.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import astropy.units as u
import dill
from ndcube import NDCube
from astropy.wcs import WCS
from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler
from joblib import Parallel, delayed
from tqdm import tqdm
from .utils import tqdm_joblib, distance_to_angle


def load_atmosphere(pkl_file: str, metadata_line: str = None) -> tuple:
    """
    Load synthetic atmosphere cube from pickle file.
    
    Creates a summed cube from all line cubes in the synthesis results.
    All line cubes are interpolated onto the wavelength grid of the metadata_line
    before summing to handle different wavelength grids for different lines.
    
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
    
    # Use the metadata line's wavelength grid as the reference
    ref_cube = line_cubes[metadata_line]
    ref_wavelengths = ref_cube.axis_world_coords(-1)[0]
    
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
        
        # Vectorized interpolation for the entire cube
        # Reshape data to (n_pixels, n_wavelengths) for batch interpolation
        cube_data_reshaped = cube.data.reshape(-1, len(cube_wavelengths))
        
        # Batch interpolation using numpy.interp
        interpolated = np.array([
            np.interp(ref_wavelengths.value, cube_wavelengths.value, spectrum, left=0.0, right=0.0)
            for spectrum in cube_data_reshaped
        ])
        
        # Add to summed data
        summed_data += interpolated.reshape(ny, nx, len(ref_wavelengths))
    
    # Create new metadata combining info from all lines
    combined_meta = ref_cube.meta.copy()
    combined_meta.update({
        "combined_lines": line_names,
        "n_lines": len(line_names),
        "metadata_source": metadata_line,
        "summed_intensity": True,
        "dynamic_mode": dynamic_mode_info,
    })
    
    # Create the summed cube using the reference cube's WCS
    summed_cube = NDCube(
        summed_data,
        wcs=ref_cube.wcs,
        unit=ref_cube.unit,
        meta=combined_meta
    )
    
    return summed_cube, dynamic_mode_info


def resample_ndcube_spectral_axis(ndcube, spectral_axis, output_resolution, ncpu=-1):
    """
    Resample the spectral axis of an NDCube using FluxConservingResampler.

    Parameters
    ----------
    ndcube : NDCube
        The input NDCube.
    spectral_axis : int
        The index of the spectral axis (e.g., 0, 1, or 2).
    output_resolution : astropy.units.Quantity
        The desired output spectral resolution (e.g., 0.01 * u.nm).
    ncpu : int, optional
        Number of CPU cores to use for parallel processing. Default is -1 (use all cores).

    Returns
    -------
    NDCube
        A new NDCube with the spectral axis resampled.
    """
    # Get the world coordinates of the spectral axis
    spectral_world = ndcube.axis_world_coords(spectral_axis)[0]
    spectral_world = spectral_world.to(output_resolution.unit)

    # Define new spectral grid
    new_spec_grid = np.arange(
        spectral_world.min().value,
        spectral_world.max().value + output_resolution.value,
        output_resolution.value
    ) * output_resolution.unit

    n_spec = len(new_spec_grid)

    # Move spectral axis to last for easier iteration
    data = np.moveaxis(ndcube.data, spectral_axis, -1)
    shape = data.shape
    flat_data = data.reshape(-1, shape[-1])

    resampler = FluxConservingResampler(extrapolation_treatment="zero_fill")
    resampled = np.zeros((flat_data.shape[0], n_spec))

    def _resample_pixel(i):
        spec = Spectrum(flux=flat_data[i] * ndcube.unit, spectral_axis=spectral_world)
        res = resampler(spec, new_spec_grid)
        return res.flux.value

    with tqdm_joblib(tqdm(total=flat_data.shape[0], desc="Resampling spectral axis", unit="pixel", leave=False)):
        results = Parallel(n_jobs=ncpu)(
            delayed(_resample_pixel)(i) for i in range(flat_data.shape[0])
        )
    resampled = np.vstack(results)

    # Reshape back to original spatial shape, but with new spectral length
    new_shape = list(shape[:-1]) + [n_spec]
    resampled = resampled.reshape(new_shape)

    # Move spectral axis back to original position
    resampled = np.moveaxis(resampled, -1, spectral_axis)

    # Update WCS for new spectral axis
    new_wcs = ndcube.wcs.deepcopy()

    wcs_axis = new_wcs.wcs.naxis - 1 - spectral_axis  # Reverse axis order for WCS
    center_pixel = (n_spec + 1) / 2  # 1-based index (FITS convention)
    new_wcs.wcs.crpix[wcs_axis] = center_pixel
    new_wcs.wcs.crval[wcs_axis] = new_spec_grid[int(center_pixel - 1)].to_value(new_wcs.wcs.cunit[wcs_axis])
    new_wcs.wcs.cdelt[wcs_axis] = (new_spec_grid[1] - new_spec_grid[0]).to_value(new_wcs.wcs.cunit[wcs_axis])

    return NDCube(resampled, wcs=new_wcs, unit=ndcube.unit, meta=ndcube.meta)


def reproject_ndcube_heliocentric_to_helioprojective(new_cube_spec, sim, det):
    """ Reproject an NDCube from heliocentric to helioprojective coordinates. """

    nx, ny, _ = new_cube_spec.shape
    wcs_hc = new_cube_spec.wcs

    dx = wcs_hc.wcs.cdelt[2] * wcs_hc.wcs.cunit[2]
    dy = wcs_hc.wcs.cdelt[1] * wcs_hc.wcs.cunit[1]
    x_angle = distance_to_angle(dx)
    y_angle = distance_to_angle(dy)
    
    crval_x_hc = wcs_hc.wcs.crval[2] * u.Unit(wcs_hc.wcs.cunit[2])
    crval_y_hc = wcs_hc.wcs.crval[1] * u.Unit(wcs_hc.wcs.cunit[1])
    crval_x_hp = distance_to_angle(crval_x_hc).to_value(u.arcsec)
    crval_y_hp = distance_to_angle(crval_y_hc).to_value(u.arcsec)

    wcs_hp = WCS(naxis=3)
    wcs_hp.wcs.ctype = [wcs_hc.wcs.ctype[0], 'HPLT-TAN', 'HPLN-TAN']
    wcs_hp.wcs.cunit = [wcs_hc.wcs.cunit[0], 'arcsec', 'arcsec']
    wcs_hp.wcs.crpix = [wcs_hc.wcs.crpix[0],
                        (ny + 1) / 2,
                        (nx + 1) / 2]
    wcs_hp.wcs.crval = [wcs_hc.wcs.crval[0], crval_y_hp, crval_x_hp]
    wcs_hp.wcs.cdelt = [wcs_hc.wcs.cdelt[0], y_angle.to_value(u.arcsec), x_angle.to_value(u.arcsec)]
    new_cube_spec_hp = NDCube(new_cube_spec.data, wcs=wcs_hp, unit=new_cube_spec.unit, meta=new_cube_spec.meta)

    nx_in, ny_in, nl_in = new_cube_spec_hp.shape
    fov_x = nx_in * x_angle
    fov_y = ny_in * y_angle
    pitch_x = sim.slit_width
    pitch_y = det.plate_scale_angle
    nx_out = int(np.floor((fov_x / pitch_x).decompose().value))
    ny_out = int(np.floor((fov_y / pitch_y).decompose().value))
    shape_out = [nx_out, ny_out, nl_in]

    crpix_spec = (shape_out[2] + 1) / 2
    crpix_y = (shape_out[1] + 1) / 2
    crpix_x = (shape_out[0] + 1) / 2

    wcs_tgt = WCS(naxis=3)
    wcs_tgt.wcs.ctype = [wcs_hc.wcs.ctype[0], 'HPLT-TAN', 'HPLN-TAN']
    wcs_tgt.wcs.cunit = [wcs_hc.wcs.cunit[0], 'arcsec', 'arcsec']
    wcs_tgt.wcs.crpix = [crpix_spec, crpix_y, crpix_x]
    wcs_tgt.wcs.crval = [wcs_hc.wcs.crval[0], crval_y_hp, crval_x_hp]
    wcs_tgt.wcs.cdelt = [wcs_hc.wcs.cdelt[0],
                        (det.plate_scale_angle * u.pix).to_value(u.arcsec),
                        (sim.slit_width).to_value(u.arcsec)]

    new_cube_spec_hp_spat = new_cube_spec_hp.reproject_to(
        wcs_tgt,
        shape_out=shape_out,
        algorithm='interpolation',
        parallel=True,
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
    print("  Spectral rebinning to instrument resolution (nx,ny,*nl*)...")

    cube_spec = resample_ndcube_spectral_axis(cube_sim, spectral_axis=2, output_resolution=det.wvl_res*u.pix, ncpu=sim.ncpu)

    print("  Spatially rebinning to plate scale (nx,*ny*,nl) and slit width (*nx*,ny,nl)...")
    cube_det = reproject_ndcube_heliocentric_to_helioprojective(
        cube_spec,
        sim,
        det
    )

    return cube_det
