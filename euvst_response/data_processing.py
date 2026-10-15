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
from scipy.sparse import csr_matrix
from scipy.special import erf
from tqdm import tqdm
from .pinhole_diffraction import _filter_transmission, _pinholes_on
from .radiometric import (_carry_photon_meta, across_slit_fwhm, photons_per_energy,
                          slit_image_width, spatial_psf_fwhm, spectral_blur_fwhm,
                          spectral_psf_fwhm, spectral_psf_margin)
from .sampling import centred_edges, light_onto_pixels, photons_onto_pixels, pixel_weights
from .utils import (_bin_edges, distance_to_angle, _fwhm_to_sigma, has_wrong_velocity_sign,
                    onto_wavelength_bins)


def load_atmosphere(pkl_file: str, metadata_line: str = None) -> tuple:
    """
    Read one spectral window of a synthesis file as a single cube, blends included.

    Every line in the file within the window of `metadata_line` is added
    onto that line's wavelengths at its own brightness. This is what
    `eclipse` observes for `reference_line`. Synthesis pickles from ECLIPSE
    0.11.0 and earlier can also be read. Reading a pickle runs any code it
    holds, so only read pickles you trust.

    Parameters
    ----------
    pkl_file : str
        Synthesis file or synthesis pickle.
    metadata_line : str, optional
        Line whose window to read. Default the file's first line.

    Returns
    -------
    summed_cube : NDCube
        Spectra of the window, indexed [y, x, wavelength].
    dynamic_mode_info : dict
        Settings of the deprecated dynamic mode the file was made in, or
        `{"enabled": False}` for a synthesis of one snapshot.
    """
    from .atmosphere import _is_pickle
    from .synthesis_file import (is_synthesis_file, read_synthesis, read_synthesis_products,
                                 synthesis_line_names)

    if is_synthesis_file(pkl_file):
        names = synthesis_line_names(pkl_file)
        if metadata_line is None and names:
            metadata_line = names[0]
        synthesis = read_synthesis(pkl_file, metadata_line)
        if not synthesis.evenly_spaced(metadata_line):
            raise ValueError(
                f"{pkl_file}: the wavelengths of {metadata_line} are not evenly spaced, which "
                f"the WCS of an NDCube cannot describe. Read the file with read_synthesis.")
        dynamic_mode_info = read_synthesis_products(pkl_file, keys="dynamic_mode").get(
            "dynamic_mode", {"enabled": False})
        return (synthesis.summed_cube(metadata_line, meta={"dynamic_mode": dynamic_mode_info}),
                dynamic_mode_info)

    if not _is_pickle(Path(pkl_file)):
        raise ValueError(
            f"{pkl_file} is neither a synthesis file, which is HDF5, nor a synthesis pickle "
            f"as older versions of ECLIPSE wrote.")
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
    """Refuse line cubes written by older ECLIPSE versions in a form now misread."""
    for cube in line_cubes.values():
        meta = cube.meta or {}
        # Refuse files from before the axis order changed (issue #12). They
        # store (x, y, wavelength), not (y, x, wavelength), so would come out
        # transposed. The WCS axis order tells the two apart.
        old_first_spatial = {"z": "SOLY", "x": "SOLZ", "y": "SOLZ"}
        axis = meta.get("integration_axis")
        if axis in old_first_spatial and cube.wcs.wcs.ctype[1] == old_first_spatial[axis]:
            raise ValueError(
                f"{path} was written by an older ECLIPSE that stored cubes "
                "as (x, y, wavelength). Cubes are now (y, x, wavelength). "
                "Re-run the synthesis with this version to regenerate the file."
            )

        # Refuse files with the old velocity convention, which has the
        # opposite sign for views along x and z.
        if has_wrong_velocity_sign(meta):
            axis = meta.get("integration_axis", "z")
            raise ValueError(
                f"{path} was written by an older ECLIPSE, which took the "
                "line-of-sight velocity straight from the simulation's velocity "
                "component. ECLIPSE now uses the velocity away from the observer. "
                f"For this view along {axis}, the two conventions have opposite "
                "signs. Re-run the synthesis with this version to regenerate "
                "the file."
            )


def sum_line_cubes(line_cubes: dict, reference_line: str) -> NDCube:
    """
    Every line's cube summed onto the wavelength grid of `reference_line`.

    Each cube is averaged onto the reference bins, conserving flux and
    taken as zero outside its own grid, so lines outside the reference
    window add nothing. The result keeps the reference cube's WCS, unit and
    metadata, plus the names of the lines it holds.
    """
    ref_cube = line_cubes[reference_line]
    ref_wavelengths = ref_cube.axis_world_coords_values(-1)[0]
    line_names = list(line_cubes.keys())

    # Get spatial dimensions from reference cube
    ny, nx, nw = ref_cube.data.shape

    # Initialize summed data with the reference wavelength grid
    summed_data = np.zeros((ny, nx, nw))

    for line_name, cube in tqdm(line_cubes.items(), desc="Summing line cubes", unit="line", leave=False):
        # Get wavelength grid for this cube
        cube_wavelengths = cube.axis_world_coords_values(-1)[0]

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
    Resample the spectral axis of an NDCube conserving flux, as in :func:`resample_spectra`.

    Parameters
    ----------
    ndcube : NDCube
        Input cube.
    spectral_axis : int
        Index of the spectral axis (e.g. 0, 1 or 2).
    output_resolution : astropy.units.Quantity
        Output spectral resolution (e.g. 0.01 * u.nm).
    ncpu : int, optional
        Unused, kept for existing calls. The resampling is one matrix
        product, which uses the threads numpy is given.

    Returns
    -------
    NDCube
        New cube with the spectral axis resampled.
    """
    # Get the world coordinates of the spectral axis
    spectral_world = ndcube.axis_world_coords_values(spectral_axis)[0]

    # Move spectral axis last, as resample_spectra expects
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
    """CRPIX, CRVAL and CDELT, in `unit`, of the evenly spaced `grid`."""
    cdelt = (grid[1] - grid[0]).to_value(unit)
    # The reference pixel is the centre of the axis, between two pixels for
    # an even number of them. CRVAL must be the wavelength at that point, not
    # at the pixel below, or the axis is labelled half a pixel low and every
    # fitted velocity comes out half a pixel blue.
    center_pixel = (len(grid) + 1) / 2  # 1-based index (FITS convention)
    return center_pixel, grid[0].to_value(unit) + (center_pixel - 1) * cdelt, cdelt


def resample_spectra(data: np.ndarray, spectral_world: u.Quantity,
                     output_resolution: u.Quantity) -> tuple:
    """
    Spectra resampled onto an evenly spaced wavelength grid, conserving flux.

    Each input wavelength covers the interval halfway to its neighbours, and
    each new pixel gets the mean of what overlaps it, so the integral over
    wavelength is kept exactly, out to the outermost intervals.

    Parameters
    ----------
    data : np.ndarray
        Any number of spectra, with wavelength on the last axis.
    spectral_world : astropy.units.Quantity
        Increasing wavelength of each pixel along the last axis, not
        necessarily evenly spaced.
    output_resolution : astropy.units.Quantity
        Spacing of the new grid. The grid starts at the first wavelength and
        steps in whole pixels until it passes the last, so its last pixel
        can lie beyond the outermost interval and stay empty. Where the
        outermost intervals of a coarse grid reach past either end, whole
        pixels are added there too.

    Returns
    -------
    tuple
        `(resampled, new_grid)`: the spectra on the new grid, wavelength
        last, and the new grid in the unit of `output_resolution`.
    """
    centres = spectral_world.to_value(output_resolution.unit)
    grid = _spectral_grid(centres, output_resolution.value)
    resampled = onto_wavelength_bins(data.reshape(-1, data.shape[-1]), centres, grid)
    return resampled.reshape(data.shape[:-1] + grid.shape), grid * output_resolution.unit


def _spectral_grid(centres: np.ndarray, step: float, margin: int = 0) -> np.ndarray:
    """
    Detector wavelength pixels for spectra at `centres`.

    The pixels run from the first wavelength in steps of `step` until past
    the last, as described in :func:`resample_spectra`, with `margin` more
    at each end.
    """
    grid = np.arange(centres.min(), centres.max() + step, step)
    # Where the input is coarser than the pixels at its ends, its outermost
    # intervals reach past the first and last pixels, and their light would
    # be lost, so pixels are added either side without moving the grid. A
    # reach under a part in 1e9 of a pixel is rounding.
    edges = _bin_edges(centres)
    below = margin + max(0, int(np.ceil((grid[0] - step / 2 - edges[0]) / step - 1e-9)))
    above = margin + max(0, int(np.ceil((edges[-1] - (grid[-1] + step / 2)) / step - 1e-9)))
    return np.concatenate([grid[0] - step * np.arange(below, 0, -1), grid,
                           grid[-1] + step * np.arange(1, above + 1)])


def _whole_pixels(extent: u.Quantity, pitch: u.Quantity) -> int:
    """
    Number of whole pixels of `pitch` in `extent`.

    A field of view of a whole number of pixels, as round angles often are,
    is off by a rounding error after conversion to a length and back. A
    count within a part in 1e9 of the next whole number is therefore taken
    as reaching it, so the last pixel is not lost.
    """
    ratio = (extent / pitch).decompose().value
    return int(np.floor(ratio * (1 + 1e-9)))


def reproject_ndcube_heliocentric_to_helioprojective(new_cube_spec, sim, det, ncpu=-1, *,
                                                     sigma_x=0.0, sigma_y=0.0, extend=False):
    """Reproject an NDCube from heliocentric to helioprojective coordinates.

    Pixels are the plate scale apart along the slit and the slit width
    apart across it; a raster keeps its own columns. Each pixel holds the
    mean of the cells over its footprint, each cell taken as uniform, so
    structure finer than a pixel is averaged as the detector averages it.

    Parameters
    ----------
    new_cube_spec : NDCube
        Input NDCube in heliocentric coordinates
    sim : Simulation
        Simulation configuration object
    det : Detector
        Detector configuration object
    ncpu : int, optional
        Unused, kept for existing calls. The averaging is a matrix product,
        which uses the threads numpy is given.
    sigma_x, sigma_y : float, optional
        Standard deviations in arcsec of a Gaussian blur across and along
        the slit, applied to the scene before pixel averaging. A raster gets
        no blur across the slit, as its columns have already been through
        the slit.
    extend : bool, optional
        Extend the scene past its edges with its edge cells, for the blur
        to bring in.
    """

    ny, nx, _ = new_cube_spec.shape
    wcs_hc = new_cube_spec.wcs

    dx = wcs_hc.wcs.cdelt[1] * wcs_hc.wcs.cunit[1]
    dy = wcs_hc.wcs.cdelt[2] * wcs_hc.wcs.cunit[2]
    x_angle = distance_to_angle(dx)
    y_angle = distance_to_angle(dy)

    # The reference pixel is put at the middle of each spatial axis below, so
    # CRVAL must be the input's coordinate there, wherever the input's own
    # reference pixel is.
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
    # A raster already has one column per slit position, so its scan axis is
    # kept and only the slit axis is put on the plate scale.
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

    # Each pixel holds the mean of the uniform cells over its footprint,
    # after the optics blur if given. Both grids are centred on the middle
    # of the input.
    rows = pixel_weights(centred_edges(ny_in, y_angle.to_value(u.arcsec)),
                         centred_edges(ny_out, (pitch_y * u.pix).to_value(u.arcsec)),
                         sigma_y, extend=extend)
    columns = pixel_weights(centred_edges(nx_in, x_angle.to_value(u.arcsec)),
                            centred_edges(nx_out, pitch_x.to_value(u.arcsec)),
                            0.0 if raster else sigma_x, extend=extend)
    # Each pixel covers only nearby cells, so the weights are stored sparse.
    rows, columns = csr_matrix(rows), csr_matrix(columns)

    def onto_pixels(cube):
        cube = (rows @ cube.reshape(ny_in, -1)).reshape(ny_out, nx_in, nl_in)
        cube = (columns @ np.moveaxis(cube, 1, 0).reshape(nx_in, -1)).reshape(
            nx_out, ny_out, nl_in)
        return np.ascontiguousarray(np.moveaxis(cube, 0, 1))

    # At one wavelength the photons scale with the data, so the photon
    # metadata is carried with the same weights.
    data, meta = _carry_photon_meta(onto_pixels, new_cube_spec)
    meta = dict(meta or {})
    # Put the WCS in wcslib's own units, metres and degrees, as reprojected
    # cubes are expected to be.
    wcs_tgt.wcs.set()
    # Record the scene's extent along the slit, which gives the rows it
    # covers at any plate scale, for main to check off-chip binning against.
    return NDCube(data, wcs=wcs_tgt, unit=new_cube_spec.unit,
                  meta={**meta, "fov_along_slit": fov_y.to(u.arcsec)})


def _through_the_optics(tel, sim) -> bool:
    """Whether the focusing optics' PSF goes into the rebinning."""
    if tel is None or not sim.psf:
        return False
    if tel.psf_type.lower() != "gaussian":
        raise ValueError(f"Unsupported PSF type: {tel.psf_type}. Supported: 'gaussian'.")
    return True


def _spectra_on_the_detector(data: np.ndarray, spectral_world: u.Quantity, det, sim,
                             tel) -> tuple:
    """
    Spectra laid onto the detector's wavelength pixels in one step.

    The spectra go from their own wavelengths onto the pixels, through the
    spectral PSF if `sim.psf` is on. Blurring them on the pixels, as the
    Monte Carlo does for a cube without `psf_applied`, moves a line narrower
    than a pixel toward its middle; here each wavelength keeps its place.
    The pixels extend :func:`~euvst_response.radiometric.spectral_psf_margin`
    further at each end, as :func:`pad_spectral_axis` does for a PSF on the
    pixels.

    Each wavelength's light is converted to photons at that wavelength,
    with the telescope's area and the photon energy there
    (:func:`~euvst_response.radiometric.photons_per_energy`), before the
    blur. The Monte Carlo counts a pixel's photons at the pixel's
    wavelength, so the returned spectra are the radiance that gives each
    pixel its photons that way. A photon frees electrons by its own energy,
    so the wavelengths of the mean and RMS photon energies in each pixel are
    returned too. With the pinholes on, so are the share of the photons'
    light the filter passed, and the wavelengths of the mean and RMS
    energies of the light it blocked, which a pinhole lets through, each
    worked out at its own wavelength.

    Returns `(spectra, grid, meta)`, with the grid in cm and `meta` the
    cube's photon `meta` entries, named as in
    :func:`~euvst_response.radiometric._carry_photon_meta`.
    """
    step = (det.wvl_res * u.pix).to_value(u.cm)
    centres = spectral_world.to_value(u.cm)
    grid = _spectral_grid(centres, step, spectral_psf_margin(tel, det, sim.slit_width))
    sigma = width = 0.0
    if sim.psf and sim.spectral_psf == "quadrature":
        sigma = _fwhm_to_sigma(spectral_psf_fwhm(tel, det, sim.slit_width)) * step
    elif sim.psf:
        sigma = _fwhm_to_sigma(spectral_blur_fwhm(tel, det)) * step
        width = slit_image_width(sim.slit_width, det) * step
    pixel_edges = np.concatenate([grid - step / 2, [grid[-1] + step / 2]])
    cells = _bin_edges(centres)
    share = light_onto_pixels(cells[:-1], cells[1:], pixel_edges, sigma, width)
    # Photons from each cell's light at its own wavelength, zero where the
    # telescope's tables do not reach.
    own = np.nan_to_num(photons_per_energy(tel, centres * u.cm).value)
    in_cells = np.asarray(data, dtype=float) * (np.diff(cells) * own)
    photons, photon_wavelength, rms_wavelength = photons_onto_pixels(share, in_cells, centres)
    # Radiance giving each pixel's photons at the pixel's own wavelength. A
    # pixel no light reaches keeps its own wavelength.
    pixel = photons_per_energy(tel, grid * u.cm).value
    spectra = np.divide(photons, step * pixel, out=np.zeros(photons.shape),
                        where=pixel > 0)

    def with_own(wavelength):
        return np.where(np.isfinite(wavelength), wavelength, grid) * u.cm

    meta = {"photon_wavelength": with_own(photon_wavelength),
            "photon_rms_wavelength": with_own(rms_wavelength)}
    if _pinholes_on(sim):
        # Photons the filter blocked from each cell's light, which a pinhole
        # lets through: the passed photons divided by the transmission, less
        # the passed photons.
        passed = _filter_transmission(tel, centres * u.cm)
        blocked = in_cells * np.divide(1.0 - passed, passed, out=np.zeros(passed.shape),
                                       where=passed > 0)
        on_pixels, blocked_wavelength, blocked_rms = photons_onto_pixels(share, blocked, centres)
        before = photons + on_pixels
        meta.update({
            "photon_filter_transmission": np.divide(photons, before, out=np.ones(before.shape),
                                                    where=before > 0),
            "blocked_photon_wavelength": with_own(blocked_wavelength),
            "blocked_photon_rms_wavelength": with_own(blocked_rms)})
    return spectra, grid * u.cm, meta


def _spatial_blur(tel, det, sim) -> dict:
    """
    Blur along and across the slit, as standard deviations in arcsec, for
    :func:`reproject_ndcube_heliocentric_to_helioprojective`.
    """
    along = (_fwhm_to_sigma(spatial_psf_fwhm(tel, det))
             * (det.plate_scale_angle * u.pix).to_value(u.arcsec))
    across = across_slit_fwhm(tel)
    return {"sigma_y": along,
            "sigma_x": 0.0 if across is None else _fwhm_to_sigma(across.to_value(u.arcsec)),
            "extend": sim.psf_boundary == "replicate"}


def rebin_atmosphere(cube_sim, det, sim, use_dask=False, *, tel=None):
    """
    Rebin synthetic atmosphere cube to instrument resolution and spatial sampling.

    Each detector pixel holds the mean of the scene over its footprint, in
    wavelength and in space. Given `tel`, each wavelength's light is counted
    in photons at its own wavelength. The cube's `meta` then holds
    `photon_wavelength` and `photon_rms_wavelength`, the wavelengths of the
    mean and RMS energies of each pixel's photons, and, with the pinholes of
    `sim` on, what they need of the light the filter blocked. With `sim.psf`
    on as well, the focusing optics' PSF blurs the scene on its own grids
    before the pixels average it, which is exact however fine the scene is.
    `meta` records this in `psf_applied`, so the Monte Carlo does not blur
    it again.

    Parameters
    ----------
    cube_sim : NDCube
        Input synthetic atmosphere cube
    det : Detector_SWC or Detector_EIS
        Detector configuration
    sim : Simulation
        Simulation configuration
    use_dask : bool, optional
        Unused, kept for existing calls.
    tel : Telescope_EUVST or Telescope_EIS, optional
        Telescope the Monte Carlo observes through. Without it, the cube is
        the scene on the pixels, as for the ground truth.

    Returns
    -------
    NDCube
        Rebinned cube at instrument resolution
    """
    print("  Spectral rebinning to instrument resolution (ny,nx,*nl*)...")
    if tel is None:
        cube_spec = resample_ndcube_spectral_axis(cube_sim, spectral_axis=2,
                                                  output_resolution=det.wvl_res*u.pix,
                                                  ncpu=sim.ncpu)
        print("  Spatially rebinning to plate scale (*ny*,nx,nl) and slit width (ny,*nx*,nl)...")
        return reproject_ndcube_heliocentric_to_helioprojective(cube_spec, sim, det,
                                                                ncpu=sim.ncpu)

    blurred = _through_the_optics(tel, sim)
    data, grid, photon_meta = _spectra_on_the_detector(
        cube_sim.data, cube_sim.axis_world_coords_values(2)[0], det, sim, tel)
    wcs = cube_sim.wcs.deepcopy()
    unit = wcs.wcs.cunit[0]
    wcs.wcs.crpix[0], wcs.wcs.crval[0], wcs.wcs.cdelt[0] = _even_grid_wcs(grid, unit)
    cube_spec = NDCube(data, wcs=wcs, unit=cube_sim.unit,
                       meta={**(cube_sim.meta or {}), **photon_meta,
                             **({"psf_applied": True} if blurred else {})})
    print("  Spatially rebinning to plate scale (*ny*,nx,nl) and slit width (ny,*nx*,nl)...")
    return reproject_ndcube_heliocentric_to_helioprojective(
        cube_spec, sim, det, ncpu=sim.ncpu, **(_spatial_blur(tel, det, sim) if blurred else {}))


def rebin_spectra(synthesis, reference_line: str, det, sim, summed=None, meta=None, *,
                  tel=None) -> NDCube:
    """
    A synthesis file's spectra at instrument resolution and spatial sampling.

    The lines reaching the window of `reference_line` are summed on its
    wavelengths, as in :func:`sum_line_cubes`. They then go through the
    same two steps as in :func:`rebin_atmosphere`, with wavelengths from the
    file rather than a WCS: a flux-conserving resampling straight from their
    own grid, which need not be evenly spaced, then rebinning onto the plate
    scale and the slit.

    Parameters
    ----------
    synthesis : euvst_response.synthesis_file.Synthesis
        Spectra read from a synthesis file.
    reference_line : str
        Line whose window is observed and whose velocity is measured.
    det : Detector_SWC or Detector_EIS
        Detector configuration
    sim : Simulation
        Simulation configuration
    summed : u.Quantity, optional
        `synthesis.summed(reference_line)`, if already computed.
    meta : dict, optional
        Cube metadata beyond what
        :meth:`~euvst_response.synthesis_file.Synthesis.summed_meta` gives,
        such as the synthesis's `dynamic_mode`, or a time series' `raster`
        entries, whose columns are its exposures and are kept as they are.
    tel : Telescope_EUVST or Telescope_EIS, optional
        Telescope, as for :func:`rebin_atmosphere`.

    Returns
    -------
    NDCube
        Rebinned cube at instrument resolution
    """
    print("  Spectral rebinning to instrument resolution (ny,nx,*nl*)...")
    reference = synthesis.lines[reference_line]
    radiance = synthesis.summed(reference_line) if summed is None else summed
    blurred = _through_the_optics(tel, sim)
    if tel is not None:
        data, grid, photon_meta = _spectra_on_the_detector(
            radiance.value, reference.wavelength, det, sim, tel)
        meta = {**(meta or {}), **photon_meta,
                **({"psf_applied": True} if blurred else {})}
    else:
        data, grid = resample_spectra(radiance.value, reference.wavelength, det.wvl_res * u.pix)

    # Synthesis cube WCS: wavelength in cm, then x and y on the Sun in Mm,
    # each referenced to the middle of its axis.
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
                       meta={**synthesis.summed_meta(reference_line), **(meta or {})})

    print("  Spatially rebinning to plate scale (*ny*,nx,nl) and slit width (ny,*nx*,nl)...")
    return reproject_ndcube_heliocentric_to_helioprojective(
        cube_spec, sim, det, ncpu=sim.ncpu, **(_spatial_blur(tel, det, sim) if blurred else {}))


def pad_spectral_axis(cube: NDCube, n: int) -> NDCube:
    """
    `cube` with `n` empty pixels added at each end of its wavelength axis.

    Wavelength is the last data axis, as in every detector-grid cube, and
    so the first WCS axis, since the WCS lists axes in reverse. The
    reference pixel moves with the data, so existing pixels keep their
    wavelengths. Used to widen a synthesis window for a spectral PSF that
    reaches beyond its margin
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
    Make a cube of one Gaussian line of known total intensity on the detector pixels.

    This is the cube `eclipse` observes for a single line of known intensity,
    ready for `monte_carlo` or `simulate_once`. Each wavelength pixel holds
    the line integrated over that pixel, so the cube holds all of the line
    within its wavelengths, however narrow the line. The tails beyond
    `n_sigma_extent` are left out.

    Parameters
    ----------
    total_intensity : u.Quantity
        Total line intensity, such as `5000 * u.erg / (u.s * u.cm**2 * u.sr)`.
    rest_wavelength : u.Quantity
        Rest wavelength, such as `195.119 * u.AA`.
    thermal_width : u.Quantity
        Line width as a 1-sigma velocity, such as `20 * u.km / u.s`.
    det : Detector_SWC or Detector_EIS
        Detector whose pixels the cube is on.
    sim : Simulation
        Simulation, which sets the slit.
    n_sigma_extent : float, optional
        Number of sigma of the line to hold either side of its centre.
        Default 8.
    n_slit_pixels : int, optional
        Number of identical pixels along the slit. Set it to the off-chip
        binning, so each binned pixel sums that many, each with its own
        noise. Default 1.
    tel : Telescope_EUVST or Telescope_EIS, optional
        Telescope. If given, the wavelength range is wide enough to hold the
        line after the spectral PSF blurs it. Give it if the PSF will be on.
        Default None.

    Returns
    -------
    NDCube
        Cube of shape (`n_slit_pixels`, 1, wavelength), in
        erg / (s cm2 sr cm).
    """
    if n_slit_pixels < 1:
        raise ValueError(f"n_slit_pixels must be >= 1, got {n_slit_pixels}")

    # --- Spectral grid --------------------------------------------------
    lam0 = rest_wavelength.to(u.cm)

    # Convert velocity width to wavelength width: sigma_lam = lam0 * v / c
    sigma_lam = (lam0 * thermal_width / const.c).to(u.cm)

    # Detector pixel pitch in cm
    dlam = det.wvl_res.to(u.cm / u.pix) * u.pix  # strip per-pixel to cm

    # The grid must hold the line as measured, not as emitted, so add the
    # spectral PSF to the thermal width in quadrature, as Gaussian widths
    # add. The PSF is often the broader: at the default 20 km/s the line is
    # 0.77 pixels against a PSF of 1.08. The grid is always widened, not only
    # when psf is set, because psf may be swept and is not known when the
    # cube is built and cached. The PSF is the one for this slit, with the
    # slit added in quadrature, which is at least as broad as the slit
    # convolved with the optics, so the grid holds the line under either
    # spectral_psf.
    sigma_total = sigma_lam
    if tel is not None:
        sigma_psf = _fwhm_to_sigma(spectral_psf_fwhm(tel, det, sim.slit_width)) * dlam
        sigma_total = np.sqrt(sigma_lam**2 + sigma_psf**2)

    half_range = n_sigma_extent * sigma_total
    n_pix_half = int(np.ceil((half_range / dlam).decompose().value))
    n_lam = 2 * n_pix_half + 1  # always odd, centred on rest wavelength

    # --- Gaussian profile -----------------------------------------------
    # Each pixel holds the line integrated between its edges and divided by
    # its width, as resample_spectra gives a synthesised spectrum, so the
    # pixels sum to total_intensity less the tails beyond n_sigma_extent,
    # whatever the line width. Sampling the Gaussian at pixel centres only
    # does that for a sigma above about half a pixel: a line of 0.3 pixels
    # centred on a pixel, such as Fe VIII 185.21 at its formation
    # temperature, would come out 35 per cent too bright. Edges are counted
    # in pixels from the line centre, since absolute wavelengths would lose
    # a part in 1e12 of a pixel to rounding. Each pixel's upper edge is the
    # next one's lower edge, so nothing is counted twice or missed.
    edges = ((np.arange(n_lam + 1) - n_pix_half - 0.5) * dlam
             / (np.sqrt(2.0) * sigma_lam)).decompose().value
    profile = (total_intensity * 0.5 * np.diff(erf(edges)) / dlam).to(
        u.erg / (u.s * u.cm**2 * u.sr * u.cm)
    )
    # Tile the profile along the slit axis. Every slit pixel holds the same
    # intensity but gets its own noise downstream, which rebin_slit_offchip
    # needs to sum them.
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