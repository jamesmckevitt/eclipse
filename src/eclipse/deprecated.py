"""
Deprecated functions and routes, which will be removed in a future release.

Each can still be imported from where it was before, such as
`eclipse.frame.apply_spectral_psf`, so existing scripts still run. The
synthesis from MURaM's own files and its dynamic mode are here as well; they
warn when `synthesise-spectra` takes them.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import astropy.units as u
import numpy as np
from ndcube import NDCube
from scipy.ndimage import zoom
from scipy.signal import convolve2d
from tqdm import tqdm

from .atmosphere import AXES, Atmosphere
from .radiometric import _spectral_kernel
from .utils import _warn_outside, angle_to_distance, require_downsample_divides

if TYPE_CHECKING:
    from .band import BandLines
    from .full_frame import _Scene


def _warn(name: str, reason: str) -> None:
    """Warn that `name` is deprecated, attributed to the first caller outside ECLIPSE."""
    _warn_outside(f"{name} is deprecated and will be removed in a future release: {reason}")




def apply_spectral_psf(rows: u.Quantity, telescope, det, slit_width: u.Quantity,
                       spectral_psf: str = "quadrature", margin: int = 0) -> u.Quantity:
    """
    Blur a frame's rows with the instrument's spectral response.

    Along the dispersion a pixel is a row. The response is that of
    :func:`~eclipse.radiometric.apply_focusing_optics_psf` for a
    synthesis through the same slit, on the same rows: for `spectral_psf`
    `"quadrature"` a Gaussian of
    :func:`~eclipse.radiometric.spectral_psf_fwhm`, and for
    `"convolution"` :func:`~eclipse.radiometric.spectral_line_spread`.
    The slit's image is part of the line profile, so a wider slit gives a
    wider response: a FWHM of 2.54 rows for the 0.2 arcsec slit and 3.35
    for the 0.4 arcsec slit.

    Nothing is assumed beyond the rows given: light blurred past either end
    is lost, as off the edge of a detector, and none comes in. The edge rows
    of a CCD do receive light from just off the chip, and at the butted edge
    the spectrum continues across the gap. To include this, lay the spectrum
    with a `margin` of at least
    :func:`~eclipse.radiometric.spectral_psf_reach` rows past each
    end (:func:`~eclipse.frame.photons_from_lines` and
    :func:`~eclipse.frame.photons_from_spectrum` take one), and pass
    the same margin here. The result is then the chip's own rows. The
    kernel is normalised, so flux that stays within the rows is conserved.

    Deprecated since 0.12.0: blurring the rows after the light is on them
    moves a line narrower than a row toward the middle of its row, by up to
    a few km/s.
    Give `spectral_psf` and `det` to
    :func:`~eclipse.frame.photons_from_lines` or
    :func:`~eclipse.frame.photons_from_spectrum` instead, which blur
    the light as it is laid onto the rows.
    """
    _warn("apply_spectral_psf",
          "blurring the rows once the light is on them moves a line narrower than a row "
          "toward the middle of its row. Give spectral_psf and det to photons_from_lines or "
          "photons_from_spectrum instead.")
    margin = int(margin)
    kernel = _spectral_kernel(telescope, det, slit_width, spectral_psf)
    kernel = kernel / kernel.sum()
    half = kernel.size // 2
    if margin and margin < half:
        raise ValueError(
            f"A margin of {margin} rows is less than the {half} the spectral "
            f"response reaches, so the edge rows would miss light."
        )
    unit = rows.unit if hasattr(rows, "unit") else 1
    values = np.asarray(rows.value if hasattr(rows, "value") else rows, dtype=float)
    if values.size <= 2 * margin:
        raise ValueError(
            f"{values.size} rows leave nothing inside a margin of {margin} at each end."
        )
    blurred = np.convolve(np.pad(values, half), kernel, mode="valid")
    return blurred[margin:values.size - margin] * unit


def get_dem_data_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    DEM data held in the results, which no version of ECLIPSE has written there.

    Deprecated since 0.12.0. The DEM and emission measure are in the
    synthesis file, the emission measure only if synthesised with
    `--keep-vdem`. Read them with
    `read_synthesis_products(path, keys=["dem_map", "em_tv"])`.

    Parameters
    ----------
    results : dict
        Results from `load_instrument_response_results`.

    Returns
    -------
    dict
        `results["dem_data"]`.

    Raises
    ------
    KeyError
        If the results hold no DEM; results from `eclipse` never do.
    """
    _warn("get_dem_data_from_results",
          "no released version wrote DEM data into the results. The DEM is in the synthesis "
          "file; read it with "
          "eclipse.read_synthesis_products(path, keys=['dem_map', 'em_tv']).")
    if "dem_data" not in results:
        raise KeyError(
            "DEM data is not in the results, as no released version wrote it there. It is "
            "in the synthesis file: read it with "
            "eclipse.read_synthesis_products(path, keys=['dem_map', 'em_tv'])."
        )

    return results["dem_data"]


def _load_psf_ascii(fname: Path, skip: int) -> np.ndarray:
    """Load PSF data from ASCII file."""
    _warn("_load_psf_ascii", "ECLIPSE does not use it. "
          "numpy.loadtxt(fname, skiprows=skip, encoding='utf-16 LE') does the same.")
    return np.loadtxt(fname, skiprows=skip, encoding="utf-16 LE")


def _resample_psf(psf: np.ndarray, res_in: u.Quantity, res_out: u.Quantity) -> np.ndarray:
    """Resample PSF to different pixel scale."""
    _warn("_resample_psf", "ECLIPSE does not use it. scipy.ndimage.zoom does the same.")
    factor = (res_in / res_out).decompose().value
    return zoom(psf, factor, order=1)


def _combine_psfs(psf_focus: np.ndarray, psf_mesh: np.ndarray, crop: float = 0.99, size: int | None = None) -> np.ndarray:
    """Convolve the focus and mesh PSFs and crop to an energy fraction or size."""
    _warn("_combine_psfs", "ECLIPSE does not use it. scipy.signal.convolve2d does the "
          "convolution.")
    psf = convolve2d(psf_focus, psf_mesh, mode="same")
    if size is not None:
        if size % 2 == 0:
            size += 1
        r0, c0 = np.array(psf.shape) // 2
        half = size // 2
        psf = psf[r0 - half : r0 + half + 1, c0 - half : c0 + half + 1]
    else:
        flat = psf.ravel()
        idx = flat.argsort()[::-1]
        csum = flat[idx].cumsum()
        thr = flat[idx[np.searchsorted(csum, flat.sum() * crop)]]
        flat[flat < thr] = 0
        rows, cols = np.where(flat.reshape(psf.shape))
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        side = max(r1 - r0, c1 - c0) + 1
        r0 = (r0 + r1) // 2 - side // 2
        c0 = (c0 + c1) // 2 - side // 2
        psf = psf[r0 : r0 + side, c0 : c0 + side]
    return psf / psf.sum()


# ---------------------------------------------------------------------------
#  Synthesis from MURaM's own files, and its dynamic mode
# ---------------------------------------------------------------------------

def load_cube(
    file_path: str | Path,
    shape: Tuple[int, int, int] = (512, 768, 256),
    unit: Optional[u.Unit] = None,
    downsample: int | bool = False,
    precision: type = np.float32,
    voxel_dx: Optional[u.Quantity] = None,
    voxel_dy: Optional[u.Quantity] = None,
    voxel_dz: Optional[u.Quantity] = None,
    create_ndcube: bool = False,
) -> np.ndarray | u.Quantity | NDCube:
    """
    Read a Fortran-ordered float32 binary cube, optionally as an NDCube.

    The file stores the cube as (x, z, y). It is returned as (z, y, x), so
    `data[k]` is a horizontal slice indexed `[y, x]`.

    Parameters
    ----------
    file_path : str | Path
        Path to the binary file.
    shape : Tuple[int, int, int]
        Full cube dimensions in the file's storage order, `(nx, nz, ny)`:
        the vertical axis is second, not last.
    unit : astropy.units.Unit, optional
        Unit to attach (e.g. u.K or u.g/u.cm**3). If None, returns a plain
        ndarray.
    downsample : int | bool
        Integer factor; if not False, keep every `downsample`-th cell along
        each axis (simple stride).
    precision : type
        np.float32 or np.float64 for the returned dtype.
    voxel_dx, voxel_dy, voxel_dz : u.Quantity, optional
        Voxel sizes in the file, at full resolution. With `downsample`, the
        returned cube's WCS uses them multiplied by it. Required if
        `create_ndcube` is True.
    create_ndcube : bool, optional
        If True, return an NDCube with WCS coordinates.

    Returns
    -------
    ndarray, Quantity, or NDCube
        Array of shape (nz', ny', nx'), or an NDCube with its coordinates.
    """
    if downsample:
        require_downsample_divides(shape, downsample)

    data = np.fromfile(file_path, dtype=np.float32).reshape(shape, order="F")
    data = data.transpose(1, 2, 0)  # (z,y,x)

    if downsample:
        data = data[::downsample, ::downsample, ::downsample]
        # Each kept cell spans `downsample` cells of the file. Make new
        # quantities rather than use *=, which would scale the caller's voxel
        # sizes and compound across calls.
        voxel_dx, voxel_dy, voxel_dz = (
            None if size is None else size * downsample
            for size in (voxel_dx, voxel_dy, voxel_dz))

    data = data.astype(precision, copy=False)
    
    if unit is not None:
        data = data * unit
        
    if create_ndcube:
        from .synthesis import create_atmosphere_ndcube

        return create_atmosphere_ndcube(data, voxel_dx, voxel_dy, voxel_dz)
    else:
        return data


def read_timestep_time(file_path: Path) -> float:
    """
    Read simulation time from MHD header file.
    
    The header file contains a single line with 9 space-separated values:
    nx ny nz dx dy dz time dt va_max
    
    The time is the 7th value (index 6) in seconds.
    
    Parameters
    ----------
    file_path : Path
        Path to the header file.
        
    Returns
    -------
    float
        Simulation time in seconds.
    """
    with open(file_path, 'r') as f:
        line = f.read().strip()
        values = line.split()
        if len(values) < 7:
            raise ValueError(f"Header file has insufficient values: {file_path} ({len(values)} values)")
        return float(values[6])


def discover_timesteps(
    time_dir: Path,
    time_filename: str,
) -> Dict[str, float]:
    """
    Discover all available timesteps and their simulation times.
    
    Parameters
    ----------
    time_dir : Path
        Directory containing header files.
    time_filename : str
        Filename prefix before the timestep suffix (e.g., "Header").
        
    Returns
    -------
    dict
        Mapping of timestep suffix to simulation time in seconds.
        E.g., {"0270000": 26729.535, "0280000": 27571.395, ...}
    """
    if not time_dir.is_dir():
        raise FileNotFoundError(f"Time directory not found: {time_dir}")
    
    timesteps = {}
    
    for file_path in sorted(time_dir.iterdir()):
        if file_path.is_file() and file_path.name.startswith(time_filename):
            # Extract suffix after the filename prefix
            suffix = file_path.name[len(time_filename):]
            if suffix.startswith('.'):
                suffix = suffix[1:]  # Remove leading dot
            
            try:
                sim_time = read_timestep_time(file_path)
                timesteps[suffix] = sim_time
            except Exception as e:
                warnings.warn(f"Could not read time from {file_path}: {e}")
    
    if not timesteps:
        raise ValueError(f"No valid timestep files found in {time_dir} with prefix '{time_filename}'")
    
    return timesteps


def get_file_for_timestep(
    directory: Path,
    filename: str,
    suffix: str,
) -> Path:
    """
    Get the file path for a specific timestep.
    
    Parameters
    ----------
    directory : Path
        Directory containing the data files.
    filename : str
        Filename prefix before the suffix (e.g., "eosT").
    suffix : str
        Timestep suffix (e.g., "0270000").
        
    Returns
    -------
    Path
        Full path to the file.
    """
    file_path = directory / f"{filename}.{suffix}"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path


def compute_slice_timestep_mapping_mhd(
    nx_mhd: int,
    voxel_dx: u.Quantity,
    slit_width: u.Quantity,
    slit_rest_time: u.Quantity,
    timestep_times: Dict[str, float],
    crop_x: Optional[Tuple[u.Quantity, u.Quantity]] = None,
) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Timestep suffix to use for each MHD X-slice.

    The spectrometer scans from right to left (high X to low X), and each
    slit position covers `slit_width` converted to Mm. Each MHD slice is
    assigned to a slit position, then to the timestep at that position's
    observation time. The time is worked out from the slice's absolute
    position, so errors do not accumulate.

    With `crop_x`, the scan starts at the right edge of the crop region.
    Slices to the right of it get the first timestep.

    Parameters
    ----------
    nx_mhd : int
        Number of MHD X-slices.
    voxel_dx : u.Quantity
        MHD voxel size along X, as a length (e.g. Mm).
    slit_width : u.Quantity
        Slit width as an angle (e.g. arcsec).
    slit_rest_time : u.Quantity
        Slit rest time per position.
    timestep_times : dict
        Simulation time in seconds for each timestep suffix.
    crop_x : tuple of u.Quantity, optional
        (x_min, x_max) crop boundaries. The scan starts at x_max, the right
        edge of the crop; slices to the right of it use the first timestep.

    Returns
    -------
    slice_mapping : list
        Timestep suffix for each MHD X-slice.
    grouped : dict
        MHD slice indices for each timestep suffix.
    """
    rest_time_sec = slit_rest_time.to_value(u.s)
    
    # Convert slit width from angular to physical distance
    slit_physical = angle_to_distance(slit_width).to(u.Mm)
    voxel_physical = voxel_dx.to(u.Mm)
    
    # Sort timesteps by simulation time
    sorted_timesteps = sorted(timestep_times.items(), key=lambda x: x[1])
    suffixes = [s for s, t in sorted_timesteps]
    times = np.array([t for s, t in sorted_timesteps])
    first_suffix = suffixes[0]
    
    # Check if observation time range extends beyond available MHD time range
    t0 = times[0]
    t_final = times[-1]
    mhd_duration = t_final - t0
    
    # Total physical extent of the full domain
    total_extent = nx_mhd * voxel_physical
    
    # The scan starts at the right edge of the domain, or of the crop if given.
    if crop_x is not None:
        x_min_crop = crop_x[0].to(u.Mm)
        x_max_crop = crop_x[1].to(u.Mm)
        
        # The WCS centres X on zero (x=0 is at nx/2), so convert to absolute
        # positions from the left edge.
        domain_center = total_extent / 2
        scan_start_position = domain_center + x_max_crop  # Right edge of crop in absolute coords
        scan_end_position = domain_center + x_min_crop    # Left edge of crop in absolute coords
        
        # Calculate observation duration for just the cropped region
        crop_extent = x_max_crop - x_min_crop
        n_slit_positions_crop = int(np.ceil((crop_extent / slit_physical).decompose().value))
        observation_duration = n_slit_positions_crop * rest_time_sec
        
        print(f"  Crop region: X = [{x_min_crop:.3f}, {x_max_crop:.3f}]")
        print(f"  Observation starts at X = {x_max_crop:.3f} (right edge of crop)")
    else:
        scan_start_position = total_extent  # Right edge of full domain
        scan_end_position = 0 * u.Mm
        n_slit_positions_crop = int(np.ceil((total_extent / slit_physical).decompose().value))
        observation_duration = n_slit_positions_crop * rest_time_sec
    
    if observation_duration > mhd_duration:
        print(f"  WARNING: Observation duration ({observation_duration:.1f} s) exceeds MHD time range ({mhd_duration:.1f} s)")
        print(f"  Slices observed after t={t_final:.1f} s will use the last available timestep")
    
    slice_mapping = []
    
    # Find the slit position of each MHD X-slice. The scan runs right to left,
    # so scan_start_position is observed first, at t=t0.
    for mhd_slice_idx in range(nx_mhd):
        # Physical position of this slice (center of voxel) in absolute coords
        x_physical = (mhd_slice_idx + 0.5) * voxel_physical
        
        # Distance from the scan start position (right edge of observation region)
        distance_from_scan_start = scan_start_position - x_physical
        
        if distance_from_scan_start < 0:
            # Outside the crop on the right: use the first timestep.
            slice_mapping.append(first_suffix)
            continue
        
        # Slit position 0 is at scan_start_position.
        slit_position = int(np.floor((distance_from_scan_start / slit_physical).decompose().value))
        
        # Observation time for this slit position
        observation_time = t0 + slit_position * rest_time_sec
        
        # Find the latest timestep that doesn't exceed observation_time
        idx = np.searchsorted(times, observation_time, side='right') - 1
        # Clamp to valid range (0 to len(times)-1)
        idx = max(0, min(idx, len(times) - 1))
        slice_mapping.append(suffixes[idx])
    
    # Group slices by timestep for efficient processing
    grouped = {}
    for slice_idx, suffix in enumerate(slice_mapping):
        if suffix not in grouped:
            grouped[suffix] = []
        grouped[suffix].append(slice_idx)
    
    # Print statistics
    if crop_x is not None:
        print(f"  Cropped region extent: {crop_extent:.3f}")
    else:
        print(f"  Physical domain extent: {total_extent:.3f}")
    print(f"  Slit physical width: {slit_physical:.3f}")
    print(f"  Number of slit positions: {n_slit_positions_crop}")
    print(f"  Total raster time: {observation_duration:.1f} s")
    
    return slice_mapping, grouped


def apply_cube_cropping(
    temp_cube: NDCube,
    rho_cube: NDCube,
    vel_cube: NDCube,
    crop_x: Optional[List[str]],
    crop_y: Optional[List[str]],
    crop_z: Optional[List[str]],
) -> Tuple[NDCube, NDCube, NDCube]:
    """
    Apply cropping to temperature, density, and velocity cubes.
    
    Parameters
    ----------
    temp_cube, rho_cube, vel_cube : NDCube
        Input cubes to crop.
    crop_x, crop_y, crop_z : list of str or None
        Crop boundaries for each axis as [min, max] strings with units.
        
    Returns
    -------
    tuple of NDCube
        Cropped (temp_cube, rho_cube, vel_cube).
    """
    # Crop points are given in world axis order, which is (SOLX, SOLY, SOLZ).
    point1 = []
    point2 = []

    if crop_x:
        point1.append(u.Quantity(crop_x[0]))
        point2.append(u.Quantity(crop_x[1]))
    else:
        point1.append(None)
        point2.append(None)

    if crop_y:
        point1.append(u.Quantity(crop_y[0]))
        point2.append(u.Quantity(crop_y[1]))
    else:
        point1.append(None)
        point2.append(None)

    if crop_z:
        point1.append(u.Quantity(crop_z[0]))
        point2.append(u.Quantity(crop_z[1]))
    else:
        point1.append(None)
        point2.append(None)
    
    # keepdims keeps an axis cropped to one cell, since the synthesis needs
    # all three axes.
    temp_cube = temp_cube.crop(point1, point2, keepdims=True)
    rho_cube = rho_cube.crop(point1, point2, keepdims=True)
    vel_cube = vel_cube.crop(point1, point2, keepdims=True)
    
    return temp_cube, rho_cube, vel_cube


def build_composite_cubes_mhd(
    base_dir: Path,
    temp_dir: str,
    temp_filename: str,
    rho_dir: str,
    rho_filename: str,
    vel_dir: str,
    vel_filename: str,
    slice_mapping: List[str],
    grouped_slices: Dict[str, List[int]],
    cube_shape: Tuple[int, int, int],
    voxel_dx: u.Quantity,
    voxel_dy: u.Quantity,
    voxel_dz: u.Quantity,
    downsample: int | bool,
    precision: type,
) -> Tuple[NDCube, NDCube, NDCube]:
    """
    Build composite temp, rho, vel cubes from multiple timesteps at MHD resolution.

    Each MHD X-slice comes from its timestep in `slice_mapping`. There is no
    spatial rebinning: the output is at full MHD resolution.
    
    Parameters
    ----------
    base_dir : Path
        Base directory for atmosphere data.
    temp_dir, temp_filename : str
        Directory and filename prefix for temperature files.
    rho_dir, rho_filename : str
        Directory and filename prefix for density files.
    vel_dir, vel_filename : str
        Directory and filename prefix for velocity files.
    slice_mapping : list
        Timestep suffix for each MHD X-slice.
    grouped_slices : dict
        Slices grouped by timestep for efficient processing.
    cube_shape : tuple
        Original cube dimensions in the file's storage order.
    voxel_dx, voxel_dy, voxel_dz : u.Quantity
        Voxel sizes in the files, at full resolution; `load_cube` applies the
        downsampling.
    downsample : int or bool
        Downsampling factor.
    precision : type
        Numerical precision.
        
    Returns
    -------
    tuple
        (temp_composite, rho_composite, vel_composite) NDCubes at MHD resolution.
    """
    nx_mhd = len(slice_mapping)
    
    # Load first timestep to determine dimensions and get reference WCS
    first_suffix = list(grouped_slices.keys())[0]
    temp_file = get_file_for_timestep(base_dir / temp_dir, temp_filename, first_suffix)
    temp_cube_ref = load_cube(
        temp_file, shape=cube_shape, unit=u.K,
        downsample=downsample, precision=precision,
        voxel_dx=voxel_dx, voxel_dy=voxel_dy, voxel_dz=voxel_dz,
        create_ndcube=True
    )
    
    nz, ny, nx = temp_cube_ref.data.shape
    reference_wcs = temp_cube_ref.wcs

    # Verify dimensions match slice mapping
    if nx != nx_mhd:
        raise ValueError(f"Cube X dimension ({nx}) doesn't match slice mapping ({nx_mhd})")

    # Initialise composite arrays
    temp_composite = np.zeros((nz, ny, nx), dtype=precision)
    rho_composite = np.zeros((nz, ny, nx), dtype=precision)
    vel_composite = np.zeros((nz, ny, nx), dtype=precision)
    
    # Process each timestep
    for suffix, slice_indices in tqdm(grouped_slices.items(), desc="Loading timesteps", unit="timestep"):
        # Load cubes for this timestep
        temp_file = get_file_for_timestep(base_dir / temp_dir, temp_filename, suffix)
        rho_file = get_file_for_timestep(base_dir / rho_dir, rho_filename, suffix)
        vel_file = get_file_for_timestep(base_dir / vel_dir, vel_filename, suffix)
        
        temp_cube = load_cube(
            temp_file, shape=cube_shape, unit=u.K,
            downsample=downsample, precision=precision,
            voxel_dx=voxel_dx, voxel_dy=voxel_dy, voxel_dz=voxel_dz,
            create_ndcube=True
        )
        rho_cube = load_cube(
            rho_file, shape=cube_shape, unit=u.g/u.cm**3,
            downsample=downsample, precision=precision,
            voxel_dx=voxel_dx, voxel_dy=voxel_dy, voxel_dz=voxel_dz,
            create_ndcube=True
        )
        vel_cube = load_cube(
            vel_file, shape=cube_shape, unit=u.cm/u.s,
            downsample=downsample, precision=precision,
            voxel_dx=voxel_dx, voxel_dy=voxel_dy, voxel_dz=voxel_dz,
            create_ndcube=True
        )
        
        # Copy relevant slices to composite (no rebinning - direct copy)
        for slice_idx in slice_indices:
            temp_composite[:, :, slice_idx] = temp_cube.data[:, :, slice_idx]
            rho_composite[:, :, slice_idx] = rho_cube.data[:, :, slice_idx]
            vel_composite[:, :, slice_idx] = vel_cube.data[:, :, slice_idx]
    
    # Create NDCubes with proper WCS (at MHD resolution)
    temp_ndcube = NDCube(temp_composite * u.K, wcs=reference_wcs, meta={"source": "composite_dynamic"})
    rho_ndcube = NDCube(rho_composite * (u.g/u.cm**3), wcs=reference_wcs, meta={"source": "composite_dynamic"})
    vel_ndcube = NDCube(vel_composite * (u.cm/u.s), wcs=reference_wcs, meta={"source": "composite_dynamic"})
    
    return temp_ndcube, rho_ndcube, vel_ndcube


def load_muram_files(
    args,
    integration_axis: str,
    downsample: int | bool = False,
    crop_x=None, crop_y=None, crop_z=None,
) -> Atmosphere:
    """
    Read the MURaM files named by the deprecated static options.

    They are downsampled, cropped and checked as in
    :func:`load_atmosphere_file`. x and y are centred on zero and z = 0 is at
    the centre of the bottom cell, as ECLIPSE has always placed a MURaM box,
    so the crop options keep the meaning they had before atmosphere files.
    """
    data_dir = Path(args.data_dir)
    files = {
        "temperature": (args.temp_file, u.K),
        "mass_density": (args.rho_file, u.g / u.cm**3),
        f"velocity_{integration_axis}": (getattr(args, f"v{integration_axis}_file"),
                                         u.cm / u.s),
    }
    cubes = {}
    for name, (file_name, unit) in files.items():
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        cubes[name] = load_cube(path, shape=tuple(args.cube_shape), unit=unit)

    nz, ny, nx = cubes["temperature"].shape
    voxel = {axis: u.Quantity(getattr(args, f"voxel_d{axis}")) for axis in AXES}
    atmosphere = Atmosphere(
        x_edges=(np.arange(nx + 1) - nx / 2) * voxel["x"],
        y_edges=(np.arange(ny + 1) - ny / 2) * voxel["y"],
        z_edges=(np.arange(nz + 1) - 0.5) * voxel["z"],
        source="MURaM", **cubes)
    from .synthesis import _prepare_atmosphere

    return _prepare_atmosphere(atmosphere, f"the MURaM files in {data_dir}",
                               integration_axis, downsample, crop_x, crop_y, crop_z)


def _dynamic_mode_cubes(args, view: str, integration_axis: str, band, downsample, precision,
                        print_mem):
    """
    Temperature, mass density and velocity cubes of the deprecated dynamic
    mode of `synthesise-spectra`, built from MURaM's own files as the slit
    steps across them, with the mode's settings and the cell sizes.
    """
    # Validate dynamic mode requirements
    if band is not None or getattr(args, "image_x", None):
        raise ValueError("Dynamic mode, which is deprecated, takes neither --band nor "
                         "--image-x; synthesise from an atmosphere file.")
    if args.slit_width is None:
        raise ValueError("--slit-width is required for dynamic mode (when --slit-rest-time is specified)")
    # The snapshots are laid out along x, which the slit steps across in a
    # view along z or y. In a view along x, x is the line of sight, and
    # every pixel would sum cells from all the snapshots.
    if integration_axis == "x":
        raise ValueError(
            "Dynamic mode lays its snapshots across x, which is the line of sight "
            "of a view along x, so every pixel would add up all of them. View along "
            "z or y.")
    if view != integration_axis:
        raise ValueError(
            f"Dynamic mode, which is deprecated, looks only from the side of the box "
            f"that --integration-axis {integration_axis} names. To look from "
            f"{view}, synthesise from an atmosphere file with --atmosphere.")

    base_dir = Path(args.data_dir)
    # Voxel sizes in the MURaM files. load_cube scales them itself when it
    # downsamples, so they are passed to it as given.
    file_voxel_dz = u.Quantity(args.voxel_dz)
    file_voxel_dx = u.Quantity(args.voxel_dx)
    file_voxel_dy = u.Quantity(args.voxel_dy)

    # Voxel sizes of the cubes as synthesised, for the path length along
    # the line of sight, the slice timing and the saved metadata.
    voxel_dz = file_voxel_dz * (downsample or 1)
    voxel_dx = file_voxel_dx * (downsample or 1)
    voxel_dy = file_voxel_dy * (downsample or 1)

    # Parse slit rest time and slit width
    slit_rest_time = u.Quantity(args.slit_rest_time)
    slit_width = u.Quantity(args.slit_width)
    
    # Determine which velocity direction to use
    if integration_axis == "x":
        vel_dir = args.vx_dir or "vx"
        vel_filename = args.vx_filename
    elif integration_axis == "y":
        vel_dir = args.vy_dir or "vy"
        vel_filename = args.vy_filename
    else:  # "z"
        vel_dir = args.vz_dir or "vz"
        vel_filename = args.vz_filename
    
    # Set directory defaults
    temp_dir = args.temp_dir or "temp"
    rho_dir = args.rho_dir or "rho"
    
    print(f"DYNAMIC MODE - Time-varying synthesis at MHD resolution (deprecated)")
    print(f"  Slit width: {slit_width}")
    print(f"  Slit rest time: {slit_rest_time}")
    print(f"  Voxel dx: {voxel_dx}")
    print()
    
    # Discover available timesteps
    time_dir = base_dir / args.time_dir
    print(f"Discovering timesteps from {time_dir}...")
    timestep_times = discover_timesteps(time_dir, args.time_filename)
    print(f"  Found {len(timestep_times)} timesteps")
    for suffix, sim_time in sorted(timestep_times.items(), key=lambda x: x[1]):
        print(f"    {suffix}: {sim_time:.3f} s")
    print()
    
    # Calculate MHD cube dimensions
    cube_shape_tuple = tuple(args.cube_shape)
    nx_mhd = cube_shape_tuple[0]
    if downsample:
        nx_mhd = nx_mhd // downsample
    
    # Prepare crop_x for slice mapping if specified
    crop_x_for_mapping = None
    if args.crop_x:
        crop_x_for_mapping = (u.Quantity(args.crop_x[0]), u.Quantity(args.crop_x[1]))
    
    # Compute slice-to-timestep mapping at MHD resolution
    print(f"Computing slice-to-timestep mapping at MHD resolution...")
    slice_mapping, grouped_slices = compute_slice_timestep_mapping_mhd(
        nx_mhd, voxel_dx, slit_width, slit_rest_time, timestep_times,
        crop_x=crop_x_for_mapping
    )
    print(f"  MHD slices per timestep:")
    for suffix, indices in sorted(grouped_slices.items(), key=lambda x: min(x[1])):
        print(f"    {suffix}: {len(indices)} slices (indices {min(indices)}-{max(indices)})")
    print()
    
    # Build composite cubes at MHD resolution
    print(f"Building composite atmosphere cubes at MHD resolution ({print_mem()})...")
    temp_cube, rho_cube, vel_cube = build_composite_cubes_mhd(
        base_dir=base_dir,
        temp_dir=temp_dir,
        temp_filename=args.temp_filename,
        rho_dir=rho_dir,
        rho_filename=args.rho_filename,
        vel_dir=vel_dir,
        vel_filename=vel_filename,
        slice_mapping=slice_mapping,
        grouped_slices=grouped_slices,
        cube_shape=cube_shape_tuple,
        voxel_dx=file_voxel_dx,
        voxel_dy=file_voxel_dy,
        voxel_dz=file_voxel_dz,
        downsample=downsample,
        precision=precision,
    )
    print(f"  Composite cube shape: {temp_cube.data.shape}")
    
    # Apply cropping if requested in dynamic mode
    if args.crop_x or args.crop_y or args.crop_z:
        print(f"Applying cropping ({print_mem()})")
        temp_cube, rho_cube, vel_cube = apply_cube_cropping(
            temp_cube, rho_cube, vel_cube,
            args.crop_x, args.crop_y, args.crop_z
        )
        print(f"  Cropped cubes to shape: {temp_cube.data.shape}")
    
    rho = rho_cube.data << rho_cube.unit

    # Dynamic mode metadata for output (no spatial rebinning in synthesis)
    dynamic_mode_metadata = {
        "enabled": True,
        "slit_width": slit_width,
        "slit_rest_time": slit_rest_time,
        "scan_direction": "right_to_left",
        "slice_timesteps": slice_mapping,
        "available_timesteps": timestep_times,
        "spatially_rebinned": False,  # Output is at MHD resolution
    }
    return temp_cube, rho, vel_cube, dynamic_mode_metadata, voxel_dx, voxel_dy, voxel_dz


def _check_dynamic_mode_synthesis(dynamic_mode_info: dict, sweep_dims: dict,
                                  sim_fixed: dict) -> None:
    """
    For a synthesis made in the deprecated dynamic mode, check that the run
    observes it with the slit width and exposure time it was made for.
    """
    is_dynamic_mode = dynamic_mode_info.get("enabled", False)
    if is_dynamic_mode:
        print("Synthesis was done in DYNAMIC MODE (time-varying atmosphere)")
        print(f"  Slit width: {dynamic_mode_info['slit_width']}")
        print(f"  Slit rest time: {dynamic_mode_info['slit_rest_time']}")
        print(f"  Timesteps used: {len(dynamic_mode_info['available_timesteps'])}")

        synth_slit_width = dynamic_mode_info["slit_width"]
        synth_rest_time = dynamic_mode_info["slit_rest_time"]

        slit_width_vals = sweep_dims.get(
            "simulation.slit_width", [sim_fixed["slit_width"]]
        )
        expos_vals = sweep_dims.get(
            "simulation.expos", [sim_fixed["expos"]]
        )

        if len(slit_width_vals) != 1:
            raise ValueError(
                f"Dynamic mode synthesis requires exactly one slit width. "
                f"Config specifies {len(slit_width_vals)}: {slit_width_vals}. "
                f"Please provide only the synthesis slit width: {synth_slit_width}"
            )
        if not np.isclose(
            slit_width_vals[0].to_value(u.arcsec),
            synth_slit_width.to_value(u.arcsec),
            rtol=1e-6,
        ):
            raise ValueError(
                f"Slit width mismatch: synthesis was done with {synth_slit_width}, "
                f"but config specifies {slit_width_vals[0]}."
            )
        if len(expos_vals) != 1:
            raise ValueError(
                f"Dynamic mode synthesis requires exactly one exposure time. "
                f"Config specifies {len(expos_vals)}: {expos_vals}. "
                f"Please provide only the synthesis slit rest time: {synth_rest_time}"
            )
        if not np.isclose(
            expos_vals[0].to_value(u.s),
            synth_rest_time.to_value(u.s),
            rtol=1e-6,
        ):
            raise ValueError(
                f"Exposure time mismatch: synthesis was done with {synth_rest_time}, "
                f"but config specifies {expos_vals[0]}."
            )
        print("  Dynamic mode parameters validated successfully!")


# ---------------------------------------------------------------------------
#  The lines and continuum arguments of full_ccd_frame
# ---------------------------------------------------------------------------

def _band_scene(path: str | Path, lines: BandLines, continuum: Optional[tuple],
                slit_position: u.Quantity, slit_width: u.Quantity,
                psf_across_slit: Optional[u.Quantity], extend: bool, n_workers: int) -> _Scene:
    """
    Spectrum of every line in a band under the slit, from the emission
    measure a synthesis file from ECLIPSE keeps, for the deprecated `lines`
    and `continuum` arguments of `full_ccd_frame`.
    """
    import h5py

    from .band import (BAND_STEP, band_spectra, band_wavelengths, densities_on_grid,
                       same_temperatures)
    from .continuum import continuum_spectra
    from .full_frame import _Scene, _centred, _slit_columns
    from .synthesis import _temperature_bins
    from .synthesis_file import read_synthesis_products
    from .utils import _bin_edges

    path = Path(path).expanduser()
    with h5py.File(path, "r") as f:
        x_edges = u.Quantity(f["x_edges"][()], f["x_edges"].attrs["unit"])
        y_edges = u.Quantity(f["y_edges"][()], f["y_edges"].attrs["unit"])
        products = f.get("synthesis", {})
        missing = [key for key in ("em_tv", "dem_map", "electron_density", "logT_grid",
                                   "logN_grid", "vel_grid") if key not in products]
        if missing == ["em_tv"]:
            raise ValueError(
                f"{path} has no em_tv, which the spectrum of a band's lines is worked out "
                f"from. Synthesise with --keep-vdem to keep it, or leave out lines for a "
                f"frame of the spectra the file holds.")
        if missing:
            raise ValueError(
                f"{path} has no {', '.join(missing)}, which the spectrum of a band's lines "
                f"is worked out from: it was not made by ECLIPSE's synthesis. Leave out "
                f"lines for a frame of the spectra the file holds.")
        columns, weights = _slit_columns(x_edges, slit_position, slit_width, psf_across_slit,
                                         extend)
        em = products["em_tv"][:, columns, :, :].astype(float)
        dem = products["dem_map"][:, columns, :].astype(float)
        density = products["electron_density"][:, columns, :].astype(float)
        logT = products["logT_grid"][()]
        logN = products["logN_grid"][()]
        velocity = u.Quantity(products["vel_grid"][()],
                              products["vel_grid"].attrs.get("unit", "cm / s"))
    if not same_temperatures(logT, lines.logT_grid):
        raise ValueError("The band's lines are on another grid of temperature than the "
                         "synthesis; work them out on the synthesis's own.")
    if not densities_on_grid(logN, lines.logN_grid):
        raise ValueError(
            f"The synthesis takes the contribution functions at densities, log10 n_e "
            f"{logN[0]:.2f} to {logN[-1]:.2f}, that are not all on the "
            f"band's grid, {lines.logN_grid[0]:.2f} to {lines.logN_grid[-1]:.2f}; work the "
            f"lines out on a grid that has them.")
    abundance = (read_synthesis_products(path, keys=["config"]).get("config")
                 or {}).get("abundance")
    if abundance is not None and abundance != lines.abundance:
        raise ValueError(f"The synthesis used the abundances {abundance} and the band's lines "
                         f"{lines.abundance}; work the lines out with the synthesis's.")
    wavelength = band_wavelengths(lines.band, BAND_STEP)
    spectra = band_spectra(lines, em, density, velocity, wavelength * u.Angstrom, n_workers,
                           column_weights=weights)
    names, edges, radiance = ["band"], [_bin_edges(wavelength)], [spectra.value]
    if continuum is not None:
        grid, free, two_photon = continuum
        expected = (lines.logN_grid.size, lines.logT_grid.size, u.Quantity(grid).size)
        if np.shape(two_photon) != expected or np.shape(free) != expected[1:]:
            raise ValueError(
                f"The continuum is shaped {np.shape(free)} and {np.shape(two_photon)}, not "
                f"{expected[1:]} and {expected}: work it out on the band's grids.")
        grid = u.Quantity(grid).to_value(u.Angstrom)
        if grid.ndim != 1 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise ValueError("The continuum's wavelengths must be finite and increase.")
        # The DEM is per unit log T, and also holds plasma faster than the
        # velocity grid, which the continuum, not Doppler shifted, needs.
        total = dem * _temperature_bins(logT)[0]
        names.append("continuum")
        edges.append(_bin_edges(grid))
        # One column at a time, to hold one column's spectrum.
        radiance.append(sum(
            weight * continuum_spectra(total[:, [k]], density[:, [k]], lines.logN_grid,
                                       np.asarray(free), np.asarray(two_photon))[:, 0]
            for k, weight in enumerate(weights)))
    return _Scene(y_edges=_centred(y_edges), names=names, edges=edges, radiance=radiance)


