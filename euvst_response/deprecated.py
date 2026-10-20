"""
Deprecated functions, which warn when called and will be removed in a future release.

Each can still be imported from where it was before, such as
`euvst_response.frame.apply_spectral_psf`, so existing scripts still run.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import astropy.units as u
import numpy as np
from scipy.ndimage import zoom
from scipy.signal import convolve2d

from .radiometric import spectral_line_spread, spectral_psf_fwhm, spectral_psf_reach
from .utils import _fwhm_to_sigma


def _warn(name: str, reason: str) -> None:
    """Warn that `name` is deprecated, attributed to the first caller outside ECLIPSE."""
    frame, level = sys._getframe(1), 2
    package = __name__.split(".")[0] + "."
    while frame is not None and frame.f_globals.get("__name__", "").startswith(package):
        frame, level = frame.f_back, level + 1
    warnings.warn(f"{name} is deprecated and will be removed in a future release: {reason}",
                  FutureWarning, stacklevel=level)


_PSF_FILES = ("ECLIPSE does not use it. scipy.signal.convolve2d and scipy.ndimage.zoom "
              "do the same.")


def apply_spectral_psf(rows: u.Quantity, telescope, det, slit_width: u.Quantity,
                       spectral_psf: str = "quadrature", margin: int = 0) -> u.Quantity:
    """
    Blur a frame's rows with the instrument's spectral response.

    Along the dispersion a pixel is a row. The response is that of
    :func:`~euvst_response.radiometric.apply_focusing_optics_psf` for a
    synthesis through the same slit, on the same rows: for `spectral_psf`
    `"quadrature"` a Gaussian of
    :func:`~euvst_response.radiometric.spectral_psf_fwhm`, and for
    `"convolution"` :func:`~euvst_response.radiometric.spectral_line_spread`.
    The slit's image is part of the line profile, so a wider slit gives a
    wider response: a FWHM of 2.54 rows for the 0.2 arcsec slit and 3.35
    for the 0.4 arcsec slit.

    Nothing is assumed beyond the rows given: light blurred past either end
    is lost, as off the edge of a detector, and none comes in. The edge rows
    of a CCD do receive light from just off the chip, and at the butted edge
    the spectrum continues across the gap. To include this, lay the spectrum
    with a `margin` of at least
    :func:`~euvst_response.radiometric.spectral_psf_reach` rows past each
    end (:func:`~euvst_response.frame.photons_from_lines` and
    :func:`~euvst_response.frame.photons_from_spectrum` take one), and pass
    the same margin here. The result is then the chip's own rows. The
    kernel is normalised, so flux that stays within the rows is conserved.

    Deprecated since 0.12.0: blurring the rows after the light is on them
    moves a line narrower than a row toward the middle of its row, by up to
    a few km/s.
    Give `spectral_psf` and `det` to
    :func:`~euvst_response.frame.photons_from_lines` or
    :func:`~euvst_response.frame.photons_from_spectrum` instead, which blur
    the light as it is laid onto the rows.
    """
    _warn("apply_spectral_psf",
          "blurring the rows once the light is on them moves a line narrower than a row "
          "toward the middle of its row. Give spectral_psf and det to photons_from_lines or "
          "photons_from_spectrum instead.")
    margin = int(margin)
    if spectral_psf == "quadrature":
        sigma = _fwhm_to_sigma(spectral_psf_fwhm(telescope, det, slit_width))
        reach = spectral_psf_reach(telescope, det, slit_width)
        offsets = np.arange(-reach, reach + 1)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    elif spectral_psf == "convolution":
        kernel = spectral_line_spread(telescope, det, slit_width)
    else:
        raise ValueError(
            f"spectral_psf must be 'quadrature' or 'convolution', got {spectral_psf!r}."
        )
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
    synthesis file. Read them with
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
          "euvst_response.read_synthesis_products(path, keys=['dem_map', 'em_tv']).")
    if "dem_data" not in results:
        raise KeyError(
            "DEM data is not in the results, as no released version wrote it there. It is "
            "in the synthesis file: read it with "
            "euvst_response.read_synthesis_products(path, keys=['dem_map', 'em_tv'])."
        )

    return results["dem_data"]


def _load_psf_ascii(fname: Path, skip: int) -> np.ndarray:
    """Load PSF data from ASCII file."""
    _warn("_load_psf_ascii", _PSF_FILES)
    return np.loadtxt(fname, skiprows=skip, encoding="utf-16 LE")


def _resample_psf(psf: np.ndarray, res_in: u.Quantity, res_out: u.Quantity) -> np.ndarray:
    """Resample PSF to different pixel scale."""
    _warn("_resample_psf", _PSF_FILES)
    factor = (res_in / res_out).decompose().value
    return zoom(psf, factor, order=1)


def _combine_psfs(psf_focus: np.ndarray, psf_mesh: np.ndarray, crop: float = 0.99, size: int | None = None) -> np.ndarray:
    """Convolve the focus and mesh PSFs and crop to an energy fraction or size."""
    _warn("_combine_psfs", _PSF_FILES)
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
