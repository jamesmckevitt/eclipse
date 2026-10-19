"""
Deprecated: helpers for tabulated PSF files, which ECLIPSE does not use.

They are now in :mod:`euvst_response.deprecated` and will be removed in a
future release.
"""
import warnings

from .deprecated import _combine_psfs, _load_psf_ascii, _resample_psf  # noqa: F401

warnings.warn("euvst_response.psf is deprecated and will be removed in a future release.",
              FutureWarning, stacklevel=2)
