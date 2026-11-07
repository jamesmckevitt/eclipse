"""The continuum and the frame, as in steps 2 and 3 of the full-CCD frame page."""
import astropy.units as u
import numpy as np

from eclipse import full_ccd_frame, read_band_lines, write_full_frame
from eclipse.continuum import compute_continuum_fiasco

if __name__ == "__main__":
    lines = read_band_lines("band_lines.h5")
    wavelength = np.linspace(170.0, 214.0, 10481) * u.AA
    free, two_photon = compute_continuum_fiasco(wavelength, lines.logT_grid, lines.logN_grid,
                                                abundance=lines.abundance)
    frame = full_ccd_frame("run/input/muram_ar_fe12.h5", lines, slit_position=0 * u.arcsec,
                           exposure=60 * u.s, slit_width=0.4 * u.arcsec, slit_mask=True,
                           n_workers=16, continuum=(wavelength, free, two_photon))
    write_full_frame(frame, "frame.h5")
