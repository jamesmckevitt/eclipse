"""The docs' full-CCD frame, with the continuum, with the slit through the active region's core."""
import astropy.units as u
import numpy as np

from eclipse import Simulation, full_ccd_frame, read_band_lines, write_full_frame
from eclipse.continuum import compute_continuum_fiasco

if __name__ == "__main__":
    lines = read_band_lines("frame/band_lines.h5")
    wavelength = np.linspace(170.0, 214.0, 10481) * u.AA
    free, two_photon = compute_continuum_fiasco(wavelength, lines.logT_grid, lines.logN_grid,
                                                abundance=lines.abundance)
    sim = Simulation(slit_width=0.4 * u.arcsec, psf=True, noise=True, slit_mask=True)
    frame = full_ccd_frame("synth/fe12.h5", lines, slit_position=0 * u.arcsec,
                           exposure=60 * u.s, sim=sim, n_workers=16,
                           continuum=(wavelength, free, two_photon))
    write_full_frame(frame, "frame/frame.h5")
