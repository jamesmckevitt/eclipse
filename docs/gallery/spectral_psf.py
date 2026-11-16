import astropy.units as u
import matplotlib.pyplot as plt

from eclipse import Detector_SWC, Telescope_EUVST, spectral_psf_fwhm

telescope = Telescope_EUVST()
detector = Detector_SWC()
slits = [0.2, 0.4, 0.8, 1.6] * u.arcsec

# spectral_psf_fwhm gives the width in detector pixels.
fwhm = [spectral_psf_fwhm(telescope, detector, slit) for slit in slits] * u.pix
fwhm = (fwhm * detector.wvl_res).to(u.mAA)

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
ax.plot(slits.value, fwhm.value, "o-")
ax.set_xlabel("Slit width [arcsec]")
ax.set_ylabel("Spectral PSF FWHM [mAngstrom]")
plt.show()
