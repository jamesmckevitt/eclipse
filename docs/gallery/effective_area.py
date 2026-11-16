import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from eclipse import Detector_SWC, Telescope_EUVST

telescope = Telescope_EUVST()
detector = Detector_SWC()
wavelength = np.linspace(170, 214, 441) * u.AA

effective_area = telescope.ea_and_throughput(wavelength) * detector.qe_euv

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
ax.plot(wavelength.value, effective_area.to_value(u.cm**2))
ax.set_xlabel("Wavelength [Angstrom]")
ax.set_ylabel("Effective area [cm$^2$]")
ax.set_title("EUVST-SW")
plt.show()
