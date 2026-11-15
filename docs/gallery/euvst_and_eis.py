import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from eclipse import Detector_SWC, Telescope_EIS, Telescope_EUVST

wavelength = np.linspace(170, 212, 421) * u.AA

euvst = Telescope_EUVST().ea_and_throughput(wavelength) * Detector_SWC().qe_euv
eis = {
    "EIS, pre-flight": Telescope_EIS(),
    "EIS, 2022": Telescope_EIS(calibration="dz2025", date="2022-01-01"),
}

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
ax.plot(wavelength.value, euvst.to_value(u.cm**2), label="EUVST-SW")
for label, telescope in eis.items():
    effective_area = telescope.effective_area(wavelength)
    ax.plot(wavelength.value, effective_area.to_value(u.cm**2), label=label)
ax.set_yscale("log")
ax.set_xlabel("Wavelength [Angstrom]")
ax.set_ylabel("Effective area [cm$^2$]")
ax.legend()
plt.show()
