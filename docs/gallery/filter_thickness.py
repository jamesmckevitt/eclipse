import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from eclipse import AluminiumFilter, Detector_SWC, Telescope_EUVST

detector = Detector_SWC()
wavelength = np.linspace(170, 214, 441) * u.AA

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
for thickness in [1000, 1485, 2000] * u.AA:
    telescope = Telescope_EUVST(filter=AluminiumFilter(al_thickness=thickness))
    effective_area = telescope.ea_and_throughput(wavelength) * detector.qe_euv
    ax.plot(wavelength.value, effective_area.to_value(u.cm**2),
            label=f"{thickness:.0f}")
ax.set_xlabel("Wavelength [Angstrom]")
ax.set_ylabel("Effective area [cm$^2$]")
ax.legend(title="Aluminium")
plt.show()
