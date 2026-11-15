import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from eclipse import Telescope_EUVST

telescope = Telescope_EUVST()
wavelength = np.linspace(170, 214, 441) * u.AA

parts = {
    "Primary mirror": telescope.primary_mirror_efficiency(wavelength),
    "Mirror roughness": telescope.microroughness_efficiency(wavelength),
    "Grating": telescope.grating_efficiency(wavelength),
    "Filter": telescope.filter.total_throughput(wavelength),
}

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
for name, throughput in parts.items():
    ax.plot(wavelength.value, np.asarray(throughput), label=name)
ax.set_xlabel("Wavelength [Angstrom]")
ax.set_ylabel("Throughput")
ax.legend()
plt.show()
