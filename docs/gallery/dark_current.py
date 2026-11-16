import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from eclipse import Detector_EIS, Detector_SWC

temperature = np.linspace(-75, -30, 46) * u.deg_C

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
for name, detector in [("EUVST-SW", Detector_SWC), ("EIS", Detector_EIS)]:
    dark_current = [detector(ccd_temperature=t).dark_current.value
                    for t in temperature]
    ax.plot(temperature.value, dark_current, label=name)
ax.set_yscale("log")
ax.set_xlabel("CCD temperature [Celsius]")
ax.set_ylabel("Dark current [electron / pixel / s]")
ax.legend()
plt.show()
