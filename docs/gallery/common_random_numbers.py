import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from eclipse import (Detector_SWC, Simulation, Telescope_EUVST,
                     create_uniform_intensity_cube, monte_carlo)

telescope = Telescope_EUVST()
simulation = Simulation(slit_width=0.2 * u.arcsec, expos=20 * u.s, psf=True)
intensity = 300 * u.erg / (u.s * u.cm**2 * u.sr)
cube = create_uniform_intensity_cube(
    intensity, 195.119 * u.AA, 20 * u.km / u.s, Detector_SWC(), simulation,
    tel=telescope)
temperatures = np.arange(-70, -29, 2.5) * u.deg_C

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
for label, common in [("Independent", False), ("Common", True)]:
    precision = []
    for temperature in temperatures:
        detector = Detector_SWC(ccd_temperature=temperature)
        if common:
            np.random.seed(1234)   # the same value before each run
        _, fits, _, _ = monte_carlo(
            cube, simulation.expos, detector, telescope, simulation,
            n_iter=50, fit_signals="dn", uniform_mode=True,
            dark_current_inverse_transform=common)
        velocity = fits["components"][fits["primary_component"]]["velocity"]
        precision.append(velocity["std"].squeeze().to_value(u.km / u.s))
    ax.plot(temperatures.value, precision, "o-", label=label)
ax.set_xlabel("CCD temperature [Celsius]")
ax.set_ylabel("Velocity precision [km / s]")
ax.legend(title="Random numbers")
plt.show()
