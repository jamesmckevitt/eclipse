import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from eclipse import (Detector_SWC, Simulation, Telescope_EUVST,
                     create_uniform_intensity_cube, monte_carlo)

telescope = Telescope_EUVST()
detector = Detector_SWC()
intensities = np.geomspace(100, 10000, 5) * u.erg / (u.s * u.cm**2 * u.sr)

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
for slit in [0.2, 0.4] * u.arcsec:
    simulation = Simulation(slit_width=slit, expos=20 * u.s, psf=True)
    precision = []
    for intensity in intensities:
        # Fe XII 195.12, made again for each slit as the PSF is applied here.
        cube = create_uniform_intensity_cube(
            intensity, 195.119 * u.AA, 20 * u.km / u.s, detector, simulation,
            tel=telescope)
        _, fits, _, _ = monte_carlo(
            cube, simulation.expos, detector, telescope, simulation,
            n_iter=100, fit_signals="dn", uniform_mode=True)
        velocity = fits["components"][fits["primary_component"]]["velocity"]
        precision.append(velocity["std"].squeeze().to_value(u.km / u.s))
    ax.plot(intensities.value, precision, "o-", label=f"{slit}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Intensity [erg / cm$^2$ / s / sr]")
ax.set_ylabel("Velocity precision [km / s]")
ax.set_title("Fe XII 195.12, 20 s")
ax.legend()
plt.show()
