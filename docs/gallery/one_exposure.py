import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from eclipse import (Detector_SWC, Simulation, Telescope_EUVST,
                     create_uniform_intensity_cube, fit_cube_gauss,
                     simulate_once)

telescope = Telescope_EUVST()
detector = Detector_SWC()
simulation = Simulation(slit_width=0.2 * u.arcsec, psf=True)

# Fe XII 195.12
intensity = 1000 * u.erg / (u.s * u.cm**2 * u.sr)
cube = create_uniform_intensity_cube(
    intensity, 195.119 * u.AA, 20 * u.km / u.s, detector, simulation,
    tel=telescope)
# simulate_once gives the signal after each step; the last is in DN.
dn = simulate_once(cube, 20 * u.s, detector, telescope, simulation,
                   uniform_mode=True)[-1]
fit, units = fit_cube_gauss(dn, n_jobs=1)

peak, centre, sigma, background = fit[0, 0]
centre = (centre * units[1]).to_value(u.AA)
sigma = (sigma * units[2]).to_value(u.AA)
wavelength = dn.axis_world_coords("em.wl")[0].to_value(u.AA)
fine = np.linspace(wavelength[0], wavelength[-1], 500)
gaussian = background + peak * np.exp(-0.5 * ((fine - centre) / sigma) ** 2)

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
ax.step(wavelength, dn.data[0, 0], where="mid", label="Simulated")
ax.plot(fine, gaussian, label="Fit")
ax.set_xlabel("Wavelength [Angstrom]")
ax.set_ylabel("Signal [DN / pixel]")
ax.set_title("Fe XII 195.12, 20 s")
ax.legend()
plt.show()
