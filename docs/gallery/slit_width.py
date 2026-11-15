import astropy.units as u
import matplotlib.pyplot as plt

from eclipse import (Detector_SWC, Simulation, Telescope_EUVST,
                     create_uniform_intensity_cube, simulate_once)

telescope = Telescope_EUVST()
detector = Detector_SWC()
intensity = 1000 * u.erg / (u.s * u.cm**2 * u.sr)

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
for slit in [0.2, 0.4, 0.8, 1.6] * u.arcsec:
    simulation = Simulation(slit_width=slit, psf=True, noise=False)
    # Fe XII 195.12
    cube = create_uniform_intensity_cube(
        intensity, 195.119 * u.AA, 20 * u.km / u.s, detector, simulation,
        tel=telescope)
    dn = simulate_once(cube, 20 * u.s, detector, telescope, simulation,
                       uniform_mode=True)[-1]
    wavelength = dn.axis_world_coords("em.wl")[0].to_value(u.AA)
    ax.step(wavelength, dn.data[0, 0], where="mid", label=f"{slit}")
ax.set_xlim(194.8, 195.44)
ax.set_xlabel("Wavelength [Angstrom]")
ax.set_ylabel("Signal [DN / pixel]")
ax.set_title("Fe XII 195.12, 20 s, without noise")
ax.legend(title="Slit")
plt.show()
