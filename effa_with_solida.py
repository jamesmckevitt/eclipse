## TODO: add a point of real EIS data (e.g. from Brooks, Warren 2011 doi:10.1088/2041-8205/727/1/L13 paper) to show our calculation for EIS is correct
## TODO: make these plots for different cases (AR, quiet Sun, flare)

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from matplotlib.colors import LogNorm
from astropy import units as u

swc_ea = 1.25 * u.cm**2  # effective area
eis_ea = 0.25 * u.cm**2
swc_ss = 0.159 * u.arcsec  # spatial sampling
eis_ss = 1.5 * u.arcsec

e_in = 1e11 * u.erg/u.cm**2/u.s/u.sr/u.cm  # incident energy
wvl0 = 195.119 * u.angstrom

ph_e = ( (const.h * const.c) / wvl0 ).cgs.value * (u.erg/u.ph)  # photon energy
e_in_photon = e_in / ph_e; assert e_in_photon.unit == u.ph/u.cm**2/u.s/u.sr/u.cm







cm_per_pixel = 1.69e-10  # cm/pixel

e_in_photon = e_in_photon * cm_per_pixel  # Energy in photons/cm^2/s/sr/pixel

theta_range_arcsec = np.linspace(0.05, 2.0, 400)  # theta values in arcseconds
theta_range = np.radians(theta_range_arcsec / 3600)  # Convert arcseconds to radians
e_out_photon_values = [0.01, 0.1, 1.0]

# Define the effective area limits corresponding to the minimum and maximum E_out values.
theta_min_rad = np.radians(0.05 / 3600)
theta_max_rad = np.radians(2.0 / 3600)
ea_lower = 0.01 / (e_in_photon * (4 * np.tan(theta_max_rad / 2) ** 2))
ea_upper = 1.0 / (e_in_photon * (4 * np.tan(theta_min_rad / 2) ** 2))

# Create a grid for effective area (log-spaced) and theta (using theta_range_arcsec)
ea_vals = np.logspace(np.log10(ea_lower), np.log10(ea_upper), 400)
Theta_grid, EA_grid = np.meshgrid(theta_range_arcsec, ea_vals)
theta_rad_grid = np.radians(Theta_grid / 3600)

# For each (theta, EA) pair, compute the E_out using the relation:
#    ea = E_out / (e_in_photon * (4 * tan(theta/2)^2))
# => E_out = ea * e_in_photon * (4 * tan(theta/2)^2)
e_out_grid = EA_grid * e_in_photon * (4 * np.tan(theta_rad_grid / 2) ** 2)

plt.figure()

# draw the heatmap first (with low alpha so it's faint)
c = plt.pcolormesh(
  Theta_grid,
  EA_grid,
  e_out_grid,
  shading='auto',
  cmap='plasma',
  norm=LogNorm(vmin=0.1, vmax=10),
  alpha=.5
)

# overplot the three effective‐area curves
for e_out in e_out_photon_values:
  ea = e_out / (e_in_photon * (4 * np.tan(theta_range / 2) ** 2))
  plt.plot(theta_range_arcsec, ea, label=f"E_out_photon = {e_out}")

# SWC / EIS horizontal lines
plt.axhline(swc_effective_area, color='red',   linestyle='--',
      label=fr"EUVST-SWC ($A_{{eff}}$={swc_effective_area} cm$^2$, $\theta$={swc_theta_arcsec}{"\u2033"})")
plt.axhline(eis_effective_area, color='blue',  linestyle='--',
      label=fr"EIS ($A_{{eff}}$={eis_effective_area} cm$^2$, $\theta$={eis_theta_arcsec}{"\u2033"})")

# SWC / EIS vertical lines
plt.axvline(swc_theta_arcsec, color='red',  linestyle='--')
plt.axvline(eis_theta_arcsec, color='blue', linestyle='--')

plt.xlabel("Theta (arcsec)")
plt.ylabel(r"Effective Area (cm$^2$)")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.colorbar(c, label="Energy at detector [ph/pix/s]")
plt.tight_layout()
plt.show(block=False)