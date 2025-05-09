## TODO: add a point of real EIS data (e.g. from Brooks, Warren 2011 doi:10.1088/2041-8205/727/1/L13 paper) to show our calculation for EIS is correct
## TODO: make these plots for different cases (AR, quiet Sun, flare)

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from matplotlib.colors import LogNorm

swc_effective_area = 1.25  # cm²
swc_theta_arcsec = 0.159 # arcseconds
swc_theta_rad = np.radians(swc_theta_arcsec / 3600)  # Convert arcseconds to radians

eis_effective_area = 0.25  # cm²
eis_theta_arcsec = 1.5  # arcseconds
eis_theta_rad = np.radians(eis_theta_arcsec / 3600)  # Convert arcseconds to radians

e_in_j = 1e4  # Observed energy in J/cm^2/s/sr/cm
wav = 1.95e-8  # Wavelength in meters (195 A)
photon_energy = (const.h.to('J.s').value * const.c.to('m/s').value) / wav  # Energy of a photon in Joules

e_in_photon = e_in_j / photon_energy  # Energy in photons/cm^2/s/sr/cm

cm_per_pixel = 1.69e-10  # cm/pixel

e_in_photon = e_in_photon * cm_per_pixel  # Energy in photons/cm^2/s/sr/pixel

theta_range_arcsec = np.linspace(0.05, 2.0, 400)  # theta values in arcseconds
theta_range = np.radians(theta_range_arcsec / 3600)  # Convert arcseconds to radians
e_out_photon_values = [0.01, 0.1, 1.0]

plt.figure(figsize=(8, 6))
for e_out in e_out_photon_values:
  ea = e_out / (e_in_photon * (4 * np.tan(theta_range / 2) ** 2))
  plt.plot(theta_range_arcsec, ea, label=f"E_out_photon = {e_out}")

# Plot horizontal lines for SWC and EIS indicating their effective areas.
plt.axhline(swc_effective_area, color='red', linestyle='--',
      label=f"SWC: EA = {swc_effective_area} cm², Theta = {swc_theta_arcsec} arcsec")
plt.axhline(eis_effective_area, color='blue', linestyle='--',
      label=f"EIS: EA = {eis_effective_area} cm², Theta = {eis_theta_arcsec} arcsec")

# Plot vertical lines for SWC and EIS indicating their theta values.
plt.axvline(swc_theta_arcsec, color='red', linestyle=':', label=f"SWC: Theta = {swc_theta_arcsec} arcsec")
plt.axvline(eis_theta_arcsec, color='blue', linestyle=':', label=f"EIS: Theta = {eis_theta_arcsec} arcsec")

plt.xlabel("Theta (arcsec)")
plt.ylabel("Effective Area (cm²)")
plt.yscale("log")  # Set y-axis to log scale
plt.title("Effective Area vs Theta for Various E_out_photon Values")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show(block=False)

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

# Plot the heatmap
plt.figure(figsize=(8, 6))
c = plt.pcolormesh(Theta_grid, EA_grid, e_out_grid, shading='auto', cmap='viridis', norm=LogNorm(vmin=0.1, vmax=10))
plt.xlabel("Theta (arcsec)")
plt.ylabel("Effective Area (cm²)")
plt.yscale("log")
plt.xlim(swc_theta_arcsec, eis_theta_arcsec)
plt.ylim(eis_effective_area, swc_effective_area)
plt.title("E_out vs Theta and Effective Area")
plt.colorbar(c, label="E_out_photon")
plt.tight_layout()
plt.show(block=False)



# Combined plot: line curves with faint heatmap background
plt.figure(figsize=(8, 6))

# draw the heatmap first (with low alpha so it's faint)
c = plt.pcolormesh(
  Theta_grid,
  EA_grid,
  e_out_grid,
  shading='auto',
  cmap='viridis',
  norm=LogNorm(vmin=0.1, vmax=10),
  alpha=0.3
)

# overplot the three effective‐area curves
for e_out in e_out_photon_values:
  ea = e_out / (e_in_photon * (4 * np.tan(theta_range / 2) ** 2))
  plt.plot(theta_range_arcsec, ea, label=f"E_out_photon = {e_out}")

# SWC / EIS horizontal lines
plt.axhline(swc_effective_area, color='red',   linestyle='--',
      label=f"SWC: EA = {swc_effective_area} cm², Theta = {swc_theta_arcsec}″")
plt.axhline(eis_effective_area, color='blue',  linestyle='--',
      label=f"EIS: EA = {eis_effective_area} cm², Theta = {eis_theta_arcsec}″")

# SWC / EIS vertical lines
plt.axvline(swc_theta_arcsec, color='red',  linestyle=':',
      label=f"SWC: Theta = {swc_theta_arcsec}″")
plt.axvline(eis_theta_arcsec, color='blue', linestyle=':',
      label=f"EIS: Theta = {eis_theta_arcsec}″")

plt.xlabel("Theta (arcsec)")
plt.ylabel("Effective Area (cm²)")
plt.yscale("log")
plt.title("Effective Area vs Theta with E_out_photon Heatmap")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.colorbar(c, label="E_out_photon")
plt.tight_layout()
plt.show(block=False)

input("Press Enter to continue...")