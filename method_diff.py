import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import find_peaks

# ----------------------------------------------------------------------
# Load the two synthetic data cubes and integrate line intensity
# ----------------------------------------------------------------------
dill.load_session('final_state_tei_synth.pkl')
fe12_tei_int = I_cubes['Fe12_195.1190'].cgs.value
d_te         = (gofnt_dict['Fe12_195.1190']['wl_grid'][1] -
                gofnt_dict['Fe12_195.1190']['wl_grid'][0]).cgs.value

dill.load_session('final_state_mck_synth.pkl')
fe12_mee_int = goft['Fe12_195.1190']['si']
d_me         = (goft['Fe12_195.1190']['wl_grid'][1] -
                goft['Fe12_195.1190']['wl_grid'][0]).cgs.value

# Integrated (1-D) intensities [erg/cm2/s/sr]
I_tei = np.sum(fe12_tei_int, axis=2) * d_te
I_mee = np.sum(fe12_mee_int, axis=2) * d_me

# ----------------------------------------------------------------------
# Difference metrics
# ----------------------------------------------------------------------
log_tei   = np.log10(I_tei, where=I_tei > 0)
log_mee   = np.log10(I_mee, where=I_mee > 0)
log_diff  = log_tei - log_mee
perc_diff = log_diff / log_mee * 100.0

# ----------------------------------------------------------------------
# Prepare axes extents
# ----------------------------------------------------------------------
nx, ny, _  = fe12_tei_int.shape
dx_pix     = voxel_dx.to("Mm").value
dy_pix     = voxel_dy.to("Mm").value
extent     = (0, nx * dx_pix, 0, ny * dy_pix)

# ----------------------------------------------------------------------
# Plot difference map
# ----------------------------------------------------------------------
fig, ax = plt.subplots()
im = ax.imshow(perc_diff.T, origin='lower', cmap='PiYG', vmin=-9, vmax=9, extent=extent)
ax.set_xlabel('x (Mm)')
ax.set_ylabel('y (Mm)')
cbar = fig.colorbar(im, ax=ax, orientation='horizontal', extend='both', location='top', label='Relative log-intensity difference (%)')
cbar.set_ticks([-9, -6, -3, 0, 3, 6, 9])
plt.tight_layout()
plt.savefig('fe12_combined_diff.png', dpi=300)
plt.close()





def find_xy_by_peak_separation(cube, wl_grid, threshold_frac=0.5, n=1):
  """
  Find pixels sorted by how far apart their two strongest peaks are.
  If n=1 returns the pixel with the largest separation,
  if n=2 the second largest, etc.
  To get all sorted pixels, call with n=None and receive a list of (x, y) tuples.

  Returns:
    (x, y) tuple if n is int,
    or a list of (x, y) tuples if n is None.
  """
  nx, ny, nw = cube.shape
  wl = np.asarray(wl_grid)

  seps = []  # list of (sep, (x, y))
  for x in range(nx):
    for y in range(ny):
      spec = cube[x, y, :]
      if np.all(np.isnan(spec)):
        continue

      peaks, _ = find_peaks(spec)
      if peaks.size < 2:
        continue

      heights = spec[peaks]
      threshold = threshold_frac * np.nanmax(heights)
      strong = peaks[heights > threshold]
      if strong.size < 2:
        continue

      # two peaks with max wavelength separation
      idx_min, idx_max = strong.min(), strong.max()
      sep = abs(wl[idx_max] - wl[idx_min])
      seps.append((sep, (x, y)))

  # sort descending by sep
  seps.sort(key=lambda item: item[0], reverse=True)
  coords = [xy for _, xy in seps]

  if n is None:
    return coords
  if isinstance(n, int) and 1 <= n <= len(coords):
    return coords[n - 1]
  return None





xy_two_peaks = find_xy_by_peak_separation(fe12_tei_int,
                 gofnt_dict['Fe12_195.1190']['wl_grid'].cgs.value,
                 threshold_frac=0.5, n=750)

#### BUMPY PIXEL: 415,190 (n=50)
#### SEP PIXEL: 411,215 (n=1000) mee<tei
#### TILTY PIXEL: 429,168 (n=1010) mee<tei
#### TWOBUMP PIXEL: 424,172 (n=750) mee<tei




# quick plot the spectra for the found pixel
plt.figure(figsize=(8, 4))
x, y = xy_two_peaks
# spectrum = fe12_tei_int[x, y, :]
tei_spectra = fe12_tei_int[x, y, :]
mee_spectra = fe12_mee_int[x, y, :]
plt.plot(gofnt_dict['Fe12_195.1190']['wl_grid'].to("AA").value, tei_spectra, label='tei')
plt.plot(goft['Fe12_195.1190']['wl_grid'].to("AA").value, mee_spectra, label='mee', linestyle='--')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Intensity (erg/cm²/s/sr)')
plt.title(f'Spectrum at pixel ({x}, {y})')
plt.legend()
plt.tight_layout()
plt.savefig('fe12_spectrum_two_peaks.png', dpi=300)
plt.close()