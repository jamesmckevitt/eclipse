import os
import gc
import numpy as np
from scipy.io import readsav
import astropy.units as u
import astropy.constants as const
from tqdm import tqdm
import psutil
import matplotlib.pyplot as plt

downsample = 2

g_mu  = 1.29  # DOI: 10.1051/0004-6361:20041507, appendix A
Fe_amu = 55.845 * u.g / u.mol  # atomic weight of iron

vel_res = 5 * u.km/u.s
vel_lim = 50 * u.km/u.s
spt_res = 0.192 * u.Mm
wvl0 = 195.119 * u.angstrom
wvl_res = (vel_res.cgs * wvl0.cgs / const.c.cgs)

if downsample:
    spt_res = spt_res * downsample

def load_cube(path, shape=(512,768,256), unit=None, downsample=False):
    """
    Load a binary cube and optionally attach an astropy unit.
    Returns an ndarray or Quantity of shape (nx, ny, nz).
    """
    data = np.fromfile(path, dtype=np.float32).reshape(shape, order='F')
    data = np.transpose(data, (0, 2, 1))

    if downsample:
        data = data[::downsample, ::downsample, ::downsample]

    return data * unit if unit is not None else data

def read_contribution_funcs(savfile):
    """
    Read CHIANTI G(T,N) structure from a .sav file.
    Returns a dict: { line_name: { 'wl0': Quantity(cm),
                                   'atom': int,
                                   'g_tn': ndarray [erg cm3 / s],
                                   'logT': array, 'logN': array } }
    """
    raw = readsav(savfile)
    goft = {}
    for entry in raw['goftarr']:
        line = entry[0].decode()
        # assume line is like "Fe11_195.119"
        wl_angstrom = float(line.split('_')[1]) * u.AA
        goft[line] = {
            'wl0' : wl_angstrom.to(u.cm),
            'atom': int(entry[1]),
            'g_tn': entry[4] * (u.erg * u.cm**3 / u.s),
            'logT': raw['logTarr'],
            'logN': raw['logNarr']
        }
    return goft

# File paths
base = '/home/jm/solar/solc/solc_euvst_sw_response/tei_synthesis_ver2.2_20250422/input/AR_64x192x192'
files = dict(
    T   = 'temp/eosT.0270000',
    rho = 'rho/result_prim_0.0270000',
    vx  = 'vx/result_prim_1.0270000',
    vy  = 'vy/result_prim_3.0270000',
    vz  = 'vz/result_prim_2.0270000'
)
paths = {k:os.path.join(base,fn) for k,fn in files.items()}

print(f"Loading static cubes ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
temp = load_cube(paths['T']  , downsample=downsample, unit=u.K)
rho  = load_cube(paths['rho'], downsample=downsample, unit=u.g/u.cm**3)
vx   = load_cube(paths['vx'] , downsample=downsample, unit=u.cm/u.s)
vy   = load_cube(paths['vy'] , downsample=downsample, unit=u.cm/u.s)
vz   = load_cube(paths['vz'] , downsample=downsample, unit=u.cm/u.s)

assert temp.shape == rho.shape == vx.shape == vy.shape == vz.shape, "Cubes must have the same shape"
nx, ny, nz = temp.shape

print("Load the goft function")
tmp = read_contribution_funcs('G_of_T.sav')
goft = tmp['Fe12_195.1190']['g_tn']
goft_logN = tmp['Fe12_195.1190']['logN']
goft_logT = tmp['Fe12_195.1190']['logT']

print(f"Precomputing lookup indices...")
logT_cube = np.log10(temp.value)
ne        = (rho/(g_mu*const.u)).cgs
logN_cube = np.log10(ne.value)

print(f"Calculating the contribution function per voxel...")
logT_flat = logT_cube.ravel()
logN_flat = logN_cube.ravel()
idx_T = np.searchsorted(goft_logT, logT_flat)
idx_N = np.searchsorted(goft_logN, logN_flat)
idx_T = np.clip(idx_T, 0, len(goft_logT) - 1)
idx_N = np.clip(idx_N, 0, len(goft_logN) - 1)
Garray = goft[idx_N, idx_T].reshape((nx, ny, nz))

# get the indices of the voxel with the maximum G
max_idx = np.unravel_index(np.argmax(Garray), Garray.shape)
i, j, k = max_idx
print(f"  logT = {logT_cube[i, j, k]:.2f}, logN = {logN_cube[i, j, k]:.2f}, Ne = {ne[i, j, k]:.2e}, rho = {rho[i, j, k]:.2e}")
print(f"  G = {Garray[i, j, k]:.2e}")

print(f"Calculating the integrated intensity per voxel [erg/s/cm3/sr]...")
Carray = Garray * (1/(4*np.pi*u.sr)) * ne**2

# imshow the Carray summed along the z axis
plt.imshow(np.log10((Carray.sum(axis=2).T*spt_res.cgs).value), aspect='equal', cmap='inferno')
plt.colorbar(label='Log(Intensity)')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.title('Log Integrated Intensity')
plt.show()

print(f"Calculating thermal width per voxel...")
probable_speed = np.sqrt(2 * const.k_B * temp / (Fe_amu / const.N_A)).cgs  # v_p = sqrt(2kT/m), m=M/N_A : cm/s
# broadening along LOS: g(v) = 1/sqrt(pi*sigma) * exp(-v^2/(wdth^2)), sigma = sqrt(kT/m), therefore sigma = v_p/sqrt(2)
gauss_wdth = probable_speed / np.sqrt(2)

print(f"Calculating the specific intensity for each voxel [erg/s/cm2/sr/cm]...")
gauss_peak = 1/(np.sqrt(2*np.pi*gauss_wdth**2))  # s/cm
gauss_cent = vz  # cm/s

gauss_x = np.arange(-vel_lim.to(u.km/u.s).value, vel_lim.to(u.km/u.s).value + vel_res.to(u.km/u.s).value, vel_res.to(u.km/u.s).value) * u.km/u.s
gauss_x = gauss_x.to(u.cm/u.s)

gx = gauss_x[None, None, None, :]                # shape (1,1,1,nvel)
gc = gauss_cent[..., None]                       # shape (nx,ny,nz,1)
gw = gauss_wdth[..., None]                       # shape (nx,ny,nz,1)
thermal_gauss = gauss_peak[..., None] * np.exp(-0.5 * ((gx - gc) / gw)**2)
normalised_gauss = thermal_gauss.value / (thermal_gauss.value.sum(axis=3, keepdims=True) + 1e-10)  # shape (nx,ny,nz,nvel)

# # Calculate the Gaussian for each voxel
# for i in tqdm(range(nx), desc="Calculating gaussians", unit="x"):
#     for j in range(ny):
#         for k in range(nz):
#             gx = gauss_x.value
#             gc = gauss_cent[i, j, k].value
#             gw = gauss_wdth[i, j, k].value
#             thermal_gauss = gauss_peak[i, j, k] * np.exp(-0.5 * ((gx - gc) / gw)**2)
#             normalised_gauss = thermal_gauss / thermal_gauss.sum()

spectral_grid = normalised_gauss * Carray[..., None] * spt_res.cgs

print(f"Integrating along the z axis...")
output = spectral_grid.sum(axis=2)  # shape: (nx, ny, nvel)

fig, ax = plt.subplots()
img = ax.imshow(
  np.log10(output.sum(axis=2).T.value),
  aspect='equal',
  cmap='inferno',
  origin='upper'
)
plt.colorbar(img, ax=ax, label='Log(Intensity)')
ax.set_xlabel('X pixel')
ax.set_ylabel('Y pixel')
ax.set_title('Log Integrated Intensity')

def onclick(event):
  if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
    # round to nearest pixel
    x_pix = int(round(event.xdata))
    y_pix = int(round(event.ydata))
    # extract spectrum at that pixel
    spectrum = output[x_pix, y_pix, :].value
    # velocity axis in km/s
    vel = gauss_x.to(u.km/u.s).value

    # plot the clicked spectrum
    fig2, ax2 = plt.subplots()
    ax2.plot(vel, spectrum)
    ax2.set_xlabel('Velocity (km/s)')
    ax2.set_ylabel('Intensity')
    ax2.set_title(f'Spectrum at pixel ({x_pix}, {y_pix})')
    plt.show()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=False)

print("Loading up Tei'sans atmosphere for comparison...")
tmp = readsav("/home/jm/solar/solc/solc_euvst_sw_response/SI_Fe_XII_1952_d0_xy_0270000.sav")
atmosphere = tmp['si_xy_dl']
atmosphere = atmosphere.transpose((2, 1, 0))  # shape (nx, ny, nwvl)

# imshow the atmosphere integrated intensity (integrated over the spectral dimension)
fig_atm, ax_atm = plt.subplots()
atm_intensity = atmosphere.sum(axis=2)
img_atm = ax_atm.imshow(np.log10(atm_intensity.T), aspect='equal', cmap='inferno', origin='upper')
plt.colorbar(img_atm, ax=ax_atm, label='Log(Intensity)')
ax_atm.set_xlabel('X pixel')
ax_atm.set_ylabel('Y pixel')
ax_atm.set_title('Log Integrated Atmosphere Intensity')

# define click event to display the atmospheric spectrum at the clicked pixel
def onclick_atm(event):
  if event.inaxes is ax_atm and event.xdata is not None and event.ydata is not None:
    # round to nearest pixel
    x_pix = int(round(event.xdata))
    y_pix = int(round(event.ydata))
    # extract the spectrum at that pixel (wavelength dimension)
    spectrum = atmosphere[x_pix, y_pix, :]
    # define a wavelength axis as pixel index (modify if you have a real wavelength calibration)
    wvl = np.arange(atmosphere.shape[2])
    
    fig2, ax2 = plt.subplots()
    ax2.plot(wvl, spectrum)
    ax2.set_xlabel('Wavelength Pixel')
    ax2.set_ylabel('Intensity')
    ax2.set_title(f'Atmospheric Spectrum at Pixel ({x_pix}, {y_pix})')
    plt.show()

cid_atm = fig_atm.canvas.mpl_connect('button_press_event', onclick_atm)
plt.show()