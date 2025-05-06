import os
import gc
import numpy as np
from scipy.io import readsav
import astropy.units as u
import astropy.constants as const
from tqdm import tqdm
import psutil
import matplotlib.pyplot as plt


g_mu  = 1.29
Fe_amu = 55.845 * u.g / u.mol  # atomic weight of iron

vel_res = 10 * u.km/u.s
vel_lim = 100 * u.km/u.s
spt_res = 0.192 * u.Mm
wvl0 = 195.119 * u.angstrom
wvl_res = (vel_res.cgs * wvl0.cgs / const.c.cgs)


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
temp = load_cube(paths['T']  , downsample=2, unit=u.K)
rho  = load_cube(paths['rho'], downsample=2, unit=u.g/u.cm**3)
vx   = load_cube(paths['vx'] , downsample=2, unit=u.cm/u.s)
vy   = load_cube(paths['vy'] , downsample=2, unit=u.cm/u.s)
vz   = load_cube(paths['vz'] , downsample=2, unit=u.cm/u.s)

nx, ny, nz = temp.shape

los_length = spt_res.cgs * nz

print("Load the goft function")
tmp = read_contribution_funcs('G_of_T.sav')
goft = tmp['Fe12_195.1190']['g_tn']
goft_logN = tmp['Fe12_195.1190']['logN']
goft_logT = tmp['Fe12_195.1190']['logT']

print(f"Precomputing lookup indices...")
logT_cube = np.log10(temp.value)
ne        = (rho/(g_mu*const.u.cgs)).to(1/u.cm**3)
logN_cube = np.log10(ne.value)

print(f"Calculating thermal width per voxel...")
thermal_widths = np.sqrt(2 * const.k_B.cgs * temp.cgs / (Fe_amu / const.N_A.cgs))

print(f"Calculating the contribution function per voxel...")
logT_flat = logT_cube.ravel()
logN_flat = logN_cube.ravel()
idx_T = np.searchsorted(goft_logT, logT_flat)
idx_N = np.searchsorted(goft_logN, logN_flat)
idx_T = np.clip(idx_T, 0, len(goft_logT) - 1)
idx_N = np.clip(idx_N, 0, len(goft_logN) - 1)
Garray = goft[idx_T, idx_N].reshape((nx, ny, nz))

Carray = Garray / (4*np.pi) * (10**logN_cube)**2

print(f"Calculating the spectral grid...")
gauss_wdth = thermal_widths / np.sqrt(2)
gauss_peak = 1/np.sqrt(2*np.pi*gauss_wdth)
gauss_cent = vz

gauss_x = np.arange(-vel_lim.value, vel_lim.value + vel_res.value, vel_res.value) * u.km/u.s
gauss_x = gauss_x.to(u.cm/u.s)

gx = gauss_x[None, None, None, :]                # shape (1,1,1,nvel)
gc = gauss_cent[..., None]                       # shape (nx,ny,nz,1)
gw = gauss_wdth[..., None]                       # shape (nx,ny,nz,1)
thermal_widths = gauss_peak[..., None] * np.exp(-0.5 * ((gx - gc) / gw)**2)
spectral_grid = thermal_widths * Carray[..., None] * spt_res

print(f"Integrating along the z axis...")
output = spectral_grid.sum(axis=2)  # shape: (nx, ny, nvel)

# imshow the output summed along the wavelength axis
plt.imshow(np.log10(output.sum(axis=2).T.value), aspect='equal', cmap='inferno')
plt.colorbar(label='Log(Intensity)')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.title('Log Integrated Intensity')
plt.show()