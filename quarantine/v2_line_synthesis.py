import os
import gc
import numpy as np
from scipy.io import readsav
import astropy.units as u
import astropy.constants as const
from tqdm import tqdm
import psutil
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

downsample = False

g_mu  = 1.29  # DOI: 10.1051/0004-6361:20041507, appendix A
Fe_amu = 55.845 * u.g / u.mol  # atomic weight of iron

vel_res = 6 * u.km/u.s
vel_lim = 300 * u.km/u.s
spt_res = 0.064 * u.Mm  # in z direction (note different in x/y)
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

def interp2d_spline(goft, goft_logT, goft_logN, logT_cube, logN_cube):
    """
    Bilinear interpolation via a B-spline of degree 1 in each direction.
    """
    # Build a spline of degree 1 (i.e. linear) in both N and T
    spline = RectBivariateSpline(
        goft_logN,        # sorted 1D array of logN
        goft_logT,        # sorted 1D array of logT
        goft,             # 2D array shape (nN, nT)
        kx=1, ky=1        # degree of spline = 1 => bilinear
    )
    # Flatten the target grid, interpolate, then reshape back
    pts_N = logN_cube.ravel()
    pts_T = logT_cube.ravel()
    G_flat = spline.ev(pts_N, pts_T)
    return G_flat.reshape(logT_cube.shape)


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

# print("Load the goft function")
# tmp = read_contribution_funcs('G_of_T.sav')
# goft = tmp['Fe12_195.1190']['g_tn']
# goft_logN = tmp['Fe12_195.1190']['logN']
# goft_logT = tmp['Fe12_195.1190']['logT']
# del tmp
tmp = readsav("/home/jm/solar/solc/solc_euvst_sw_response/quarantine/tei_synthesis_ver2.2_20250422/G_of_T_N/G_of_T_N_Fe_XII_1952.sav")
goft = tmp['g_of_t_n']
goft_logN = tmp['lognarr']
goft_logT = tmp['logtarr']
del tmp

print(f"Precomputing lookup indices...")
logT_cube = np.log10(temp.value)
ne        = (rho/(g_mu*const.u)).cgs
logN_cube = np.log10(ne.value)
del ne

print(f"Calculating the contribution function per voxel...")
Garray = interp2d_spline(goft, goft_logT, goft_logN, logT_cube, logN_cube) * (u.erg * u.cm**3 / u.s)  # note, CHIANTI says erg/cm3/s

print(f"Calculating the integrated intensity per voxel [erg/s/cm3/sr]...")
with np.errstate(over='ignore', invalid='ignore'):
    ne2 = ( 10**logN_cube * (1/u.cm**3) )**2
    ne2[~np.isfinite(ne2)] = 0.0
Carray = Garray * (1/(4*np.pi*u.sr)) * ne2
del Garray, ne2

li = Carray.sum(axis=2)*spt_res.cgs

print(f"Calculating the specific intensity for each voxel [erg/s/cm2/sr/cm]...")
probable_speed = np.sqrt(2 * const.k_B * temp / (Fe_amu / const.N_A)).cgs  # v_p = sqrt(2kT/m), m=M/N_A : cm/s
gauss_wdth = probable_speed / np.sqrt(2)  # broadening along LOS: g(v) = 1/sqrt(pi*sigma) * exp(-v^2/(wdth^2)), sigma = sqrt(kT/m), therefore sigma = v_p/sqrt(2)
del probable_speed

gauss_peak = 1.0 / (np.sqrt(2 * np.pi * gauss_wdth**2))

gauss_x = np.arange(-vel_lim.to(u.km/u.s).value,vel_lim.to(u.km/u.s).value + vel_res.to(u.km/u.s).value,vel_res.to(u.km/u.s).value) * u.km/u.s
gauss_x_cgs = gauss_x.cgs.value

spt_res_cgs = spt_res.cgs.value
vel_res_cgs = vel_res.cgs.value
wvl_res_cgs = wvl_res.cgs.value

nl = gauss_x_cgs.size
spectral_grid = np.zeros((nx, ny, nl))

for i in tqdm(range(nx), desc="Calculating gaussians", unit="x"):
    peak_i = gauss_peak[i, :, :]
    vz_i   = vz[i, :, :].cgs.value
    gw_i   = gauss_wdth[i, :, :].value
    C_i    = Carray[i, :, :]

    diff = (gauss_x_cgs[None, None, :] - vz_i[:, :, None]) / gw_i[:, :, None]

    thermal_gauss = peak_i[:, :, None] * np.exp(-0.5 * diff**2)

    tmp = thermal_gauss * C_i[:, :, None] * spt_res_cgs

    spectral_grid[i, :, :] = tmp.sum(axis=1)

spectral_grid_2 = spectral_grid * vel_res_cgs / wvl_res_cgs
spectral_grid_2 = spectral_grid_2 * (u.erg / u.s / u.cm**2 / u.sr / u.cm)

print(f"Integrating along the z axis...")
output = spectral_grid_2.sum(axis=2)  # shape: (nx, ny, nvel)

fig, ax = plt.subplots()
img = ax.imshow(
    np.log10(output.T.value),
    aspect='equal',
    cmap='inferno',
    origin='upper'
)
plt.colorbar(img, ax=ax, label='Log(Intensity)')
ax.set_xlabel('X pixel')
ax.set_ylabel('Y pixel')
ax.set_title('Log Integrated Intensity')

def onclick(event, ax=ax, cube=spectral_grid_2, vel_axis=gauss_x):
    """
    On mouse‐click inside our image axes, round to the nearest pixel,
    pull the full spectrum from `cube` and plot it against `vel_axis`.
    """
    if event.inaxes is not ax:
        return

    if event.xdata is None or event.ydata is None:
        return

    # round to the nearest pixel indices
    x_pix = int(round(event.xdata))
    y_pix = int(round(event.ydata))

    # guard against clicking outside the valid range
    nx, ny, _ = cube.shape
    if not (0 <= x_pix < nx and 0 <= y_pix < ny):
        return

    # extract spectrum (plain ndarray) at that (x,y)
    spectrum = cube[x_pix, y_pix, :].value

    # velocity axis in km/s as plain ndarray
    vel = vel_axis.to(u.km/u.s).value

    # plot the clicked spectrum
    fig2, ax2 = plt.subplots()
    ax2.plot(vel, spectrum)
    ax2.set_xlabel('Velocity (km/s)')
    ax2.set_ylabel('Intensity')
    ax2.set_title(f'Spectrum at pixel ({x_pix}, {y_pix})')
    plt.show()

# connect the click handler and show
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()