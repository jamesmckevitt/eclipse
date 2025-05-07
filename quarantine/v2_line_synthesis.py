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

#### DEBUGGING
const_c_cgs = 29979245800.000000
const_u = 1.66e-24

downsample = False

g_mu  = 1.29  # DOI: 10.1051/0004-6361:20041507, appendix A
Fe_amu = 55.845 * u.g / u.mol  # atomic weight of iron

vel_res = 6 * u.km/u.s
vel_lim = 300 * u.km/u.s
# spt_res = 0.192 * u.Mm
spt_res = 0.064 * u.Mm
wvl0 = 195.119 * u.angstrom
wvl_res = (vel_res.cgs * wvl0.cgs / const_c_cgs)

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
ne        = (rho/(g_mu*const_u)).cgs
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

# # imshow the Carray summed along the z axis
# plt.imshow(np.log10(li.T.value), aspect='equal', cmap='inferno')
# plt.colorbar(label='Intensity')
# plt.xlabel('X pixel')
# plt.ylabel('Y pixel')
# plt.title('Log Integrated Intensity')
# plt.show()

print(f"Calculating thermal width per voxel...")
probable_speed = np.sqrt(2 * const.k_B * temp / (Fe_amu / const.N_A)).cgs  # v_p = sqrt(2kT/m), m=M/N_A : cm/s
# broadening along LOS: g(v) = 1/sqrt(pi*sigma) * exp(-v^2/(wdth^2)), sigma = sqrt(kT/m), therefore sigma = v_p/sqrt(2)
gauss_wdth = probable_speed / np.sqrt(2)
del probable_speed

print(f"Calculating the specific intensity for each voxel [erg/s/cm2/sr/cm]...")
gauss_peak = 1/(np.sqrt(2*np.pi*gauss_wdth**2))  # s/cm
gauss_cent = vz  # cm/s

gauss_x = np.arange(-vel_lim.to(u.km/u.s).value, vel_lim.to(u.km/u.s).value + vel_res.to(u.km/u.s).value, vel_res.to(u.km/u.s).value) * u.km/u.s
gauss_x = gauss_x.to(u.cm/u.s)

nl = gauss_x.shape[0]

# gx = gauss_x[None, None, None, :]                # shape (1,1,1,nvel)
# gc = gauss_cent[..., None]                       # shape (nx,ny,nz,1)
# gw = gauss_wdth[..., None]                       # shape (nx,ny,nz,1)
# del gauss_cent, gauss_wdth

# thermal_gauss = gauss_peak[..., None] * np.exp(-0.5 * ((gx - gc) / gw)**2)
# normalised_gauss = thermal_gauss.value / (thermal_gauss.value.sum(axis=3, keepdims=True) + 1e-10)  # shape (nx,ny,nz,nvel)

from IPython import embed;embed()

spectral_grid = np.zeros((nx, ny, nl))  #  * (u.erg / u.s / u.cm**2 / u.sr / u.cm)  # shape (nx, ny, nvel)
for i in tqdm(range(nx), desc="Calculating gaussians", unit="x"):
    for j in range(ny):

        tmp = np.zeros((nz, nl))

        for k in range(nz):
            gx = gauss_x.value
            gc = gauss_cent[i, j, k].value
            gw = gauss_wdth[i, j, k].value
            thermal_gauss = gauss_peak[i, j, k] * np.exp(-0.5 * ((gx - gc) / gw)**2)
            tmp[k, :] = thermal_gauss * Carray[i, j, k] * spt_res.cgs.value
            # del gx, gc, gw, thermal_gauss, normalised_gauss

        spectral_grid[i, j, :] = tmp.sum(axis=0)  # shape (nvel)

# # Calculate the Gaussian for each voxel
# for i in tqdm(range(nx), desc="Calculating gaussians", unit="x"):
#     for j in range(ny):
#         for k in range(nz):
#             gx = gauss_x.value
#             gc = gauss_cent[i, j, k].value
#             gw = gauss_wdth[i, j, k].value
#             thermal_gauss = gauss_peak[i, j, k] * np.exp(-0.5 * ((gx - gc) / gw)**2)
#             normalised_gauss = thermal_gauss / thermal_gauss.sum()

# spectral_grid = normalised_gauss * Carray[..., None] * spt_res.cgs

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