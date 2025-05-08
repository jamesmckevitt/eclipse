import os
import gc
import numpy as np
from scipy.io import readsav
from scipy.ndimage import map_coordinates
import astropy.units as u
import astropy.constants as const
from tqdm import tqdm
import psutil
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

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
        goft[line] = {
            'wl0' : (float(line.split('_')[1]) * u.AA).to(u.cm),  # line format: "Fe11_195.119"
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

def calculate_specific_intensity(C_cube, gauss_peak, gauss_wdth, gauss_x_cgs, spt_res_cgs, vel_res_cgs, wvl_res_cgs, vz_cube):

    nx, ny, nz = C_cube.shape
    nl = gauss_x_cgs.size

    I_cube = np.zeros((nx, ny, nl))
    for i in tqdm(range(nx), desc='Looping through x axis', unit='x'):
        peak_i = gauss_peak[i, :, :]
        vz_i = vz_cube[i, :, :].cgs.value
        gw_i = gauss_wdth[i, :, :].cgs.value
        C_i = C_cube[i, :, :].cgs.value

        thermal_gauss = peak_i[:, :, None] * np.exp(-((gauss_x_cgs[None, None, :] - vz_i[:, :, None])**2) / (2 * gw_i[:, :, None]**2))
        I_cube[i, :, :] = np.sum( thermal_gauss * C_i[:, :, None] * spt_res_cgs * (vel_res_cgs / wvl_res_cgs), axis=1 ) * (u.erg / u.s / u.cm**2 / u.sr / u.cm)
    return I_cube


def main():

    raster_centre = (0.0, 0.0)  # location of raster on Sun in arcsec (x,y)
    vel_res     = 6 * u.km/u.s  # velocity bin width
    vel_lim     = 300 * u.km/u.s  # +-velocity range
    spt_res     = 0.064 * u.Mm  # spatial resolution
    vel_grid = np.arange(-vel_lim.to(u.cm/u.s).value,
                       vel_lim.to(u.cm/u.s).value + vel_res.to(u.cm/u.s).value,
                       vel_res.to(u.cm/u.s).value) * (u.cm/u.s)  # velocity grid
    nl = vel_grid.size  # spectral resolution

    downsample = False

    g_mu  = 1.29  #####################* (1/u.u)  # mean molecular weight solar (amu) [doi:10.1051/0004-6361:20041507]
    Fe_amu = 55.845 * u.g / u.mol  # atomic weight of iron

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
    temp_cube = load_cube(paths['T']  , downsample=downsample, unit=u.K)
    rho_cube  = load_cube(paths['rho'], downsample=downsample, unit=u.g/u.cm**3)
    vx_cube   = load_cube(paths['vx'] , downsample=downsample, unit=u.cm/u.s)
    vy_cube   = load_cube(paths['vy'] , downsample=downsample, unit=u.cm/u.s)
    vz_cube   = load_cube(paths['vz'] , downsample=downsample, unit=u.cm/u.s)
    assert temp_cube.shape == rho_cube.shape == vx_cube.shape == vy_cube.shape == vz_cube.shape, "Cubes must have the same shape"

    logT_cube = np.log10(temp_cube.value)
    ne        = (rho_cube/(g_mu*const.u.cgs)).to(1/u.cm**3)  # density / ()
    logN_cube = np.log10(ne.value)
    del ne, rho_cube

    print(f"Loading G(T,N) tables ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
    gofnt_dict = read_contribution_funcs('G_of_T.sav')

    ###### TEMP FOR DEBUG ######
    gofnt_dict = {k:v for k,v in gofnt_dict.items() if k == 'Fe12_195.1190'}
    ###### TEMP FOR DEBUG ######

    print(f"Calculating the contribution function [erg cm3/s] per voxel ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
    G_cubes = {}
    for line, info in tqdm(gofnt_dict.items(), desc='Lines', unit='line', leave=False):
        G_cubes[line] = interp2d_spline(info['g_tn'], info['logT'], info['logN'], logT_cube, logN_cube) * (u.erg * u.cm**3 / u.s)

    print(f"Calculating the integrated intensity [erg/s/cm3/sr] per voxel ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
    with np.errstate(over='ignore', invalid='ignore'):
        ne2 = (10**logN_cube * (1/u.cm**3))**2
        ne2[~np.isfinite(ne2)] = 0.0
        C_cubes = {line: G_cubes[line] * (1/(4*np.pi*u.sr)) * ne2 for line in tqdm(G_cubes.keys(), desc='Lines', unit='line', leave=False)}
    del G_cubes, ne2

    print(f"Calculating the specific intensity [erg/s/cm2/sr/cm] for each voxel ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
    probable_speed = np.sqrt(2 * const.k_B.cgs * temp_cube / (Fe_amu / const.N_A)).cgs  # v_p = sqrt(2kT/m), m=M/N_A : cm/s
    del temp_cube
    gauss_wdth = probable_speed / np.sqrt(2)  # broadening along LOS: g(v) = 1/sqrt(pi*sigma) * exp(-v^2/(wdth^2)), sigma = sqrt(kT/m), therefore sigma = v_p/sqrt(2) : cm/s
    del probable_speed

    gauss_peak = 1.0 / (np.sqrt(2 * np.pi * gauss_wdth**2))  # s / cm

    gauss_x_cgs = vel_grid.cgs.value  # velocity grid in cm/s

    spt_res_cgs = spt_res.to(u.cm).value  # save these to avoid repeated unit calculation
    vel_res_cgs = vel_res.to(u.cm/u.s).value

    I_cubes = {}
    for line, C_cube in tqdm(C_cubes.items(), desc='Lines', unit='line', leave=False):
        wvl0_cgs = gofnt_dict[line]['wl0'].cgs.value  # rest wavelength in cm
        wvl_res_cgs = vel_res_cgs * wvl0_cgs / const.c.cgs.value  # wavelength resolution in cm
        I_cubes[line] = calculate_specific_intensity(C_cube, gauss_peak, gauss_wdth, gauss_x_cgs, spt_res_cgs, vel_res_cgs, wvl_res_cgs, vz_cube)
    del C_cubes, gauss_wdth, gauss_peak, gauss_x_cgs

    print(f"Saving cubes to disk ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
    filename = 'I_cubes.npz'
    np.savez_compressed('I_cubes.npz', **{line: I_cubes[line] for line in I_cubes.keys()})
    print(f"File {filename} saved with total size {os.path.getsize(filename)/1e9:.2f} GB")
    print(f"Reload this using:")
    print(f"  tmp = np.load('{filename}')")
    print("  I_cubes = {line: tmp[line] for line in tmp.files}")

    # reload the .npz file
    tmp = np.load('I_cubes.npz')
    # remake the dictionary
    # I_cubes = {line: I_cubes[line] for line in I_cubes.files}
    I_cubes = {line: tmp[line] for line in tmp.files}

    I_cube = I_cubes['Fe12_195.1190']

    fig, ax = plt.subplots()
    img = ax.imshow(np.log10(I_cube.sum(axis=2).T), aspect='equal', cmap='inferno', origin='lower')
    plt.colorbar(img, ax=ax, label='Log(Intensity)')
    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')
    def onclick(event, ax=ax, cube=I_cube, vel_axis=vel_grid.to(u.km/u.s).value):
        if event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x_pix = int(round(event.xdata))
        y_pix = int(round(event.ydata))
        nx, ny, _ = cube.shape
        if not (0 <= x_pix < nx and 0 <= y_pix < ny):
            return
        spectrum = cube[x_pix, y_pix, :]
        vel = vel_axis
        fig2, ax2 = plt.subplots()
        ax2.plot(vel, spectrum)
        ax2.set_xlabel('Velocity (km/s)')
        ax2.set_ylabel('Intensity')
        ax2.set_title(f'Spectrum at pixel ({x_pix}, {y_pix})')
        plt.show()
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    globals().update(locals())
if __name__ == '__main__':
    main()