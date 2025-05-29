## TODO: Improve method. Currently assumes atmosphere is isotopic (temperature and density) at each grid point.
##    Fix by summing temperature and density through LOS first to get temperature, density profile.
##    Plot as discrete points connected (linear interpolation between points) to get a smooth profile.
##    Then convolve with the G(n,T) function to get the contribution function for each LOS pixel.

## NOTE: This needs the G(n,T) function .sav file generated in IDL using my make_goft.pro script.
##     ChiantiPy and fiasco do not work properly and so this can't be done in Python yet. 

import os
import numpy as np
from scipy.io import readsav
import astropy.units as u
import astropy.constants as const
from tqdm import tqdm
import psutil
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import dill

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

def interp2d_spline(goft, goft_logT, goft_logN, logT_cube, logN_cube, cube_precision=np.float64):
    """
    Bilinear interpolation via a B-spline of degree 1 in each direction.
    """
    spline = RectBivariateSpline(
        goft_logN,        # sorted 1D array of logN
        goft_logT,        # sorted 1D array of logT
        goft,             # 2D array shape (nN, nT)
        kx=1, ky=1        # degree of spline = 1 => bilinear
    )
    pts_N = logN_cube.ravel()
    pts_T = logT_cube.ravel()
    G_flat = spline.ev(pts_N, pts_T)
    return G_flat.reshape(logT_cube.shape).astype(cube_precision)

def calculate_specific_intensity(C_cube, gauss_peak, gauss_wdth, gauss_x_cgs, spt_res_z_cgs, vel_res_cgs, wvl_res_cgs, vz_cube, cube_precision=np.float64, full_vectorization=False, ncpu=False):
    """
    Calculate the specific intensity for a given contribution function cube.
    The function uses a Gaussian velocity profile which is then converted to wavelength to calculate the intensity.
    WARNING: Full vecotirisation is likely to crash the code unless running on a large cluster (RAM>500 GB).
    """

    nx, ny, nz = C_cube.shape
    nl = gauss_x_cgs.size

    if full_vectorization:
        
        C_val    = C_cube.cgs.value      if hasattr(C_cube, 'cgs')    else np.asarray(C_cube)
        peak_val = np.asarray(gauss_peak)
        gw_val   = gauss_wdth.cgs.value
        vz_val   = vz_cube.cgs.value

        C_b    = C_val[..., None]
        peak_b = peak_val[..., None]
        gw_b   = gw_val[..., None]
        vz_b   = vz_val[..., None]

        x_b = gauss_x_cgs[None, None, None, :]
        exponent = -((x_b - vz_b)**2) / (2.0 * gw_b**2)

        thermal_gauss = peak_b * np.exp(exponent)
        prefactor = spt_res_z_cgs * (vel_res_cgs / wvl_res_cgs)
        emissivity = thermal_gauss * C_b * prefactor
        I_val = np.sum(emissivity, axis=2)
        I_cube = I_val

    else:

        if ncpu:
            # PARALLEL
            def _compute_intensity_slice(i):
                peak_i = gauss_peak[i, :, :]
                vz_i   = vz_cube[i, :, :].cgs.value
                gw_i   = gauss_wdth[i, :, :].cgs.value
                C_i    = C_cube[i, :, :].cgs.value
                thermal_gauss = peak_i[:, :, None] * np.exp(-((gauss_x_cgs[None, None, :] - vz_i[:, :, None])**2) / (2.0 * gw_i[:, :, None]**2))
                emissivity = thermal_gauss * C_i[:, :, None] * spt_res_z_cgs * (vel_res_cgs / wvl_res_cgs)
                return np.sum(emissivity, axis=1)
            I_slices = Parallel(n_jobs=ncpu)(delayed(_compute_intensity_slice)(i) for i in range(nx))
            I_cube = np.stack(I_slices, axis=0)

        else:
            # SERIAL
            I_cube = np.zeros((nx, ny, nl), dtype=cube_precision)
            for i in tqdm(range(nx), desc='Looping through x axis', unit='x', leave=False):
                peak_i = gauss_peak[i, :, :]
                vz_i = vz_cube[i, :, :].cgs.value
                gw_i = gauss_wdth[i, :, :].cgs.value
                C_i = C_cube[i, :, :].cgs.value
                thermal_gauss = peak_i[:, :, None] * np.exp(-((gauss_x_cgs[None, None, :] - vz_i[:, :, None])**2) / (2 * gw_i[:, :, None]**2))
                I_cube[i, :, :] = np.sum( thermal_gauss * C_i[:, :, None] * spt_res_z_cgs * (vel_res_cgs / wvl_res_cgs), axis=1)

    return I_cube.astype(cube_precision)

def combine_spectra(I_cubes, gofnt_dict, prime_line, simple_sum=False):
  """
  Combine all line spectra into one spectrum per pixel.

  If simple_sum is True:
    - Simply sum all lines (background and primary) together, interpolating to the prime_line wavelength grid.
    - background_spectrum and background_spectrum_line are both the sum of all background lines.

  If simple_sum is False:
    - Background lines: Connect the two highest peaks with a straight line.
      If the line goes below 0 anywhere in the wavelength range, connect the highest peak with y=0 at xmin or xmax.
    - Primary lines: sum their full profiles.
    - Final spectrum = baseline + sum(primary profiles).
  """
  nx, ny, nl = I_cubes[prime_line].shape

  combined = np.zeros_like(I_cubes[prime_line].value)
  background_spectrum = np.zeros_like(I_cubes[prime_line].value)
  background_spectrum_line = np.zeros_like(I_cubes[prime_line].value)

  wl_grid_prime = gofnt_dict[prime_line]['wl_grid'].to(u.AA).value

  background_lines = [line for line in I_cubes.keys() if gofnt_dict[line]['background']]

  if simple_sum:
    # Sum all lines (background and primary) together
    for line, I_cube in I_cubes.items():
      wl_grid = gofnt_dict[line]['wl_grid'].to(u.AA).value
      for i in range(nx):
        for j in range(ny):
          spectrum = I_cube[i, j, :].value
          interpolated = np.interp(wl_grid_prime, wl_grid, spectrum)
          combined[i, j, :] += interpolated
          if line in background_lines:
            background_spectrum[i, j, :] += interpolated
            background_spectrum_line[i, j, :] += interpolated
    return combined, background_spectrum, background_spectrum_line

  # Original method: fit a line to the background
  for line, I_cube in tqdm(I_cubes.items(), desc='Making background', unit='line', leave=False):
    if line in background_lines:
      wl_grid = gofnt_dict[line]['wl_grid'].to(u.AA).value
      for i in range(nx):
        for j in range(ny):
          spectrum = I_cube[i, j, :].value
          interpolated = np.interp(wl_grid_prime, wl_grid, spectrum)
          background_spectrum[i, j, :] += interpolated

  peak1_idx = np.argmax(background_spectrum, axis=2)
  point2_idx = np.where(peak1_idx < nl // 2, nl - 1, 0)

  for i in range(nx):
    for j in range(ny):
      x1, y1 = wl_grid_prime[peak1_idx[i,j]], background_spectrum[i,j,peak1_idx[i,j]]
      x2, y2 = wl_grid_prime[point2_idx[i,j]], 0
      slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
      intercept = y1 - slope * x1
      background_spectrum_line[i,j,:] = slope * wl_grid_prime + intercept

      if np.any(background_spectrum[i,j,:] > background_spectrum_line[i,j,:]):
        peak3_idx = np.argmax(background_spectrum[i,j,:] - background_spectrum_line[i,j,:])
        x1, y1 = wl_grid_prime[peak1_idx[i,j]], background_spectrum[i,j,peak1_idx[i,j]]
        x2, y2 = wl_grid_prime[peak3_idx], background_spectrum[i,j,peak3_idx]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        intercept = y1 - slope * x1
        background_spectrum_line[i,j,:] = slope * wl_grid_prime + intercept

  for line, I_cube in I_cubes.items():
    if line not in background_lines:
      wl_grid = gofnt_dict[line]['wl_grid'].to(u.AA).value
      for i in range(nx):
        for j in range(ny):
          spectrum = I_cube[i, j, :].value
          interpolated = np.interp(wl_grid_prime, wl_grid, spectrum)
          combined[i, j, :] += interpolated

  combined += background_spectrum_line

  return combined, background_spectrum, background_spectrum_line

def main():

    print("Warning: This will take lots of memory and so can crash if not enough is available.")
    print("   Regular outputs of memory used and total available will be printed.")

    ncpu = -1  # number of CPUs to use for parallel processing (-1 = all available)
    cube_precision = np.float64  # set to np.float64 for double precision, np.float32 for single precision
    if cube_precision == np.float32:
        print("WARNING: Using SINGLE PRECISION for all cubes to save memory. This may cause accuracy issues.")

    if os.path.exists('_C_cubes.npz') or os.path.exists('_I_cubes.npz') or os.path.exists('_G_cubes.npz'):
        print("WARNING: One or more of the files _C_cubes.npz, _I_cubes.npz, or _G_cubes.npz exist.")
        print("   This means that you are going to be loading at least one file, and that you should be sure you are using the same settings (resolution, lines, etc.) as when they were created.")
        print("   If you are not sure, delete the files and rerun the code.")

    vel_res     = 6 * u.km/u.s  # velocity bin width
    vel_lim     = 300 * u.km/u.s  # +-velocity range
    spt_res_z     = 0.064 * u.Mm  # spatial resolution in z
    spt_res_x, spt_res_y = 0.192 * u.Mm, 0.192 * u.Mm  # spatial resolution in x and y
    vel_grid = np.arange(-vel_lim.to(u.cm/u.s).value,
                       vel_lim.to(u.cm/u.s).value + vel_res.to(u.cm/u.s).value,
                       vel_res.to(u.cm/u.s).value) * (u.cm/u.s)  # velocity grid
    primary_lines = ['Fe12_195.1190', 'Fe12_195.1790']  # used for if there is a strong line (blend) which shouldn't be used as any old background line (e.g. 195.179 is a blend of 195.119)
    prime_line = 'Fe12_195.1190'  # primary line to use for the velocity grid

    ## Debugging options
    downsample = False  # number or False, used to speed up and reduce memory usage during testing by reducing the size of the cubes by sparse sampling.
    limit_to_lines = False  # e.g. ['Fe12_195.1190'] or False used to limit the lines to only those specified in the list to speed up and reduce memory usage.
    ## End of debugging options

    g_mu  = 1.29  # mean molecular weight solar (unitless as multiplied by u in kg) [doi:10.1051/0004-6361:20041507]
    Fe_amu = 55.845 * u.g / u.mol  # atomic weight of iron

    base = './data/atmosphere'
    files = dict(
        T   = 'temp/eosT.0270000',  # https://stacks.stanford.edu/file/dv883vb9686/eosT.0270000
        rho = 'rho/result_prim_0.0270000',  # https://stacks.stanford.edu/file/dv883vb9686/result_prim_0.0270000
        vx  = 'vx/result_prim_1.0270000',  # https://stacks.stanford.edu/file/dv883vb9686/result_prim_1.0270000
        vy  = 'vy/result_prim_3.0270000',  # https://stacks.stanford.edu/file/dv883vb9686/result_prim_3.0270000
        vz  = 'vz/result_prim_2.0270000'  # https://stacks.stanford.edu/file/dv883vb9686/result_prim_2.0270000
    )
    paths = {k:os.path.join(base,fn) for k,fn in files.items()}

    print(f"Loading static cubes ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
    temp_cube = load_cube(paths['T']  , downsample=downsample, unit=u.K).astype(cube_precision)
    rho_cube  = load_cube(paths['rho'], downsample=downsample, unit=u.g/u.cm**3).astype(cube_precision)
    vx_cube   = load_cube(paths['vx'] , downsample=downsample, unit=u.cm/u.s).astype(cube_precision)
    vy_cube   = load_cube(paths['vy'] , downsample=downsample, unit=u.cm/u.s).astype(cube_precision)
    vz_cube   = load_cube(paths['vz'] , downsample=downsample, unit=u.cm/u.s).astype(cube_precision)
    assert temp_cube.shape == rho_cube.shape == vx_cube.shape == vy_cube.shape == vz_cube.shape, "Cubes must have the same shape"
    del vx_cube, vy_cube  # not used as LOS is through z

    logT_cube = np.log10(temp_cube.value).astype(cube_precision)
    ne        = (rho_cube/(g_mu*const.u.cgs)).to(1/u.cm**3)
    logN_cube = np.log10(ne.value).astype(cube_precision)
    del ne, rho_cube, temp_cube

    print(f"Loading G(T,N) tables ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
    gofnt_dict = read_contribution_funcs('gofnt.sav')
    for line, info in gofnt_dict.items():
        info['g_tn'] = info['g_tn'].astype(cube_precision)
        info['logT'] = info['logT'].astype(cube_precision)
        info['logN'] = info['logN'].astype(cube_precision)
    if limit_to_lines:
        gofnt_dict = {k:v for k,v in gofnt_dict.items() if k in limit_to_lines}

    print(f"Calculating the wavelength grid from the velocity grid for each line ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
    for line, info in gofnt_dict.items():
        info['wl_grid'] = (vel_grid * info['wl0'] / const.c + info['wl0']).cgs

    print(f"Assigning background status to the lines as appropriate.")
    for line, info in gofnt_dict.items():
        if line in primary_lines:
            info['background'] = False
        else:
            info['background'] = True

    filename = '_G_cubes.npz'
    if os.path.exists(filename):
        print(f"Loading saved contribution function [erg cm3/s] per voxel from {filename} ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
        tmp = np.load(filename)
        G_cubes = {line: tmp[line] * (u.erg * u.cm**3 / u.s) for line in tmp.files}
        del tmp
    else: 
        print(f"Calculating the contribution function [erg cm3/s] per voxel ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
        G_cubes = {line: np.empty_like(logT_cube, dtype=cube_precision) for line in gofnt_dict.keys()}
        for line, info in tqdm(gofnt_dict.items(), desc='Lines', unit='line', leave=False):
            G_cubes[line] = interp2d_spline(info['g_tn'], info['logT'], info['logN'], logT_cube, logN_cube, cube_precision=cube_precision) * (u.erg * u.cm**3 / u.s)
        np.savez(filename, **{line: G_cubes[line].value for line in G_cubes.keys()})
    for line, info in gofnt_dict.items():
        del info['g_tn'], info['logT'], info['logN']

    filename = '_C_cubes.npz'
    if os.path.exists(filename):
        print(f"Loading saved integrated intensity [erg/s/cm3/sr] per voxel from {filename} ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
        tmp = np.load(filename)
        C_cubes = {line: tmp[line] * (u.erg / u.s / u.cm**3 / u.sr) for line in tmp.files}
        del tmp
    else:
        print(f"Calculating the integrated intensity [erg/s/cm3/sr] per voxel ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
        with np.errstate(over='ignore', invalid='ignore'):
            ne2 = (10**logN_cube * (1/u.cm**3))**2
            ne2[~np.isfinite(ne2)] = 0.0
            C_cubes = {line: np.zeros_like(G_cubes[line], dtype=cube_precision) for line in G_cubes.keys()}
            C_cubes = {line: (G_cubes[line] * (1/(4*np.pi*u.sr)) * ne2).astype(cube_precision) for line in tqdm(G_cubes.keys(), desc='Lines', unit='line', leave=False)}  # note, can't turn ne2 to float32 directly as values outside of single precision range (larger than 1e38)
        np.savez(filename, **{line: C_cubes[line].value for line in C_cubes.keys()})
        del ne2
    del G_cubes, logN_cube

    filename = '_I_cubes.npz'
    if os.path.exists(filename):
        print(f"Loading saved specific intensity [erg/s/cm2/sr/cm] for each voxel from {filename} ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
        tmp = np.load(filename)
        I_cubes = {line: tmp[line] * (u.erg / u.s / u.cm**2 / u.sr / u.cm) for line in tmp.files}
        del tmp
    else:
        print(f"Calculating the specific intensity [erg/s/cm2/sr/cm] for each LOS pixel ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
        probable_speed = np.sqrt(2 * const.k_B.cgs * (10**logT_cube * (u.K)) / (Fe_amu / const.N_A)).cgs  # v_p = sqrt(2kT/m), m=M/N_A : cm/s
        gauss_wdth = probable_speed / np.sqrt(2)  # broadening along LOS: g(v) = 1/sqrt(pi*sigma) * exp(-v^2/(wdth^2)), sigma = sqrt(kT/m), therefore sigma = v_p/sqrt(2) : cm/s
        gauss_peak = 1.0 / (np.sqrt(2 * np.pi * gauss_wdth**2))  # s / cm
        gauss_x_cgs = vel_grid.cgs.value  # velocity grid in cm/s
        spt_res_z_cgs = spt_res_z.to(u.cm).value  # save these to avoid repeated unit calculation
        vel_res_cgs = vel_res.to(u.cm/u.s).value
        I_cubes = {line: np.empty((C_cubes[line].shape[0], C_cubes[line].shape[1], gauss_x_cgs.size), dtype=cube_precision) for line in C_cubes.keys()}
        pbar = tqdm(C_cubes.items(), desc='Lines', unit='line', leave=False)
        for line, C_cube in pbar:
            pbar.set_postfix(mem=f"{psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB")
            wvl_res_cgs = gofnt_dict[line]['wl_grid'][1].cgs.value - gofnt_dict[line]['wl_grid'][0].cgs.value  # wavelength resolution in cm
            I_cubes[line] = calculate_specific_intensity(C_cube, gauss_peak, gauss_wdth, gauss_x_cgs, spt_res_z_cgs, vel_res_cgs, wvl_res_cgs, vz_cube, cube_precision=cube_precision, ncpu=ncpu) * (u.erg / u.s / u.cm**2 / u.sr / u.cm)
        np.savez(filename, **{line: I_cubes[line].value for line in I_cubes.keys()})
        del probable_speed, gauss_wdth, gauss_peak, gauss_x_cgs, spt_res_z_cgs, vel_res_cgs, wvl_res_cgs
    del C_cubes, logT_cube, vz_cube

    filename = '_I_cube.npz'
    if os.path.exists(filename):
        print(f"Loading saved combined spectra [erg/s/cm2/sr/cm] per LOS pixel from {filename} ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
        tmp = np.load(filename)
        I_cube = tmp['I_cube'] * (u.erg / u.s / u.cm**2 / u.sr / u.cm)
        del tmp
    else:
        print(f"Combining into one spectra per LOS pixel [erg/s/cm2/sr/cm] (primary + background) ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
        I_cube, background_spectrum, background_spectrum_line = combine_spectra(I_cubes, gofnt_dict, prime_line, simple_sum=True)
        I_cube *= (u.erg / u.s / u.cm**2 / u.sr / u.cm)
        np.savez(filename, I_cube=I_cube.cgs.value, wl_grid=gofnt_dict[prime_line]['wl_grid'].cgs.value, vel_grid=vel_grid.cgs.value, spt_res_x=spt_res_x.cgs.value, spt_res_y=spt_res_y.cgs.value, spt_res_z=spt_res_z.cgs.value, prime_line=prime_line, wl0=gofnt_dict[prime_line]['wl0'].cgs.value)

    I_cube_total = I_cube.sum(axis=2).value * (gofnt_dict[prime_line]['wl_grid'][1].cgs.value - gofnt_dict[prime_line]['wl_grid'][0].cgs.value)
    np.savez('_I_cube_total.npz', I_cube_total=I_cube_total, wl_grid=gofnt_dict[prime_line]['wl_grid'].cgs.value, vel_grid=vel_grid.cgs.value, spt_res_x=spt_res_x.cgs.value, spt_res_y=spt_res_y.cgs.value, spt_res_z=spt_res_z.cgs.value, prime_line=prime_line, wl0=gofnt_dict[prime_line]['wl0'].cgs.value)

    filepath = 'session.pkl'
    dill.dump_session(filepath)

    globals().update(locals());raise ValueError("Kicking back to ipython")

    wl_resolution = gofnt_dict[prime_line]['wl_grid'][1].cgs.value - gofnt_dict[prime_line]['wl_grid'][0].cgs.value
    fig, ax = plt.subplots()
    img = ax.imshow(np.log10(I_cube.sum(axis=2).T.value * wl_resolution), aspect='equal', cmap='inferno', origin='lower')
    plt.colorbar(img, ax=ax, label='Log(Intensity)')
    plt.savefig('synthesised_spectra.png', dpi=300)
    plt.close(fig)


    # plot the total summed spectra from I_cube for the median intensity pixel
    median_x, median_y = np.unravel_index(np.argmax(I_cube.sum(axis=2).value), I_cube.shape[:2])
    print(f"Median pixel at ({median_x}, {median_y}) with intensity {I_cube[median_x, median_y, :].sum().value:.2e} erg/s/cm2/sr/cm")
    fig, ax = plt.subplots()
    wl_grid = gofnt_dict[prime_line]['wl_grid'].to(u.AA).value
    spec_tot = I_cube[median_x, median_y, :].value
    ax.plot(wl_grid, spec_tot, label='Total Spectrum', color='black', lw=2)
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Intensity (erg/s/cm2/sr/cm)')
    ax.set_title(f'Spectrum at pixel ({median_x}, {median_y})')
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig('synthesised_spectrum_median_pixel.png', dpi=300)
    plt.close()






    fig, ax = plt.subplots()
    img = ax.imshow(np.log10(I_cube.sum(axis=2).T.value), aspect='equal', cmap='inferno', origin='lower')
    plt.colorbar(img, ax=ax, label='Log(Intensity)')

    def onclick(event,
          ax=ax,
          total_cube=I_cube,
          back_cube=background_spectrum,
          cubes=I_cubes,
          gofnt=gofnt_dict):
        if event.inaxes is not ax:
          return
        if event.xdata is None or event.ydata is None:
          return
        x_pix = int(round(event.xdata))
        y_pix = int(round(event.ydata))
        nx, ny, _ = total_cube.shape
        if not (0 <= x_pix < nx and 0 <= y_pix < ny):
          return

        # Prepare the two-panel figure
        fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

        # Plot each line's spectrum
        for name, cube_line in cubes.items():
          wl_grid = gofnt[name]['wl_grid'].to(u.AA).value
          spec = cube_line[x_pix, y_pix, :]
          y_line = spec.value if hasattr(spec, 'value') else spec
          ax1.plot(wl_grid, y_line, label=name, alpha=0.7)
          ax2.plot(wl_grid, y_line, label=name, alpha=0.7)

        # Plot total and background
        wl_tot = gofnt[prime_line]['wl_grid'].to(u.AA).value
        spec_tot = total_cube[x_pix, y_pix, :]
        spec_back = back_cube[x_pix, y_pix, :]
        # spec_back_line = back_line_cube[x_pix, y_pix, :]

        y_tot = spec_tot.value if hasattr(spec_tot, 'value') else spec_tot

        ax1.plot(wl_tot, y_tot, 'k-', lw=2, label='Total')
        ax2.plot(wl_tot, y_tot, 'k-', lw=2, label='Total')

        ax1.plot(wl_tot, spec_back, 'r--', lw=2, label='Background')
        ax2.plot(wl_tot, spec_back, 'r--', lw=2, label='Background')

        ax1.set_ylabel('Intensity')
        ax1.set_title(f'Spectrum at pixel ({x_pix}, {y_pix}) — linear scale')
        ax1.legend(loc='best', fontsize='small')

        ax2.set_yscale('log')
        ax2.set_ylim(bottom=1e-1)
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Intensity')
        ax2.set_title(f'Spectrum at pixel ({x_pix}, {y_pix}) — log scale')
        ax2.legend(loc='best', fontsize='small')

        wl0_A = gofnt[prime_line]['wl0'].to(u.AA).value
        c_km_s = const.c.to(u.km / u.s).value

        def wl_to_vel(wl):
            return (wl - wl0_A) / wl0_A * c_km_s

        def vel_to_wl(v):
            return (v / c_km_s) * wl0_A + wl0_A

        secax = ax1.secondary_xaxis('top', functions=(wl_to_vel, vel_to_wl))
        secax.set_xlabel('Velocity (km/s)')

        plt.tight_layout()
        plt.show()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    globals().update(locals())
if __name__ == '__main__':
    main()