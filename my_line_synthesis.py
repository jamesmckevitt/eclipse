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

# Physical constants
amu   = const.u.cgs                         # atomic mass unit [g]
kB    = const.k_B.cgs                      # Boltzmann constant [erg/K]
c_cgs = const.c.cgs                         # speed of light [cm/s]
g_mu  = 1.29                                # mean molecular weight (amu)


def load_cube(path, shape=(512,768,256), unit=None):
    """
    Load a binary cube and optionally attach an astropy unit.
    Returns an ndarray or Quantity of shape (nx, ny, nz).
    """
    data = np.fromfile(path, dtype=np.float32).reshape(shape, order='F')
    data = np.transpose(data, (0, 2, 1))
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


def los_unit_vector(disk_center_arcsec):
    """
    From disk-centre offsets (x_off,y_off) in arcsec, 
    compute the LOS unit vector [x,y,mu].
    """
    arcsec_to_rad = (1*u.arcsec).to(u.rad).value
    x_off, y_off = disk_center_arcsec
    x_rad = x_off * arcsec_to_rad
    y_rad = y_off * arcsec_to_rad
    sin2 = x_rad**2 + y_rad**2
    mu   = np.sqrt(max(0.0, 1 - sin2))
    return np.array([x_rad, y_rad, mu])


def build_lookup_indices(logT_cube, logN_cube, logT, logN):
    """
    Build integer lookup arrays iT,iN for sampling the 
    2D G(T,N) tables.
    """
    nT, nN = len(logT), len(logN)
    Tmin, Tmax = logT.min(), logT.max()
    Nmin, Nmax = logN.min(), logN.max()
    scaleT = (nT-1)/(Tmax-Tmin)
    scaleN = (nN-1)/(Nmax-Nmin)
    iT = np.clip(((logT_cube-Tmin)*scaleT).round().astype(int), 0, nT-1)
    iN = np.clip(((logN_cube-Nmin)*scaleN).round().astype(int), 0, nN-1)
    return iT, iN


def integrate_along_los(j_vol, los_vec, spatial_res):
    """
    Integrate j_vol (nz,nx,ny) along an arbitrary LOS 
    by trilinear interpolation → returns 2D image (ny,nx).
    """
    dz = (spatial_res / los_vec[2]).to(u.cm).value
    nx, ny, nz = j_vol.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    img = np.zeros((ny, nx)) * j_vol.unit
    for k in range(nz):
        xp = X + (los_vec[0]/los_vec[2]) * k
        yp = Y + (los_vec[1]/los_vec[2]) * k
        coords = [ xp, yp, np.full_like(xp, k) ]
        vals = map_coordinates(j_vol.value, coords, order=1, mode='constant', cval=0.0)
        img += vals * dz * j_vol.unit
    return img


def compute_spectral_cube_per_wavelength(j_vol, v_los, temp_cube, atom, vel_grid, vel_res, wvl0):
    """
    Build a spectral cube per cm of wavelength, one velocity bin at a time.
    j_vol      : Quantity (nx,ny,nz) emissivity [erg/cm3/s/sr]
    v_los      : Quantity (nx,ny,nz) LOS-projected velocity [cm/s]
    temp_cube  : Quantity (nx,ny,nz) temperature [K]
    atom       : int                 atomic mass number
    vel_grid   : Quantity (n_vel,)   velocities [cm/s]
    vel_res    : Quantity            velocity bin width [cm/s]
    wvl0       : Quantity            rest wavelength [cm]
    
    returns spec_lambda : Quantity (ny,nx,n_vel)      [erg/cm2/s/sr/cm]
    """

    print("Performing precomputations...")

    # precompute thermal width per voxel
    sigma = np.sqrt(2 * const.k_B.cgs * temp_cube.cgs / (atom * amu.cgs))

    # calculate the wavelength resolution in cm
    dlam = (vel_res * wvl0 / const.c).to(u.cm).value

    # calculate the wavelength grid in cm
    wvl_grid = vel_grid * wvl0 / const.c + wvl0

    nx, ny, nz = j_vol.shape
    n_vel      = vel_grid.size
    spec = np.zeros((nx, ny, n_vel))

    ## Serial (less memory, faster)
    for iv in tqdm(range(n_vel), desc='Velocity bins', unit='bin'):
        v0 = vel_grid[iv]
        # 3D Gaussian phi(v0; each voxel)
        phi  = np.exp(-((v0 - v_los)**2) / (sigma**2)) / (sigma * np.sqrt(np.pi))
        # integrate along z 2D (nx,ny)
        spec_vel = (j_vol * phi).sum(axis=2) * vel_res
        # store as (x,y,iv)
        spec[:, :, iv] = (spec_vel / dlam)
        # free this bin’s scratch space
        del phi, spec_vel

    ## Parallel (more memory, slower)
    # # define a helper that computes one velocity bin
    # def _process_bin(iv):
    #   v0 = vel_grid[iv]
    #   # line profile at this voxel
    #   phi = np.exp(-((v0 - v_los)**2) / (sigma**2)) / (sigma * np.sqrt(np.pi))
    #   # integrate emissivity × profile along z
    #   spec_vel = (j_vol * phi).sum(axis=2) * vel_res
    #   # convert to per‐wavelength and return raw array
    #   return (spec_vel / dlam).value
    # # run in parallel over velocity bins
    # spec_list = Parallel(n_jobs=-1)(
    #   delayed(_process_bin)(iv) for iv in range(n_vel)
    # )
    # # stack into final cube (nx, ny, n_vel) and reattach unit
    # spec = np.stack(spec_list, axis=2)

    # free the large arrays
    del j_vol, v_los, temp_cube, sigma
    gc.collect()  # force garbage collection

    return spec * u.erg/(u.cm**2*u.s*u.sr*u.cm), wvl_grid


def main():
    # Parameters
    raster_centre = (0.0, 0.0)        # arcsec offsets (x,y)
    vel_res     = 10 * u.km/u.s
    vel_lim     = 20 * u.km/u.s

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

    # Load static cubes
    print(f"Loading static cubes ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
    temp = load_cube(paths['T'],   unit=u.K)
    rho  = load_cube(paths['rho'], unit=u.g/u.cm**3)
    vx   = load_cube(paths['vx'],  unit=u.cm/u.s)
    vy   = load_cube(paths['vy'],  unit=u.cm/u.s)
    vz   = load_cube(paths['vz'],  unit=u.cm/u.s)

    # Precompute lookup indices
    print(f"Precomputing lookup indices...")

    logT_cube = np.log10(temp.value)
    ne        = (rho/(g_mu*amu)).to(1/u.cm**3)
    logN_cube = np.log10(ne.value)

    print(f"Loading G(T,N) tables ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
    goft = read_contribution_funcs('G_of_T.sav')
    ### TEMP
    # remove all lines which aren't "Fe12_195.1190"
    goft = {k:v for k,v in goft.items() if k == 'Fe12_195.1190'}
    ### TEMP
    sample = next(iter(goft.values()))
    print(f"Building lookup indices ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
    iT, iN = build_lookup_indices(logT_cube, logN_cube, sample['logT'], sample['logN'])

    # LOS vector & velocity grid
    print(f"Building LOS vector and velocity grid ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
    los_vec = los_unit_vector(raster_centre)
    v_los_cube = (vx*los_vec[0] + vy*los_vec[1] + vz*los_vec[2]).to(u.cm/u.s)
    vgrid = (np.arange(-vel_lim.to(u.cm/u.s).value,
                       vel_lim.to(u.cm/u.s).value + vel_res.to(u.cm/u.s).value,
                       vel_res.to(u.cm/u.s).value) * (u.cm/u.s))

    spectra = {}
    for line, info in tqdm(goft.items(), desc='Lines', unit='line'):
        
        print("Calculating emissivity...")
        g_tn = info['g_tn'].value  # g(T,N) [erg cm3/s]
        j_vol = (g_tn[iT,iN] * ne**2 / (4*np.pi)) * (u.erg/(u.cm**3*u.s*u.sr))  # emissivity [erg/cm3/s/sr]

        # build spectral cube
        print(f"Building spectral cube for {line} ({psutil.virtual_memory().used/1e9:.2f}/{psutil.virtual_memory().total/1e9:.2f} GB)...")
        spec, wvl_grid = compute_spectral_cube_per_wavelength(
            j_vol,             # volumetric emissivity
            v_los_cube,        # full LOS-projected velocity
            temp,              # temperature cube
            info['atom'],      # atomic mass
            vgrid,             # 1D vel grid
            vel_res.to(u.cm/u.s),
            info['wl0']        # rest wavelength
        )

        # add the wavelength grid to the spectra dict
        spectra[line] = {
            'spec' : spec,
            'wvl'  : wvl_grid
        }

        # free this line's large arrays
        del j_vol
        gc.collect()  # force garbage collection


    first = next(iter(spectra.keys()))
    spec_first = spectra[first]['spec']

    # integrate over the velocity (wavelength) axis
    integrated_map = spec_first.sum(axis=2).T.value  # transpose so [row, col] = [y, x]
    # plot the map
    fig_map, ax_map = plt.subplots()
    im = ax_map.imshow(integrated_map, origin='lower', cmap='inferno', aspect='equal')  # set aspect to 'equal'
    ax_map.set_title('Integrated Intensity Map')
    ax_map.set_xlabel('X pixel')
    ax_map.set_ylabel('Y pixel')
    plt.colorbar(im, ax=ax_map, label=f'Integrated Intensity [erg/cm2/s/sr/cm]')

    def onclick(event):
        if event.inaxes == ax_map and event.xdata is not None:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            spec_pixel = spec_first[x, y, :]

            _fig, ax_spec = plt.subplots()
            ax_spec.plot(vgrid.to(u.km/u.s).value, spec_pixel.value, '-o')
            ax_spec.set_title(f'Spectrum at pixel (x={x}, y={y})')
            ax_spec.set_xlabel('v (km/s)')
            ax_spec.set_ylabel('I (erg/cm2/s/sr/A)')
            plt.show()

    # connect and display
    fig_map.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    globals().update(locals())
if __name__ == '__main__':
    main()
