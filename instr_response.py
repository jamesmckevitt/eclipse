## TODO: compare calculated velocities to real velocities (to show systematic errors): 
#      - fit high-res emission line (remake synthetic cubes at v high wavelength res)
## TODO: include the QE as a function of wavelength, rather than a constant factor when doing multiple wavelengths
## TODO: show the tgt fit of the cube before the instrument response in the summary plots
## TODO: Tidy up summary plots code and make individual plots bigger
## TODO: make it possible to plot all summary plots after all monte carlo iterations have been done

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
from scipy.signal import convolve2d
from scipy.io import readsav
from scipy.interpolate import interp1d
from tqdm import tqdm
import astropy.constants as const
import astropy.units as u
from joblib import Parallel, delayed
import warnings
from scipy.optimize import OptimizeWarning
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler

# Detector
swc_vis_qe = 1                                # Quantum efficiency at visible wavelengths
swc_euv_qe = 0.76                             # Quantum efficiency at target wavelength
swc_e_per_euv_ph = 18.0 * (u.electron/u.ph)   # Electrons produced per EUV photon
swc_e_per_vis_ph = 2.0 * (u.electron/u.ph)    # Electrons produced per VIS photon
swc_read_noise_e = 10.0 * (u.electron/u.pix)  # CCD readout noise
swc_dn_per_e = 19.0 * (u.DN/u.electron)       # Conversion factor
swc_dark_current = 1.0 *(u.electron/u.pix)    # Dark current
swc_pix_size = (13.5*u.um).cgs/u.pix          # Pixel size
swc_wvl_res = (16.9*u.mAA).cgs/u.pix          # Spectral resolution per pixel
# swc_spt_res = (0.159 * (u.arcsec)).to(u.rad)  # Spatial resolution
swc_spt_res = (4. * (u.arcsec)).to(u.rad)  # Spatial resolution
swc_spt_res = const.au.cgs * np.tan(swc_spt_res/2) * 2  # Spatial resolution at the Sun

# Telescope
tel_collecting_area = (.5*np.pi*(0.28*u.m/2)**2).cgs  # Entrance pupil diameter (aperture stop) is .28 m; *half* pi r^2 as only half of beam for SW
tel_pm_efficiency = 0.161                             # Primary mirror efficiency
tel_ega_efficiency = 0.0623                           # Grating efficiency
tel_filter_transmission = 0.507                       # Filter transmission efficiency
tel_focus_psf_filename = "data/swc/psf_euvst_v20230909_195119_focus.txt"  # PSF filename
tel_mesh_psf_filename = "data/swc/psf_euvst_v20230909_derived_195119_mesh.txt"  # Filter PSF filename
tel_focus_psf_input_res = 0.5 * u.um / u.pix
tel_mesh_psf_input_res = 6.12e-4 * u.mm / u.pix

# Synthetic data
dat_filename = "_I_cube.npz"

# Simulation parameters
sim_n = 5                                # Number of Monte Carlo iterations per exposure time
sim_t = [2] * u.s               # Exposure time (s) 90s for quiet Sun, 40s for active region, 5s for flare (1s before x-band loss on EIS).
sim_stray_light_s = 1 * (u.ph/u.s/u.pix)  # Visible stray light photon/s/pixel
sim_ncpu = -1                              # Number of CPU cores to use for parallel processing (-1 for all available cores)

def wvl_to_vel(wl, sim_wvl0):return (wl - sim_wvl0) / sim_wvl0 * const.c.cgs
def vel_to_wvl(v, sim_wvl0):return (v / const.c.cgs) * sim_wvl0 + sim_wvl0

def gaussian(wave, peak, cent, sigma, background):
    return peak * np.exp(-0.5 * ((wave - cent) / sigma) ** 2) + background

def fit_spectra(data_cube, wave_axis):
    """
    Fit a Gaussian to each spectrum in the data cube.
    """
    # Get the dimensions of the data cube: n_scan (number of scans),
    # n_slit (number of spectra per scan) and n_spec (spectral points per spectrum)
    n_scan, n_slit, n_spec = data_cube.shape
    
    # Define an inner function to process a single scan direction.
    def process_scan(i):
        # Create an array to store the fit parameters for all spectra in this scan
        row_fit_params = np.zeros((n_slit, 4))
        # Loop over each spectrum (slit) in the current scan
        for j in range(n_slit):
            # Fit a Gaussian to the spectrum at data_cube[i, j, :]
            fit_result = fit_gaussian_profile(wave_axis, data_cube[i, j, :])
            row_fit_params[j, :] = fit_result['params']
        return row_fit_params
    
    # Use joblib.Parallel to execute the outer loop concurrently.
    # n_jobs=-1 utilises all available cores.
    results = Parallel(n_jobs=sim_ncpu)(
        delayed(process_scan)(i) for i in tqdm(range(n_scan), desc=f"Fitting spectra using {sim_ncpu} parallel cores", unit="scan", leave=False)
    )
    
    # Combine the results for each scan back into a single numpy array.
    fit_params = np.array(results)
    return fit_params

def guess_initial_params(wave, profile):
    """
    Estimate initial guess parameters for Gaussian fitting.
    """
    background = np.min(profile)
    # Remove the background for the peak and centre estimation
    profile_corr = profile - background
    profile_corr[profile_corr < 0] = 0
    peak = np.max(profile_corr)
    # Compute centre as the weighted average
    if np.sum(profile_corr) > 0:
        cent = np.sum(wave * profile_corr) / np.sum(profile_corr)
    else:
        cent = wave[len(wave) // 2]
    # Rough estimate for sigma using the total area under the curve
    sigma = np.abs(wave[1] - wave[0]) * np.sum(profile_corr) / (peak * np.sqrt(2 * np.pi))
    return [peak, cent, sigma, background]

def fit_gaussian_profile(wave, profile):
    """
    Fit a Gaussian function to a spectral profile.
    """
    valid = profile >= 0  # Only fit for non-negative values
    p0 = guess_initial_params(wave[valid], profile[valid])
    try:
        # Estimate uncertainties using the square root of the profile values
        noise = np.sqrt(np.maximum(profile[valid], 1))
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", OptimizeWarning)
          popt, pcov = curve_fit(gaussian, wave[valid], profile[valid], p0=p0, sigma=noise)
        perr = np.sqrt(np.diag(pcov))
        model = gaussian(wave, *popt)
        converged = True
    except Exception as err:
        popt = [0, 0, 0, 0]
        perr = [0, 0, 0, 0]
        model = np.zeros_like(wave)
        converged = False

    return {'params': popt, 'errors': perr, 'model': model, 'converged': converged}

def add_poisson_noise(data_cube):
    unit = data_cube.unit
    return np.random.poisson(data_cube.value) * unit

def load_psf(filename, skiprows=0):
    psf = np.loadtxt(filename, skiprows=skiprows, encoding='UTF-16 LE')
    return psf

def resample_psf(psf, old_spacing, new_spacing):
    scale_factor = old_spacing.to(u.cm/u.pix).value / new_spacing.to(u.cm/u.pix).value
    psf_resampled = zoom(psf, scale_factor, order=1)
    return psf_resampled

def combine_normalise_psf(psf1, psf2, crop_frac=0.99, max_size=None):
    """
    Combine two PSF arrays by convolving them, then crop and normalise.
    """
    psf = convolve2d(psf1, psf2, mode='same', boundary='fill')
    nr, nc = psf.shape

    if max_size is not None:
        if max_size % 2 == 0 or max_size < 1:
            raise ValueError("max_size must be a positive odd integer")
        # Find peak pixel
        idx = np.argmax(psf)
        r0c, c0c = np.unravel_index(idx, psf.shape)
        half = max_size // 2
        # Compute bounds
        r0 = max(0, r0c - half)
        c0 = max(0, c0c - half)
        r1 = min(nr, r0 + max_size)
        c1 = min(nc, c0 + max_size)
        # Adjust if at edge
        if (r1 - r0) < max_size:
            r0 = max(0, r1 - max_size)
        if (c1 - c0) < max_size:
            c0 = max(0, c1 - max_size)
        cropped = psf[r0:r0 + max_size, c0:c0 + max_size]
    else:
        # Intensity‑based cropping
        total = psf.sum()
        flat = psf.ravel()
        idx_desc = np.argsort(flat)[::-1]
        cumsum = np.cumsum(flat[idx_desc])
        th_idx = np.searchsorted(cumsum, total * crop_frac)
        th_val = flat[idx_desc[th_idx]]
        mask = psf >= th_val
        rows, cols = np.where(mask)
        if rows.size == 0:
            cropped = psf.copy()
        else:
            r0, r1 = rows.min(), rows.max()
            c0, c1 = cols.min(), cols.max()
            h, w = r1 - r0 + 1, c1 - c0 + 1
            side = max(h, w)
            center_r, center_c = (r0 + r1) // 2, (c0 + c1) // 2
            half = side // 2
            r0s = max(0, center_r - half)
            c0s = max(0, center_c - half)
            r1s, c1s = r0s + side, c0s + side
            if r1s > nr:
                r1s, r0s = nr, nr - side
            if c1s > nc:
                c1s, c0s = nc, nc - side
            cropped = psf[r0s:r1s, c0s:c1s]

    return cropped/cropped.sum()

def convolve_cube_with_psf(cube, psf):
    """
    Convolve each (spatial, spectral) plane in the cube with the instrument PSF.
    """
    n_scan, n_slit, n_spec = cube.shape
    conv_cube = np.zeros_like(cube)
    for i in range(n_scan):
        plane = cube[i, :, :]
        plane_unit = plane.unit
        plane_conv = convolve2d(plane.value, psf, mode='same', boundary='fill') * plane_unit
        conv_cube[i, :, :] = plane_conv
    return conv_cube

def read_out_photons(counts_cube):
    """
    Read out the photons from the detector, including CCD readout noise, dark current, and conversion to electrons.
    """
    electrons = counts_cube * swc_euv_qe * swc_e_per_euv_ph
    electrons += swc_dark_current
    noise = np.random.normal(loc=0.0, scale=swc_read_noise_e.value, size=electrons.shape) * (u.electron / u.pix)
    electrons_noisy = electrons + noise
    electrons_noisy[electrons_noisy < 0] = 0.0
    return electrons_noisy

def add_vis_stray_light(electrons_cube, sim_t_i):
    """
    Add visible stray light, in electrons, to the cube.
    """
    out_cube = electrons_cube.copy()
    sim_stray_light = sim_stray_light_s * sim_t_i
    sim_stray_light_unit = sim_stray_light.unit
    stray_light_vis_photons = np.random.poisson(sim_stray_light.value, size=electrons_cube.shape) * sim_stray_light_unit
    stray_light_electrons = stray_light_vis_photons * swc_e_per_vis_ph * swc_vis_qe
    out_cube += stray_light_electrons
    out_cube[out_cube < 0] = 0.0
    return out_cube

def convert_counts_to_dn(electrons_cube):
    """
    Convert the cube of electrons to digital numbers (DN).
    """
    return electrons_cube * swc_dn_per_e

def load_synthetic_atmosphere(filepath):
    """
    Load synthesised atmosphere from synthesise_spectra.py.
    """
    tmp = np.load(filepath)  # all in cgs units
    tmp = {item: tmp[item] for item in tmp.files}
    assert tmp['spt_res_x'] == tmp['spt_res_y'], "Spatial resolution in x and y are not equal"
    sim_spt_res = tmp['spt_res_x'] * u.cm
    sim_wvl_grid = tmp['wl_grid'] * u.cm
    sim_wvl_res = (tmp['wl_grid'][1] - tmp['wl_grid'][0]) * (u.cm/u.pix)
    sim_vel_grid = tmp['vel_grid'] * (u.cm / u.s)
    sim_vel_res = (tmp['vel_grid'][1] - tmp['vel_grid'][0]) * (u.cm / u.s)
    atm_cube = tmp['I_cube'] * (u.erg / (u.s * u.cm**2 * u.sr * u.cm))
    # sim_wvl0 = tmp['wl0'] * u.cm
    sim_wvl0 = (195.119 * u.AA).cgs
    return atm_cube, sim_wvl_grid, sim_wvl_res, sim_vel_grid, sim_vel_res, sim_spt_res, sim_wvl0

def rebin_atmosphere(atm_cube, sim_wvl_grid, sim_wvl_res, sim_vel_grid, sim_spt_res, swc_wvl_res, swc_spt_res, sim_wvl0):
    """
    Resample the synthetic atmosphere to the swc spatial and spectral resolution.
    """
    swc_wvl_grid = np.arange(sim_wvl_grid[0].cgs.value, sim_wvl_grid[-1].cgs.value + swc_wvl_res.to(u.cm/u.pix).value, swc_wvl_res.to(u.cm/u.pix).value) * u.cm

    resampled_atm_cube = np.zeros((atm_cube.shape[0], atm_cube.shape[1], len(swc_wvl_grid)))
    fluxc_resample = FluxConservingResampler(extrapolation_treatment='nan_fill')
    def _resample_scan(i):
        row = np.zeros((atm_cube.shape[1], len(swc_wvl_grid)))
        for j in range(atm_cube.shape[1]):
            spec = Spectrum1D(flux=atm_cube[i, j, :],
                      spectral_axis=sim_wvl_grid)
            row[j, :] = fluxc_resample(spec, swc_wvl_grid).flux.value
        return row
    results = Parallel(n_jobs=sim_ncpu)(
        delayed(_resample_scan)(i)
        for i in tqdm(range(atm_cube.shape[0]), desc=f"Resampling spectra using {sim_ncpu} parallel cores", unit="scan", leave=False))
    resampled_atm_cube = np.stack(results, axis=0)
    nans = np.isnan(resampled_atm_cube[0,0,:])
    assert np.all(nans == np.isnan(resampled_atm_cube)), "Not all pixels have the same number of nans in the last axis. Something weird happened with detecting overlapping bins."
    resampled_atm_cube = resampled_atm_cube[:, :, ~nans]
    swc_wvl_grid = swc_wvl_grid[~nans]
    scale_factor = sim_spt_res / swc_spt_res
    atm_cube = zoom(resampled_atm_cube, (scale_factor, scale_factor, 1), order=1)
    resampled_atm_cube = np.clip(resampled_atm_cube, 0, None) * (u.erg / (u.s * u.cm**2 * u.sr * u.cm))
    swc_vel_grid = wvl_to_vel(swc_wvl_grid, sim_wvl0)
    return resampled_atm_cube, swc_wvl_grid, swc_vel_grid

def convert_to_photons(data_cube, wvl0):
    """
    Convert the cube of intensity to photons.
    """
    out_cube = data_cube.copy()
    out_cube = out_cube.to(u.J/u.cm**2/u.sr/u.cm)
    photon_energy = (const.h.to('J.s') * const.c.cgs / wvl0) * 1/u.photon
    out_cube /= photon_energy
    out_cube = np.round(out_cube)  # Round to nearest integer as can only measure whole photons
    return out_cube

def add_partial_effective_area(data_cube):
    """
    Add the partial effective area of the telescope to the data cube.
    """
    out_cube = data_cube.copy()
    tel_partial_ea = tel_collecting_area.cgs * tel_pm_efficiency * tel_ega_efficiency * tel_filter_transmission  # Telescope partial effective area (excluding CCD QE)
    out_cube *= tel_partial_ea
    return out_cube

def get_per_pixel(data_cube):
    """
    Convert the cube of photons to per pixel.
    """
    out_cube = data_cube.copy()
    solid_angle = swc_spt_res.cgs** 2 / const.au.cgs** 2 * u.sr # steradians ( / CCD row )
    out_cube *= solid_angle  # photon/cm ( / CCD row )
    out_cube *= swc_wvl_res.to(u.cm/u.pix)  # Spectral sampling per pixel
    return out_cube

def calculate_velocity(fit_params, dat_rest_wave):
    """
    Calculate the Doppler velocity from the fitted Gaussian parameters.
    """
    cent = fit_params[:, :, 1]
    sigma = fit_params[:, :, 2]
    velocity = (cent - dat_rest_wave.cgs.value) / dat_rest_wave.cgs.value * const.c.cgs.value
    return velocity

def monte_carlo(atm_cube_i, wvl_grid, swc_vel_grid, psf_combined, sim_t_i, sim_wvl0):
    """
    Perform a Monte Carlo simulation of the instrument response.
    """
    fit_cubes = []
    for it in tqdm(range(sim_n), desc="Monte Carlo iterations", unit="iteration", leave=False):

        poisson_cube = add_poisson_noise(atm_cube_i)                ; assert poisson_cube.unit == (u.erg/u.cm**2/u.sr/u.cm)
        photon_cube = convert_to_photons(poisson_cube, sim_wvl0)    ; assert photon_cube.unit == (u.photon/u.cm**2/u.sr/u.cm)
        area_cube = add_partial_effective_area(photon_cube)         ; assert area_cube.unit == (u.photon/u.sr/u.cm)
        pixel_cube = get_per_pixel(area_cube)                       ; assert pixel_cube.unit == (u.photon/u.pix)
        psf_cube = convolve_cube_with_psf(pixel_cube, psf_combined) ; assert psf_cube.unit == (u.photon/u.pix)
        electron_cube = read_out_photons(psf_cube)                  ; assert electron_cube.unit == (u.electron/u.pix)
        sl_cube = add_vis_stray_light(electron_cube, sim_t_i)       ; assert sl_cube.unit == (u.electron/u.pix)
        dn_cube = convert_counts_to_dn(sl_cube)                     ; assert dn_cube.unit == (u.DN/u.pix)

        fit_cube = fit_spectra(dn_cube.value, wvl_grid.value)
        fit_cubes.append(fit_cube)

        if it == 0:
            poisson_cube_0 = poisson_cube
            photon_cube_0 = photon_cube
            area_cube_0 = area_cube
            pixel_cube_0 = pixel_cube
            psf_cube_0 = psf_cube
            electron_cube_0 = electron_cube
            sl_cube_0 = sl_cube
            dn_cube_0 = dn_cube
            fit_cube_0 = fit_cube

    return {
        'fit_cubes': fit_cubes,
        'poisson_cube_0': poisson_cube_0,
        'photon_cube_0': photon_cube_0,
        'area_cube_0': area_cube_0,
        'pixel_cube_0': pixel_cube_0,
        'psf_cube_0': psf_cube_0,
        'electron_cube_0': electron_cube_0,
        'sl_cube_0': sl_cube_0,
        'dn_cube_0': dn_cube_0,
        'fit_cube_0': fit_cube_0
    }

def analysis(fit_cubes, tgt_fit_cube, sim_wvl0, units):
    """
    Analyse the velocities.
    """
    vel_tgt = calculate_velocity(tgt_fit_cube, sim_wvl0)
    vel_vals = np.array([calculate_velocity(fit_cube, sim_wvl0) for fit_cube in fit_cubes])
    vel_mean = np.mean(vel_vals, axis=0)
    vel_std = np.std(vel_vals, axis=0)
    vel_err = vel_tgt - vel_mean

    return {
        'velocity_vals': vel_vals*(u.cm/u.s).to(units),
        'velocity_mean': vel_mean*(u.cm/u.s).to(units),
        'velocity_std': vel_std*(u.cm/u.s).to(units),
        'velocity_err': vel_err*(u.cm/u.s).to(units),
    }

def plot_atmosphere(swc_atm_cube, swc_wvl_grid, swc_vel_grid, sim_atm_cube, sim_wvl_grid, sim_wvl_res, sim_vel_grid, sim_wvl0, block=True):

    fig, ax = plt.subplots(2, 1)
    img_swc = ax[0].imshow(np.log10(swc_atm_cube.sum(axis=2).T.value*swc_wvl_res.to(u.cm/u.pix).value), aspect='equal', cmap='inferno', origin='lower')
    ax[0].set_title('sw rebinned')
    ax[0].set_xlabel('Pixel X')
    ax[0].set_ylabel('Pixel Y')
    plt.colorbar(img_swc, ax=ax[0], label='Log(Intensity [erg/s/cm2/sr])')

    img_sim = ax[1].imshow(np.log10(sim_atm_cube.sum(axis=2).T.value*sim_wvl_res.to(u.cm/u.pix).value), aspect='equal', cmap='inferno', origin='lower')
    ax[1].set_title('simulated')
    ax[1].set_xlabel('Pixel X')
    ax[1].set_ylabel('Pixel Y')
    plt.colorbar(img_sim, ax=ax[1], label='Log(Intensity [erg/s/cm2/sr])')

    marker, = ax[0].plot([], [], 'ro', markersize=5)  # Red marker for clicked point on SWC plot
    marker2, = ax[1].plot([], [], 'ro', markersize=5)  # Red marker for clicked point on SIM plot

    def onclick(event):
        if event.inaxes is ax[0]:  # Clicked on SWC plot
            swc_x_pix, swc_y_pix = int(round(event.xdata)), int(round(event.ydata))
            sim_x_pix = int(np.round(event.xdata * sim_atm_cube.shape[0] / swc_atm_cube.shape[0]))
            sim_y_pix = int(np.round(event.ydata * sim_atm_cube.shape[1] / swc_atm_cube.shape[1]))
        elif event.inaxes is ax[1]:  # Clicked on SIM plot
            sim_x_pix, sim_y_pix = int(round(event.xdata)), int(round(event.ydata))
            swc_x_pix = int(np.round(event.xdata * swc_atm_cube.shape[0] / sim_atm_cube.shape[0]))
            swc_y_pix = int(np.round(event.ydata * swc_atm_cube.shape[1] / sim_atm_cube.shape[1]))
        else:
            return

        marker.set_data([swc_x_pix], [swc_y_pix])  # Update marker for SWC
        marker2.set_data([sim_x_pix], [sim_y_pix])  # Update marker for SIM
        fig.canvas.draw()

        swc_total_int = np.sum(swc_atm_cube[swc_x_pix, swc_y_pix, :].value) * swc_wvl_res.to(u.cm/u.pix).value
        sim_total_int = np.sum(sim_atm_cube[sim_x_pix, sim_y_pix, :].value) * sim_wvl_res.to(u.cm/u.pix).value
        percent_diff = (swc_total_int - sim_total_int) / sim_total_int * 100

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.step(swc_wvl_grid.to(u.AA).value, swc_atm_cube[swc_x_pix, swc_y_pix, :].value, where='mid', label='SWC Spectrum')
        ax2.step(sim_wvl_grid.to(u.AA).value, sim_atm_cube[sim_x_pix, sim_y_pix, :].value, where='mid', label='SIM Spectrum')
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Intensity')
        ax2.set_title(f'Spectrum at SWC pixel ({swc_x_pix}, {swc_y_pix}) and SIM pixel ({sim_x_pix}, {sim_y_pix})\nPercent difference: {percent_diff:.2f}%')
        ax2.legend(loc='best', fontsize='small')
        plt.tight_layout()
        plt.show()

        fig2.canvas.mpl_connect('close_event', on_close)  # Remove markers when the pop-up is closed

    def on_close(event):
        # Remove the markers when the pop-up is closed
        marker.set_data([], [])
        marker2.set_data([], [])
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=block)













def plot_summaries(mc_res, analysis_res,
                   swc_atm_cube, swc_wvl_res, swc_wvl_grid, swc_vel_grid,
                   sim_t_i, sim_label,
                   photon0, pixel0, psf0, electron0, sl0, dn0):

    tot_int = (swc_atm_cube.sum(axis=2).value * swc_wvl_res.to(u.cm/u.pix).value)
    flat = tot_int.ravel()
    mean_I = flat.mean()
    std_I  = flat.std()
    targets = [mean_I-std_I, mean_I, mean_I+std_I]
    coords = []
    for tgt in targets:
        idx = np.argmin(np.abs(flat - tgt))
        coords.append(np.unravel_index(idx, tot_int.shape))

    fig1, axs1 = plt.subplots(4, 1, figsize=(6, 10), sharex=True)
    maps = [tot_int, (analysis_res['velocity_mean']), (analysis_res['velocity_std']), (analysis_res['velocity_err'])]
    titles = [f'Total intensity (t={sim_t_i.value:.0f}s)', 'Mean Doppler velocity', 'Std of Doppler velocity', 'Error (tgt-mean)']
    cmaps = ['inferno','RdYlBu_r','RdYlBu_r','RdYlBu_r']
    # limits = [None, (-100, 100), (0, 100), (-100, 100)]
    limits = [None, (-50, 50), (0, 2), (-2, 2)]
    markers = ['x','o','+']

    for ax, data, title, cmap, lims in zip(axs1, maps, titles, cmaps, limits):
        if "intensity" in title:
            data = np.log10(data)
        im = ax.imshow(data.T, origin='lower', cmap=cmap, aspect='equal')
        ax.set_title(title)
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        plt.colorbar(im, ax=ax)
        for m, (ix,iy) in zip(markers, coords):
            # ax.plot(ix, iy, m, ms=10, mfc='none', mew=2, color='white')
            ax.plot(ix, iy, m, color='white')
        if lims is not None:
            im.set_clim(lims)

    plot_target_spectra(swc_wvl_grid, sim_t_i, mc_res, photon0, pixel0, psf0, electron0, sl0, dn0, coords)

    plot_int_vs_doppler_errs(tot_int, analysis_res)

    markers_click = [ax.plot([],[], 'r+', ms=12)[0] for ax in axs1]

    def onclick(event):
        if event.inaxes not in axs1:
            return
        ax_idx = axs1.tolist().index(event.inaxes)
        xpix = int(round(event.xdata))
        ypix = int(round(event.ydata))

        # update all red markers
        for mk in markers_click:
            mk.set_data([],[])
        # # place on whichever map was clicked
        # markers_click[ax_idx].set_data([xpix],[ypix])
        # fig1.canvas.draw()
        # place on all maps
        for mk in markers_click:
            mk.set_data([xpix],[ypix])
        fig1.canvas.draw()

        # now plot the overlaid spectra for that pixel
        plot_click_spectra(xpix, ypix,
                           swc_wvl_grid, sim_t_i,
                           mc_res, photon0, pixel0, psf0, electron0, sl0, dn0)

    fig1.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show(block=False)


def plot_target_spectra(swc_wvl_grid, sim_t_i, mc_res, photon0, pixel0, psf0, electron0, sl0, dn0, coords):
    stages = [photon0, pixel0, psf0, electron0, sl0, dn0]
    names  = ['photon_cube','pixel_cube','psf_cube',
              'electron_cube','sl_cube','dn_cube']

    fig2, axs2 = plt.subplots(3, 6, figsize=(9,5), sharex=True)
    for col, (cube, nm) in enumerate(zip(stages, names)):
        for row, (ix, iy) in enumerate(coords):
            spec = cube[ix, iy, :].value
            ax = axs2[row, col]
            ax.step(swc_wvl_grid.to(u.AA).value, spec, where='mid')
            if row == 2:
                ax.set_xlabel('Wavelength (Å)')
                ax2 = ax.twiny()
                v = (swc_wvl_grid - swc_wvl_grid.mean())/swc_wvl_grid.mean()*const.c.to('km/s').value
                ax2.set_xlim(v.min(), v.max())
                ax2.set_xlabel('Velocity (km/s)')
            if col == 0:
                ax.set_ylabel(f'Row {row}')
            if col == len(stages)-1:
                # plot the fit from mc_res
                fit = mc_res['fit_cube_0'][ix, iy, :]
                yvals = gaussian(swc_wvl_grid.value, *fit)
                ax.plot(swc_wvl_grid.to(u.AA).value, yvals, 'r--', lw=1)
            ax.set_title(nm)

    fig2.suptitle(f'Spectra at pixel ({ix},{iy}), t={sim_t_i.value}s', y=0.92)
    plt.tight_layout()
    plt.show(block=False)

def plot_click_spectra(ix, iy,
                       swc_wvl_grid, sim_t_i,
                       mc_res, photon0, pixel0, psf0, electron0, sl0, dn0):
    stages = [photon0, pixel0, psf0, electron0, sl0, dn0]
    names  = ['photon_cube','pixel_cube','psf_cube',
              'electron_cube','sl_cube','dn_cube']

    fig2, axs2 = plt.subplots(1, 6, figsize=(9,2), sharex=True)
    for col, (cube, nm) in enumerate(zip(stages, names)):
        spec = cube[ix, iy, :].value
        ax = axs2[col]
        ax.step(swc_wvl_grid.to(u.AA).value, spec, where='mid')
        if col == 0:
            ax.set_ylabel('Intensity')
        if col == len(stages)-1:
            fit = mc_res['fit_cube_0'][ix, iy, :]
            yvals = gaussian(swc_wvl_grid.value, *fit)
            ax.plot(swc_wvl_grid.to(u.AA).value, yvals, 'r--', lw=1)
        ax.set_title(nm)

    fig2.suptitle(f'Spectra at pixel ({ix},{iy}), t={sim_t_i.value}s', y=0.92)
    plt.tight_layout()
    plt.show(block=False)

def plot_int_vs_doppler_errs(tot_int, analysis_res):
    "Make a scatter plot of total intensity vs. the different Doppler errors all on one plot"
    fig3, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(tot_int.ravel(), analysis_res['velocity_err'].ravel(), s=1, alpha=0.5, color='k')
    ax.scatter(tot_int.ravel(), analysis_res['velocity_std'].ravel(), s=1, alpha=0.5, color='r')
    ax.scatter(tot_int.ravel(), analysis_res['velocity_mean'].ravel(), s=1, alpha=0.5, color='b')
    ax.set_xlabel('Total intensity (erg/s/cm2/sr)')
    ax.set_ylabel('Doppler error (km/s)')
    ax.set_title('Total intensity vs. Doppler error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show(block=False)















def main(): 

    print("Loading PSFs...")
    psf_mesh = load_psf(tel_mesh_psf_filename, skiprows=16)
    psf_focus = load_psf(tel_focus_psf_filename, skiprows=21)

    print("Resampling PSFs to the instrument spatial/spectral resolution...")
    psf_mesh_resampled = resample_psf(psf_mesh, tel_mesh_psf_input_res, swc_pix_size)
    psf_focus_resampled = resample_psf(psf_focus, tel_focus_psf_input_res, swc_pix_size)

    print("Combining and normalising PSFs...")
    psf_combined = combine_normalise_psf(psf_mesh_resampled, psf_focus_resampled, max_size=5)

    print("Loading the synthetic atmosphere...")
    sim_atm_cube, sim_wvl_grid, sim_wvl_res, sim_vel_grid, sim_vel_res, sim_spt_res, sim_wvl0 = load_synthetic_atmosphere(dat_filename)

    print("Rebinning the atmosphere to the instrument spatial/spectral resolution...")
    swc_atm_cube, swc_wvl_grid, swc_vel_grid = rebin_atmosphere(sim_atm_cube, sim_wvl_grid, sim_wvl_res, sim_vel_grid, sim_spt_res, swc_wvl_res, swc_spt_res, sim_wvl0)

    # plot_atmosphere(swc_atm_cube, swc_wvl_grid, swc_vel_grid, sim_atm_cube, sim_wvl_grid, sim_wvl_res, sim_vel_grid, sim_wvl0, block=False)

    print("Fitting spectra before instrument effects...")
    tgt_fit_cube = fit_spectra(sim_atm_cube.value, sim_wvl_grid.value)

    all_results = []
    for sim_t_i in tqdm(sim_t, desc="Exposure times", unit="exposure time"):
        swc_atm_cube_i = swc_atm_cube*sim_t_i
        monte_carlo_results = monte_carlo(swc_atm_cube_i, swc_wvl_grid, swc_vel_grid, psf_combined, sim_t_i, sim_wvl0)
        analysis_results = analysis(monte_carlo_results['fit_cubes'], tgt_fit_cube, sim_wvl0, u.km/u.s)

    globals().update(locals());raise ValueError("Kicking back to ipython")

    mc_res = monte_carlo_results
    plot_summaries(
      mc_res, analysis_results,
      swc_atm_cube_i, swc_wvl_res, swc_wvl_grid, swc_vel_grid,
      sim_t_i, f'{sim_t_i.value}s',
      mc_res['photon_cube_0'], mc_res['pixel_cube_0'], mc_res['psf_cube_0'],
      mc_res['electron_cube_0'], mc_res['sl_cube_0'], mc_res['dn_cube_0']
    )

    globals().update(locals())
if __name__ == "__main__":
    main()