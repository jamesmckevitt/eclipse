## TODO: add background spectras

import numpy as np
import matplotlib
matplotlib.use('Agg')
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
import h5py

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Detector
swc_vis_qe = 1                   # Quantum efficiency at visible wavelengths
swc_euv_qe = 0.76                 # Quantum efficiency at target wavelength (function of wavelength)
swc_e_per_euv_ph = 18.0           # Electrons produced per EUV photon
swc_e_per_vis_ph = 2.0            # Electrons produced per VIS photon
swc_read_noise_e = 10.0           # CCD readout noise (electrons)
swc_dn_per_e = 19.0               # Conversion factor (DN per electron)
swc_dark_current = 1.0            # Dark current (electrons)
swc_pixel_size_um = 13.5          # Pixel size (microns)
swc_pixel_size = swc_pixel_size_um * 1e-6  # Pixel size (m)
swc_spatial_sampling_a = 0.159    # Spatial sampling (arcsec/pixel)
swc_spatial_sampling = swc_spatial_sampling_a * np.pi / (180 * 3600)  # Spatial sampling in radians/pixel
swc_spatial_sampling_at_sun = const.au.to('m').value * np.tan(swc_spatial_sampling/2) * 2  # Spatial sampling at the Sun per spatial pixel (m)
swc_spectral_sampling_ma = 16.9   # Spectral sampling (mA/pixel)
swc_spectral_sampling = swc_spectral_sampling_ma * 1e-13  # Spectral sampling in m/pixel

# Telescope
tel_collecting_area = .5*np.pi*(0.28/2)**2  # Entrance pupil diameter (aperture stop) is .28 m; *half* pi r^2 as only half of beam for SW
tel_pm_efficiency = 0.161         # Primary mirror efficiency
tel_ega_efficiency = 0.0623       # Grating efficiency
tel_filter_transmission = 0.507   # Filter transmission efficiency
tel_focus_psf_filename = "psf_euvst_v20230909_195119_focus.txt"  # PSF filename
tel_mesh_psf_filename = "psf_euvst_v20230909_derived_195119_mesh.txt"  # Filter PSF filename
tel_focus_psf_input_res_um = 0.5  # Input resolution (microns)
tel_focus_psf_input_res = tel_focus_psf_input_res_um * 1e-6  # Input resolution (m)
tel_mesh_psf_input_res_mm = 6.1200E-04  # Input resolution (millimeters)
tel_mesh_psf_input_res = tel_mesh_psf_input_res_mm * 1e-3  # Input resolution (m)

# Synthetic data
dat_filename = "SI_Fe_XII_1952_d0_xy_0270000.sav"  # MuRAM file
dat_rest_wave = 195.12e-10        # Rest wavelength (m)
dat_velocity_range_cm = 600e5     # Velocity range in the wavelength direction (m/s) (from MuRAM readme)
dat_velocity_range = dat_velocity_range_cm * 1e-2  # Velocity range in the wavelength direction (m/s)
dat_pixel_scale_cm = 0.192e8      # Pixel scale (cm) (from MuRAM readme)
dat_pixel_scale = dat_pixel_scale_cm * 1e-2  # Pixel scale (m) (from MuRAM readme)

# Simulation parameters
sim_n = 50                       # Number of Monte Carlo iterations per exposure time
sim_t = [1,2,5,10,20]    # Exposure time (s) 90s for quiet Sun, 40s for active region, 5s for flare (1s before x-band loss on EIS).
sim_stray_light_s = 1             # Visible stray light photon/s/pixel

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def gaussian(wave, peak, cent, sigma, background):
    """
    Gaussian function for curve-fitting.

    Parameters:
        wave : numpy.ndarray
            Array of wavelengths.
        peak : float
            Peak amplitude of the Gaussian.
        cent : float
            Centre (mean) of the Gaussian.
        sigma : float
            Standard deviation (width) of the Gaussian.
        background : float
            Constant background level.
    
    Returns:
        numpy.ndarray: Evaluated Gaussian profile plus background.
    """
    return peak * np.exp(-0.5 * ((wave - cent) / sigma) ** 2) + background


def fit_spectra(data_cube, wave_axis):
    """
    Fit a Gaussian to each spectrum in the data cube.

    Parameters:
        data_cube : numpy.ndarray
            Cube of spectra (3D array).
        wave_axis : numpy.ndarray
            Wavelength axis corresponding to the spectral dimension.
    
    Returns:
        numpy.ndarray: Fitted parameters for each spectrum.
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
    results = Parallel(n_jobs=-1)(
        delayed(process_scan)(i) for i in tqdm(range(n_scan), desc="Fitting spectra", unit="scan", leave=False)
    )
    
    # Combine the results for each scan back into a single numpy array.
    fit_params = np.array(results)
    return fit_params


def guess_initial_params(wave, profile):
    """
    Estimate initial guess parameters for Gaussian fitting.

    Parameters:
        wave : numpy.ndarray
            Wavelength (or spectral bin) array.
        profile : numpy.ndarray
            Measured intensity profile.
    
    Returns:
        list: Initial guesses [peak, cent, sigma, background].
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

    Parameters:
        wave : numpy.ndarray
            Wavelength or spectral bin axis.
        profile : numpy.ndarray
            Measured intensity profile (in DN).
    
    Returns:
        dict: Contains fit parameters, parameter errors, fitted model, and a flag of convergence.
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

    return {
        'params': popt,
        'errors': perr,
        'model': model,
        'converged': converged,
    }


def add_poisson_noise(data_cube):
    """
    Add Poisson noise to the spectra in each spatial position.
    This simulates the photon counting statistics of the signal.

    Parameters:
        data_cube : numpy.ndarray
            Input cube of signal (photon counts).
    
    Returns:
        numpy.ndarray: Noisy data cube.
    """
    return np.random.poisson(data_cube)


def load_psf(filename, skiprows=0):
    """
    Load the PSF from a file.

    Parameters:
        filename : str
            Path to the PSF file.
    
    Returns:
        numpy.ndarray: PSF kernel (2D).
    """
    psf = np.loadtxt(filename, skiprows=skiprows, encoding='UTF-16 LE')
    return psf


def resample_psf(psf, old_spacing, new_spacing):
    """
    Resample the PSF array from its original pixel spacing to that of the detector.

    Parameters:
        psf : numpy.ndarray
            Input PSF (2D array).
        old_spacing : float
            Original PSF pixel spacing (microns).
        new_spacing : float
            Detector pixel size (microns).
    
    Returns:
        numpy.ndarray: Resampled PSF.
    """
    scale_factor = old_spacing / new_spacing
    psf_resampled = zoom(psf, scale_factor, order=1)
    return psf_resampled


def combine_normalise_psf(psf1, psf2, crop_frac=0.99, max_size=None):
  """
  Combine two PSF arrays by convolving them, then crop and normalize.

  You can either:
    - Crop by including a fraction of total intensity (crop_frac, default 0.99)
    - Or specify max_size (odd integer) for a centered square crop on the peak.

  Parameters:
    psf1, psf2 : numpy.ndarray
    Input PSFs.
    crop_frac : float, optional
    Fraction of total intensity to include when intensity‑based cropping.
    max_size : int, optional
    Maximum output size (odd). If set, overrides crop_frac.

  Returns:
    numpy.ndarray: Cropped & normalized combined PSF.
  """
  # Convolve
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

  # Normalize to unit sum
  cropped /= cropped.sum()
  return cropped


def convolve_cube_with_psf(cube, psf):
    """
    Convolve each (spatial, spectral) plane in the cube with the instrument PSF.

    The cube is assumed to have shape (n_scan, n_slit, n_spectral). For each scan position,
    the (n_slit, n_spectral) plane is convolved with the PSF.

    Parameters:
        cube : numpy.ndarray
            Input data cube.
        psf : numpy.ndarray
            PSF kernel (2D).
    
    Returns:
        numpy.ndarray: Convolved data cube.
    """
    n_scan, n_slit, n_spec = cube.shape
    conv_cube = np.zeros_like(cube)
    for i in range(n_scan):
        plane = cube[i, :, :]
        plane_conv = convolve2d(plane, psf, mode='same', boundary='fill')
        conv_cube[i, :, :] = plane_conv
    return conv_cube


def read_out_photons(counts_cube):
    """
    Read out the photons from the detector, including CCD readout noise, dark current, and conversion to electrons.

    Parameters:
        counts_cube : numpy.ndarray
            Cube of photon counts.

    Returns:
        numpy.ndarray: Cube of electrons after readout noise and dark current.
    """
    # Convert counts to electrons
    electrons = counts_cube * swc_euv_qe * swc_e_per_euv_ph

    # Add dark current (in electrons)
    electrons += swc_dark_current

    # Add readout noise (Gaussian noise with mean 0, std = swc_read_noise_e)
    noise = np.random.normal(loc=0.0, scale=swc_read_noise_e, size=electrons.shape)
    electrons_noisy = electrons + noise

    # No negative electrons
    electrons_noisy[electrons_noisy < 0] = 0.0

    return electrons_noisy


def add_vis_stray_light(electrons_cube, sim_t_i):
    """
    Add visible stray light, in electrons, to the cube.

    Parameters:
        electrons_cube : numpy.ndarray
            Cube of electrons (output of read_out_photons).

    Returns:
        numpy.ndarray: Cube of electrons with added stray light.
    """

    out_cube = electrons_cube.copy()

    # Add stray light signal
    sim_stray_light = sim_stray_light_s * sim_t_i
    
    stray_light_vis_photons = np.random.poisson(sim_stray_light, size=electrons_cube.shape)

    stray_light_electrons = stray_light_vis_photons * swc_e_per_vis_ph * swc_vis_qe

    out_cube += stray_light_electrons

    # No negative electrons
    out_cube[out_cube < 0] = 0.0

    return out_cube


def convert_counts_to_dn(electrons_cube):
    """
    Convert the cube of electrons to digital numbers (DN).

    Parameters:
        electrons_cube : numpy.ndarray
            Cube of electrons (output of read_out_photons).
    
    Returns:
        numpy.ndarray: Cube of DN values.
    """
    return electrons_cube / swc_dn_per_e


def load_muram_atmosphere(filepath, sim_t_i):
    """
    Load a synthesised MuRAM atmosphere from an IDL .sav file for the Fe XII 195.12 emission line and include the exposure time.
    
    Parameters
    ----------
    filepath : str
        Path to the IDL .sav file (e.g. 'SI_Fe_XII_1952_d0_xy_0270000.sav').
        Units are erg/s/cm^2/sr/cm.
    
    Returns
    -------
    atmosphere_cube : numpy.ndarray
        The 3D synthesised atmosphere cube with dimensions [n1, n2, n_wave],
        where n_wave is the number of wavelength bins.
        erg/cm^2/sr/cm.
    wave_axis : numpy.ndarray
        Wavelength axis corresponding to the spectral dimension, in Angstrom.
    
    Notes
    -----
    The IDL save file is assumed to have a variable with a name formatted as:
        "si_<plane>_dl"   (e.g. si_xy_dl for the XY plane).
    The units of the synthesised intensity are erg/s/cm^2/sr/cm.
    
    The wavelength axis is constructed as follows:
      - The number of spectral points (nw) is taken from the third dimension of the cube.
      - A velocity grid is made spanning -300x1e5 to +300x1e5 [cm/s].
      - This is then converted to a wavelength shift using the relation: 
            dλ (cm) = (velocity / c) * la_0.
      - Finally, the wavelength axis is centred on la_0 and converted to Angstrom 
        (1 cm = 1e8 Angstrom).
    """

    # Load the IDL .sav file using SciPy
    data = readsav(filepath)
    data_keys = list(data.keys())
    assert len(data_keys) == 1
    key = data_keys[0]

    # Extract the synthesised intensity cube
    atmosphere_cube = np.copy(data[key])  # erg/s/cm^2/sr/cm
    atmosphere_cube = np.transpose(atmosphere_cube, (2, 1, 0))  # Correct dimensions from IDL import artifact (so [n1, n2, n_wave])

    # Calculate the spectral scale
    n_wave = atmosphere_cube.shape[2]
    velocity_grid = np.linspace(-dat_velocity_range*.5, dat_velocity_range*.5, n_wave)  # m/s
    wavelength_shift = velocity_grid / const.c.to('m/s').value * dat_rest_wave  # m
    wave_axis = dat_rest_wave + wavelength_shift  # m

    # Rebin the spectral axis to swc_spectral_resolution
    new_wave_axis = np.arange(wave_axis[0], wave_axis[-1], swc_spectral_sampling)  # m
    interp_func = interp1d(wave_axis, atmosphere_cube, axis=2, bounds_error=False, fill_value=0)
    atmosphere_cube = interp_func(new_wave_axis)
    wave_axis = new_wave_axis  # m

    # Rebin the spatial axes of the cube to the spatial dimensions of the SWC
    scale_factor = dat_pixel_scale / swc_spatial_sampling_at_sun
    atmosphere_cube = zoom(atmosphere_cube, (scale_factor, scale_factor, 1), order=5)  # Rebin spatially
    atmosphere_cube = np.clip(atmosphere_cube, 0, None)  # No negative values

    # Multiply by the exposure time
    atmosphere_cube *= sim_t_i  # Convert to erg/cm^2/sr/cm

    return atmosphere_cube, wave_axis


def convert_to_photons(data_cube):
    """
    Convert the cube of intensity to photons.

    Parameters:
        data_cube : numpy.ndarray
            Cube of intensity (erg/cm^2/sr/cm).
    
    Returns:
        numpy.ndarray: Cube of photons (photon/cm^2/sr/cm).
    """
    out_cube = data_cube.copy()

    # Convert to J/cm^2/sr/cm
    out_cube = out_cube.astype(np.float64) * 1e-7

    # Convert to photon/cm^2/sr/cm
    photon_energy = const.h.to('J.s').value * const.c.to('m/s').value / (dat_rest_wave)  # in J
    out_cube /= photon_energy

    # Round to nearest integer as can only measure whole photons
    out_cube = np.round(out_cube)

    return out_cube


def add_partial_effective_area(data_cube):
    """
    Add the partial effective area of the telescope to the data cube.

    Parameters:
        data_cube : numpy.ndarray
            Cube of photons (photon/cm^2/sr/cm).
    
    Returns:
        numpy.ndarray: Cube of photons (photon/sr/cm).
    """

    out_cube = data_cube.copy()

    # Calculate the partial effective area
    tel_partial_ea = tel_collecting_area * tel_pm_efficiency * tel_ega_efficiency * tel_filter_transmission  # Telescope partial effective area (excluding CCD QE) in m^2
    tel_partial_ea_cm2 = tel_partial_ea * 1e4  # Convert to cm^2
    out_cube *= tel_partial_ea_cm2  # Convert to photon/sr/cm
    return out_cube


def get_per_pixel(data_cube):
    """
    Convert the cube of photons to per pixel.

    Parameters:
        data_cube : numpy.ndarray
            Cube of photons (photon/sr/cm).

    Returns:
        numpy.ndarray: Cube of photons (photon/pixel).
    """

    out = data_cube.copy()

    # Solid angle per spatial pixel (CCD row)
    solid_angle = swc_spatial_sampling_at_sun ** 2 / const.au.to('m').value ** 2  # steradians / CCD row
    out *= solid_angle  # photon/cm ( / CCD row )

    # Spectral sampling
    out *= swc_spectral_sampling * 1e2  # cm

    return out


def plot_spectra(key_pixels, wave_axis, savename):
    """
    Plot the spectra for the key pixels.

    Parameters:
        key_pixels : dict
            Dictionary of key pixel information.
        wave_axis : numpy.ndarray
            Wavelength axis corresponding to the spectral dimension.
        it : int
            Current iteration number.
    """

    fig, axs = plt.subplots(4, 9, figsize=(30, 15))
    for i, key in enumerate(key_pixels.keys()):

        axs[i, 0].step(wave_axis*1e10, key_pixels[key]['spectra_sim'], where='mid', color='black')
        axs[i, 1].step(wave_axis*1e10, key_pixels[key]['spectra_poisson'], where='mid', color='black')
        axs[i, 2].step(wave_axis*1e10, key_pixels[key]['spectra_photon'], where='mid', color='black')
        axs[i, 3].step(wave_axis*1e10, key_pixels[key]['spectra_area'], where='mid', color='black')
        axs[i, 4].step(wave_axis*1e10, key_pixels[key]['spectra_pixel'], where='mid', color='black')
        axs[i, 5].step(wave_axis*1e10, key_pixels[key]['spectra_psf'], where='mid', color='black')
        axs[i, 6].step(wave_axis*1e10, key_pixels[key]['spectra_el'], where='mid', color='black')
        axs[i, 7].step(wave_axis*1e10, key_pixels[key]['spectra_sl'], where='mid', color='black')
        axs[i, 8].step(wave_axis*1e10, key_pixels[key]['spectra_dn'], where='mid', color='black')

        fit_x = np.linspace(wave_axis[0], wave_axis[-1], 100)
        fit_y = gaussian(fit_x, *key_pixels[key]['fit_params'])
        axs[i, 8].plot(fit_x*1e10, fit_y, label='Fitted Gaussian', color='red', linestyle='--')

        for j in range(9):
            titles = {
                0: 'Simulated',
                1: 'Poisson',
                2: 'Photon',
                3: 'Area',
                4: 'Pixel',
                5: 'PSF',
                6: 'Electrons',
                7: 'Stray Light',
                8: 'DN'
            }
            yaxes = {
                0: 'erg/cm^2/sr/cm)',
                1: 'erg/cm^2/sr/cm)',
                2: 'photon/cm^2/sr/cm)',
                3: 'photon/sr/cm)',
                4: 'photon/pixel)',
                5: 'photon/pixel)',
                6: 'electrons/pixel',
                7: 'electrons/pixel',
                8: 'DN/pixel'
            }
            axs[i, j].set_title(f"{key} - {titles[j]}")
            axs[i, j].set_xlabel('Wavelength (Angstrom)')
            axs[i, j].set_ylabel(yaxes[j])
            axs[i, j].grid()
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(savename)
    plt.close()


def calculate_velocity(fit_params):
    """
    Calculate the Doppler velocity from the fitted Gaussian parameters.

    Parameters:
        fit_params : numpy.ndarray
            Fitted parameters for each spectrum.
    
    Returns:
        numpy.ndarray: Velocity values.
    """
    # Extract the fitted parameters
    cent = fit_params[:, :, 1]  # Centre of the Gaussian
    sigma = fit_params[:, :, 2]  # Width of the Gaussian

    # Calculate the Doppler velocity
    velocity = (cent - dat_rest_wave) / dat_rest_wave * const.c.to('m/s').value  # m/s
    return velocity


def monte_carlo_analysis(sim_cube, wave_axis, psf, sim_t_i):
    # Make a dictionary of dictionaries to store information about the key pixels
    total = np.sum(sim_cube, axis=2)
    key_pixels = {
        'max': {'ipix': np.unravel_index(np.argmax(total), total.shape)},
        'p75': {'ipix': np.unravel_index(np.abs(total - np.percentile(total, 75)).argmin(), total.shape)},
        'p50': {'ipix': np.unravel_index(np.abs(total - np.percentile(total, 50)).argmin(), total.shape)},
        'p25': {'ipix': np.unravel_index(np.abs(total - np.percentile(total, 25)).argmin(), total.shape)}
    }
    for key in key_pixels.keys():key_pixels[key]['spectra_sim'] = sim_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

    # Preallocate arrays for storing velocity values
    velocity_vals = np.zeros((sim_n, sim_cube.shape[0], sim_cube.shape[1]))

    for it in tqdm(range(sim_n), desc="Monte Carlo iterations", unit="iteration", leave=False):

        # Add Poisson noise
        poisson_cube = add_poisson_noise(sim_cube)
        for key in key_pixels.keys():key_pixels[key]['spectra_poisson'] = poisson_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Convert to photons (photon/cm^2/sr/cm)
        photon_cube = convert_to_photons(poisson_cube)
        for key in key_pixels.keys():key_pixels[key]['spectra_photon'] = photon_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Add partial effective area (photon/sr/cm)
        area_cube = add_partial_effective_area(photon_cube)
        for key in key_pixels.keys():key_pixels[key]['spectra_area'] = area_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Add solid angle and wavelength scale (photon/pixel)
        pixel_cube = get_per_pixel(area_cube)
        for key in key_pixels.keys():key_pixels[key]['spectra_pixel'] = pixel_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Convolve with the PSF
        psf_cube = convolve_cube_with_psf(pixel_cube, psf)
        for key in key_pixels.keys():key_pixels[key]['spectra_psf'] = psf_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Read out electrons
        el_cube = read_out_photons(psf_cube)
        for key in key_pixels.keys():key_pixels[key]['spectra_el'] = el_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Add stray light signal
        sl_cube = add_vis_stray_light(el_cube, sim_t_i)
        for key in key_pixels.keys():key_pixels[key]['spectra_sl'] = sl_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Convert to DN
        dn_cube = convert_counts_to_dn(sl_cube)
        for key in key_pixels.keys():key_pixels[key]['spectra_dn'] = dn_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Fit the spectrum
        fit_params = fit_spectra(dn_cube, wave_axis)
        for key in key_pixels.keys():
            key_pixels[key]['fit_params'] = fit_params[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Calculate the Doppler velocity and excess broadening
        velocity_vals[it, :, :] = calculate_velocity(fit_params)

        # Plot the velocity map and spectra for the first iteration
        if it == 0:
            plt.figure()
            plt.imshow(velocity_vals[it, :, :]/1000, cmap='seismic', interpolation='nearest', vmin=-30, vmax=30)
            plt.colorbar(label='Velocity (km/s)')
            plt.title(f"Velocity at iteration {it}")
            plt.savefig(f"velocity_{sim_t_i}_{it}.png")
            plt.close()

            plot_spectra(key_pixels, wave_axis, f'spectra_{sim_t_i}_{it}.png')

    # Calculate the standard deviation (uncertainty) maps
    velocity_std = np.nanstd(velocity_vals, axis=0)

    # Plot the Doppler velocity standard deviation map in km/s
    plt.figure()
    plt.imshow(velocity_std/1000, cmap='seismic', interpolation='nearest', vmin=-5, vmax=5)
    plt.colorbar(label='Standard Deviation of Doppler Velocity (km/s)')
    plt.title('Doppler Velocity Standard Deviation Map (km/s)')
    plt.savefig(f'velocity_std_map_{sim_t_i}.png')
    plt.close()

    return {'velocity_vals': velocity_vals, 'velocity_std': velocity_std, 'key_pixels': key_pixels}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def main(): 

    # Load the PSFs
    psf_mesh = load_psf(tel_mesh_psf_filename, skiprows=16)
    psf_focus = load_psf(tel_focus_psf_filename, skiprows=21)

    # Resample the PSFs to the detector pixel size
    psf_mesh_resampled = resample_psf(psf_mesh, tel_mesh_psf_input_res, swc_pixel_size)
    psf_focus_resampled = resample_psf(psf_focus, tel_focus_psf_input_res, swc_pixel_size)

    # Combine the PSFs
    psf_combined = combine_normalise_psf(psf_mesh_resampled, psf_focus_resampled, max_size=5)

    # Loop over the exposure times and store results
    all_results = []
    for sim_t_i in tqdm(sim_t, desc="Exposure times", unit="exposure time"):
      # Load the synthetic atmosphere cube.
      sim_cube, wave_axis = load_muram_atmosphere(dat_filename, sim_t_i)

      # Perform Monte Carlo analysis over sim_n iterations
      results = monte_carlo_analysis(sim_cube, wave_axis, psf_combined, sim_t_i)
      all_results.append(results)

    # After the loop: plot exposure time vs. Doppler velocity uncertainty for each key pixel
    exposure_times = sim_t
    keys = list(all_results[0]['key_pixels'].keys())
    markers = ['o', 's', '^', 'v']  # one per key

    plt.figure(figsize=(8, 6))
    for key, m in zip(keys, markers):
      # get the pixel indices for this key
      ipix = all_results[0]['key_pixels'][key]['ipix']
      # extract the std dev at that pixel for each exposure time
      std_vals = []
      for res in all_results:
        std_vals.append(res['velocity_std'][ipix[0], ipix[1]] / 1000.0) # convert to km/s

      plt.plot(exposure_times, std_vals,
           marker=m, linestyle='-',
           label=key)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Exposure Time (s)")
    plt.ylabel("Doppler Velocity Uncertainty (km/s)")
    plt.title("Velocity Uncertainty vs Exposure Time")
    plt.legend(title="Key Pixels")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("velocity_uncertainty_vs_exposure_time.png")
    plt.close()

    # After plotting, save all_results in a structured HDF5 file
    with h5py.File('solar_mc_results.h5', 'w') as hf:
      # Global attributes
      hf.attrs['sim_n'] = sim_n
      hf.attrs['exposure_times'] = sim_t

      # Per‐exposure groups
      for t_val, res in zip(sim_t, all_results):
        grp = hf.create_group(f"exposure_{t_val}s")
        grp.create_dataset('velocity_vals', data=res['velocity_vals'])
        grp.create_dataset('velocity_std', data=res['velocity_std'])
        # Key pixels metadata
        kp_grp = grp.create_group('key_pixels')
        for key, info in res['key_pixels'].items():
          sub = kp_grp.create_group(key)
          sub.attrs['ipix'] = info['ipix']
    # Also save a backup using the simple numpy save function
    np.savez('solar_mc_results.npz', all_results=all_results)

if __name__ == "__main__":
    main()