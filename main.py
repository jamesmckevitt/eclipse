import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
from scipy.signal import convolve2d
from scipy.io import readsav
from tqdm import tqdm
import astropy.constants as const
import astropy.units as u
import os

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Detector
swc_qe = 0.7                      # Quantum efficiency
swc_e_per_euv_ph = 18.0           # Electrons produced per EUV photon
swc_e_per_vis_ph = 2.0            # Electrons produced per VIS photon
swc_read_noise_e = 10.0           # CCD readout noise (electrons)
swc_dn_per_e = 19.0               # Conversion factor (DN per electron)
swc_dark_current = 1.0            # Dark current (electrons)
swc_pixel_size = 13.5e-6          # Pixel size (m)
swc_instrument_width = 0.1123e-10 # Instrumental width (m) ### TODO: REVISIT USING REAL PSF ###

# Telescope
tel_psf_filename = "psf.txt"      # PSF filename
tel_psf_input_res = None          # Input resolution (m)
tel_psf_output_res = None         # Output resolution (m)

# Synthetic data
dat_filename = "SI_Fe_XII_1952_d0_xy_0270000.sav"  # MuRAM file
dat_rest_wave = 195.12e-10        # Rest wavelength (m)

# Simulation parameters
sim_n = 1                         # Number of Monte Carlo iterations

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
    n_scan, n_slit, n_spec = data_cube.shape
    fit_params = np.zeros((n_scan, n_slit, 4))  # [peak, cent, sigma, background]
    for i in range(n_scan):
        for j in range(n_slit):
            fit_result = fit_gaussian_profile(wave_axis, data_cube[i, j, :])
            fit_params[i, j, :] = fit_result['params']
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


# def rebin_psf(psf, old_spacing, new_spacing):
#     """
#     Rebin (resample) the PSF array from its original pixel spacing to that of the detector.

#     Parameters:
#         psf : numpy.ndarray
#             Input PSF (2D array).
#         old_spacing : float
#             Original PSF pixel spacing (microns).
#         new_spacing : float
#             Detector pixel size (microns).
    
#     Returns:
#         numpy.ndarray: Rebinned PSF.
#     """
#     scale_factor = old_spacing / new_spacing
#     psf_rebinned = zoom(psf, scale_factor, order=1)
#     # Normalise to conserve total energy
#     total_before = psf.sum()
#     total_after = psf_rebinned.sum()
#     if total_after > 0:
#         psf_rebinned *= (total_before / total_after)
#     return psf_rebinned


# def convolve_cube_with_psf(cube, psf):
#     """
#     Convolve each (spatial, spectral) plane in the cube with the instrument PSF.

#     The cube is assumed to have shape (n_scan, n_slit, n_spectral). For each scan position,
#     the (n_slit, n_spectral) plane is convolved with the PSF.

#     Parameters:
#         cube : numpy.ndarray
#             Input data cube.
#         psf : numpy.ndarray
#             PSF kernel (2D).
    
#     Returns:
#         numpy.ndarray: Convolved data cube.
#     """
#     n_scan, n_slit, n_spec = cube.shape
#     conv_cube = np.zeros_like(cube)
#     for i in range(n_scan):
#         plane = cube[i, :, :]
#         plane_conv = convolve2d(plane, psf, mode='same', boundary='fill')
#         conv_cube[i, :, :] = plane_conv
#     return conv_cube


# def convert_counts_to_dn(counts_cube, qe=DEFAULT_QE, e_per_euv_ph=DEFAULT_E_PER_EUV_PH,
#                          read_noise=DEFAULT_READ_NOISE_E, dn_per_e=DEFAULT_DN_PER_E,
#                          dark_current=DEFAULT_DARK_CURRENT):
#     """
#     Convert photon counts into digital numbers (DN) via conversion to electrons.

#     This function multiplies the counts by the quantum efficiency and electrons-per-photon,
#     adds a constant dark current and readout noise (sampled from a Gaussian), and converts
#     the resulting electrons to DN.

#     Parameters:
#         counts_cube : numpy.ndarray
#             Cube with photon counts.
#         qe : float
#             Detector quantum efficiency.
#         e_per_ph : float
#             Electrons produced per photon.
#         read_noise : float
#             CCD readout noise (in electrons).
#         dn_per_e : float
#             Conversion factor (DN per electron).
#         dark_current : float
#             Dark current in electrons.
    
#     Returns:
#         numpy.ndarray: Data cube in DN.
#     """
#     # Convert counts to electrons and add dark current
#     electrons = counts_cube * qe * e_per_euv_ph + dark_current
#     # Add read noise
#     noise = np.random.normal(loc=0.0, scale=read_noise, size=electrons.shape)
#     electrons_noisy = electrons + noise
#     electrons_noisy[electrons_noisy < 0] = 0.0  # no negative electrons
#     # Convert to DN
#     return electrons_noisy / dn_per_e


def load_muram_atmosphere(filepath):
    """
    Load a synthesised MuRAM atmosphere from an IDL .sav file for the Fe XII 195.12 emission line.
    
    Parameters
    ----------
    filepath : str
        Path to the IDL .sav file (e.g. 'SI_Fe_XII_1952_d0_xy_0270000.sav').
    
    Returns
    -------
    atmosphere_cube : numpy.ndarray
        The 3D synthesised atmosphere cube with dimensions [n1, n2, n_wave],
        where n_wave is the number of wavelength bins.
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

    # Extract the synthesised intensity cube.
    atmosphere_cube = np.copy(data[key])  # erg/s/cm^2/sr/cm
    atmosphere_cube = np.transpose(atmosphere_cube, (2, 1, 0))  # Correct dimensions from IDL import artifact (so [n1, n2, n_wave])

    # Determine the number of spectral points from the cube's third dimension.
    n_wave = atmosphere_cube.shape[2]

    # Construct a velocity grid spanning -300e5 to +300e5 cm/s.
    # (The readme specifies vr = 600e5 cm/s as the full range.)
    vr = 600.0e5   # Total velocity range in cm/s
    # Generate a uniform velocity grid from -vr/2 to +vr/2
    velocity_grid = np.linspace(-vr/2, vr/2, n_wave)
    
    # Calculate the velocity spacing (cm/s)
    dv = velocity_grid[1] - velocity_grid[0]
    
    # Speed of light in cm/s
    c = 2.99792458e10
    
    # Convert the velocity grid to a wavelength shift (in cm)
    dlar = velocity_grid / c * dat_rest_wave
    
    # Create the wavelength axis in cm, centred on the line centre la_0
    wave_cm = dat_rest_wave + dlar
    
    # Convert the wavelength axis to Angstroms (1 cm = 1e8 Angstrom)
    wave_axis = wave_cm * 1e8

    # Convert to J/s/cm^2/sr/cm
    atmosphere_cube *= 1e-7

    # Convert to photon/s/cm^2/sr/cm
    photon_energy = const.h.to('J.s').value * const.c.to('m/s').value / (dat_rest_wave * 1e-10)  # in J
    atmosphere_cube /= photon_energy

    # Convert to photon/s/pixel/cm
    cube_pixel_size = 0.192 * 1e6  # m (from readme)
    solid_angle = (cube_pixel_size / const.au.to('m').value) ** 2  # steradians
    atmosphere_cube *= (swc_pixel_size ** 2) * solid_angle

    # Convert to photon/s/pixel
    d_wave = wave_cm[1] - wave_cm[0]  # cm
    atmosphere_cube *= d_wave

    # TEMP!!!!!!!!!!! Normalise so the maximum is 50
    atmosphere_cube *= 50.0 / np.max(atmosphere_cube)
    # TEMP!!!!!!!!!!! round everything to the nearest integer
    atmosphere_cube = np.round(atmosphere_cube).astype(np.int32)

    return atmosphere_cube, wave_axis


def plot_spectra(key_pixels, wave_axis, it):
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

    fig, axs = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle(f"Monte Carlo Iteration {it}", fontsize=16)
    for i, key in enumerate(key_pixels.keys()):
        axs[i, 0].plot(wave_axis, key_pixels[key]['spectra_sim'], label='Simulated', color='blue')
        axs[i, 1].plot(wave_axis, key_pixels[key]['spectra_poisson'], label='Poisson', color='orange')
        axs[i, 2].plot(wave_axis, key_pixels[key]['spectra_psf'], label='PSF', color='green')
        axs[i, 3].plot(wave_axis, key_pixels[key]['spectra_sl'], label='Stray Light', color='red')
        axs[i, 4].plot(wave_axis, key_pixels[key]['spectra_dn'], label='DN', color='purple')

        for j in range(5):
            axs[i, j].set_title(f"{key} - {j}")
            axs[i, j].set_xlabel('Wavelength (Angstrom)')
            axs[i, j].set_ylabel('y')
            axs[i, j].legend()
            axs[i, j].grid()
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"monte_carlo_iteration_{it}.png")
    plt.close()

def monte_carlo_analysis(sim_cube, wave_axis, psf):
    # Make a dictionary of dictionaries to store information about the key pixels
    total = np.sum(sim_cube, axis=2)
    key_pixels = {
        'max': {'ipix': np.unravel_index(np.argmax(total), total.shape)},
        'p75': {'ipix': np.unravel_index(np.abs(total - np.percentile(total, 75)).argmin(), total.shape)},
        'p50': {'ipix': np.unravel_index(np.abs(total - np.percentile(total, 50)).argmin(), total.shape)},
        'p25': {'ipix': np.unravel_index(np.abs(total - np.percentile(total, 25)).argmin(), total.shape)},
    }
    for key in key_pixels.keys():key_pixels[key]['spectra_sim'] = sim_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

    # Preallocate arrays for storing velocity and nonthermal values
    velocity_vals = np.zeros((sim_n, sim_cube.shape[0], sim_cube.shape[1]))
    nonthermal_vals = np.zeros((sim_n, sim_cube.shape[0], sim_cube.shape[1]))

    for it in tqdm(range(sim_n), desc="Monte Carlo iterations", unit="iteration"):

        # Add Poisson noise
        poisson_cube = add_poisson_noise(sim_cube)
        for key in key_pixels.keys():key_pixels[key]['spectra_poisson'] = poisson_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Convolve with the PSF
        # psf_cube = convolve_cube_with_psf(poisson_cube, psf)
        psf_cube = poisson_cube.copy()
        for key in key_pixels.keys():key_pixels[key]['spectra_psf'] = psf_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Add stray light
        # sl_cube = add_vis_stray_light(psf_cube)
        sl_cube = psf_cube.copy()
        for key in key_pixels.keys():key_pixels[key]['spectra_sl'] = sl_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Read out electrons
        # el_cube = read_out_photons(sl_cube)
        el_cube = sl_cube.copy()
        for key in key_pixels.keys():key_pixels[key]['spectra_el'] = el_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Convert to DN
        # dn_cube = convert_counts_to_dn(el_cube)
        dn_cube = el_cube.copy()
        for key in key_pixels.keys():key_pixels[key]['spectra_dn'] = dn_cube[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Fit the spectrum
        fit_params = fit_spectra(dn_cube, wave_axis)
        for key in key_pixels.keys():
            key_pixels[key]['fit_params'] = fit_params[key_pixels[key]['ipix'][0], key_pixels[key]['ipix'][1], :]

        # Calculate the Doppler velocity and excess broadening
        for key in key_pixels.keys():
            key_pixels[key]['velocity'] = None
            key_pixels[key]['nonthermal'] = None

        # Plot the spectras for this iteration
        if it % 1 == 0:
            plot_spectra(key_pixels, wave_axis, it)

    # Calculate the standard deviation (uncertainty) maps
    velocity_std = np.nanstd(velocity_vals, axis=2)
    nonthermal_std = np.nanstd(nonthermal_vals, axis=2)
    
    return {'velocity_std': velocity_std, 'nonthermal_std': nonthermal_std}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def main():

    # Load the synthetic atmosphere cube.
    sim_cube, wave_axis = load_muram_atmosphere(dat_filename)

    # Load the PSF
    # psf = load_psf(tel_psf_filename)
    # psf_rebinned = rebin_psf(psf, tel_psf_input_res, tel_psf_output_res)
    psf_rebinned = None

    # Perform Monte Carlo analysis over nsim iterations
    results = monte_carlo_analysis(sim_cube, wave_axis, psf_rebinned)


if __name__ == "__main__":
    main()