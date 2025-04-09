"""
Synthetic Observations from SOLAR-C EUVST/SW Instrument

This script synthesises realistic observations of an emission line as seen by the SOLAR-C EUVST/SW
instrument. It does so by starting from a synthetic atmosphere 
cube, then:
  
  1. Adding Poisson noise to the spectra (simulating photon counting statistics),
  2. Convolving the spatial/spectral plane with an instrument point spread function (PSF),
  3. Converting photon counts to electrons (applying quantum efficiency, electrons per photon,
     CCD readout noise and dark current), and converting these electrons to digital numbers (DN),
  4. Fitting a Gaussian profile to each spectrum to extract the line centroid (for Doppler velocity)
     and width (to compute the excess (non-thermal) broadening),
  5. Repeating the simulation (Monte Carlo style) to produce error estimates in the maps.

The final outputs are maps of the mean Doppler velocity, its uncertainty, the non-thermal broadening,
and its uncertainty. Results are plotted and saved as an image.

Author: James McKevitt
Date: 09/04/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
from scipy.signal import convolve2d
from scipy.io import readsav
from tqdm import tqdm
import argparse
import os

# ---------------------------------------------------------------------------
# Global Constants and Default Detector/Instrument Parameters
# ---------------------------------------------------------------------------

# Speed of light (km/s)
C_SPEED = 3e5
# Planck's constant (J/s)
H_PLANCK = 1.054571817e-34

# Default detector parameters
DEFAULT_QE = 0.7                  # Quantum efficiency
DEFAULT_E_PER_EUV_PH = 18.0       # Electrons produced per EUV photon
DEFAULT_E_PER_VIS_PH = 2.0        # Electrons produced per VIS photon
DEFAULT_READ_NOISE_E = 10.0       # CCD readout noise (electrons)
DEFAULT_DN_PER_E = 19.0           # Conversion factor (DN per electron)
DEFAULT_DARK_CURRENT = 1.0        # Dark current (electrons)

# Instrument spectral parameters (for the simulated emission line)
DEFAULT_WAVE0 = 195.12            # Rest wavelength in Angstroms
DEFAULT_SIGMA0 = 0.1123           # Instrumental Gaussian width (Angstroms)
DEFAULT_DWAVE = 0.0335            # Spectral dispersion (Angstrom per spectral bin)
DEFAULT_NWAVE = 32                # Number of spectral bins

# Number of Monte Carlo iterations for error analysis
DEFAULT_NSIM = 1

# ---------------------------------------------------------------------------
# Define functions for the simulation steps
# ---------------------------------------------------------------------------

def gaussian(wave, peak, cent, sigma, background):
    """
    Gaussian function for curve-fitting.

    Parameters:
        wave : numpy.ndarray
            Array of wavelengths (or spectral bins).
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


def rebin_psf(psf, old_spacing, new_spacing):
    """
    Rebin (resample) the PSF array from its original pixel spacing to that of the detector.

    Parameters:
        psf : numpy.ndarray
            Input PSF (2D array).
        old_spacing : float
            Original PSF pixel spacing (microns).
        new_spacing : float
            Detector pixel size (microns).
    
    Returns:
        numpy.ndarray: Rebinned PSF.
    """
    scale_factor = old_spacing / new_spacing
    psf_rebinned = zoom(psf, scale_factor, order=1)
    # Normalise to conserve total energy
    total_before = psf.sum()
    total_after = psf_rebinned.sum()
    if total_after > 0:
        psf_rebinned *= (total_before / total_after)
    return psf_rebinned


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


def convert_counts_to_dn(counts_cube, qe=DEFAULT_QE, e_per_euv_ph=DEFAULT_E_PER_EUV_PH,
                         read_noise=DEFAULT_READ_NOISE_E, dn_per_e=DEFAULT_DN_PER_E,
                         dark_current=DEFAULT_DARK_CURRENT):
    """
    Convert photon counts into digital numbers (DN) via conversion to electrons.

    This function multiplies the counts by the quantum efficiency and electrons-per-photon,
    adds a constant dark current and readout noise (sampled from a Gaussian), and converts
    the resulting electrons to DN.

    Parameters:
        counts_cube : numpy.ndarray
            Cube with photon counts.
        qe : float
            Detector quantum efficiency.
        e_per_ph : float
            Electrons produced per photon.
        read_noise : float
            CCD readout noise (in electrons).
        dn_per_e : float
            Conversion factor (DN per electron).
        dark_current : float
            Dark current in electrons.
    
    Returns:
        numpy.ndarray: Data cube in DN.
    """
    # Convert counts to electrons and add dark current
    electrons = counts_cube * qe * e_per_euv_ph + dark_current
    # Add read noise
    noise = np.random.normal(loc=0.0, scale=read_noise, size=electrons.shape)
    electrons_noisy = electrons + noise
    electrons_noisy[electrons_noisy < 0] = 0.0  # no negative electrons
    # Convert to DN
    return electrons_noisy / dn_per_e


def load_muram_atmosphere(filepath, plane='xy', la_0=195.119e-8):
    """
    Load a synthesised MuRAM atmosphere from an IDL .sav file for the Fe XII 195.12 emission line.
    
    Parameters
    ----------
    filepath : str
        Path to the IDL .sav file (e.g. 'SI_Fe_XII_1952_d0_xy_0270000.sav').
    plane : str, optional
        The spatial plane to load ('xy', 'xz', or 'yz'). Default is 'xy'.
    la_0 : float, optional
        The line centre wavelength in centimetres. For Fe XII 195.12, use 195.119e-8 [cm].
    
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
    
    # Construct the expected key based on the provided plane. For example, for 'xy':
    key = f"si_{plane}_dl"  # e.g. SI_xy_dl
    if key not in data:
        raise KeyError(f"Expected key '{key}' not found in file {filepath}")
    
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
    dlar = velocity_grid / c * la_0
    
    # Create the wavelength axis in cm, centred on the line centre la_0
    wave_cm = la_0 + dlar
    
    # Convert the wavelength axis to Angstroms (1 cm = 1e8 Angstrom)
    wave_axis = wave_cm * 1e8

    # Convert to J/s/cm^2/sr/cm
    atmosphere_cube *= 1e-7

    # Convert to photon/s/cm^2/sr/cm
    PLANCK_CONSTANT = 6.62607015e-34  # Planck's constant in J.s
    SPEED_OF_LIGHT = 3e8  # Speed of light in m/s
    WAVELENGTH = 195.12 / 1e-10  # Wavelength in m
    photon_energy = PLANCK_CONSTANT * SPEED_OF_LIGHT / WAVELENGTH  # in J
    atmosphere_cube /= photon_energy

    # Convert to photon/s/pixel/cm
    pixel_size = 1.35e-4  # cm (13.5 microns)
    pixel_area = pixel_size ** 2  # cm^2
    sun_distance = 1.496e11  # m
    cube_pixel_size = 0.192 * 1e6  # m
    solid_angle = (cube_pixel_size / sun_distance) ** 2  # steradians
    atmosphere_cube *= pixel_area * solid_angle

    # Convert to photon/s/pixel
    d_wave = wave_cm[1] - wave_cm[0]  # cm
    atmosphere_cube *= d_wave

    # TEMP!!!!!!!!!!! Normalise so the maximum is 50
    atmosphere_cube *= 50.0 / np.max(atmosphere_cube)
    # TEMP!!!!!!!!!!! round everything to the nearest integer
    atmosphere_cube = np.round(atmosphere_cube).astype(np.int32)

    return atmosphere_cube, wave_axis


def analyse_spectrum_fit(wave_axis, spectrum, instrument_wave0=DEFAULT_WAVE0,
                         instrument_sigma=DEFAULT_SIGMA0):
    """
    Fit a Gaussian to a spectrum and compute the Doppler velocity and non-thermal (excess) broadening.

    The Doppler velocity is calculated as:
        v = C_SPEED * (centroid - wave0) / wave0.
    The excess broadening is calculated by subtracting the instrumental broadening in quadrature
    from the measured width.

    Parameters:
        wave_axis : numpy.ndarray
            The wavelength (or spectral bin) axis in Angstroms.
        spectrum : numpy.ndarray
            The spectrum (in DN) to be fitted.
        instrument_wave0 : float
            The rest wavelength (Angstrom).
        instrument_sigma : float
            The instrumental sigma (Angstrom).
    
    Returns:
        dict: Contains keys:
            - 'velocity' : Doppler velocity (km/s),
            - 'nonthermal' : Excess broadening (km/s),
            - 'fit_success' : Boolean flag,
            - 'fit_details' : Dictionary of the full fit result.
    """
    fit_result = fit_gaussian_profile(wave_axis, spectrum)

    if not fit_result['converged']:
        return {'velocity': np.nan, 'nonthermal': np.nan, 'fit_success': False, 'fit_details': fit_result}
    
    # Extract fitted parameters: peak, centre, sigma, background
    peak, cent, sigma, background = fit_result['params']
    # Compute Doppler velocity (km/s)
    velocity = C_SPEED * (cent - instrument_wave0) / instrument_wave0

    # Convert measured sigma (in Angstroms) to km/s
    sigma_kms = C_SPEED * sigma / instrument_wave0
    instr_sigma_kms = C_SPEED * instrument_sigma / instrument_wave0
    # Compute excess broadening via quadrature subtraction (if measured > instrumental)
    if sigma_kms > instr_sigma_kms:
        nonthermal = np.sqrt(sigma_kms**2 - instr_sigma_kms**2)
    else:
        nonthermal = 0.0

    return {'velocity': velocity, 'nonthermal': nonthermal,
            'fit_success': True, 'fit_details': fit_result}


def plot_spectra(x, x_units, y, y_units, save_path, x_fit=None, y_fit=None):
    """
    Plot the spectrum and the fitted Gaussian model.

    Parameters:
        x : numpy.ndarray
            The x-axis units (e.g. Angstrom).
        x_units : str
            Units for the x-axis.
        y : numpy.ndarray
            The y-axis units (e.g. photons).
        y_units : str
            Units for the y-axis.
        y_fit : numpy.ndarray, optional
            Fitted Gaussian model to overlay on the spectrum.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Observed Spectrum', color='blue')
    if y_fit is not None:
        plt.plot(x_fit, y_fit, label='Fitted Gaussian', color='red')
    plt.xlabel(f'{x_units}')
    plt.ylabel(f'{y_units}')
    plt.title('Spectrum and Fitted Gaussian')
    plt.legend()
    plt.grid()
    plt.savefig(f'{save_path}.png')
    plt.close()


def monte_carlo_analysis(sim_cube, wave_axis, psf, qe, e_per_ph, read_noise,
                         dn_per_e, dark_current, instrument_wave0,
                         instrument_sigma, nsim=DEFAULT_NSIM):
    """
    Perform Monte Carlo error analysis of the synthetic observation process.

    For each Monte Carlo iteration the steps are:
      1. Add Poisson noise to the simulation cube.
      2. Convolve the (spatial, spectral) planes with the PSF.
      3. Convert counts to DN (via electrons, with read noise and dark current).
      4. Fit each spectrum to obtain Doppler velocity and non-thermal broadening.
    
    Finally, mean and standard deviation maps (over iterations) of the velocity and non-thermal 
    broadening are computed.

    Parameters:
        sim_cube : numpy.ndarray
            Noise-free synthetic atmosphere cube.
        wave_axis : numpy.ndarray
            Wavelength axis (Angstrom).
        psf : numpy.ndarray
            PSF kernel (2D).
        qe, e_per_ph, read_noise, dn_per_e, dark_current : float
            Detector parameters.
        instrument_wave0 : float
            Rest wavelength (Angstrom).
        instrument_sigma : float
            Instrumental sigma (Angstrom).
        nsim : int
            Number of Monte Carlo iterations.
    
    Returns:
        dict: Contains maps for mean and uncertainty of both Doppler velocity and excess broadening.
    """
    n_scan, n_slit, n_wave = sim_cube.shape
    # Initialize storage arrays for the derived parameters over nsim iterations
    velocity_maps = np.zeros((nsim, n_scan, n_slit))
    nonthermal_maps = np.zeros((nsim, n_scan, n_slit))

    for it in tqdm(range(nsim), desc="Monte Carlo iterations", unit="iteration"):
        
        max_intensity_pix = np.unravel_index(np.argmax(np.sum(sim_cube, axis=2)), sim_cube.shape[:2])

        plot_spectra(wave_axis, "Wavelength (Angstrom)", sim_cube[max_intensity_pix[0], max_intensity_pix[1], :],
                     "Intensity (photon)", f"sim_spectra_{it}")

        # Step 1: Add Poisson noise to the spectra in each spatial position
        noisy_cube = add_poisson_noise(sim_cube)

        plot_spectra(wave_axis, "Wavelength (Angstrom)", noisy_cube[max_intensity_pix[0], max_intensity_pix[1], :],
                     "Intensity (photon)", f"noisy_spectra_{it}")

        # # Step 2: Convolve with the PSF in the (slit, spectral) plane for each scan position
        # conv_cube = convolve_cube_with_psf(noisy_cube, psf)
        conv_cube = noisy_cube.copy()

        # Step 3: Convert from counts to DN (including electrons conversion and adding noise)
        dn_cube = convert_counts_to_dn(conv_cube, qe, e_per_ph, read_noise, dn_per_e, dark_current)

        plot_spectra(wave_axis, "Wavelength (Angstrom)", dn_cube[max_intensity_pix[0], max_intensity_pix[1], :],
                     "Intensity (DN)", f"dn_spectra_{it}")

        # # Step 4: Loop over each (scan, slit) pixel and fit the spectrum
        # for i in range(n_scan):
        #     for j in range(n_slit):
        #         spectrum = dn_cube[i, j, :]
        #         fit_info = analyse_spectrum_fit(wave_axis, spectrum, instrument_wave0, instrument_sigma)
        #         velocity_maps[it, i, j] = fit_info['velocity']
        #         nonthermal_maps[it, i, j] = fit_info['nonthermal']
        i = max_intensity_pix[0]
        j = max_intensity_pix[1]
        spectrum = dn_cube[i, j, :]
        fit_info = analyse_spectrum_fit(wave_axis, spectrum, instrument_wave0, instrument_sigma)
        velocity_maps[it, i, j] = fit_info['velocity']
        nonthermal_maps[it, i, j] = fit_info['nonthermal']

        # calculate y_fit based on the fit parameters
        x_fit = np.linspace(wave_axis[0], wave_axis[-1], 1000)
        fit_params = fit_info['fit_details']['params']
        y_fit = gaussian(x_fit, *fit_params)

        plot_spectra(wave_axis, "Wavelength (Angstrom)", dn_cube[max_intensity_pix[0], max_intensity_pix[1], :],
                     "Intensity (DN)", f"fit_spectra_{it}", x_fit=x_fit, y_fit=y_fit)
        
        exit()

    # Compute mean and standard deviation (uncertainty) maps
    velocity_mean = np.nanmean(velocity_maps, axis=0)
    velocity_std = np.nanstd(velocity_maps, axis=0)
    nonthermal_mean = np.nanmean(nonthermal_maps, axis=0)
    nonthermal_std = np.nanstd(nonthermal_maps, axis=0)
    
    return {'velocity_mean': velocity_mean, 'velocity_std': velocity_std,
            'nonthermal_mean': nonthermal_mean, 'nonthermal_std': nonthermal_std}


# ---------------------------------------------------------------------------
# Main function for command-line interface and overall execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate synthetic SOLAR-C EUVST/SW observations from a MuRAM atmosphere cube."
    )
    # Simulation cube parameters
    parser.add_argument("--muram_file", type=str, default=None,
              help=f"Path to the MuRAM atmosphere cube file (e.g. SI_Fe_XII_1952_d0_xy_0270000.sav). Defailt: {None}")
    parser.add_argument("--muram_plane", type=str, default=None,
              help=f"Spatial plane to load ('xy', 'xz', or 'yz'). Default: {None}")
    # parser.add_argument("--background", type=float, default=10.0,
    #           help=f"Background photon counts (default: {10.0})")
    # Monte Carlo
    parser.add_argument("--nsim", type=int, default=DEFAULT_NSIM,
              help=f"Number of Monte Carlo simulations (default: {DEFAULT_NSIM})")
    # Detector parameters
    parser.add_argument("--qe", type=float, default=DEFAULT_QE,
              help=f"Quantum efficiency (default: {DEFAULT_QE})")
    parser.add_argument("--e_per_ph", type=float, default=DEFAULT_E_PER_EUV_PH,
              help=f"Electrons per EUV photon (default: {DEFAULT_E_PER_EUV_PH})")
    parser.add_argument("--read_noise", type=float, default=DEFAULT_READ_NOISE_E,
              help=f"Read noise in electrons (default: {DEFAULT_READ_NOISE_E})")
    parser.add_argument("--dn_per_e", type=float, default=DEFAULT_DN_PER_E,
              help=f"Digital numbers per electron (default: {DEFAULT_DN_PER_E})")
    parser.add_argument("--dark_current", type=float, default=DEFAULT_DARK_CURRENT,
              help=f"Dark current (default: {DEFAULT_DARK_CURRENT})")
    # Instrument spectral parameters
    parser.add_argument("--wave0", type=float, default=DEFAULT_WAVE0,
              help=f"Rest wavelength in Angstrom (default: {DEFAULT_WAVE0})")
    parser.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA0,
              help=f"Instrumental sigma in Angstrom (default: {DEFAULT_SIGMA0})")
    parser.add_argument("--dwave", type=float, default=DEFAULT_DWAVE,
              help=f"Spectral dispersion in Angstrom (default: {DEFAULT_DWAVE})")
    # PSF parameters
    parser.add_argument("--psf_file", type=str, default="",
              help="Path to file containing PSF data. If not provided, a Gaussian PSF is generated.")
    parser.add_argument("--psf_size", type=int, default=11,
              help=f"Size (number of pixels) of the generated PSF kernel (odd integer, default: {11})")
    parser.add_argument("--psf_sigma", type=float, default=1.5,
              help=f"Gaussian sigma for the generated PSF (default: {1.5})")
    parser.add_argument("--psf_old_spacing", type=float, default=0.6,
              help=f"Original PSF pixel spacing (microns; default: {0.6})")
    parser.add_argument("--psf_new_spacing", type=float, default=13.5,
              help=f"Detector pixel size (microns; default: {13.5})")
    
    args = parser.parse_args()

    # Load the synthetic atmosphere cube.
    if args.muram_file is None or args.muram_plane is None:
        print("Error: MuRAM file must be specified.")
        return
    if not os.path.exists(args.muram_file):
        print(f"Error: MuRAM file '{args.muram_file}' does not exist.")
        return
    sim_cube, wave_axis = load_muram_atmosphere(args.muram_file, plane=args.muram_plane)
    
    # Load or generate the PSF kernel
    if args.psf_file and os.path.exists(args.psf_file):
        print("Loading PSF from file:", args.psf_file)
        psf = np.loadtxt(args.psf_file)
    else:
        print("No PSF file provided. Generating a Gaussian PSF.")
        k = args.psf_size
        # Create a coordinate grid centred at zero
        x = np.arange(0, k) - (k - 1) / 2
        y = np.arange(0, k) - (k - 1) / 2
        X, Y = np.meshgrid(x, y)
        psf = np.exp(- (X**2 + Y**2) / (2 * args.psf_sigma**2))
        psf = psf / np.sum(psf)  # normalise the PSF
    # Rebin the PSF to match the detector pixel scale
    psf_rebinned = rebin_psf(psf, args.psf_old_spacing, args.psf_new_spacing)
    
    # Perform Monte Carlo analysis over nsim iterations
    results = monte_carlo_analysis(sim_cube, wave_axis, psf_rebinned,
                                   qe=args.qe, e_per_ph=args.e_per_ph,
                                   read_noise=args.read_noise, dn_per_e=args.dn_per_e,
                                   dark_current=args.dark_current,
                                   instrument_wave0=args.wave0, instrument_sigma=args.sigma0,
                                   nsim=args.nsim)
    
    # Plot the derived maps for Doppler velocity and non-thermal broadening.
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    im0 = axs[0, 0].imshow(results['velocity_mean'], origin='lower', cmap='RdBu_r', vmin=-30, vmax=30)
    axs[0, 0].set_title("Mean Doppler Velocity (km/s)")
    plt.colorbar(im0, ax=axs[0, 0])
    
    im1 = axs[0, 1].imshow(results['velocity_std'], origin='lower', cmap='viridis')
    axs[0, 1].set_title("Velocity Uncertainty (km/s)")
    plt.colorbar(im1, ax=axs[0, 1])
    
    im2 = axs[1, 0].imshow(results['nonthermal_mean'], origin='lower', cmap='inferno', vmin=0, vmax=30)
    axs[1, 0].set_title("Mean Non-thermal Velocity (km/s)")
    plt.colorbar(im2, ax=axs[1, 0])
    
    im3 = axs[1, 1].imshow(results['nonthermal_std'], origin='lower', cmap='magma')
    axs[1, 1].set_title("Non-thermal Velocity Uncertainty (km/s)")
    plt.colorbar(im3, ax=axs[1, 1])
    
    plt.tight_layout()
    output_fig = "synthetic_observations_results.png"
    plt.savefig(output_fig)
    plt.close()
    
    print("Simulation complete. Results saved to", output_fig)


if __name__ == "__main__":
    main()