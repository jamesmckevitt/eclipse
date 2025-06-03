import warnings, pickle, types, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.ndimage import zoom
from scipy.signal import convolve2d
from tqdm import tqdm
from joblib import Parallel, delayed
import astropy.units as u
import astropy.constants as const
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler

##############################################################################
# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------
##############################################################################

def wl_to_vel(wl: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    return (wl - wl0)/wl0 * const.c.cgs

def vel_to_wl(v: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    return (v/const.c.cgs) * wl0 + wl0

def gaussian(wave, peak, cent, sigma, back):
    """Scalar Gaussian + constant background."""
    return peak * np.exp(-0.5*((wave-cent)/sigma)**2) + back

def fano_noise(E, Fano):
    """
    Add Fano noise to an expected electron count E (integer).
    Returns a non-negative integer electron count.
    """
    sigma = np.sqrt(Fano * E)
    n    = np.random.normal(loc=E, scale=sigma)
    n    = int(round(n))
    return max(n, 0)


##############################################################################
# ---------------------------------------------------------------------------
#  PSF handling
# ---------------------------------------------------------------------------
##############################################################################

def _load_psf_ascii(fname, skip: int) -> np.ndarray:
    return np.loadtxt(fname, skiprows=skip, encoding="utf-16 LE")

def _resample_psf(psf, res_in, res_out):
    factor = res_in.to(u.cm/u.pixel).value / res_out.to(u.cm/u.pixel).value
    return zoom(psf, factor, order=1)

def _combine_psfs(psf_focus, psf_mesh, crop=0.99, size=None):
    """
    Convolve -> crop (99 % encircled energy or fixed size) -> renormalise.
    """
    psf = convolve2d(psf_focus, psf_mesh, mode="same")
    if size:                                               # centre-crop
        if size % 2 == 0: size += 1
        r0, c0 = np.array(psf.shape)//2
        half   = size//2
        psf = psf[r0-half:r0+half+1, c0-half:c0+half+1]
    else:                                                  # energy crop
        flat = psf.ravel(); idx = flat.argsort()[::-1]
        csum = flat[idx].cumsum()
        flat[flat < flat[idx[np.searchsorted(csum, flat.sum()*crop)]]] = 0
        rows, cols = np.where(flat.reshape(psf.shape))
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        side   = max(r1-r0, c1-c0)+1
        r0 = (r0+r1)//2 - side//2
        c0 = (c0+c1)//2 - side//2
        psf = psf[r0:r0+side, c0:c0+side]
    return psf / psf.sum()


##############################################################################
# ---------------------------------------------------------------------------
#  Atmosphere I/O & resampling
# ---------------------------------------------------------------------------
##############################################################################

def load_atmosphere(pkl_file: str):
    """Load synthetic cube (erg s-1 cm-2 sr-1 cm-1) and metadata."""
    with open(pkl_file, "rb") as f:
        dat = pickle.load(f)
    cube      = dat["I_cube"] * (u.erg/u.s/u.cm**2/u.sr/u.cm)
    wl_grid   = dat["wl_grid"] * u.cm
    spt_res   = dat["spt_res_x"] * u.cm
    wl0       = 195.119 * u.AA
    del dat
    return cube, wl_grid, spt_res, wl0.cgs

def rebin_atmosphere(cube_sim, wl_sim, spt_sim,
                     wl_pitch_out, spt_pitch_out):
    """
    * spectral: flux-conserving resample to instrument spectral resolution
    * spatial : bilinear zoom to instrument pixel scale
    """
    wl_out = np.arange(wl_sim[0].value,
                       wl_sim[-1].value + wl_pitch_out.value,
                       wl_pitch_out.value) * u.cm

    resampler = FluxConservingResampler(extrapolation_treatment='nan_fill')

    def _do_scan(i):
        row = np.empty((cube_sim.shape[1], len(wl_out)))
        for j in range(cube_sim.shape[1]):
            spec = Spectrum1D(cube_sim[i,j,:], spectral_axis=wl_sim)
            row[j,:] = resampler(spec, wl_out).flux.value
        return row

    rows = Parallel(n_jobs=-1)(delayed(_do_scan)(i)
                               for i in tqdm(range(cube_sim.shape[0]),
                               desc="Spectral rebin", unit="scan", leave=False))
    cube = np.stack(rows)*cube_sim.unit
    cube = cube[:,:,~np.isnan(cube[0,0,:])]
    wl_out = wl_out[~np.isnan(wl_out)]

    # spatial zoom
    factor = (spt_sim/spt_pitch_out).decompose().value
    cube = zoom(cube, (factor, factor, 1), order=1)

    return cube, wl_out


##############################################################################
# ---------------------------------------------------------------------------
#  Radiometric pipeline
# ---------------------------------------------------------------------------
##############################################################################

def intensity_to_photons(I, wl0):
    """erg s-1 cm-2 sr-1 cm-1  →  photons s-1 cm-2 sr-1 cm-1"""
    E_ph = (const.h*const.c/wl0).to("erg")      # photon energy (erg)
    return (I.to("erg/(s cm2 sr cm)") / E_ph).to(1/u.s/u.cm**2/u.sr/u.cm)

def add_effective_area(ph_cm2_sr_cm_s):
    """x A_eff ⇒ photons s-1 sr-1 cm-1"""
    A_eff = TEL["collecting_area"].cgs * TEL["pm_eff"] * TEL["grat_eff"] * TEL["filt_eff"]
    return ph_cm2_sr_cm_s * A_eff

def photons_to_pixel_rate(ph_sr_cm_s, wl_pitch, spt_pitch):
    """
    x Ω_pix x Δλ_pix   →   photons s-1 pix-1 (before exposure time)
    """
    Ω_pix = (spt_pitch**2/const.au.cgs**2).to(u.sr)
    return ph_sr_cm_s * Ω_pix * wl_pitch

def to_electrons(photon_rate, t_exp):
    """Apply QE, e- per photon, dark + read noise."""
    e_per_ph = fano_noise(DET["e_per_ph_euv"].value, DET["si_fano"]) * u.electron/u.photon
    e  = photon_rate * t_exp * DET["qe_euv"] * e_per_ph
    e += DET["dark_current"] * t_exp
    e += np.random.normal(0, DET["read_noise_rms"].value, photon_rate.shape) * u.electron
    e[e<0] = 0
    return e

def to_dn(electrons):
    return electrons * DET["gain_dn_per_e"]


##############################################################################
# ---------------------------------------------------------------------------
#  Noise & stray-light models
# ---------------------------------------------------------------------------
##############################################################################

def add_poisson(data):
    """Poisson deviate for any Quantity with photon units."""
    unit = data.unit
    return np.random.poisson(data.value) * unit

def add_stray_light(electrons, t_exp):
    """Visible photons hitting the detector, converted to additional e-."""
    n_vis_ph = np.random.poisson((1*u.photon/u.s/u.pixel * t_exp).value,
                                 size=electrons.shape) * u.photon
    e_per_ph = fano_noise(DET["e_per_ph_vis"].value, DET["si_fano"]) * u.photon
    return electrons + n_vis_ph * DET["e_per_ph_vis"] * DET["qe_vis"]


##############################################################################
# ---------------------------------------------------------------------------
#  Spectral fitting (per-pixel Gaussian)
# ---------------------------------------------------------------------------
##############################################################################

def _guess_params(wv, prof):
    back = prof.min()
    prof_c = np.maximum(prof-back, 0)
    peak = prof_c.max()
    centre = np.average(wv, weights=prof_c) if prof_c.sum() else wv[len(wv)//2]
    sigma = np.abs(np.diff(wv).mean()) * prof_c.sum() / (peak*np.sqrt(2*np.pi))
    return [peak, centre, sigma, back]

def _fit_one(wv, prof):
    p0   = _guess_params(wv, prof)
    sig  = np.sqrt(np.maximum(prof,1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt,_ = curve_fit(gaussian, wv, prof, p0=p0, sigma=sig)
    return popt                  # (peak, centre, sigma, back)

def fit_cube_gauss(dn_cube: np.ndarray, wv: np.ndarray, n_jobs=-1):
    n_scan, n_slit, _ = dn_cube.shape
    def _scan(i):
        return np.array([_fit_one(wv, dn_cube[i,j,:]) for j in range(n_slit)])
    rows = Parallel(n_jobs=n_jobs)(delayed(_scan)(i)
                                   for i in tqdm(range(n_scan),
                                   desc="Gaussian fits", unit="scan", leave=False))
    return np.stack(rows)        # shape (n_scan,n_slit,4)


##############################################################################
# ---------------------------------------------------------------------------
#  Monte-Carlo wrapper
# ---------------------------------------------------------------------------
##############################################################################

def simulate_once(I_cube, wl0, psf, t_exp):
    """One detector realisation (returns DN cube)."""
    photons0 = intensity_to_photons(I_cube, wl0)
    photons0 = add_effective_area(photons0)
    photons0 = photons_to_pixel_rate(photons0, DET["wvl_res"], DET["spt_res"])

    photons0 = add_poisson(photons0)           # shot noise
    # Optical PSF
    n_scan,n_slit,_ = photons0.shape
    photons_blur = np.empty_like(photons0)
    for i in range(n_scan):
        photons_blur[i,:,:] = convolve2d(photons0[i,:,:], psf, mode="same")

    e_cube = to_electrons(photons_blur, t_exp)
    e_cube = add_stray_light(e_cube, t_exp)
    dn_cube = to_dn(e_cube)

    return dn_cube, photons0, photons_blur, e_cube

def monte_carlo(I_cube, wl_axis, wl0, psf, t_exp, n_iter=5):
    """
    Repeat the entire detector path N times.
    """
    fits, dn_all = [], []
    for _ in tqdm(range(n_iter), desc="Monte-Carlo", unit="iter", leave=False):
        dn, *_ = simulate_once(I_cube, wl0, psf, t_exp)
        dn_all.append(dn)
        fits.append(fit_cube_gauss(dn.value, wl_axis.value))
    return np.array(dn_all), np.array(fits)


##############################################################################
# ---------------------------------------------------------------------------
#  Analysis metrics
# ---------------------------------------------------------------------------
##############################################################################

def velocity_from_fit(fit_arr, wl0):
    centres = fit_arr[...,1]
    return (centres - wl0.value)/wl0.value * const.c.cgs.value   # cm/s

def analyse(fits_all, fits_true, wl0):
    v_true  = velocity_from_fit(fits_true, wl0)
    v_all   = velocity_from_fit(fits_all, wl0)   # shape (N_iter, n_scan, n_slit)
    v_mean  = v_all.mean(axis=0)
    v_std   = v_all.std(axis=0)
    v_err   = v_true - v_mean
    return dict(v_mean=v_mean, v_std=v_std, v_err=v_err,
                v_samples=v_all, v_true=v_true)


##############################################################################
# ---------------------------------------------------------------------------
#                 M A I N   W O R K F L O W
# ---------------------------------------------------------------------------
##############################################################################

def main():

    # ---------------- user-tunable parameters -----------------
    # Detector
    DET = dict(
        qe_vis          = 1.00,                           # quantum efficiency (visible light)
        qe_euv          = 0.76,                           # quantum efficiency (195 A)
        e_per_ph_euv    =  18.0 * u.electron/u.photon,    # electrons per EUV photon
        e_per_ph_vis    =   2.0 * u.electron/u.photon,    # electrons per visible photon
        read_noise_rms  =  10.0 * u.electron/u.pixel,     # read noise RMS
        dark_current    =   1.0 * u.electron/u.pixel/u.s, # dark current
        gain_dn_per_e   =  19.0 * u.DN/u.electron,        # gain
        pix_size        = (13.5 * u.um).cgs/u.pixel,      # pixel size
        wvl_res         = (16.9 * u.mAA).cgs/u.pixel,     # spectral resolution
        spt_res         = ( 1.5 * u.arcsec).to(u.rad),    # spatial resolution
        si_fano         = 0.115                           # Fano factor for silicon
    )
    # Telescope
    TEL = dict(
        D_ap           = 0.28     * u.m,          # entrance pupil diameter
        pm_eff         = 0.161,                   # primary-mirror reflectivity EUV
        grat_eff       = 0.0623,                  # grating efficiency EUV
        filt_eff       = 0.507,                   # mesh filter transmission
        psf_focus_res  = 0.5      * u.um/u.pixel, # sampling of files
        psf_mesh_res   = 6.12e-4  * u.mm/u.pixel,
        psf_focus_file = Path("data/swc/psf_euvst_v20230909_195119_focus.txt"),
        psf_mesh_file  = Path("data/swc/psf_euvst_v20230909_derived_195119_mesh.txt")
    )
    TEL["collecting_area"] = 0.5 * np.pi * (TEL["D_ap"]/2)**2      # half pupil
    # Simulation
    SIM = dict(
        expos = [1, 2, 5, 10, 20] * u.s,    # exposure times
        n_iter = 50                         # number of Monte-Carlo iterations per exposure
    )
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # PSF
    # ----------------------------------------------------------
    psf_focus = _load_psf_ascii(TEL["psf_focus_file"], skip=21)
    psf_mesh  = _load_psf_ascii(TEL["psf_mesh_file"],  skip=16)
    psf_focus = _resample_psf(psf_focus, TEL["psf_focus_res"], DET["pix_size"])
    psf_mesh  = _resample_psf(psf_mesh,  TEL["psf_mesh_res"],  DET["pix_size"])
    psf       = _combine_psfs(psf_focus, psf_mesh, size=5)

    # ----------------------------------------------------------
    # atmosphere
    # ----------------------------------------------------------
    cube_sim, wl_sim, spt_sim, wl0 = load_atmosphere("_I_cube.npz")
    cube_reb, wl_axis = rebin_atmosphere(cube_sim, wl_sim, spt_sim,
                                         DET["wvl_res"], DET["spt_res"])

    # "Ground truth" Gaussian fits (no instrument effects)
    fit_truth = fit_cube_gauss(cube_sim.value, wl_sim.value)

    # ----------------------------------------------------------
    # monte-carlo
    # ----------------------------------------------------------
    for t_exp in SIM["expos"]:
        dn_all, fits_all = monte_carlo(cube_reb, wl_axis, wl0, psf, t_exp, n_iter=SIM["n_iter"])
        stats = analyse(fits_all, fit_truth, wl0)

        # --- simple printout summarising errors ------------------------
        rms_err = np.sqrt(np.mean(stats["v_err"]**2)) * u.cm/u.s
        print(f"{t_exp.value:4.1f} s  :  RMS velocity error = "
              f"{rms_err.to(u.km/u.s):4.2f}")

        # --- save everything for later notebooks ----------------------
        out = dict(dn_all=dn_all, fits_all=fits_all, wl_axis=wl_axis,
                   stats=stats, t_exp=t_exp, wl0=wl0,
                   cube_reb=cube_reb, cube_sim=cube_sim)
        with open(f"_response_{t_exp.value:.0f}s.pkl", "wb") as f:
            pickle.dump(out, f)

    print("Done.")

    globals().update(locals());raise ValueError("Kicking back to ipython")
if __name__ == "__main__":
    main()
