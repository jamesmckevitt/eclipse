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
from tqdm import tqdm
from joblib import Parallel, delayed

##############################################################################
# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------
##############################################################################

def wl_to_vel(wl: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    return (wl - wl0)/wl0 * const.c

def vel_to_wl(v: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    return (v/const.c) * wl0 + wl0

def gaussian(wave, peak, cent, sigma, back):
    return peak * np.exp(-.5*((wave-cent)/sigma)**2) + back

def fano_noise(E, Fano):
    sigma = np.sqrt(Fano * E)
    n    = np.random.normal(loc=E, scale=sigma)
    n    = int(round(n))
    return max(n, 0)

def angle_to_distance(angle: u.Quantity) -> u.Quantity:
    if angle.unit.physical_type != "angle": raise ValueError("Input must be an angle")
    return 2 * const.au * np.tan(angle.to(u.rad) / 2)


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

def slit_sample_cube(
    cube_hr:        u.Quantity,
    dx_cm:          float,
    slit_w_cm:      float,
    scan_step_cm:   float,
) -> u.Quantity:
    """
    Slide a slit along X, integrate the high-resolution cube inside
    that window, and output the *mean specific intensity* for every scan position.

    No rounding of indices is performed; sub-pixel edges are handled
    via linear interpolation of the cumulative integral.

    Parameters
    ----------
    cube_hr      : ndarray (Nx, Ny, Nλ) with intensity units
    dx_cm        : float  - simulation pixel pitch along X  [cm]
    slit_w_cm    : float  - slit physical width             [cm]
    scan_step_cm : float  - scan step between exposures     [cm]

    Returns
    -------
    cube_out : ndarray (Nscan, Ny, Nλ)  - mean intensity per slit
    """
    Nx, Ny, Nl = cube_hr.shape
    # ------------------------------------------------------------------
    # Prefix integral along X  (axis 0);  prefix[0] = 0 for convenience
    # ------------------------------------------------------------------
    prefix = np.concatenate(
        [np.zeros((1, Ny, Nl), dtype=cube_hr.dtype),   # x = 0 edge
         np.cumsum(cube_hr, axis=0) * dx_cm],          # ∑ I·Δx
        axis=0
    )                                  # shape (Nx+1, Ny, Nλ)

    def interp_prefix(pos_cm: float):
        """Linear interp of prefix at arbitrary X-position (cm)."""
        pos_idx = pos_cm / dx_cm              # fractional index
        i0      = np.floor(pos_idx).astype(int)
        alpha   = pos_idx - i0
        i0      = np.clip(i0, 0, Nx)          # safety
        i1      = np.clip(i0+1, 0, Nx)
        return (1-alpha)*prefix[i0] + alpha*prefix[i1]

    # ------------------------------------------------------------------
    # Scan positions (left edge of slit for every exposure)
    # ------------------------------------------------------------------
    left_edges = np.arange(0, (Nx*dx_cm - slit_w_cm) + 1e-9, scan_step_cm)
    Nscan      = len(left_edges)
    cube_out   = np.empty((Nscan, Ny, Nl), dtype=cube_hr.dtype)

    for s, L in enumerate(left_edges):
        R          = L + slit_w_cm                       # right edge
        integral   = interp_prefix(R) - interp_prefix(L) # ∫_L^R I dx
        cube_out[s] = integral / slit_w_cm               # mean intensity

    return cube_out * cube_hr.unit

def load_atmosphere(npz_file: str):
    """Load synthetic cube (erg s-1 cm-2 sr-1 cm-1) and metadata."""
    dat = np.load(npz_file)
    cube      = dat["I_cube"] * (u.erg/u.s/u.cm**2/u.sr/u.cm)
    wl_grid   = dat["wl_grid"] * u.cm
    spt_res   = dat["spt_res_x"] * u.cm
    wl0       = dat["wl0"] * u.cm
    plotting = {}
    plotting["mean_idx"]  = dat["mean_idx"]
    plotting["minus_idx"] = dat["minus_idx"]
    plotting["plus_idx"]  = dat["plus_idx"]
    plotting["sigma_factor"] = dat["sigma_factor"]
    plotting["margin"] = dat["margin"]
    return cube, wl_grid, spt_res, wl0, plotting

def rebin_atmosphere(
    cube_sim:        u.Quantity,
    wl_sim:          u.Quantity,
    spt_sim:         u.Quantity,
):
    """
    1. Flux-conserving spectral resample
    2. Slide slit across X (scan direction)
    3. Bin/zoom along Y to the detector plate scale.

    Returns
    -------
    cube_out : ndarray (nscan, ny_det, nwvl_det)
    wl_det   : 1-D wavelength grid after spectral rebinning
    """
    wl_pitch_out = DET["wvl_res"]
    y_pitch_out  = DET["plate_scale_length"]
    slit_width_as = SIM["slit_width"]
    scan_step_as  = SIM["slit_scan_step"]

    # -------- spectral resample ----------------------------
    print("  Flux-conserving spectral resampling...")
    wl_det = np.arange(wl_sim[0].value,
                       wl_sim[-1].value + (wl_pitch_out*u.pixel).cgs.value,
                       (wl_pitch_out*u.pixel).cgs.value) * wl_sim.unit

    resampler = FluxConservingResampler(extrapolation_treatment='zero_fill')

    def _reb_spec(i):
        row = np.zeros((cube_sim.shape[1], len(wl_det))) * cube_sim.unit
        for j in range(cube_sim.shape[1]):
            spec = Spectrum1D(cube_sim[i, j, :], spectral_axis=wl_sim)
            row[j, :] = resampler(spec, wl_det).flux
        return row

    rows = Parallel(n_jobs=-1)(
        delayed(_reb_spec)(i)
        for i in tqdm(range(cube_sim.shape[0]), desc="Spectral rebin", unit="slice", leave=False)
    )
    print("  Stacking rows...")
    cube_spec = np.stack(rows)  # nx_sim, ny_sim, nwvl_det

    # -------- slit scanning along X -----------------------------------
    print("  Moving slit across X...")
    cube_scan = slit_sample_cube(
        cube_spec,
        spt_sim.cgs.value,
        angle_to_distance(slit_width_as).cgs.value,
        angle_to_distance(scan_step_as).cgs.value
        )  # nscan, ny_sim, nwvl_det

    # -------- rebin along Y to detector plate-scale -------------------
    print("  Rebinning along Y to detector plate scale...")
    factor_y = (spt_sim / y_pitch_out).decompose().value
    cube_det = zoom(cube_scan, (1, factor_y, 1), order=1) * cube_scan.unit  # nscan, ny_det, nwvl_det

    return cube_det, wl_det

def _find_new_key_idxs(cube_sim, spt_sim, plotting):
    """
    Find the new indices for the key pixels in the rebinned cube.
    """
    return plotting


##############################################################################
# ---------------------------------------------------------------------------
#  Radiometric pipeline
# ---------------------------------------------------------------------------
##############################################################################

def intensity_to_photons(I, wl_axis):
    """erg/s/cm2/sr/cm to ph/s/cm2/sr/cm"""
    E_ph = (const.h * const.c / wl_axis).to("erg") * (1 / u.ph)
    return (I / E_ph).to(u.ph/u.s/u.cm**2/u.sr/u.cm)

def add_effective_area(ph_cm2_sr_cm_s):
    """ph/s/cm2/sr/cm to ph/s/sr/cm"""
    A_eff = TEL["collecting_area"].cgs * TEL["pm_eff"] * TEL["grat_eff"] * TEL["filt_eff"]
    return ph_cm2_sr_cm_s * A_eff

def photons_to_pixel_rate(ph_sr_cm_s, wl_pitch, spt_pitch):
    """ph/s/sr/cm to ph/s/pix"""
    pixel_solid_angle = (( spt_pitch * u.pix )**2 / const.au**2).cgs.value * (u.sr)
    return ph_sr_cm_s * pixel_solid_angle * wl_pitch

def apply_psf(signal, psf):
    """
    Apply the optical PSF to each scan position in the signal cube.
    """
    n_scan, n_slit, _ = signal.shape
    blurred = np.empty_like(signal)
    for i in range(n_scan):
        blurred[i, :, :] = convolve2d(signal[i, :, :].value, psf, mode="same") * signal.unit
    return blurred

def to_electrons(photon_rate, t_exp):
    """ph/s/pix to e-/pix"""
    e_per_ph = fano_noise(DET["e_per_ph_euv"].value, DET["si_fano"]) * u.electron/u.photon
    e  = photon_rate * t_exp * DET["qe_euv"] * e_per_ph
    e += DET["dark_current"] * t_exp   ### ADD POISSON NOISE TO DARK CURRENT
    e += np.random.normal(0, DET["read_noise_rms"].value, photon_rate.shape) * (u.electron/u.pix)
    e[e<0] = 0
    return e

def to_dn(electrons):
    """e-/pix to DN/pix"""
    dn = electrons / DET["gain_e_per_dn"]
    dn[dn > DET["max_dn"]] = DET["max_dn"]  # clamp to max DN value
    return dn


##############################################################################
# ---------------------------------------------------------------------------
#  Noise & stray-light models
# ---------------------------------------------------------------------------
##############################################################################

def add_poisson(data):
    """Poisson deviate for any Quantity with units if present."""
    if isinstance(data, u.Quantity):
        unit = data.unit
        return np.random.poisson(data.value) * unit
    else:
        return np.random.poisson(data)

def add_stray_light(electrons, t_exp):
    """Visible photons hitting the detector, converted to additional e-."""
    n_vis_ph = np.random.poisson((1*u.photon/u.s/u.pixel * t_exp).value,
                                 size=electrons.shape) * (u.photon/u.pix)
    e_per_ph = fano_noise(DET["e_per_ph_vis"].value, DET["si_fano"]) * (u.electron/u.photon)
    return electrons + n_vis_ph * e_per_ph * DET["qe_vis"]


##############################################################################
# ---------------------------------------------------------------------------
#  Spectral fitting (per-pixel Gaussian)
# ---------------------------------------------------------------------------
##############################################################################

def _guess_params(wv, prof):
    back = prof.min()
    prof_c = prof - back
    prof_c[prof_c < 0] = 0
    peak = prof_c.max()
    centre = wv[np.nanargmax(prof_c)]
    sigma = np.trapezoid(prof_c, wv) / (peak * np.sqrt(2*np.pi))
    return [peak, centre, sigma, back]

def _fit_one(wv, prof):
    p0   = _guess_params(wv, prof)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt, _ = curve_fit(
                gaussian,
                wv,
                prof,
                p0=p0,
                maxfev=5000
            )
            # return [
            #     popt[0] * prof.unit,  # peak
            #     popt[1] * wv.unit,    # centre
            #     popt[2] * wv.unit,    # sigma
            #     popt[3] * prof.unit    # background
            # ]
            return popt
        except RuntimeError:
            # return [
            #     -1 * prof.unit,  # peak
            #     -1 * wv.unit,    # centre
            #     -1 * wv.unit,    # sigma
            #     -1 * prof.unit   # background
            # ]
            return np.array([-1, -1, -1, -1])

def fit_cube_gauss(signal_cube: np.ndarray, wv: np.ndarray, n_jobs=-1):
    n_scan, n_slit, _ = signal_cube.shape
    signal_unit = signal_cube.unit
    wv_unit = wv.unit
    def _scan(i):
        return np.array([_fit_one(wv.value, signal_cube.value[i,j,:]) for j in range(n_slit)])
    rows = Parallel(n_jobs=n_jobs)(delayed(_scan)(i)
                                   for i in tqdm(range(n_scan),
                                   desc="Gaussian fits", unit="scan", leave=False))
    return np.array(rows) * np.array([
        signal_unit, wv_unit, wv_unit, signal_unit
    ])  # shape (n_scan,n_slit,4) with units


##############################################################################
# ---------------------------------------------------------------------------
#  Monte-Carlo wrapper
# ---------------------------------------------------------------------------
##############################################################################

def simulate_once(I_cube, wl_axis, t_exp):
    """One detector realisation (returns DN cube)."""
    signal0 = add_poisson(I_cube)
    signal1 = intensity_to_photons(signal0, wl_axis)
    signal2 = add_effective_area(signal1)
    signal3 = photons_to_pixel_rate(signal2, DET["wvl_res"], DET["plate_scale_length"])
    signal4 = apply_psf(signal3, TEL["psf"])
    signal5 = to_electrons(signal4, t_exp)
    signal6 = add_stray_light(signal5, t_exp)
    signal7 = to_dn(signal6)

    return signal0, signal1, signal2, signal3, signal4, signal5, signal6, signal7

def monte_carlo(I_cube, wl_axis, t_exp, n_iter=5):
    """
    Repeat the entire detector path N times.
    """
    signals, fits = [], []
    for _ in tqdm(range(n_iter), desc="Monte-Carlo", unit="iter", leave=False):
        signals += [simulate_once(I_cube, wl_axis, t_exp)]
        fits += [fit_cube_gauss(signals[-1][-1], wl_axis.cgs)]
    return np.array(signals), np.array(fits)


##############################################################################
# ---------------------------------------------------------------------------
#  Analysis metrics
# ---------------------------------------------------------------------------
##############################################################################

def velocity_from_fit(fit_arr, wl0):

    # take the unit from applying to each value individually, to applying to the whole array
    arr = fit_arr[..., 1]
    unit = arr.flat[0].unit
    values = np.vectorize(lambda q: q.value)(arr)
    centres = values * unit

    mask_all_minus1 = np.all(fit_arr == -1, axis=-1)
    v = (centres - wl0) / wl0 * const.c  # cm/s
    v = np.where(mask_all_minus1, -1 * u.cm / u.s, v)
    return v

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
#  Output
# ---------------------------------------------------------------------------
##############################################################################

def plot_radiometric_pipeline(
    signals, wl_axis,
    idx_sim_minus, idx_sim_mean, idx_sim_plus,
    spt_pitch_sim, spt_pitch_instr,
    save="fig_radiometric_pipeline.png",
    row_labels=(r"$\mu-\sigma$", r"$\mu$", r"$\mu+\sigma$"),
    key_pixel_colors=("mediumseagreen", "black", "deeppink"),
):
    """
    Plot the evolution of the radiometric pipeline.

    Four columns, three rows (µ-σ, µ, µ+σ).
    Columns:
      0. signal1 / signal2 / signal3  – triple y-axis
      1. signal4                     – single y-axis
      2. signal5                     – single y-axis
      3. signal6 / signal7           – dual y-axis
    """

    factor = (spt_pitch_sim / spt_pitch_instr).decompose().value
    def _map(idx_sim):
        return tuple(int(round(i * factor)) for i in idx_sim)

    idxs_reb = tuple(_map(x) for x in (idx_sim_minus, idx_sim_mean, idx_sim_plus))
    nx_reb, ny_reb = signals[1].shape[:2]
    for ix, iy in idxs_reb:
        if (ix < 0 or ix >= nx_reb) or (iy < 0 or iy >= ny_reb):
            raise IndexError("Mapped index out of range.")

    wl_A = wl_axis.to(u.angstrom).value

    # Helper – extract (λ) spectra for one spatial pixel
    def spectrum(stage_idx, row_idx):
        return signals[stage_idx][idxs_reb[row_idx] + (slice(None),)]

    # ------------------------------------------------------------------
    # 1. Figure & main grid
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        3, 4, figsize=(10, 6),
        sharex="row",
        constrained_layout=True
    )
    fig.subplots_adjust(right=0.86, wspace=0.18, hspace=0.06)

    # ------------------------------------------------------------------
    # 2. Loop over rows
    # ------------------------------------------------------------------
    for row in range(3):
        colour = key_pixel_colors[row]

        # Row label
        lab_ax = axes[row, 0].inset_axes([-0.42, 0, 0.1, 1], frameon=False)
        lab_ax.set_axis_off()
        lab_ax.text(0, 0.5, row_labels[row], va="center", ha="left", rotation=90, fontsize=9)

        # Column 0 – signal1/2/3
        ax0 = axes[row, 0]
        sp1 = spectrum(1, row)
        ax0.step(wl_A, sp1, where="mid", color=colour, lw=1)
        if row == 0:
            ax0.set_title("signal1/2/3", fontsize=8)
        ax0.set_ylabel(r"ph s$^{-1}$ cm$^{-2}$ sr$^{-1}$ cm$^{-1}$", color=colour, fontsize=7)
        ax0.tick_params(direction="in", which="both", top=True, right=True)

        # 2nd y-axis: signal2
        ax_r1 = ax0.twinx()
        sp2 = spectrum(2, row)
        ax_r1.step(wl_A, sp2, where="mid", color="tab:orange", lw=1)
        ax_r1.set_ylim(sp2.min(), sp2.max())
        ax_r1.set_ylabel(r"ph s$^{-1}$ sr$^{-1}$ cm$^{-1}$", color="tab:orange", fontsize=7)
        ax_r1.yaxis.labelpad = 8
        ax_r1.tick_params(direction="in", colors="tab:orange", which="both", right=True)
        ax_r1.patch.set_visible(False)

        # 3rd y-axis: signal3
        ax_r2 = ax0.twinx()
        sp3 = spectrum(3, row)
        ax_r2.step(wl_A, sp3, where="mid", color="tab:blue", lw=1)
        ax_r2.set_ylim(sp3.min(), sp3.max())
        ax_r2.spines.right.set_position(("axes", 1.15))
        ax_r2.set_ylabel(r"ph s$^{-1}$ pix$^{-1}$", color="tab:blue", fontsize=7)
        ax_r2.yaxis.labelpad = 24
        ax_r2.tick_params(direction="in", colors="tab:blue", which="both", right=True)
        ax_r2.patch.set_visible(False)

        # Column 1 – signal4
        ax1 = axes[row, 1]
        sp4 = spectrum(4, row)
        ax1.step(wl_A, sp4, where="mid", color=colour, lw=1)
        if row == 0:
            ax1.set_title("signal4", fontsize=8)
        ax1.set_ylabel(r"ph s$^{-1}$ pix$^{-1}$ (PSF)", color=colour, fontsize=7)
        ax1.tick_params(direction="in", which="both", top=True, right=True)

        # Column 2 – signal5
        ax2 = axes[row, 2]
        sp5 = spectrum(5, row)
        ax2.step(wl_A, sp5, where="mid", color=colour, lw=1)
        if row == 0:
            ax2.set_title("signal5", fontsize=8)
        ax2.set_ylabel(r"e$^-$ pix$^{-1}$", color=colour, fontsize=7)
        ax2.tick_params(direction="in", which="both", top=True, right=True)

        # Column 3 – signal6/7
        ax3 = axes[row, 3]
        sp6 = spectrum(6, row)
        ax3.step(wl_A, sp6, where="mid", color=colour, lw=1)
        if row == 0:
            ax3.set_title("signal6/7", fontsize=8)
        ax3.set_ylabel(r"e$^-$ pix$^{-1}$ (stray)", color=colour, fontsize=7)
        ax3.tick_params(direction="in", which="both", top=True, right=True)

        ax_r3 = ax3.twinx()
        sp7 = spectrum(7, row)
        ax_r3.step(wl_A, sp7, where="mid", color="tab:red", lw=1)
        ax_r3.spines.right.set_position(("axes", 1.12))
        ax_r3.set_ylim(sp7.min(), sp7.max())
        ax_r3.set_ylabel(r"DN pix$^{-1}$", color="tab:red", fontsize=7)
        ax_r3.yaxis.labelpad = 16
        ax_r3.tick_params(direction="in", colors="tab:red", which="both", right=True)
        ax_r3.patch.set_visible(False)

        # Bottom x-label
        if row == 2:
            for col in range(4):
                axes[row, col].set_xlabel("Wavelength [Å]")

    # ------------------------------------------------------------------
    # 3. Save and close
    # ------------------------------------------------------------------
    fig.savefig(save, dpi=300)
    plt.close(fig)
    return fig


def plot_maps(
    signal_cube, wl_axis, idx_sim_minus, idx_sim_mean, idx_sim_plus, save,
    key_pixel_colors=("mediumseagreen", "black", "deeppink"),
):
    """
    Plot intensity and Doppler velocity maps from a signal cube (e.g., first MC iteration).
    Marks mean, plus, minus sigma pixels if provided.
    """

    # Intensity map: integrate over wavelength
    si = signal_cube.sum(axis=2)
    log_si = np.log10(si, where=si > 0.0, out=np.zeros_like(si))

    # Velocity map: find peak wavelength index, convert to velocity
    peak_idx = signal_cube.argmax(axis=2)
    wl0 = wl_axis[len(wl_axis)//2]  # crude guess: central wavelength
    c = 2.99792458e10  # cm/s
    v_map = (wl_axis[peak_idx] - wl0) / wl0 * c / 1e5  # km/s

    fig, (axI, axV) = plt.subplots(1, 2, figsize=(10, 5))

    imI = axI.imshow(log_si.T, origin="lower", aspect="equal", cmap="afmhot")
    axI.set_title(r"$\\log_{10} \\int I(\\lambda) d\\lambda$")
    fig.colorbar(imI, ax=axI, orientation="horizontal", label="log10 Intensity")

    imV = axV.imshow(v_map.T, origin="lower", aspect="equal", cmap="RdBu_r", vmin=-15, vmax=15)
    axV.set_title("Doppler velocity of peak intensity [km/s]")
    fig.colorbar(imV, ax=axV, orientation="horizontal", label="v [km/s]")

    # Markers for key pixels
    for idx, color in zip([idx_sim_minus, idx_sim_mean, idx_sim_plus], key_pixel_colors):
        if idx is not None:
            axI.plot(idx[0], idx[1], marker="o", color=color, markersize=8, fillstyle="none", lw=2)
            axV.plot(idx[0], idx[1], marker="o", color=color, markersize=8, fillstyle="none", lw=2)

    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close(fig)


##############################################################################
# ---------------------------------------------------------------------------
#                 M A I N   W O R K F L O W
# ---------------------------------------------------------------------------
##############################################################################

def main():

    # ---------------- user-tunable parameters -----------------

    # Detector
    DET = dict(
        qe_vis            =  1.00,                          # quantum efficiency (visible light)
        qe_euv            =  0.76,                          # quantum efficiency (195 A)
        e_per_ph_euv      =  18.0 * u.electron/u.photon,    # electrons per EUV photon
        e_per_ph_vis      =   2.0 * u.electron/u.photon,    # electrons per visible photon
        read_noise_rms    =  10.0 * u.electron/u.pixel,     # read noise RMS
        dark_current      =   1.0 * u.electron/u.pixel/u.s, # dark current
        gain_e_per_dn     =   2.0 * u.electron/u.DN,        # gain
        max_dn            = 65535 * u.DN/u.pixel,           # maximum DN value
        pix_size          = (13.5 * u.um).cgs/u.pixel,      # pixel size
        wvl_res           = (16.9 * u.mAA).cgs/u.pixel,     # spectral resolution
        plate_scale_angle = 0.159 * u.arcsec/u.pixel,       # plate scale (angle)
        si_fano           = 0.115                           # Fano factor for silicon
    )
    DET["plate_scale_length"] = (2 * const.au * np.tan(( 1*u.pix*DET["plate_scale_angle"] )/2)).to(u.cm) / u.pixel # plate scale (length at sun)

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
        expos          = [1, 2, 5, 10, 20] * u.s,     # exposure times
        n_iter         = 50,                          # number of Monte-Carlo iterations per exposure
        vis_sl         = 1 * (u.photon/u.s/u.pixel),  # stray light (visible photons)
        slit_width     = 0.2 * u.arcsec,              # slit width
        slit_scan_step = 0.2 * u.arcsec,              # slit scan step
    )

    globals().update(DET=DET, TEL=TEL, SIM=SIM)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # PSF
    # ----------------------------------------------------------
    print("Loading PSF files...")
    psf_focus = _load_psf_ascii(TEL["psf_focus_file"], skip=21)
    psf_mesh  = _load_psf_ascii(TEL["psf_mesh_file"],  skip=16)
    psf_focus = _resample_psf(psf_focus, TEL["psf_focus_res"], DET["pix_size"])
    psf_mesh  = _resample_psf(psf_mesh,  TEL["psf_mesh_res"],  DET["pix_size"])
    psf       = _combine_psfs(psf_focus, psf_mesh, size=5)
    TEL["psf"] = psf; globals().update(DET=DET, TEL=TEL, SIM=SIM)

    # ----------------------------------------------------------
    # atmosphere
    # ----------------------------------------------------------

    print("Loading atmosphere...")
    cube_sim, wl_sim, spt_sim, wl0, plotting = load_atmosphere("synthesised_spectra.npz")
    plotting = _find_new_key_idxs(cube_sim, spt_sim, plotting)

    print("Rebinning atmosphere cube to instrument resolution for each slit position...")
    cube_reb, wl_axis = rebin_atmosphere(cube_sim, wl_sim, spt_sim)

    # "Ground truth" Gaussian fits (no instrument effects)
    print("Fitting ground truth cube...")
    fit_truth = fit_cube_gauss(cube_sim.cgs, wl_sim.cgs)

    # ----------------------------------------------------------
    # monte-carlo
    # ----------------------------------------------------------
    for t_exp in tqdm(SIM["expos"], desc="Exposure time", unit="exposure"):
        signals, fits = monte_carlo(cube_reb, wl_axis, t_exp, n_iter=SIM["n_iter"])

        analysis_results = analyse(fits, fit_truth, wl0)

        plot_maps(
            signals[0][-1], wl_axis,
            plotting["minus_idx"], plotting["mean_idx"], plotting["plus_idx"],
            save=f"fig_maps_{t_exp.to_value(u.s)}.png",
            key_pixel_colors=("mediumseagreen", "black", "deeppink"),
        )

        plot_radiometric_pipeline(
            signals=signals[0],
            wl_axis=wl_axis,
            idx_sim_minus=plotting["minus_idx"],
            idx_sim_mean =plotting["mean_idx"],
            idx_sim_plus =plotting["plus_idx"],
            spt_pitch_sim=spt_sim,
            spt_pitch_instr=DET["spt_res_length"],
            save="fig_detector_pipeline.png",
        )

    print("Done.")

if __name__ == "__main__":
    main()