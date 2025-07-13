import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import glob
import dill
from instr_response import angle_to_distance, velocity_from_fit
import matplotlib
from typing import Tuple, Iterable
from tqdm import tqdm
import warnings                                   # NEW
from matplotlib.colors import ListedColormap, BoundaryNorm   # NEW
from astropy.coordinates import SkyCoord, SpectralCoord                    # NEW
from sunpy.coordinates import Helioprojective
from astropy.time import Time
import astropy.constants as const
import astropy.units as u
from ndcube import NDCube
import sunpy.map
from astropy.wcs import WCS
from matplotlib.gridspec import GridSpec                                # NEW

# ----------------------------------------------------------------------
# --- GENERIC MAP MAKING ----------------------------------------------
# ----------------------------------------------------------------------
def make_maps(
    *,
    cube: NDCube,
    analysis_per_exp: dict[float, dict] | None = None,
    precision_requirement: u.Quantity = 2.0 * u.km / u.s,
    vstd_map: u.Quantity | None = None,
    dn_cube: NDCube | None = None,
    photon_cube: NDCube | None = None,
    vel_arr: u.Quantity | None = None,
    # colour-maps
    exposure_cmap: str = "viridis",
    vstd_cmap: str = "magma",
    int_cmap: str = "afmhot",
    dn_cmap: str = "afmhot",
    photon_cmap: str = "afmhot",
    vel_cmap: str = "RdBu_r",
) -> dict[str, sunpy.map.Map]:
    """
    Convert the provided arrays into SunPy Maps.
    Only quantities explicitly passed are converted.
    """
    maps: dict[str, sunpy.map.Map] = {}

    # ---- exposure-time map ------------------------------------------
    if analysis_per_exp is not None:
        exp_times = sorted(analysis_per_exp.keys())
        nlevels   = len(exp_times)
        shape     = next(iter(analysis_per_exp.values()))["v_std"].shape
        best_exp  = np.full(shape, np.nan)
        for s in exp_times:
            vstd = analysis_per_exp[s]["v_std"].to_value(u.km / u.s)
            msk  = (vstd <= precision_requirement.to_value(u.km / u.s)) & np.isnan(best_exp)
            best_exp[msk] = s
        cmap = ListedColormap(plt.get_cmap(exposure_cmap)(np.linspace(0, 1, nlevels)))
        cmap.set_over("white"); cmap.set_bad("white")
        norm = BoundaryNorm(np.arange(-.5, nlevels + .5, 1), nlevels)

        m = sunpy.map.Map(best_exp.T, cube.wcs.celestial.swapaxes(0, 1))
        m.plot_settings.update(dict(cmap=cmap, norm=norm, origin="lower"))
        maps["exposure"] = m

    # ---- velocity-uncertainty map -----------------------------------
    if vstd_map is not None:
        m = sunpy.map.Map(vstd_map.T.to_value(u.km / u.s), cube.wcs.celestial.swapaxes(0, 1))
        m.plot_settings.update(dict(cmap=vstd_cmap, vmin=0, origin="lower"))
        maps["v_std"] = m

    # ---- integrated intensity ---------------------------------------
    if cube.data.ndim == 3:
        ii = (cube.data.sum(axis=2) *
              (cube.wcs.wcs.cdelt[0] * cube.wcs.wcs.cunit[0])).cgs.value
        log_ii = np.log10(ii, out=np.zeros_like(ii), where=ii > 0)
        m = sunpy.map.Map(log_ii.T, cube.wcs.celestial.swapaxes(0, 1))
        m.plot_settings.update(dict(cmap=int_cmap, vmin=0, origin="lower"))
        maps["intensity"] = m

    # ---- first DN ----------------------------------------------------
    if dn_cube is not None:
        log_dn = np.log10(dn_cube.data.sum(axis=2),
                          out=np.zeros_like(dn_cube.data[..., 0]), where=dn_cube.data.sum(axis=2) > 0)
        m = sunpy.map.Map(log_dn.T, cube.wcs.celestial.swapaxes(0, 1))
        m.plot_settings.update(dict(cmap=dn_cmap, vmin=0, origin="lower"))
        maps["dn"] = m

    # ---- photons -----------------------------------------------------
    if photon_cube is not None:
        log_ph = np.log10(photon_cube.data.sum(axis=2),
                          out=np.zeros_like(photon_cube.data[..., 0]), where=photon_cube.data.sum(axis=2) > 0)
        m = sunpy.map.Map(log_ph.T, cube.wcs.celestial.swapaxes(0, 1))
        m.plot_settings.update(dict(cmap=photon_cmap, vmin=0, origin="lower"))
        maps["photon"] = m

    # ---- velocity map -----------------------------------------------
    if vel_arr is not None:
        m = sunpy.map.Map(vel_arr.T.to_value(u.km / u.s),
                          cube.wcs.celestial.swapaxes(0, 1))
        m.plot_settings.update(dict(cmap=vel_cmap,
                                    vmin=(-15*u.km/u.s).value,
                                    vmax=(15*u.km/u.s).value,
                                    origin="lower"))
        maps["velocity"] = m

    return maps


# ----------------------------------------------------------------------
# --- GENERIC MAP PLOTTING --------------------------------------------
# ----------------------------------------------------------------------
def plot_maps(
    maps: dict[str, sunpy.map.Map] | list[sunpy.map.Map],
    save: str,
    *,
    layout: str | tuple[int, int] = "horizontal",
    share: bool = True,
    dpi: int = 300,
) -> None:
    """
    Draw one or many maps in a whitespace-free panel arrangement.
    layout: 'horizontal' | 'vertical' | (nrows,ncols)
    """
    if isinstance(maps, dict):
        maps = list(maps.values())
    n = len(maps)

    # --- determine grid ------------------------------------------------
    if layout == "horizontal":
        nrows, ncols = 1, n
    elif layout == "vertical":
        nrows, ncols = n, 1
    elif isinstance(layout, tuple) and len(layout) == 2:
        nrows, ncols = layout
        if nrows * ncols < n:
            raise ValueError("Grid too small")
    else:
        raise ValueError("Unknown layout")

    fig = plt.figure(figsize=(4*ncols, 4*nrows))
    gs  = GridSpec(nrows, ncols, wspace=0.0, hspace=0.0)

    for k, m in enumerate(maps):
        r, c = divmod(k, ncols)
        ax = fig.add_subplot(gs[r, c], projection=m)
        im = m.plot(axes=ax, **m.plot_settings)
        ax.set_title("")
        ax.set_aspect(abs(m.scale[1]/m.scale[0]))
        ax.grid(False)
        if share and k:
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.coords[0].set_ticklabel_visible(False)
            ax.coords[1].set_ticklabel_visible(False)

        # colour-bar if single panel
        if n == 1:
            cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                              pad=0.05, fraction=0.046)
            if m.unit != u.dimensionless_unscaled:
                cb.set_label(str(m.unit))

    plt.tight_layout(pad=0)
    plt.savefig(save, dpi=dpi, bbox_inches="tight")
    plt.close(fig)







def plot_vstd_vs_exposure_per_pix(
    analysis_per_exp: dict[float, dict],
    sel_idx: list[Tuple[int, int]] | None = None,
    colours: Iterable[str] | None = None,
    lbls: Iterable[str] | None = None,
    save: str = "fig_vstd_vs_exposure_per_pix.png",
    *,                                        # --- keyword–only after here
    log_x: bool = True,
    log_y: bool = True,
    vstd_max: float | None = None,
) -> None:
    """
    Plot std_v versus exposure time for specific detector pixels.
    """

    # ------------------------------------------------------------------
    # ---- gather σ_v ---------------------------------------------------
    exp_times = sorted(analysis_per_exp.keys())                     # ascending [s]
    curves = {idx: [] for idx in sel_idx}

    for sec in exp_times:
        v_std_map = analysis_per_exp[sec]["v_std"]
        for idx in sel_idx:
            val = v_std_map[idx].to_value(u.km / u.s)
            if vstd_max is not None and val > vstd_max:
                val = np.nan
            curves[idx].append(val)

    # ------------------------------------------------------------------
    # ---- plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.25, 3.5))
    for (idx, col, lab) in zip(sel_idx, colours, lbls):
        y = curves[idx]
        if not np.all(np.isnan(y)):
            ax.plot(exp_times, y, marker="o", color=col, label=lab)

    ax.set_xlabel("Exposure time [s]")
    ax.set_ylabel(r"$\sigma_v$ [km/s]")

    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(exp_times)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if log_y:
        ax.set_yscale("log")
    ax.set_ylim(bottom=0)
    if vstd_max is not None:
        ax.set_ylim(top=vstd_max)

    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize="small")
    ax.tick_params(direction="in", which="both", top=True, right=True)

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_int_vs_std(
    photons_per_exposure: u.Quantity,
    vstd_per_exposure: u.Quantity,
    save: str = "fig_vstd_vs_exposure_per_pix.png",
    *,                                        # --- keyword–only after here
    log_x: bool = True,
    log_y: bool = True,
    vstd_max: float | None = None,
) -> None:

    photon_cube = (photons_per_exposure.data * photons_per_exposure.unit).sum(axis=2)
    vstd_cube   = vstd_per_exposure.data * vstd_per_exposure.unit

    flat_photons = photon_cube.to_value(u.photon / u.s / u.pix).ravel()
    flat_vstd    = vstd_cube.to_value(u.km / u.s).ravel()

    # ------------------------------------------------------------------
    # ---- plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.25, 3.5))

    ax.scatter(flat_photons, flat_vstd, s=1, color="dimgray", alpha=1)

    ax.set_xlabel(r"Photon intensity [photon s$^{-1}$ pix$^{-1}$]")
    ax.set_ylabel(r"$\sigma_v$ [km/s]")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_ylim(bottom=0)
    if vstd_max is not None:
        ax.set_ylim(top=vstd_max)

    ax.grid(ls=":", alpha=0.5)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_spectra(
    dn_cube: u.Quantity,
    wl_axis: u.Quantity,
    idx_sim_minus: Tuple[int, int] | None,
    idx_sim_mean: Tuple[int, int] | None,
    idx_sim_plus: Tuple[int, int] | None,
    wl0: u.Quantity,
    *,                                       # keep following args keyword-only
    fit_cube: u.Quantity | np.ndarray,
    sigma_factor: float = 1.0,
    key_pixel_colors: Tuple[str, str, str] = ("deeppink", "black", "mediumseagreen"),
    save: str = "fig_spectra_dn.png",
    figsize: Tuple[float, float] = (6, 9.5),
) -> None:
    """
    Vertical layout (+1 σ at top, mean centre, −1 σ bottom) with a shared
    wavelength axis (bottom) and a shared velocity axis (top).  Each panel has
    an independent y-axis.  No vertical gaps between panels.
    """
    # ------------------------------------------------------------------
    # ---- handy converters --------------------------------------------
    wl_A  = wl_axis.to(u.angstrom).value
    wl0_A = wl0.to(u.angstrom).value
    c_kms = const.c.to_value(u.km / u.s)
    wl2v  = lambda wl: (wl - wl0_A) / wl0_A * c_kms
    v2wl  = lambda v: wl0_A * (1 + v / c_kms)

    # ------------------------------------------------------------------
    # ---- ordering:  +σ,  μ,  −σ --------------------------------------
    idxs   = [idx_sim_plus, idx_sim_mean, idx_sim_minus]
    titles = [
        rf"$\mu + {sigma_factor:.0f}\sigma$",
        r"$\mu$",
        rf"$\mu - {sigma_factor:.0f}\sigma$",
    ]

    # ------------------------------------------------------------------
    # ---- create stacked axes via GridSpec ----------------------------
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(nrows=3, ncols=1, hspace=0.0)
    axes = gs.subplots(sharex=True)

    # ------------------------------------------------------------------
    # ---- x-axis limits & tick locations ------------------------------
    v_lim        = (-250, 250)                               # km s⁻¹
    wl_lim_A     = v2wl(np.array(v_lim))
    v_ticks      = np.arange(-200, 201, 100)                 # every 100 km s⁻¹
    wl_tick_A    = v2wl(v_ticks)

    # ------------------------------------------------------------------
    # ---- loop over the three panels ----------------------------------
    first_secax: plt.Axes | None = None
    for ax, idx, title, color in zip(axes, idxs, titles, key_pixel_colors):
        if idx is None:
            ax.set_visible(False)
            continue

        # -------- observed spectrum -----------------------------------
        sel   = dn_cube[idx + (slice(None),)]
        spec  = sel.value if isinstance(sel, u.Quantity) else sel
        ax.step(wl_A, spec, where="mid", color=color, lw=1.5, zorder=2)

        # -------- fitted Gaussian ------------------------------------
        p = fit_cube[idx + (slice(None),)]
        if np.all(p != -1):
            peak, centre, sigma, back = (p[k].value for k in range(4))
            wl_hi = np.linspace(wl_axis.cgs.value.min(),
                                wl_axis.cgs.value.max(),
                                wl_axis.size * 10)           # ×10 sampling
            gauss = gaussian(wl_hi, peak, centre, sigma, back)
            ax.plot((wl_hi * u.cm).to_value(u.angstrom),
                    gauss, ls="--", color=color, lw=1.0, zorder=1)

        # -------- set y-axis lower limit to 0 -------------------------
        ax.set_ylim(bottom=0)

        # -------- cosmetics ------------------------------------------
        ax.set_ylabel("DN/pix")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(ls=":", alpha=0.5)

        # ticks on all four sides, pointing inwards
        ax.tick_params(direction="in", which="both", top=True, right=True)

        # -------------- velocity axis (top) --------------------------
        secax = ax.secondary_xaxis("top", functions=(wl2v, v2wl))
        secax.set_xlim(*v_lim)
        secax.set_xticks(v_ticks)
        secax.tick_params(direction="in", which="both", top=True)

        if first_secax is None:
            # first visible panel → keep labels
            # secax.set_xlabel("Velocity [km/s]")
            secax.set_xlabel("Velocity (Fe XII 195.119 Å) [km/s]")
            first_secax = secax
        else:
            # other panels: keep ticks but drop labels
            secax.set_xlabel("")
            secax.set_xticklabels([])

        # ------------------------------------------------------------------
        # Right-side “σ-label” styled like a y-axis label
        # ------------------------------------------------------------------
        ax.annotate(
            title,
            xy=(1.02, 0.5), xycoords=("axes fraction", "axes fraction"),
            rotation=90,
            ha="left", va="center",
            fontsize=ax.yaxis.label.get_size(),
        )

    # ------------------------------------------------------------------
    # ---- move scientific-notation offset to the y-label --------------
    fig.canvas.draw()  # populate offset texts
    for ax in axes:
        if not ax.get_visible():
            continue
        off_txt = ax.yaxis.get_offset_text().get_text()
        if off_txt:                                      # e.g. '1e3'
            exponent = off_txt.replace("1e", "")
            ax.yaxis.get_offset_text().set_visible(False)
            ax.set_ylabel(fr"Intensity [$10^{{{exponent}}}$ DN/pix]")

    # ------------------------------------------------------------------
    # ---- shared X-axis (bottom wavelength axis) ----------------------
    axes[-1].set_xlabel("Wavelength [Å]")
    axes[-1].set_xlim(*wl_lim_A)
    axes[-1].set_xticks(wl_tick_A)
    axes[-1].set_xticklabels([f"{w:.3f}" for w in wl_tick_A])
    axes[-1].tick_params(direction="in", which="both", top=True, right=True)

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_pkl_results(pkl_file):
    with open(pkl_file, "rb") as f:
        dat = dill.load(f)
    results = dat["results"]
    config = dat["config"]
    instrument = dat["instrument"]
    DET = dat["DET"]
    TEL = dat["TEL"]
    SIM = dat["SIM"]
    plotting = dat["plotting"]
    cube_sim = dat["cube_sim"]
    cube_reb = dat["cube_reb"]
    return results, config, instrument, DET, TEL, SIM, plotting, cube_sim, cube_reb

def get_plot_dir(base, contamination):
    return os.path.join(base, f"contam_{contamination}")




def plot_vstd_vs_contamination(results_files: list[str], save_prefix: str = "fig_vstd_vs_contamination"):
    """
    Plot Doppler velocity standard deviation vs C thickness for the mean pixel, for all specified files,
    for each available exposure time.
    """
    import dill
    import matplotlib.pyplot as plt
    import astropy.units as u
    import numpy as np
    import os

    # Gather all exposure times across all files
    all_exp_times = set()
    per_file_data = []

    for file in results_files:
        with open(file, "rb") as f:
            dat = dill.load(f)
        config = dat["config"]["config"]
        plotting = dat["plotting"]
        results = dat["results"]
        c_thick = config.get("c_thickness", "0 nm")
        c_thick = u.Quantity(c_thick)
        analysis_per_exp = results["analysis_per_exp"]
        all_exp_times.update(analysis_per_exp.keys())
        per_file_data.append({
            "c_thick": c_thick.to(u.angstrom).value,
            "analysis_per_exp": analysis_per_exp,
            "plotting": plotting,
        })

    # For each exposure time, collect vstds for all files
    for sec in sorted(all_exp_times):
        c_thicks = []
        vstds = []
        for entry in per_file_data:
            analysis_per_exp = entry["analysis_per_exp"]
            plotting = entry["plotting"]
            idx_mean = plotting.get("mean_idx")
            if sec in analysis_per_exp and idx_mean is not None:
                v_std_map = analysis_per_exp[sec]["v_std"]
                v_std_map = np.transpose(v_std_map, (0,2,1))
                vstds.append(v_std_map[idx_mean].to_value(u.km / u.s))
                c_thicks.append(entry["c_thick"])
        c_thicks = np.array(c_thicks)
        vstds = np.array(vstds)
        order = np.argsort(c_thicks)
        c_thicks = c_thicks[order]
        vstds = vstds[order]

        plt.figure(figsize=(6, 4))
        plt.plot(c_thicks, vstds, marker="o", color="black", label="mean pixel")
        plt.xlabel("C thickness [Å]")
        plt.ylabel(r"$\sigma_v$ [km/s]")
        plt.title(f"Exposure time: {sec:.1f} s")
        plt.grid(ls=":", alpha=0.5)
        plt.tight_layout()
        save = f"{save_prefix}_{sec:.1f}s.png"
        plt.savefig(save, dpi=300)
        plt.close()



def main():

    # pkl_files = sorted(glob.glob("./run/result/*.pkl"))
    pkl_files = ['./run/result/instrument_response_quick.pkl']
    if not pkl_files:
        print("No .pkl files found in current directory.")
        return

    for pkl_file in pkl_files:

        print(f"Processing {pkl_file} ...")
        (results,
         config,
         instrument,
         DET,
         TEL,
         SIM,
         plotting,
         cube_sim,
         cube_reb) = load_pkl_results(pkl_file)

        hc_coords = [plotting['plus_coords'], 
                  plotting['mean_coords'], 
                  plotting['minus_coords']]

        obstime = Time("2025-01-01T00:00:00")

        hc_skycoords = [
            SkyCoord(
                c[1],
                c[0],
                const.R_sun.to(u.Mm),
                frame="heliocentric",
                observer="earth",
                obstime=obstime
            )
            for c in hc_coords
        ]

        hp_frame = Helioprojective(observer="earth", obstime=obstime)
        hp_coords = [c.transform_to(hp_frame) for c in hc_skycoords]

        wl_axis = cube_reb.axis_world_coords(2)[0]
        wl0 = cube_reb.meta.get("rest_wav")

        sim_name = os.path.splitext(os.path.basename(pkl_file))[0]

        output_dir = f"./run/plot/{sim_name}_plots"
        ensure_dir(output_dir)

        # convert the coordinates to pixel values
        spec = SpectralCoord(wl0)
        sky = [
            SkyCoord(Tx=coord.Tx, Ty=coord.Ty,
                     frame='helioprojective', obstime=obstime)
            for coord in hp_coords
        ]

        # convert each Helioprojective coordinate to pixel indices as (x, y) pairs
        pixel_indices = []
        for coord in sky:
            x_val, y_val, _ = cube_reb.wcs.world_to_pixel(spec, coord)
            pixel_indices.append((int(np.round(x_val)), int(np.round(y_val))))








        def translate_heliocentric_coordinates(
            coords: Iterable[Tuple[u.Quantity, u.Quantity]],
            cube_sim: NDCube,
            cube_reb: NDCube,
            *,
            obstime: Time = Time("2025-01-01T00:00:00"),
        ) -> tuple[
            list[SkyCoord],
            list[SkyCoord],
            list[tuple[int, int]],
            list[tuple[int, int]],
        ]:
            """
            Convert heliocentric (x, y) coordinates to
            1. heliocentric SkyCoord objects,
            2. helioprojective SkyCoord objects,
            3. pixel indices in `cube_sim`,
            4. pixel indices in `cube_reb`.

            Parameters
            ----------
            coords
            Iterable of (y, x) heliocentric coordinates with units.
            cube_sim, cube_reb
            NDCubes for the synthetic (high-res) and simulated
            (rebinned) data.
            obstime
            Time of the observation (needed for the transforms).

            Returns
            -------
            hc_skycoords
            List of heliocentric SkyCoord objects.
            hp_skycoords
            List of helioprojective SkyCoord objects.
            pix_sim
            List of (x, y) pixel indices in `cube_sim`.
            pix_reb
            List of (x, y) pixel indices in `cube_reb`.
            """
            # ----------------------------------------------------------
            # 1. heliocentric SkyCoord objects
            # ----------------------------------------------------------
            hc_skycoords: list[SkyCoord] = [
            SkyCoord(
                c[1],                      # x
                c[0],                      # y
                const.R_sun.to(u.Mm),      # z  (radius)
                frame="heliocentric",
                observer="earth",
                obstime=obstime,
            )
            for c in coords
            ]

            # ----------------------------------------------------------
            # 2. transform to helioprojective
            # ----------------------------------------------------------
            hp_frame = Helioprojective(observer="earth", obstime=obstime)
            hp_skycoords = [c.transform_to(hp_frame) for c in hc_skycoords]

            # ----------------------------------------------------------
            # Helper to convert a list of SkyCoords to pixel indices
            # ----------------------------------------------------------
            def _coords_to_pix(
                cube: NDCube,
                skycoords: list[SkyCoord],
            ) -> list[tuple[int, int]]:
                """
                Convert a list of SkyCoord objects to integer pixel indices
                on the spatial detector plane of the provided NDCube.
                """
                # Convert only the celestial (x, y) part of the WCS – the spectral/time
                # axes are irrelevant for locating the pixel position on the detector.
                pix: list[tuple[int, int]] = []
                for sc in skycoords:
                    x_val, y_val = cube.wcs.celestial.world_to_pixel(sc)
                    pix.append((int(np.round(x_val)), int(np.round(y_val))))
                return pix

            # ----------------------------------------------------------
            # 3. pixel indices in each cube
            # ----------------------------------------------------------
            synthetic_pixel_indices = _coords_to_pix(cube_sim, hp_skycoords)
            simulated_pixel_indices = _coords_to_pix(cube_reb, hp_skycoords)

            return (
                hc_skycoords,
                hp_skycoords,
                synthetic_pixel_indices,
                simulated_pixel_indices,
            )



        hc_coords = [
            plotting['plus_coords'],
            plotting['mean_coords'],
            plotting['minus_coords']
            ]
        (
            hc_skycoords,
            hp_skycoords,
            synthetic_pixel_indices,
            simulated_pixel_indices,
        ) = translate_heliocentric_coordinates(
            coords=hc_coords,
            cube_sim=cube_sim,
            cube_reb=cube_reb,
            obstime=obstime,
        )






        # ------------------------------------------------------------------
        # ---- CREATE ALL REQUESTED MAPS IN ONE CALL ------------------------
        maps = make_maps(
            cube=cube_reb,
            analysis_per_exp=results['analysis_per_exp'],
            precision_requirement=2.0*u.km/u.s,
            vstd_map=results['analysis_per_exp'][1.0]['v_std'],
            dn_cube=results['first_dn_signal_per_exp'][1.0],
            photon_cube=results['first_photon_signal_per_exp'][1.0],
            vel_arr=results['analysis_per_exp'][1.0]['v_samples'][0, :, :],
        )

        # ------------------------------------------------------------------
        # ---- SINGLE FIGURE WITH ALL PANELS -------------------------------
        plot_maps(
            maps,                       # dict of SunPy maps
            save="TEMP_all_maps.png",
            # layout=(2, 3),              # grid: 2 rows × 3 cols
            share=True,
        )

        plot_int_vs_std(
            results['first_photon_signal_per_exp'][1.0],
            results['analysis_per_exp'][1.0]['v_std'],
            save='TEMP8.png',
        )

        plot_vstd_vs_exposure_per_pix(                        # RENAMED + UPDATED
            results['analysis_per_exp'],
            cube_reb,
            sel_idx=pixel_indices,                                # explicit pixel list
            colours=["red", "green", "blue"],
            lbls=["1", "2", "3"],
            # save=os.path.join(output_dir, "fig_vstd_vs_exposure.png"),
            save='TEMP.png',
            log_x=True,
            log_y=True,
            vstd_max=None,
        )


    results_files = [
        "./run/result/swc_slit0.2_oxide95_carbon0.pkl",
        "./run/result/swc_slit0.2_oxide95_carbon40.pkl",
        "./run/result/swc_slit0.2_oxide95_carbon80.pkl",
        "./run/result/swc_slit0.2_oxide95_carbon120.pkl",
        "./run/result/swc_slit0.2_oxide95_carbon160.pkl",
        "./run/result/swc_slit0.2_oxide95_carbon200.pkl",
    ]
    plot_vstd_vs_contamination(
        results_files,
        save_prefix="fig_vstd_vs_contamination"
    )

if __name__ == "__main__":
    main()

























































    # # ----------------------------------------------------------------------
    # # plot output
    # # ----------------------------------------------------------------------

    # # Calculate Doppler velocity map before plotting
    # print(f"Calculating Doppler velocity map ({print_mem()})")
    # v_map = calculate_doppler_map(total_si.value, v_edges)

    # key_pixel_colors = ["deeppink", "black", "mediumseagreen"]  # in order of minus, mean, plus

    # plot_maps(
    #     total_si.value, v_map, voxel_dx, voxel_dy, downsample, margin,
    #     goft[main_line]["wl_grid"].cgs.value, "fig_synthetic_maps.png",
    #     mean_idx=mean_idx, plus_sigma_idx=plus_idx, minus_sigma_idx=minus_idx,
    #     sigma_factor=sigma_factor, key_pixel_colors=key_pixel_colors
    # )

    # plot_dems(
    #     dem_map, em_tv, logT_centres, v_edges,
    #     plus_idx, mean_idx, minus_idx, sigma_factor,
    #     xlim=(5.5, 6.9),
    #     ylim_dem=(26.5, 30),
    #     ylim_2d_dem=(-50, 50),
    #     save="fig_synthetic_dems.png",
    #     goft=goft, main_line=main_line,
    #     key_pixel_colors=key_pixel_colors,
    #     logT_grid=logT_grid,
    #     logN_grid=logN_grid,
    #     figsize=(12, 6),
    #     cbar_offset=1.325,
    #     inset_axis_offset=0.77,
    # )

    # plot_g_function(
    #     goft, main_line, logT_grid, logN_grid,
    #     save="fig_g_function.png",
    #     xlim=(5.5, 6.9),
    #     show_vlines=True,
    #     figsize=(6, 3)
    # )

    # plot_spectrum(
    #     goft, total_si.value, goft[main_line]["wl_grid"].to('AA').value,
    #     minus_idx, mean_idx, plus_idx,
    #     main_line=main_line, secondary_line="Fe12_195.1790",
    #     key_pixel_colors=key_pixel_colors,
    #     sigma_factor=sigma_factor,
    #     save="fig_synthetic_spectra.png",
    #     xlim_vel=(-250, 250),
    #     yorders=9,
    #     ylimits=(None, 8e13),
    #     main_label="Fe XII 195.119",
    #     secondary_label="Fe XII 195.179",
    #     figsize=(12, 6)
    # )



##############################################################################
# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
##############################################################################

def _find_mean_sigma_pixel(ndcube, margin_frac=0.20, sigma_factor=1.0):
    """
    Return (mean_coord, plus_sigma_coord, minus_sigma_coord)
    corresponding to the pixel intensities nearest to the mean and
    mean +- (sigma_factor)*std, as SkyCoord coordinates using pixel_to_world.
    """

    # Account for margin and crop to inner region
    nx, ny = ndcube.data.shape
    margin = int(margin_frac * min(nx, ny))
    inner = ndcube.data[margin:nx - margin, margin:ny - margin]

    # Calculate mean and standard deviation
    mean_val = np.mean(inner)
    std_val = np.std(inner)
    plus_sigma_val = mean_val + sigma_factor * std_val
    minus_sigma_val = mean_val - sigma_factor * std_val

    # Find indices of the closest pixels to mean, mean + sigma, and mean - sigma
    mean_idx = np.unravel_index(np.argmin(np.abs(inner - mean_val)), inner.shape)
    plus_sigma_idx = np.unravel_index(np.argmin(np.abs(inner - plus_sigma_val)), inner.shape)
    minus_sigma_idx = np.unravel_index(np.argmin(np.abs(inner - minus_sigma_val)), inner.shape)

    # Convert to global pixel indices
    mean_idx_global = (mean_idx[0] + margin, mean_idx[1] + margin)
    plus_sigma_idx_global = (plus_sigma_idx[0] + margin, plus_sigma_idx[1] + margin)
    minus_sigma_idx_global = (minus_sigma_idx[0] + margin, minus_sigma_idx[1] + margin)

    # Use pixel_to_world to get SkyCoord coordinates
    mean_coord = ndcube.wcs.pixel_to_world(mean_idx_global[1], mean_idx_global[0])
    plus_sigma_coord = ndcube.wcs.pixel_to_world(plus_sigma_idx_global[1], plus_sigma_idx_global[0])
    minus_sigma_coord = ndcube.wcs.pixel_to_world(minus_sigma_idx_global[1], minus_sigma_idx_global[0])

    return mean_coord, plus_sigma_coord, minus_sigma_coord


def calculate_doppler_map(total_si, v_edges):
    # Helper: Gaussian function
    def gaussian(x, amp, mu, sigma, offset):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

    v_cent_km = 0.5 * (v_edges[:-1] + v_edges[1:]) * (u.cm / u.s)
    v_cent_km = v_cent_km.to(u.km / u.s).value
    nx, ny, nl = total_si.shape
    v_map = np.full((nx, ny), np.nan)

    def fit_pixel(i, j):
        spectrum = total_si[i, j, :]
        try:
            amp0 = spectrum.max()
            mu0 = v_cent_km[np.argmax(spectrum)]
            sigma0 = 10  # km/s, rough guess
            offset0 = 0
            popt, _ = curve_fit(
                gaussian, v_cent_km, spectrum,
                p0=[amp0, mu0, sigma0, offset0],
                maxfev=5000
            )
            return (i, j, popt[1])  # centroid (mu)
        except Exception:
            return (i, j, np.nan)

    pixel_indices = [(i, j) for i in range(nx) for j in range(ny)]

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(fit_pixel)(i, j) for i, j in tqdm(pixel_indices, desc="Doppler map", unit="pixel", leave=False)
    )

    for i, j, mu in results:
        v_map[i, j] = mu

    return v_map


def plot_maps(
  total_si, v_map, voxel_dx, voxel_dy, downsample, margin, wl_grid_main, save,
  mean_idx=None, plus_sigma_idx=None, minus_sigma_idx=None, sigma_factor=1.0,
  key_pixel_colors=None
):
  """
  Intensity + Doppler maps (side-by-side), with mean and +-(sigma_factor) pixels marked
  on both panels. One shared legend on the velocity panel.
  key_pixel_colors: list of 3 colors for plus_sigma, mean, minus_sigma pixels.
  """
  ds = downsample if isinstance(downsample, int) and downsample > 1 else 1
  dx_pix = voxel_dx.to(u.Mm).value * ds
  dy_pix = voxel_dy.to(u.Mm).value * ds
  nx, ny = total_si.shape[:2]
  extent = (0, nx * dx_pix, 0, ny * dy_pix)

  wl_res = wl_grid_main[1] - wl_grid_main[0]

  # Remove the Gaussian fitting logic from here, as it's now precomputed
  fig = plt.figure(figsize=(11, 5))
  gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.0)
  axI = fig.add_subplot(gs[0, 0])
  axV = fig.add_subplot(gs[0, 1], sharey=axI)

  si = total_si.sum(axis=2) * wl_res  # integrate over wavelength
  log_si = np.log10(si, where=si > 0.0, out=np.zeros_like(si))

  # Intensity panel with lower colorbar limit set to 0
  imI = axI.imshow(
    log_si.T,  # log10 of integrated intensity
    origin="lower", aspect="equal", cmap="afmhot",
    extent=extent, vmin=0.0
  )
  # margin of shortest side
  this_margin = int(margin * min(nx, ny))
  rect = Rectangle(
    (this_margin * dx_pix, this_margin * dy_pix),
    (nx - 2 * this_margin) * dx_pix, (ny - 2 * this_margin) * dy_pix,
    fill=False, edgecolor="black", linewidth=1, linestyle="--"
  )
  axI.add_patch(rect)
  axI.set_xlabel("X (Mm)")
  axI.set_ylabel("Y (Mm)")
  cbarI = fig.colorbar(imI, ax=axI, orientation="horizontal", extend="neither", aspect=35, shrink=0.95, pad=0.115)
  cbarI.set_label(
    r"$\log_{10}\!\left(\int I(\lambda)\,\mathrm{d}\lambda\mathrm{ }\left[\mathrm{erg/s/cm}^2\mathrm{/sr}\right]\right)$"
  )
  axI.tick_params(direction="in", top=True, bottom=True, left=True, right=True)

  # Doppler panel
  imV = axV.imshow(
    v_map.T, origin="lower", aspect="equal", cmap="RdBu_r",
    extent=extent, vmin=-15, vmax=15
  )
  rect = Rectangle(
    (this_margin * dx_pix, this_margin * dy_pix),
    (nx - 2 * this_margin) * dx_pix, (ny - 2 * this_margin) * dy_pix,
    fill=False, edgecolor="black", linewidth=1, linestyle="--"
  )
  axV.add_patch(rect)
  axV.tick_params(labelleft=False, direction="in", top=True, bottom=True, right=True, left=True)
  axV.set_xlabel("X (Mm)")

  # Doppler panel colorbar (thin, below the axis)
  cbarV = fig.colorbar(imV, ax=axV, orientation="horizontal", extend="both", aspect=35, shrink=0.95, pad=0.115)
  cbarV.set_label(r"$v$ [km/s]")

  # Markers on both panels
  if mean_idx and plus_sigma_idx and minus_sigma_idx:
    # idxs = [minus_sigma_idx, mean_idx, plus_sigma_idx]
    # base_labels = [
    #   rf"$\mu - {sigma_factor:.0f}\sigma$",
    #   r"$\mu$",
    #   rf"$\mu + {sigma_factor:.0f}\sigma$"
    # ]
    # markers = ["1", "3", "2"]
    idxs = [plus_sigma_idx, mean_idx, minus_sigma_idx]
    base_labels = [
      rf"$\mu + {sigma_factor:.0f}\sigma$",
      r"$\mu$",
      rf"$\mu - {sigma_factor:.0f}\sigma$"
    ]
    # # Add intensity and velocity to each label
    # for i, idx in enumerate(idxs):
    #   I_1d = total_si[idx[0], idx[1], :]
    #   I_val = I_1d.sum() * wl_res
    #   logI_val = np.log10(I_val) if I_val > 0 else -np.inf
    #   v_val = v_map[idx[0], idx[1]]
    #   base_labels[i] += f"I={I_val:.2e}, v={v_val:.1f} km/s"
    markers = ["2", "3", "1"]
    if key_pixel_colors is None:
      colors = ["tab:blue", "tab:green", "tab:orange"]
    else:
      # reverse the order to match idxs
      colors = key_pixel_colors[::-1]
    for base_label, idx, marker, color in zip(base_labels, idxs, markers, colors):
      I_1d = total_si[idx[0], idx[1], :]
      I_val = I_1d.sum() * wl_res
      logI_val = np.log10(I_val) if I_val > 0 else -np.inf
      v_val = v_map[idx[0], idx[1]]
      label = f"{base_label}"

      # Mark intensity panel
      axI.scatter(
        idx[0] * dx_pix + dx_pix / 2,
        idx[1] * dy_pix + dy_pix / 2,
        color=color, s=250, marker=marker, linewidth=2,
      )
      # Mark velocity panel + legend
      axV.scatter(
        idx[0] * dx_pix + dx_pix / 2,
        idx[1] * dy_pix + dy_pix / 2,
        color=color, s=250, marker=marker, linewidth=2,
        label=label
      )
    axV.legend(loc="upper right", fontsize="small")

  plt.tight_layout()
  plt.savefig(save, dpi=600, bbox_inches="tight")
  plt.close(fig)


def plot_dems(
  dem_map,
  em_tv,
  logT_centres,
  v_edges,
  plus_idx,
  mean_idx,
  minus_idx,
  sigma_factor,
  xlim=(5.5, 7.0),
  ylim_dem=(25, 29),
  ylim_2d_dem=None,
  save="dem_and_2d_dem.png",
  goft=None,
  main_line="Fe12_195.1190",
  key_pixel_colors=None,
  logT_grid=None,
  logN_grid=None,
  figsize=(15, 7),
  cbar_offset=1.25,
  inset_axis_offset=0.8,
):
  """
  Plot DEM(T) (top row) and DEM(T,v) maps (bottom row) for three pixels.
  A slim, transparent inset axis overlays each map on the left, showing
  the line profile (intensity vs. velocity) without resizing the map.
  The right y-axis (wavelength) is now exactly aligned with the left y-axis (velocity).
  key_pixel_colors: list of 3 colors for plus_sigma, mean, minus_sigma pixels.
  """

  idxs   = [minus_idx, mean_idx, plus_idx]
  # titles = [r"$\mu-\sigma$", r"$\mu$", r"$\mu+\sigma$"]
  titles = [
    rf"$\mu - {sigma_factor:.0f}\sigma$",
    r"$\mu$",
    rf"$\mu + {sigma_factor:.0f}\sigma$"
  ]

  v_centres = 0.5 * (v_edges[:-1] + v_edges[1:]) * u.cm/u.s
  v_centres_kms = v_centres.to(u.km/u.s).value
  extent = (logT_centres[0], logT_centres[-1],
        v_centres_kms[0],    v_centres_kms[-1])

  # define primary->secondary and secondary->primary for wavelength axis
  if goft and main_line in goft:
    wl0   = goft[main_line]["wl0"].to(u.angstrom).value
    c_kms = const.c.to(u.km/u.s).value
    v2wl  = lambda v: wl0 * (1 + v / c_kms)
    wl2v  = lambda wl: (wl - wl0) / wl0 * c_kms

  fig, axes = plt.subplots(
    2, 3, figsize=figsize,
    sharey="row", gridspec_kw=dict(wspace=0.0, hspace=0.0)
  )

  # -------------------------------------------------------- top row DEM(T)
  if key_pixel_colors is None:
      colours = ["tab:blue", "tab:green", "tab:orange"]
  else:
      colours = key_pixel_colors
  for ax, idx, title, c in zip(axes[0], idxs, titles, colours):
    log_dem = np.log10(dem_map[idx], where=dem_map[idx] > 0,
                out=np.zeros_like(dem_map[idx]))
    ax.step(logT_centres, log_dem, where="mid", color=c, lw=1.8)
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim_dem)
    ax.set_xlabel(r"$\log_{10} T$  [K]")
    ax.grid(ls=":")
  axes[0, 0].set_ylabel(
    r"$\log_{10}\left(\xi\:\mathrm{[1/cm}^{5}/\mathrm{dex}\mathrm{]}\right)$"
  )

  # ------------------------------------------- build 2-D DEM arrays + limits
  datas, vmin, vmax = [], None, None
  for idx in idxs:
    data = np.log10(em_tv[idx].T, where=em_tv[idx].T > 0,
              out=np.zeros_like(em_tv[idx].T))
    data[data == 0] = np.nan
    datas.append(data)

    mx = (logT_centres >= xlim[0]) & (logT_centres <= xlim[1])
    my = np.ones_like(v_centres_kms, dtype=bool)
    if ylim_2d_dem:
      my &= (v_centres_kms >= ylim_2d_dem[0]) & (v_centres_kms <= ylim_2d_dem[1])
    sub = data[np.ix_(my, mx)]
    vmin = np.nanmin(sub) if vmin is None else min(vmin, np.nanmin(sub))
    vmax = np.nanmax(sub) if vmax is None else max(vmax, np.nanmax(sub))

  # ----------------------------------------------------- bottom row maps
  ims = []
  for i, ax in enumerate(axes[1]):
    im = ax.imshow(
      datas[i], origin="lower", aspect="auto",
      cmap="Purples", extent=extent,
      vmin=vmin, vmax=vmax
    )
    ims.append(im)
    ax.set_xlim(*xlim)
    if ylim_2d_dem:
      ax.set_ylim(*ylim_2d_dem)
    ax.set_xlabel(r"$\log_{10}\left(T\:\mathrm{[K]}\right)$")
    ax.grid(ls=":")
    # inset profile
    if goft and main_line in goft:
      spec = goft[main_line]["si"][idxs[i]]

      stick = inset_axes(
        ax, width="50%", height="100%",
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=ax.transAxes,
        loc="lower left", borderpad=0,
      )
      stick.set_facecolor("none")

      stick.set_axisbelow(False)
      for spine in stick.spines.values():
        spine.set_zorder(3)

      stick.plot(spec, v_centres_kms, color="red", lw=1.3, zorder=1)

      stick.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
      stick.xaxis.set_ticks_position("top")
      stick.xaxis.set_label_position("top")
      stick.tick_params(axis="x", labelsize=7, direction="in")

      fig.canvas.draw()
      offset_text = stick.xaxis.get_offset_text().get_text().replace("1e", "")
      stick.xaxis.get_offset_text().set_visible(False)

      stick.set_xlabel(
        f"Fe XII 195.119 $\\AA$ intensity \n [$10^{{{offset_text}}}$ erg/s/cm$^{{2}}$/sr/cm]",
        fontsize=8, zorder=3
      )

      ticks = stick.get_xticks()
      ticks = ticks[ticks != 0]
      stick.set_xticks(ticks)

      stick.yaxis.set_ticks([])

      stick.spines[['right', 'bottom', 'left']].set_visible(False)
      stick.spines['top'].set_position(('axes', inset_axis_offset))

      stick.set_xlim(0, spec.max()*1.05)
      stick.set_ylim(ax.get_ylim())

  # ------------------------------------------------ right‐hand wavelength axis on bottom panels
  if goft and main_line in goft:
    for ax in axes[1]:
      v_ticks = ax.get_yticks()
      vmin_ax, vmax_ax = ax.get_ylim()
      v_ticks = v_ticks[(v_ticks >= vmin_ax) & (v_ticks <= vmax_ax)]
      wl_ticks = v2wl(v_ticks)
      wl_ticklabels = [f"{wl:.3f}" for wl in wl_ticks]
      ax_r = ax.secondary_yaxis("right")
      ax_r.set_yticks(v_ticks)
      ax_r.set_yticklabels(wl_ticklabels)
      ax_r.set_ylabel(r"Wavelength (Fe XII 195.119 $\AA$) [$\AA$]")
      ax.tick_params(axis='y', direction='in', which='both')
      ax_r.tick_params(axis='y', direction='in', which='both')

  # ------------------------------------------------ Tick parameters for all main panels
  for row in axes:
    for ax in row:
      ax.tick_params(direction="in", top=True, bottom=True)

      # add ticks on the right y axis
      ax.yaxis.set_ticks_position("both")
      ax.yaxis.set_tick_params(which="both", direction="in", right=True)

  axes[1, 0].set_ylabel("Velocity [km/s]")

  # ------------------------------------------------ shared colourbar (move to right of all panels)
  cax = inset_axes(
      axes[1, -1], width="3%", height="90%",
      loc="center left",
      bbox_to_anchor=(cbar_offset, 0., 1, 1),
      bbox_transform=axes[1, -1].transAxes,
      borderpad=0,
  )
  cbar = fig.colorbar(
    ims[0], cax=cax, orientation="vertical", extend="min"
  )
  cbar.set_label(
    r"$\log_{10}\,\left(\Xi\:\mathrm{[1/cm}^{5}/\mathrm{dex}/\Delta v] \right)$"
  )

  # --------------------------------------------------------
  # Overplot the contribution function
  # --------------------------------------------------------
  if goft and main_line in goft and logT_grid is not None and logN_grid is not None:
          g_tn = goft[main_line]["g_tn"]
          integrated_g_over_T = np.trapz(g_tn, logT_grid, axis=1)
          optimal_density_idx = np.argmax(integrated_g_over_T)
          optimal_density = logN_grid[optimal_density_idx]
          g_at_optimal_density = g_tn[optimal_density_idx, :]

          # Compute mean and standard deviation of logT weighted by G
          mean_logT = np.average(logT_grid, weights=g_at_optimal_density)
          std_logT = np.sqrt(np.average((logT_grid - mean_logT)**2, weights=g_at_optimal_density))

          # Positions of vertical lines at mean +- xsigma
          vlines = [mean_logT - 2*std_logT, mean_logT + 2*std_logT]

          # --------------------------------------------------------
          # Add vertical lines to all panels
          # --------------------------------------------------------
          for ax_row in axes:
                  for ax in ax_row:
                          for vline in vlines:
                                #   ax.axvline(vline, color='grey', linestyle='--', linewidth=1, alpha=1.0, zorder=0)
                                  ax.axvline(vline, color='grey', linestyle=(0,(5,10)), linewidth=1, alpha=1.0, zorder=0)

          # --------------------------------------------------------
          # Add annotation for vertical lines at top right
          # --------------------------------------------------------
          # Add a custom legend entry for the vertical lines at mean +- xsigma
          custom_line = Line2D([0], [0], color='grey', linestyle=(0,(5,5)), linewidth=1, alpha=1.0)
          handles, labels = axes[0, -1].get_legend_handles_labels()
          handles.append(custom_line)
          labels.append(fr"$G_{{\mathrm{{Fe\,XII\,195.119}}}}(T,\,N_{{e}}={optimal_density:.1f})\pm2\sigma_{{G}}$")
          axes[0, -1].legend(handles=handles, labels=labels, loc="upper right", fontsize="small")

  plt.tight_layout()
  plt.savefig(save, dpi=300, bbox_inches="tight")
  plt.close(fig)

def plot_spectrum(
    goft: dict,
    total_si: np.ndarray,
    wl_grid_main: np.ndarray,
    minus_idx: int,
    mean_idx: int,
    plus_idx: int,
    main_line: str = "Fe12_195.1190",
    secondary_line: str = "Fe12_195.1790",
    main_label: str | None = None,
    secondary_label: str | None = None,
    key_pixel_colors: tuple[str, str, str] | None = None,
    sigma_factor: float = 1.0,
    xlim_vel: tuple[float, float] | None = None,   # (v_min, v_max) km s-1
    yorders: float | None = None,                  # log-y span below ymax
    ylimits: tuple[float, float] | None = None,     # optional (ymin, ymax) for bottom row
    save: str = "spectrum.png",
    figsize: tuple[float, float] = (15, 8),
) -> None:
    """
    Plot three spectra (μ - sigma, μ, μ + sigma) in two rows (linear + log y).

    New parameters:
    main_label, secondary_label:
        If provided, these override the legend labels for main_line
        and secondary_line respectively.
    """

    # ---------------------------------------------------------------------
    # constants and helpers
    # ---------------------------------------------------------------------
    wl0_A = goft[main_line]["wl0"].to(u.AA).value
    c_kms = const.c.to(u.km / u.s).value

    wl2v = lambda wl: (wl - wl0_A) / wl0_A * c_kms          # A -> km s-1
    v2wl = lambda v: wl0_A * (1 + v / c_kms)                # km s-1 -> A

    # velocity limits -> wavelength limits
    wl_min, wl_max = (None, None)
    if xlim_vel is not None:
        wl_min, wl_max = v2wl(xlim_vel[0]), v2wl(xlim_vel[1])

    # default total-spectrum colours
    if key_pixel_colors is None:
        key_pixel_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:3]

    # determine legend labels
    label_main = main_label if main_label is not None else main_line
    label_sec  = secondary_label if secondary_label is not None else secondary_line

    # ---------------------------------------------------------------------
    # figure & axes grid
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(
        2, 3,
        figsize=figsize,
        sharex="col", sharey="row",
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
    )

    pixel_indices = (minus_idx, mean_idx, plus_idx)
    col_titles = (
        rf"$\mu - {sigma_factor:.0f}\sigma$",
        r"$\mu$",
        rf"$\mu + {sigma_factor:.0f}\sigma$",
    )

    # ---------------------------------------------------------------------
    # main plotting loop
    # ---------------------------------------------------------------------
    for col, (pix, title, tot_colour) in enumerate(
        zip(pixel_indices, col_titles, key_pixel_colors)
    ):
        ax_lin = axes[0, col]   # linear y-axis (upper row)
        ax_log = axes[1, col]   # log    y-axis (lower row)

        # plot every individual line
        for line_name, info in goft.items():
            wl_src = info["wl_grid"].to(u.AA).value
            spec_int = np.interp(wl_grid_main, wl_src, info["si"][pix],
                                 left=0.0, right=0.0)

            if line_name == main_line:
                colr, z, lw = "red", 3, 1.2
            elif line_name == secondary_line:
                colr, z, lw = "blue", 3, 1.2
            else:
                colr, z, lw = "darkgrey", 1, 0.8

            ax_lin.plot(wl_grid_main, spec_int, c=colr, lw=lw, zorder=z)
            ax_log.plot(wl_grid_main, spec_int, c=colr, lw=lw, zorder=z)

        # summed spectrum
        ax_lin.plot(wl_grid_main, total_si[pix], c=tot_colour, lw=2.0, zorder=4)
        ax_log.plot(wl_grid_main, total_si[pix], c=tot_colour, lw=2.0, zorder=4)

        # basic cosmetics
        ax_lin.set_title(title)
        ax_lin.grid(ls=":", alpha=0.5)
        ax_log.grid(ls=":", alpha=0.5)
        ax_lin.tick_params(direction="in", which="both", top=True, right=True)
        ax_log.tick_params(direction="in", which="both", top=True, right=True)

        ax_log.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
        ax_log.set_yscale("log")

        # enforce wavelength limits (velocity follows automatically)
        if wl_min is not None:
            ax_lin.set_xlim(wl_min, wl_max)

        # ================================================================
        # velocity & wavelength ticks
        # ================================================================
        if xlim_vel is not None:
            v_min, v_max = xlim_vel
        else:
            v_min, v_max = wl2v(ax_lin.get_xlim())

        v_locator = MaxNLocator(nbins=5, symmetric=True)
        v_ticks   = v_locator.tick_values(v_min, v_max)

        sec = ax_lin.secondary_xaxis("top", functions=(wl2v, v2wl))
        sec.set_xlabel(f"Velocity ({main_label}) [km/s]")
        sec.set_xticks(v_ticks)
        sec.set_xticklabels([f"{v:.0f}" for v in v_ticks])
        sec.tick_params(direction="in", which="both", top=True, right=True)

        wl_ticks = v2wl(v_ticks)
        wl_labels = [f"{wl:.3f}" for wl in wl_ticks]

        fixed_loc = FixedLocator(wl_ticks)
        fixed_fmt = FixedFormatter(wl_labels)
        ax_lin.xaxis.set_major_locator(fixed_loc)
        ax_lin.xaxis.set_major_formatter(fixed_fmt)
        ax_log.xaxis.set_major_locator(fixed_loc)
        ax_log.xaxis.set_major_formatter(fixed_fmt)

    # ---------------------------------------------------------------------
    # optional tweaks: log-y lower limit & y-axis sci-format
    # ---------------------------------------------------------------------
    for ax in axes[0]:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.set_ylim(bottom=0)

    if yorders is not None:
        for ax in axes[1]:
            _, ymax = ax.get_ylim()
            ax.set_ylim(ymax / 10 ** yorders, ymax)

    # override bottom-row y-limits if provided
    if ylimits is not None:
        ymin, ymax = ylimits
        for ax in axes[1]:
            ax.set_ylim(ymin, ymax)

    # ---------------------------------------------------------------------
    # set top and bottom y-axis labels
    # ---------------------------------------------------------------------
    fig.canvas.draw()
    offset_text = axes[0, 0].yaxis.get_offset_text().get_text().replace("1e", "")
    axes[0, 0].yaxis.get_offset_text().set_visible(False)
    axes[0, 0].set_ylabel(
        fr"$\mathrm{{Intensity\:[10^{{{offset_text}}}\:erg/s/cm^2/sr/cm]}}$"
    )
    axes[1, 0].set_ylabel(
        r"Intensity [erg/s/cm$^2$/sr/cm]"
    )

    # # ---------------------------------------------------------------------
    # # legend (upper-left panel)
    # # ---------------------------------------------------------------------
    # lines_for_total = [
    #   plt.Line2D([0], [0], c=key_pixel_colors[0], lw=2.0),
    #   plt.Line2D([0], [0], c=key_pixel_colors[1], lw=2.0),
    #   plt.Line2D([0], [0], c=key_pixel_colors[2], lw=2.0),
    # ]
    # handles = [
    #   plt.Line2D([0], [0], c="red",      lw=1.2, label=label_main),
    #   plt.Line2D([0], [0], c="blue",     lw=1.2, label=label_sec),
    #   plt.Line2D([0], [0], c="darkgrey", lw=0.8, label="Background"),
    #   tuple(lines_for_total),
    # ]
    # labels = [label_main, label_sec, "Background", "Total"]
    # axes[0, 0].legend(
    #   handles=handles,
    #   labels=labels,
    #   loc="upper left",
    #   fontsize="small",
    #   handler_map={tuple: HandlerTuple(ndivide=None)}
    # )

    # ------------------------------------------------------------------
    #  custom handler: three coloured strokes separated by "/"
    # ------------------------------------------------------------------
    class HandlerTripleWithSlash(HandlerBase):
        """
        Draw three lines separated by forward slashes, with a little
        horizontal padding around every slash.
        """
        def __init__(self, pad_frac=0.08, **kw):
            super().__init__(**kw)
            self.pad_frac = pad_frac    # fraction of the full handle-width

        def create_artists(self, legend, tup, x0, y0, width, height, fontsize, trans):
            # unpack the three Line2D that came in as a tuple
            l1, l2, l3 = tup

            # total free space taken up by two slashes  (each drawn as "/")
            slash_width = width * 0.08
            pad = width * self.pad_frac

            # compute segment widths: split remaining space equally in three
            w_line = (width - 2*slash_width - 4*pad) / 3.0

            # helpful y-coordinate (centre of legend handle)
            yc = y0 + height * 0.5

            artists = []

            # ------- first coloured stroke -------
            x_left = x0
            x_right = x_left + w_line
            artists.append(Line2D([x_left, x_right], [yc, yc],
                                color=l1.get_color(), lw=l1.get_linewidth(),
                                solid_capstyle='butt', transform=trans))

            # ------- slash -------
            x_left = x_right + pad
            x_right = x_left + slash_width
            artists.append(Line2D([x_left, x_right], [y0 + height*0.1,
                                                    y0 + height*0.9],
                                color="k", lw=1.0, transform=trans))

            # ------- second coloured stroke -------
            x_left = x_right + pad
            x_right = x_left + w_line
            artists.append(Line2D([x_left, x_right], [yc, yc],
                                color=l2.get_color(), lw=l2.get_linewidth(),
                                solid_capstyle='butt', transform=trans))

            # ------- second slash -------
            x_left = x_right + pad
            x_right = x_left + slash_width
            artists.append(Line2D([x_left, x_right], [y0 + height*0.1,
                                                    y0 + height*0.9],
                                color="k", lw=1.0, transform=trans))

            # ------- third coloured stroke -------
            x_left = x_right + pad
            x_right = x_left + w_line
            artists.append(Line2D([x_left, x_right], [yc, yc],
                                color=l3.get_color(), lw=l3.get_linewidth(),
                                solid_capstyle='butt', transform=trans))

            return artists

    lines_for_total = [plt.Line2D([0], [0], c=c, lw=2.0) for c in key_pixel_colors]

    handles = [
        Line2D([0], [0], c='red',  lw=1.2, label=label_main),
        Line2D([0], [0], c='blue', lw=1.2, label=label_sec),
        Line2D([0], [0], c='grey', lw=0.8, label='Background'),
        tuple(lines_for_total),
    ]

    labels = [label_main, label_sec, "Background", "Total"]

    axes[0, 0].legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            fontsize="small",
            handlelength=3,                 # make the handle box a bit wider
            handler_map={tuple: HandlerTripleWithSlash(pad_frac=0.1)})

    # ---------------------------------------------------------------------
    # finish up
    # ---------------------------------------------------------------------
    plt.tight_layout(pad=0.1)
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_g_function(
    goft,
    line_name,
    logT_grid,
    logN_grid,
    save="g_function.png",
    xlim=None,
    ylim=None,
    show_vlines=True,
    figsize=(6, 5)
):
    """
    Plot G(logT, logN) for the specified emission line as a 2D colored image.
    Optionally restrict x/y limits and plot vertical lines at mean+-2sigma for the
    optimal density (where G is maximized when integrated over T).
    """
    g_data = goft[line_name]["g_tn"]

    # Optionally crop logT_grid and g_data to xlim
    if xlim is not None:
        mask_T = (logT_grid >= xlim[0]) & (logT_grid <= xlim[1])
        g_data = g_data[:, mask_T]
        logT_grid_plot = logT_grid[mask_T]
    else:
        logT_grid_plot = logT_grid

    # Optionally crop logN_grid and g_data to ylim
    if ylim is not None:
        mask_N = (logN_grid >= ylim[0]) & (logN_grid <= ylim[1])
        g_data = g_data[mask_N, :]
        logN_grid_plot = logN_grid[mask_N]
    else:
        logN_grid_plot = logN_grid

    fig, ax = plt.subplots(figsize=figsize)
    # # use pcolormesh so the axes “know” the exact grid values
    # # X runs over logT, Y over logN, and g_data[i,j] corresponds to (logT_grid_plot[j], logN_grid_plot[i])
    # X, Y = np.meshgrid(logT_grid_plot, logN_grid_plot)
    # im = ax.pcolormesh(
    #   X, Y, g_data,
    #   shading='nearest',
    #   cmap='Greys'
    # )
    extent = (
        logT_grid_plot[0], logT_grid_plot[-1],
        logN_grid_plot[0], logN_grid_plot[-1]
    )
    im = ax.imshow(
        g_data,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="Greys",
    )
    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    # Move the order of magnitude from the colorbar to the label
    fig.canvas.draw()
    offset_text = cbar.ax.yaxis.get_offset_text().get_text().replace("1e", "")
    cbar.ax.yaxis.get_offset_text().set_visible(False)
    cbar.set_label(rf"$G_{{\mathrm{{Fe\,XII\,195.119\,\AA}}}}(T,N)$ [$10^{{{offset_text}}}$ erg cm$^3$/s]")
    ax.set_xlabel(r"$\log_{10}(T\:\mathrm{[K]})$")
    ax.set_ylabel(r"$\log_{10}(N_e\:\mathrm{[1/cm^{3}]})$")

    # Plot vertical lines at mean+-2sigma for the optimal density
    if show_vlines:
        g_tn = goft[line_name]["g_tn"]
        # Use possibly cropped logT_grid and logN_grid for vlines
        g_tn_plot = g_tn
        if ylim is not None:
            g_tn_plot = g_tn_plot[mask_N, :]
        if xlim is not None:
            g_tn_plot = g_tn_plot[:, mask_T]
        # Integrate G over T for each density
        integrated_g_over_T = np.trapz(g_tn_plot, logT_grid_plot, axis=1)
        optimal_density_idx = np.argmax(integrated_g_over_T)
        optimal_density = logN_grid_plot[optimal_density_idx]
        g_at_optimal_density = g_tn_plot[optimal_density_idx, :]

        # Compute mean and std of logT weighted by G
        mean_logT = np.average(logT_grid_plot, weights=g_at_optimal_density)
        std_logT = np.sqrt(np.average((logT_grid_plot - mean_logT) ** 2, weights=g_at_optimal_density))
        vlines = [mean_logT - 2 * std_logT, mean_logT + 2 * std_logT]
        for vline in vlines:
            ax.axvline(
                vline,
                color='grey',
                linestyle=(0, (5, 10)),
                linewidth=1,
                alpha=1.0,
                zorder=2,
            )
        # Add annotation for the vlines
        ax.legend(
            [plt.Line2D([0], [0], color='grey', linestyle=(0, (5, 5)), linewidth=1)],
            # [fr"$G(T,{{N_{{e}}={optimal_density:.1f}}})\pm 2\sigma_{{G}}$"],
            [fr"$G_{{\mathrm{{Fe\,XII\,195.119\,\AA}}}}(T,\,N_{{e}}={optimal_density:.1f})\pm2\sigma_{{G}}$"],
            loc="upper right",
            fontsize="small",
        )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)




##############################################################################
# ---------------------------------------------------------------------------
#  Interactive quick-look viewer
# ---------------------------------------------------------------------------
##############################################################################

def launch_viewer(
    *,                                 # force keyword use
    total_si     : np.ndarray,
    goft         : dict,
    dem_map      : np.ndarray,
    wl_ref       : np.ndarray,
    v_edges      : np.ndarray,
    logT_centres : np.ndarray,
    main_line    : str,
) -> None:
    """
    Opens one overview window (intensity + Doppler map).  Clicking on a pixel
    launches two further windows:

        - DEM(T)    - y-axis in log-scale
        - Spectra   - upper panel linear-y, lower panel log-y
                      - one curve per emission line
                      - thick black curve = summed spectrum
    """

    # ----------------- helper lambdas -----------------
    wl0_A   = goft[main_line]["wl0"].to(u.AA).value
    c_km_s  = const.c.to(u.km/u.s).value
    wl2v    = lambda wl:  (wl - wl0_A) / wl0_A * c_km_s         # A -> km s-1
    v2wl    = lambda vel: (vel / c_km_s)    * wl0_A + wl0_A     # km s-1 -> A

    # velocity grid centres that correspond to wl_ref
    v_centres_km = (0.5 * (v_edges[:-1] + v_edges[1:]) * u.cm/u.s)\
                    .to(u.km/u.s).value

    nx, ny, _ = total_si.shape
    wl_res     = wl_ref[1] - wl_ref[0]

    # ----------------- overview figure -----------------
    fig_ov, (ax_I, ax_V) = plt.subplots(2, 1, figsize=(7, 10))

    si = total_si.sum(axis=2) * wl_res  # integrate over wavelength
    log_si = np.log10(si, where=si > 0.0, out=np.zeros_like(si))

    imI = ax_I.imshow(
        log_si.T,  # log10 of integrated intensity
        origin="lower", aspect="equal", cmap="inferno"
    )
    ax_I.set_title(r"$\log_{10}\!\int I(\lambda)\,\mathrm{d}\lambda$")
    fig_ov.colorbar(imI, ax=ax_I, label=r"$\log_{10} I$  [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ dex$^{-1}$]")

    peak_idx = total_si.argmax(axis=2)             # index of lambda of max signal
    v_map    = v_centres_km[peak_idx]              # Doppler map
    imV = ax_V.imshow(
        v_map.T, origin="lower", aspect="equal",
        cmap="RdBu_r"
    )
    ax_V.set_title("Doppler velocity of peak intensity  [km s-1]")
    imV.set_clim(-30, 30)  # set color limits for velocity map
    fig_ov.colorbar(imV, ax=ax_V, label="v  (km s-1)")

    plt.tight_layout()
    plt.show(block=False)          # keep UI responsive

    # ----------------- click callback -----------------
    def _on_click(event):
      if event.inaxes not in (ax_I, ax_V):        # click somewhere else
        return
      if event.xdata is None or event.ydata is None:
        return

      # pixel indices (round - the images are pixel-aligned)
      x, y = map(int, map(round, (event.xdata, event.ydata)))
      if not (0 <= x < nx and 0 <= y < ny):
        return

      # -------- DEM window --------
      fig_dem, ax_dem = plt.subplots(figsize=(5, 4))
      dem_1d = dem_map[x, y, :]
      log_dem_1d = np.log10(dem_1d, where=dem_1d > 0.0, out=np.zeros_like(dem_1d))
      ax_dem.plot(logT_centres, log_dem_1d, where='mid', lw=1.8)
      ax_dem.set_xlabel(r"\log_{10} T  [K]")
      ax_dem.set_ylabel(r"\log_{10} DEM  [cm$^{-5}$ dex$^{-1}$]")
      ax_dem.set_title(f"DEM(T)  -  pixel ({x},{y})")
      ax_dem.grid(ls=":")
      fig_dem.tight_layout()
      plt.show(block=False)

      # -------- spectra window --------
      fig_sp, (ax_lin, ax_log) = plt.subplots(2, 1, sharex=True,
                          figsize=(7, 8))
      line_names   = list(goft.keys())
      n_lines      = len(line_names)
      cmap         = get_cmap("tab10", n_lines)

      # summed spectrum (already on wl_ref grid)
      summed = total_si[x, y, :]

      # plot each emission line - thin coloured curves
      for i, name in enumerate(line_names):
        spec_px   = goft[name]["si"][x, y, :]
        wl_src    = goft[name]["wl_grid"].to(u.AA).value
        spec_int  = np.interp(wl_ref, wl_src, spec_px, left=0.0, right=0.0)
        lbl       = f"{name}" + ("  (bg)" if goft[name]["background"] else "")
        ax_lin.plot(wl_ref, spec_int, color=cmap(i), lw=1.0, label=lbl)
        ax_log.plot(wl_ref, spec_int, color=cmap(i), lw=1.0)

      # summed spectrum - thick black curve on top
      ax_lin.plot(wl_ref, summed, color="k", lw=2.0, label="total")
      ax_log.plot(wl_ref, summed, color="k", lw=2.0)

      # axis cosmetics for linear panel
      ax_lin.set_ylabel("I  (linear)")
      ax_lin.set_title(f"Spectrum - pixel ({x},{y})")
      ax_lin.grid(ls=":")

      # axis cosmetics for log panel
      ax_log.set_yscale("log")
      ax_log.set_xlabel("Wavelength  (A)")
      ax_log.set_ylabel("I  (log)")
      # ax_log.grid(ls:")

      # add secondary x-axis (velocity) to both panels
      for ax in (ax_lin, ax_log):
        sec = ax.secondary_xaxis("top", functions=(wl2v, v2wl))
        sec.set_xlabel("Velocity  (km s-1)")

      # adjust log y-axis limits: bottom 10 orders below top
      y_top = ax_log.get_ylim()[1]
      ax_log.set_ylim(y_top / 1e10, y_top)

      ax_lin.legend(fontsize="small", ncol=2)

      fig_sp.tight_layout()
      plt.show(block=False)

    # connect callback
    fig_ov.canvas.mpl_connect("button_press_event", _on_click)



    # # ----------------------------------------------------------------------
    # # call viewer after all processing is complete
    # # ----------------------------------------------------------------------
    # launch_viewer(
    #     total_si     = total_si.value,
    #     goft         = goft,
    #     dem_map      = dem_map,
    #     wl_ref       = goft[main_line]["wl_grid"].to(u.AA).value,
    #     v_edges      = v_edges,
    #     logT_centres = logT_centres,
    #     main_line    = main_line,
    # )