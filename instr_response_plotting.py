import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import glob
import dill
from instr_response import angle_to_distance
import matplotlib
from typing import Tuple, Iterable
from tqdm import tqdm

def plot_radiometric_pipeline(
    signals: Tuple[u.Quantity, ...],
    wl_axis: u.Quantity,
    idx_sim_minus: Tuple[int, int] | None,
    idx_sim_mean: Tuple[int, int] | None,
    idx_sim_plus: Tuple[int, int] | None,
    spt_pitch_sim: u.Quantity,          # kept for API-compatibility (unused)
    spt_pitch_instr: u.Quantity,        # kept for API-compatibility (unused)
    save: str = "fig_radiometric_pipeline.png",
    row_labels: Iterable[str] = (r"$\mu-\sigma$", r"$\mu$", r"$\mu+\sigma$"),
    key_pixel_colors: Iterable[str] = ("mediumseagreen", "black", "deeppink"),
) -> plt.Figure:
    # indices are already expressed in detector pixels – no further scaling
    idxs_reb = (idx_sim_minus, idx_sim_mean, idx_sim_plus)

    wl_A = wl_axis.to(u.angstrom).value
    def spectrum(stage_idx: int, row_idx: int) -> np.ndarray:
        return signals[stage_idx][idxs_reb[row_idx] + (slice(None),)]

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(10, 6),
        sharex="row",
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
    )
    fig.subplots_adjust(right=0.86)

    for row in range(3):
        colour = key_pixel_colors[row]
        lab_ax = axes[row, 0].inset_axes([-0.42, 0, 0.1, 1], frameon=False)
        lab_ax.set_axis_off()
        lab_ax.text(0, 0.5, row_labels[row], va="center", ha="left", rotation=90, fontsize=9)

        ax0 = axes[row, 0]
        sp1 = spectrum(1, row)
        ax0.step(wl_A, sp1, where="mid", color=colour, lw=1)
        if row == 0:
            ax0.set_title("signal1/2/3", fontsize=8)
        ax0.set_ylabel(r"ph s$^{-1}$ cm$^{-2}$ sr$^{-1}$ cm$^{-1}$", color=colour, fontsize=7)
        ax0.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

        ax_r1 = ax0.twinx()
        sp2 = spectrum(2, row)
        ax_r1.step(wl_A, sp2, where="mid", color="tab:orange", lw=1)
        ax_r1.set_ylim(sp2.min(), sp2.max())
        ax_r1.set_ylabel(r"ph s$^{-1}$ sr$^{-1}$ cm$^{-1}$", color="tab:orange", fontsize=7)
        ax_r1.yaxis.labelpad = 8
        ax_r1.tick_params(direction="in", colors="tab:orange", which="both", top=True, bottom=True, right=True)
        ax_r1.patch.set_visible(False)

        ax_r2 = ax0.twinx()
        sp3 = spectrum(3, row)
        ax_r2.step(wl_A, sp3, where="mid", color="tab:blue", lw=1)
        ax_r2.set_ylim(sp3.min(), sp3.max())
        ax_r2.spines.right.set_position(("axes", 1.15))
        ax_r2.set_ylabel(r"ph s$^{-1}$ pix$^{-1}$", color="tab:blue", fontsize=7)
        ax_r2.yaxis.labelpad = 24
        ax_r2.tick_params(direction="in", colors="tab:blue", which="both", top=True, bottom=True, right=True)
        ax_r2.patch.set_visible(False)

        ax1 = axes[row, 1]
        sp4 = spectrum(4, row)
        ax1.step(wl_A, sp4, where="mid", color=colour, lw=1)
        if row == 0:
            ax1.set_title("signal4", fontsize=8)
        ax1.set_ylabel(r"ph s$^{-1}$ pix$^{-1}$ (PSF)", color=colour, fontsize=7)
        ax1.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

        ax2 = axes[row, 2]
        sp5 = spectrum(5, row)
        ax2.step(wl_A, sp5, where="mid", color=colour, lw=1)
        if row == 0:
            ax2.set_title("signal5", fontsize=8)
        ax2.set_ylabel(r"e$^-$ pix$^{-1}$", color=colour, fontsize=7)
        ax2.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

        ax3 = axes[row, 3]
        sp6 = spectrum(6, row)
        ax3.step(wl_A, sp6, where="mid", color=colour, lw=1)
        if row == 0:
            ax3.set_title("signal6/7", fontsize=8)
        ax3.set_ylabel(r"e$^-$ pix$^{-1}$ (stray)", color=colour, fontsize=7)
        ax3.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

        ax_r3 = ax3.twinx()
        sp7 = spectrum(7, row)
        ax_r3.step(wl_A, sp7, where="mid", color="tab:red", lw=1)
        ax_r3.spines.right.set_position(("axes", 1.12))
        ax_r3.set_ylim(sp7.min(), sp7.max())
        ax_r3.set_ylabel(r"DN pix$^{-1}$", color="tab:red", fontsize=7)
        ax_r3.yaxis.labelpad = 16
        ax_r3.tick_params(direction="in", colors="tab:red", which="both", top=True, bottom=True, right=True)
        ax_r3.patch.set_visible(False)

        if row == 2:
            for col in range(4):
                axes[row, col].set_xlabel("Wavelength [Å]")
        for col in range(4):
            axes[row, col].tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

    fig.savefig(save, dpi=300)
    plt.close(fig)
    return fig


def plot_maps(
    signal_cube: u.Quantity,
    fit_cube: u.Quantity,
    wl_axis: u.Quantity,
    wl0: u.Quantity,
    idx_sim_minus: Tuple[int, int] | None,
    idx_sim_mean: Tuple[int, int] | None,
    idx_sim_plus: Tuple[int, int] | None,
    photon_cube: u.Quantity | np.ndarray,   # signal6  (photons / pix / λ)
    save: str,
    *,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    key_pixel_colors: Iterable[str] = ("deeppink", "mediumseagreen", "black"),
    previous: dict | None = None,
    save_data_path: str | None = None,
) -> None:
    # --------------------------------------------------------------
    # photons in each CCD row:  Σ_λ  photon_cube(x,y,λ)
    # --------------------------------------------------------------
    ph_int = photon_cube.sum(axis=2) if isinstance(photon_cube, np.ndarray) \
             else photon_cube.sum(axis=2).value          # photons  pix⁻¹

    # ---------------- log10 intensity ----------------------------

    n_scan, n_slit = ph_int.shape
    x_pix_size = SIM.slit_scan_step.to(u.arcsec).value
    y_pix_size = DET.plate_scale_angle.to(u.arcsec / u.pix).value

    x = (np.arange(n_scan) - n_scan // 2) * x_pix_size
    y = (np.arange(n_slit) - n_slit // 2) * y_pix_size
    extent = [
        x[0] - x_pix_size / 2,
        x[-1] + x_pix_size / 2,
        y[0] - y_pix_size / 2,
        y[-1] + y_pix_size / 2,
    ]

    nrows = 2 if previous else 1
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(11, 5),
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
        sharey="row",       # y-axis shared within each row (axV shares y with axI)
    )
    if nrows == 1:
        axes = axes.reshape(1, 2)

    axI, axV = axes[0]
    # imI = axI.imshow(ph_int.T, origin="lower", aspect="auto",
    imI = axI.imshow(np.log10(ph_int, out=np.zeros_like(ph_int), where=ph_int > 0).T,
                     origin="lower", aspect="auto",  #  vmin=0,
                     cmap="afmhot", extent=extent)
    v_map = velocity_from_fit(fit_cube, wl0).to(u.km / u.s)
    imV = axV.imshow(v_map.T.value, origin="lower", aspect="auto",
                     cmap="RdBu_r", vmin=-15, vmax=15, extent=extent)

    # ------------------------------------------------------------------
    # photon colour-bar
    # ------------------------------------------------------------------
    cbarI = fig.colorbar(imI, ax=axI, orientation="horizontal",
                         pad=0.14, shrink=0.95, aspect=35)
    # cbarI.set_label(r"ph/row ")
    cbarI.set_label(
        r"$\log_{10}\!\left(\sum_{\lambda\,\mathrm{pix}}I(\lambda)\:\mathrm{ }\left[\mathrm{ph/s/pix}\right]\right)$"
    )

    # ------------------------------------------------------------------
    # velocity colour-bar (unchanged)
    # ------------------------------------------------------------------
    cbarV = fig.colorbar(imV, ax=axV, orientation="horizontal",
                         pad=0.14, extend="both", shrink=0.95, aspect=35)
    cbarV.set_label(r"$v$ [km/s]")

    # ---------------- spatial zoom if requested ----------------
    if xlim:
        axI.set_xlim(*xlim)
        axV.set_xlim(*xlim)
    if ylim:
        axI.set_ylim(*ylim)
        axV.set_ylim(*ylim)

    # ---------------- formatting, markers, save ----------------
    def _format(ax):
        ax.set_xlabel("X [arcsec]")
        if ax is axI:
            ax.set_ylabel("Y [arcsec]")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        interval = 15.0
        max_x = max(abs(extent[0]), abs(extent[1]))
        max_y = max(abs(extent[2]), abs(extent[3]))
        xticks = np.arange(-np.ceil(max_x / interval) * interval, np.ceil(max_x / interval) * interval + interval / 2, interval)
        yticks = np.arange(-np.ceil(max_y / interval) * interval, np.ceil(max_y / interval) * interval + interval / 2, interval)
        xticks = xticks[(xticks >= extent[0]) & (xticks <= extent[1])]
        yticks = yticks[(yticks >= extent[2]) & (yticks <= extent[3])]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        # pixels are already expressed in arcsec; enforce equal scaling
        ax.set_aspect(1.0)
        ax.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

    _format(axI)
    _format(axV)

    markers = ["2", "3", "1"]
    labels = [r"$\mu-1\sigma$", r"$\mu$", r"$\mu+1\sigma$"]

    # for idx, color, marker in zip(
    for idx, color, marker, label in zip(
        # [idx_sim_minus, idx_sim_mean, idx_sim_plus],
        list(reversed([idx_sim_minus, idx_sim_mean, idx_sim_plus])),
        key_pixel_colors,
        markers,
        # labels
        # list(reversed(key_pixel_colors)),  # reverse this order (for some reason...)
        # list(reversed(markers)),
        list(reversed(labels))
    ):
        if idx is not None:
            x_pos = (idx[0] - n_scan // 2) * x_pix_size
            y_pos = (idx[1] - n_slit // 2) * y_pix_size
            for ax in (axI, axV):
                ax.scatter(x_pos, y_pos, marker=marker, color=color, s=250, linewidth=2, label=label)

    axV.legend(
        loc="upper right",
        fontsize="small")

    plt.tight_layout()
    plt.savefig(save, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_exposure_time_map(
    analysis_per_exp: dict[float, dict],
    precision_requirement: float,
    x_pix_size: float,
    y_pix_size: float,
    save: str,
    cmap: str = "viridis",
) -> None:
    """
    Draw a *discrete* map showing the minimum exposure time that fulfils the
    requested Doppler-velocity precision.

    Pixels that never reach the required precision remain white.  
    The colour-bar’s “extend-max” triangle is also painted white so it is
    visually associated with those pixels.
    """
    # ------------------------------------------------------------------
    # ---- build per-pixel “best exposure” map (in *index* space) -------
    # ------------------------------------------------------------------
    exp_times_sorted = sorted(analysis_per_exp.keys())          # e.g. [0.5, 1, 2 …]
    n_levels         = len(exp_times_sorted)

    shape   = next(iter(analysis_per_exp.values()))["v_std"].shape
    idx_map = np.full(shape, np.nan)                           # pixel → exp-index

    for idx, exp_time in enumerate(exp_times_sorted):
        v_std_map = analysis_per_exp[exp_time]["v_std"].to_value(u.km / u.s)
        mask      = (v_std_map <= precision_requirement) & np.isnan(idx_map)
        idx_map[mask] = idx                                       # store index

    # ------------------------------------------------------------------
    # ---- spatial coordinates -----------------------------------------
    n_scan, n_slit = idx_map.shape
    x = (np.arange(n_scan) - n_scan // 2) * x_pix_size
    y = (np.arange(n_slit) - n_slit // 2) * y_pix_size
    extent = [
        x[0] - x_pix_size / 2, x[-1] + x_pix_size / 2,
        y[0] - y_pix_size / 2, y[-1] + y_pix_size / 2,
    ]

    # ------------------------------------------------------------------
    # ---- discrete colormap & normalisation ---------------------------
    cmap_discrete = ListedColormap(
        plt.get_cmap(cmap)(np.linspace(0, 1, n_levels))
    )
    cmap_discrete.set_over("white")   # colour for “extend-max” triangle
    cmap_discrete.set_bad("white")    # NaNs (unreached precision) → white

    norm = BoundaryNorm(np.arange(-0.5, n_levels + 0.5, 1), n_levels)

    # ------------------------------------------------------------------
    # ---- plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        np.ma.masked_invalid(idx_map).T,      # mask NaNs → white
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap_discrete,
        norm=norm,
    )

    # ---- colour-bar with discrete ticks ------------------------------
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.12,
        ticks=np.arange(n_levels),
        shrink=0.95,
        aspect=35,
        extend="max",                 # show white triangle on the right
    )
    cbar.set_ticklabels([f"{t:g} s" for t in exp_times_sorted])
    cbar.set_label(
        rf"Minimum exposure time to reach $\sigma_v \leq {precision_requirement:.1f}\:\mathrm{{km/s}}$"
    )

    # ------------------------------------------------------------------
    # ---- cosmetics ---------------------------------------------------
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    ax.set_aspect(1)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_maps(
    v_map: u.Quantity,
    v_std_map: u.Quantity,
    save: str,
    x_pix_size: float,
    y_pix_size: float,
    key_pixel_colors: Iterable[str] = ("mediumseagreen", "black", "deeppink"),
    idx_minus: Tuple[int, int] | None = None,
    idx_mean: Tuple[int, int] | None = None,
    idx_plus: Tuple[int, int] | None = None,
    vmin: float = -15,
    vmax: float = 15,
    std_vmax: float = 5,
) -> None:
    n_scan, n_slit = v_map.shape
    x = (np.arange(n_scan) - n_scan // 2) * x_pix_size
    y = (np.arange(n_slit) - n_slit // 2) * y_pix_size
    extent = [x[0] - x_pix_size / 2, x[-1] + x_pix_size / 2, y[0] - y_pix_size / 2, y[-1] + y_pix_size / 2]

    fig, (axV, axStd) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw=dict(wspace=0.0, hspace=0.0), sharey=True)

    # Doppler velocity map (left panel)
    imV = axV.imshow(v_map.T.to(u.km / u.s).value, origin="lower", aspect="auto",
                     cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent)
    cbarV = fig.colorbar(imV, ax=axV, orientation="horizontal", pad=0.14, extend="both", shrink=0.95, aspect=35)
    cbarV.set_label(r"$v$ [km/s]")

    # Velocity standard deviation map (right panel)
    imStd = axStd.imshow(v_std_map.T.to(u.km / u.s).value, origin="lower", aspect="auto",
                         cmap="magma", vmin=0, vmax=std_vmax, extent=extent)
    cbarStd = fig.colorbar(imStd, ax=axStd, orientation="horizontal", pad=0.14, shrink=0.95, aspect=35, extend="max")
    cbarStd.set_label(r"$\sigma_v$ [km/s]")

    # Formatting
    for ax in [axV, axStd]:
        ax.set_xlabel("X [arcsec]")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect(1.0)
        ax.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

        interval = 15.0
        max_x = max(abs(extent[0]), abs(extent[1]))
        max_y = max(abs(extent[2]), abs(extent[3]))
        xticks = np.arange(-np.ceil(max_x / interval) * interval, np.ceil(max_x / interval) * interval + interval / 2, interval)
        yticks = np.arange(-np.ceil(max_y / interval) * interval, np.ceil(max_y / interval) * interval + interval / 2, interval)
        xticks = xticks[(xticks >= extent[0]) & (xticks <= extent[1])]
        yticks = yticks[(yticks >= extent[2]) & (yticks <= extent[3])]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

    axV.set_ylabel("Y [arcsec]")

    # # Markers for key pixels
    # markers = ["2", "3", "1"]
    # labels = [r"$\mu-1\sigma$", r"$\mu$", r"$\mu+1\sigma$"]
    # for idx, color, marker, label in zip(
    #     [idx_minus, idx_mean, idx_plus],
    #     key_pixel_colors,
    #     markers,
    #     labels
    # ):
    #     if idx is not None:
    #         x_pos = (idx[0] - n_scan // 2) * x_pix_size
    #         y_pos = (idx[1] - n_slit // 2) * y_pix_size
    #         for ax in (axV, axStd):
    #             ax.scatter(x_pos, y_pos, marker=marker, color=color, s=250, linewidth=2, label=label)

    # axStd.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.savefig(save, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_multi_maps(
    npz_files: list[str | Path],
    map_type: str = "intensity",                    # "intensity" | "velocity"
    *,
    key_pixel_colors: Iterable[str] = ("mediumseagreen", "black", "deeppink"),
    markers: list[str] = ("2", "3", "1"),
    labels: list[str]  = (r"$\mu-1\sigma$", r"$\mu$", r"$\mu+1\sigma$"),
    save: str = "fig_multi_maps.png",
    figsize: tuple = (10, 6.75),
):
    """
    Stack several intensity / velocity maps vertically.

    If `map_type == "intensity"` the quantity shown is the *integrated photon
    count per CCD row* (Σ_λ ph pix⁻¹ s⁻¹) exactly like in `plot_maps`, not the
    specific intensity.
    """
    # ------------------------------------------------------------------
    # ---- load maps + ancillary info ----------------------------------
    maps, dxs, dys, idxs_list = [], [], [], []
    for npz_file in npz_files:
        dat = np.load(npz_file, allow_pickle=True)

        # ---- map -----------------------------------------------------
        if map_type == "intensity":
            # Prefer integrated photon map derived from first_signals[4]
            if "first_signals" in dat.files:
                photon_cube = dat["first_signals"][4]          # signal4
                if isinstance(photon_cube, np.ndarray):
                    m = photon_cube.sum(axis=2)                       # ph  pix⁻¹ s⁻¹
                else:                                                 # Quantity
                    m = photon_cube.sum(axis=2).value
            elif "si_map" in dat:
                m = dat["si_map"]                                     # fallback
            elif "log_si" in dat:
                m = 10 ** dat["log_si"]
            else:
                raise ValueError(f"no suitable intensity map in {npz_file}")
        elif map_type == "velocity":
            if "analysis_res" in dat:
                m = dat["analysis_res"].item()["v_mean"].to_value(u.km / u.s)
            elif "v_map" in dat:
                m = dat["v_map"]
            else:
                raise ValueError(f"no velocity map in {npz_file}")
        else:
            raise ValueError("map_type must be 'intensity' or 'velocity'")
        maps.append(m)

        # ---- pixel sizes --------------------------------------------
        if {"x_pix_size", "y_pix_size"} <= set(dat.files):
            dxs.append(float(dat["x_pix_size"]))
            dys.append(float(dat["y_pix_size"]))
        else:
            sim = dat["SIM"].item()
            det = dat["DET"].item()
            dxs.append(sim.slit_scan_step.to_value(u.arcsec))
            dys.append(det.plate_scale_angle.to_value(u.arcsec / u.pix))

        # ---- key-pixel indices --------------------------------------
        idxs: list[tuple[int, int] | None] = [None, None, None]
        if "plotting" in dat.files:
            plotting = dat["plotting"].item()
            for n, k in enumerate(("minus_idx", "mean_idx", "plus_idx")):
                if plotting.get(k) is not None:
                    idxs[n] = tuple(plotting[k])
        idxs_list.append(idxs)

    # ------------------------------------------------------------------
    # ---- figure + adaptive GridSpec ----------------------------------
    nrows   = len(maps)
    heights = [m.shape[1] * dy for m, dy in zip(maps, dys)]
    fig     = plt.figure(figsize=figsize)
    gs      = fig.add_gridspec(nrows, 1, height_ratios=heights, hspace=0.0)
    axes    = [fig.add_subplot(gs[i, 0]) for i in range(nrows)]

    # ------------------------------------------------------------------
    # ---- global X-extent (shared across rows) ------------------------
    xmins, xmaxs = [], []
    for m, dx in zip(maps, dxs):
        n_scan = m.shape[0]
        x      = (np.arange(n_scan) - n_scan // 2) * dx
        xmins.append(x[0] - dx / 2)
        xmaxs.append(x[-1] + dx / 2)
    xlim = (min(xmins), max(xmaxs))

    # ------------------------------------------------------------------
    # ---- plot each map ----------------------------------------------
    for ax, m, dx, dy, idxs in zip(axes, maps, dxs, dys, idxs_list):
        n_scan, n_slit = m.shape
        x = (np.arange(n_scan) - n_scan // 2) * dx
        y = (np.arange(n_slit) - n_slit // 2) * dy
        extent = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]

        # ---- data + colour scaling ----------------------------------
        if map_type == "intensity":
            data = np.log10(m, out=np.zeros_like(m), where=m > 0)
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            cmap       = "afmhot"
            cbar_label = r"$\log_{10}\!\left(\sum_{\lambda\,\mathrm{pix}}I(\lambda)\:\mathrm{ }\left[\mathrm{ph/s/pix}\right]\right)$"
        else:  # velocity
            data = m
            vmax = np.nanmax(np.abs(data))
            vmin = -vmax
            cmap = "RdBu_r"
            cbar_label = r"$v$ [km/s]"

        im = ax.imshow(
            data.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )

        # ---- ticks / labels ----------------------------------------
        ax.set_xlim(*xlim)
        ax.set_aspect(1)
        ax.tick_params(direction="in", which="both", top=True, right=True)

        # ---- markers -----------------------------------------------
        for idx, color, marker, lab in zip(
            reversed(idxs),
            key_pixel_colors,
            markers,
            reversed(labels),
        ):
            if idx is None:
                continue
            # ax.scatter(
            #     (idx[0] - n_scan // 2) * dx,
            #     (idx[1] - n_slit // 2) * dy,
            #     marker=marker,
            #     color=color,
            #     s=230,
            #     lw=2,
            #     label=lab,
            # )
        # place legend only on the bottom row, lower-right corner
        # if ax is axes[-1]:
        #     ax.legend(loc="lower right", fontsize="small")

        # ---- individual colour-bar ---------------------------------
        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            pad=0.04,
            shrink=0.92,
            aspect=20,
        )
        cbar.set_label(cbar_label)

    axes[-1].set_xlabel("X [arcsec]")
    for ax in axes:
        ax.set_ylabel("Y [arcsec]")

    plt.savefig(save, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_std_map(
    v_std_map: u.Quantity,
    save: str,
    x_pix_size: float,
    y_pix_size: float,
    key_pixel_colors: Iterable[str] = ("mediumseagreen", "black", "deeppink"),
    idx_minus: Tuple[int, int] | None = None,
    idx_mean: Tuple[int, int] | None = None,
    idx_plus: Tuple[int, int] | None = None,
) -> None:
    n_scan, n_slit = v_std_map.shape
    x = (np.arange(n_scan) - n_scan // 2) * x_pix_size
    y = (np.arange(n_slit) - n_slit // 2) * y_pix_size
    extent = [x[0] - x_pix_size / 2, x[-1] + x_pix_size / 2, y[0] - y_pix_size / 2, y[-1] + y_pix_size / 2]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(v_std_map.T.to(u.km / u.s).value, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=0, vmax=5)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.1)
    cbar.set_label(r"$\sigma_v$ [km/s]")

    interval = 15.0
    max_x = max(abs(extent[0]), abs(extent[1]))
    max_y = max(abs(extent[2]), abs(extent[3]))
    xticks = np.arange(-np.ceil(max_x / interval) * interval, np.ceil(max_x / interval) * interval + interval / 2, interval)
    yticks = np.arange(-np.ceil(max_y / interval) * interval, np.ceil(max_y / interval) * interval + interval / 2, interval)
    xticks = xticks[(xticks >= extent[0]) & (xticks <= extent[1])]
    yticks = yticks[(yticks >= extent[2]) & (yticks <= extent[3])]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    ax.set_aspect(1.0)
    ax.tick_params(direction="in", which="both", top=True, bottom=True, left=True, right=True)

    for idx, color in zip([idx_minus, idx_mean, idx_plus], key_pixel_colors):
        if idx is not None:
            ax.plot(
                (idx[0] - n_scan // 2) * x_pix_size,
                (idx[1] - n_slit // 2) * y_pix_size,
                marker="o",
                color=color,
                markersize=8,
                fillstyle="none",
                lw=2,
            )

    plt.tight_layout(pad=0.1)
    plt.savefig(save, dpi=300)
    plt.close(fig)


def plot_intensity_vs_vstd(
    intensity: np.ndarray,
    v_std: u.Quantity,
    save: str,
    *,                           # keep new kw-only args after here
    fit_intensity_min: float | None = None,
    vstd_max: float | None = None,                 # NEW: upper σ_v cut-off (km/s)
    idx_minus: Tuple[int, int] | None = None,      # NEW
    idx_mean : Tuple[int, int] | None = None,      # NEW
    idx_plus : Tuple[int, int] | None = None,      # NEW
    key_pixel_colors: Tuple[str, str, str] = ("deeppink", "black", "mediumseagreen"),
    labels: Tuple[str, str, str] = (r"$\mu-1\sigma$", r"$\mu$", r"$\mu+1\sigma$"),
) -> None:
    """
    Scatter of per-pixel intensity versus 1-σ velocity uncertainty plus two
    log–log linear fits.  Optionally draws vertical lines at the intensities
    of reference pixels (μ−σ, μ, μ+σ).

    Parameters
    ----------
    …
    idx_minus / idx_mean / idx_plus : tuple(int,int), optional
        Indices of the “μ−σ”, “μ”, “μ+σ” pixels.  If provided, a vertical
        line is drawn at the corresponding intensity.
    key_pixel_colors : tuple(str,str,str), optional
        Colours for the three reference pixels.
    labels : tuple(str,str,str), optional
        Legend labels for the three reference pixels.
    """
    inten = intensity.ravel()
    vstd  = v_std.to(u.km / u.s).value.ravel()

    # ------------------------------------------------------------------
    # ---- basic masks --------------------------------------------------
    valid_scatter = (inten > 0) & (vstd > 0)
    if vstd_max is not None:                          #  ← NEW
        valid_scatter &= (vstd <= vstd_max)

    if fit_intensity_min is not None:
        valid_fit = valid_scatter & (inten >= fit_intensity_min)
    else:
        valid_fit = valid_scatter

    # ------------------------------------------------------------------
    # ---- log-space arrays --------------------------------------------
    log_i_all = np.log10(inten[valid_scatter])
    log_v_all = np.log10(vstd [valid_scatter])

    log_i_fit = np.log10(inten[valid_fit])
    log_v_fit = np.log10(vstd [valid_fit])

    # ------------------------------------------------------------------
    # ---- global linear fit (log–log) ---------------------------------
    coeff = np.polyfit(log_i_fit, log_v_fit, 1)
    fit_x = np.linspace(log_i_fit.min(), log_i_fit.max(), 100)
    fit_y = coeff[0] * fit_x + coeff[1]

    # -------- create equation string for legend -----------------------
    m_global = coeff[0]
    c_global = coeff[1]
    A_global = 10 ** c_global
    # Format A_global in scientific notation as 10^{exp}
    if A_global != 0:
        exp_A = int(np.floor(np.log10(abs(A_global))))
        mant_A = A_global / (10 ** exp_A)
        if np.isclose(mant_A, 1.0):
            A_str = fr"10^{{{exp_A}}}"
        else:
            A_str = fr"{mant_A:.2g} \times 10^{{{exp_A}}}"
    else:
        A_str = "0"
    label_global = f"$\\sigma_v = {A_str} \\cdot I^{{{m_global:.2g}}}$"

    # ------------------------------------------------------------------
    # ---- ridge (upper envelope) fit ----------------------------------
    nbins = 25
    bins  = np.linspace(log_i_fit.min(), log_i_fit.max(), nbins + 1)
    bin_cent  = 0.5 * (bins[:-1] + bins[1:])
    max_log_v = np.full(nbins, np.nan)
    for k in range(nbins):
        m = (log_i_fit >= bins[k]) & (log_i_fit < bins[k + 1])
        if np.any(m):
            max_log_v[k] = log_v_fit[m].max()
    valid_bin   = ~np.isnan(max_log_v)
    ridge_coeff = np.polyfit(bin_cent[valid_bin], max_log_v[valid_bin], 1)
    ridge_y     = ridge_coeff[0] * fit_x + ridge_coeff[1]

    # ------------------------------------------------------------------
    # ---- main figure --------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(inten[valid_scatter], vstd[valid_scatter],
               s=1, color="dimgray", alpha=1)
    ax.plot(10 ** fit_x, 10 ** fit_y,
            color="red", label=label_global)
    # ax.plot(10 ** fit_x, 10 ** ridge_y,
    #         color="limegreen", ls="--", lw=1.5, label="ridge fit")

    # ------------------------------------------------------------------
    # ---- optional vertical lines for key pixels ----------------------
    # requested = [
    #     (idx_minus, key_pixel_colors[0], labels[0]),
    #     (idx_mean,  key_pixel_colors[1], labels[1]),
    #     (idx_plus,  key_pixel_colors[2], labels[2]),
    # ]
    requested = [
        (idx_plus,  key_pixel_colors[2], labels[2]),
        (idx_mean,  key_pixel_colors[1], labels[1]),
        (idx_minus, key_pixel_colors[0], labels[0]),
    ]
    for idx, colour, lab in requested:
        if idx is None:
            continue
        inten_val = intensity[idx]
        if inten_val > 0:
            ax.axvline(inten_val, color=colour, ls="--", lw=1.3, label=lab)

    # ------------------------------------------------------------------
    # ---- final cosmetics & save --------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Intensity [erg/s/cm$^2$/sr]")
    ax.set_ylabel(r"$\sigma_v$ [km/s]")
    ax.legend(fontsize="small")
    ax.tick_params(direction="in", which="both", top=True, bottom=True,
                   left=True, right=True)

    plt.tight_layout(pad=0.1)
    plt.savefig(save, dpi=300)
    plt.close(fig)

def plot_vstd_vs_exposure(
    analysis_per_exp: dict[float, dict],
    *,                                        # --- all arguments after this are keyword-only
    idx_minus: Tuple[int, int] | None = None,
    idx_mean : Tuple[int, int] | None = None,
    idx_plus : Tuple[int, int] | None = None,
    key_pixel_colors: Tuple[str, str, str] = ("deeppink", "black", "mediumseagreen"),
    labels: Tuple[str, str, str] = (r"$\mu-1\sigma$", r"$\mu$", r"$\mu+1\sigma$"),
    save: str = "fig_vstd_vs_exposure.png",
    log_x: bool = True,
    log_y: bool = True,                     # <-- NEW
    vstd_max: float | None = None,
) -> None:
    """
    σ_v versus exposure time for the reference pixels.

    …
    log_y : bool, optional
        Set y-axis to logarithmic scale if *True*.
    """
    exp_times = sorted(analysis_per_exp.keys())                     # ascending [s]

    # ---- keep only the pixels that were actually requested ----------
    # requested = [
    #     (idx_minus, key_pixel_colors[0], labels[0]),
    #     (idx_mean,  key_pixel_colors[1], labels[1]),
    #     (idx_plus,  key_pixel_colors[2], labels[2]),
    # ]
    requested = [
        (idx_plus,  key_pixel_colors[2], labels[2]),
        (idx_mean,  key_pixel_colors[1], labels[1]),
        (idx_minus, key_pixel_colors[0], labels[0]),
    ]
    requested = [(idx, col, lab) for idx, col, lab in requested if idx is not None]
    if not requested:
        warnings.warn("No pixel indices provided; nothing to plot.")
        return

    # ---- gather σ_v --------------------------------------------------
    curves: dict[Tuple[int, int], list[float]] = {
        idx: [] for idx, _, _ in requested
    }
    for sec in exp_times:
        v_std_map = analysis_per_exp[sec]["v_std"].to_value(u.km / u.s)
        for idx in curves.keys():
            val = v_std_map[idx]
            if vstd_max is not None and val > vstd_max:
                val = np.nan
            curves[idx].append(val)

    # ---- plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.25, 3.5))
    for (idx, color, lab) in requested:
        y = curves[idx]
        if not np.all(np.isnan(y)):
            ax.plot(exp_times, y, marker="o", color=color, label=lab)

    ax.set_xlabel("Exposure time [s]")
    ax.set_ylabel(r"$\sigma_v$ [km/s]")

    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(exp_times)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if log_y:                               # <-- NEW
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
    plotting = dat["plotting"]
    cube_reb = dat["cube_reb"]
    wl_axis = dat["wl_axis"]
    wl0 = dat["wl0"]
    spt_sim = dat["spt_sim"]
    DET = dat["DET"]
    SIM = dat["SIM"]
    return results, config, plotting, cube_reb, wl_axis, wl0, spt_sim, DET, SIM

def get_plot_dir(base, contamination):
    return os.path.join(base, f"contam_{contamination}")

def plot_all_for_simulation(
    sim_name,
    results,
    plotting,
    cube_reb,
    wl_axis,
    wl0,
    spt_sim,
    DET,
    SIM,
    output_dir,
):
    for contamination, res in results.items():
        plot_dir = get_plot_dir(output_dir, contamination)
        ensure_dir(plot_dir)
        first_signal_per_exp = res["first_signal_per_exp"]
        first_fit_per_exp = res["first_fit_per_exp"]
        analysis_per_exp = res["analysis_per_exp"]

        # Per-exposure plots
        for sec, first_signals in first_signal_per_exp.items():
            first_fits = first_fit_per_exp[sec]
            analysis_res = analysis_per_exp[sec]
            sec_str = f"{sec:.1f}s"

            plot_maps(
                cube_reb,
                first_fits,
                wl_axis,
                wl0,
                plotting["minus_idx"],
                plotting["mean_idx"],
                plotting["plus_idx"],
                photon_cube=first_signals[4],
                save=os.path.join(plot_dir, f"maps_{sec_str}.png"),
                previous=None,
                save_data_path=None,
            )

            plot_radiometric_pipeline(
                signals=first_signals,
                wl_axis=wl_axis,
                idx_sim_minus=plotting["minus_idx"],
                idx_sim_mean=plotting["mean_idx"],
                idx_sim_plus=plotting["plus_idx"],
                spt_pitch_sim=spt_sim,
                spt_pitch_instr=DET.plate_scale_length,
                save=os.path.join(plot_dir, f"radiometric_pipeline_{sec_str}.png"),
            )

            plot_velocity_std_map(
                v_std_map=analysis_res["v_std"],
                save=os.path.join(plot_dir, f"vstd_{sec_str}.png"),
                x_pix_size=SIM.slit_scan_step.to(u.arcsec).value,
                y_pix_size=DET.plate_scale_angle.to(u.arcsec / u.pix).value,
                idx_minus=plotting["minus_idx"],
                idx_mean=plotting["mean_idx"],
                idx_plus=plotting["plus_idx"],
            )

            wl_pitch = (wl_axis[1] - wl_axis[0]).cgs
            intensity = (first_signals[0].sum(axis=2) * wl_pitch).value
            plot_intensity_vs_vstd(
                intensity=intensity,
                v_std=analysis_res["v_std"],
                save=os.path.join(plot_dir, f"intensity_vs_vstd_{sec_str}.png"),
                vstd_max=1e9,
                fit_intensity_min=1e2,
                idx_minus=plotting.get("minus_idx"),
                idx_mean=plotting.get("mean_idx"),
                idx_plus=plotting.get("plus_idx"),
            )

            plot_spectra(
                dn_cube=first_signals[7],
                wl_axis=wl_axis,
                idx_sim_minus=plotting["minus_idx"],
                idx_sim_mean=plotting["mean_idx"],
                idx_sim_plus=plotting["plus_idx"],
                wl0=wl0,
                fit_cube=first_fits,
                sigma_factor=plotting["sigma_factor"],
                key_pixel_colors=("mediumseagreen", "black", "deeppink"),
                save=os.path.join(plot_dir, f"spectra_dn_{sec_str}.png"),
            )

            plot_velocity_maps(
                analysis_res["v_mean"],
                analysis_res["v_std"],
                save=os.path.join(plot_dir, f"velocity_maps_{sec_str}.png"),
                x_pix_size=SIM.slit_scan_step.to_value(u.arcsec),
                y_pix_size=DET.plate_scale_angle.to_value(u.arcsec / u.pix),
                idx_minus=plotting["minus_idx"],
                idx_mean=plotting["mean_idx"],
                idx_plus=plotting["plus_idx"],
            )

        plot_exposure_time_map(
            analysis_per_exp=analysis_per_exp,
            precision_requirement=2.0,
            x_pix_size=SIM.slit_scan_step.to_value(u.arcsec),
            y_pix_size=DET.plate_scale_angle.to_value(u.arcsec / u.pix),
            save=os.path.join(plot_dir, "exposure_time_map.png"),
            cmap="viridis",
        )

        plot_vstd_vs_exposure(
            analysis_per_exp=analysis_per_exp,
            idx_minus=plotting.get("minus_idx"),
            idx_mean=plotting.get("mean_idx"),
            idx_plus=plotting.get("plus_idx"),
            save=os.path.join(plot_dir, "vstd_vs_exposure.png"),
            log_x=True,
            vstd_max=60,
        )


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

    # Find all *_results.pkl files in the current directory (or specify a directory)
    npz_files = sorted(glob.glob("*_results.pkl"))
    if not npz_files:
        print("No *_results.pkl files found in current directory.")
        return

    for npz_file in npz_files:
        print(f"Processing {npz_file} ...")
        results, config, plotting, cube_reb, wl_axis, wl0, spt_sim, DET, SIM = load_pkl_results(npz_file)

        sim_name = os.path.splitext(os.path.basename(npz_file))[0]
        output_dir = f"{sim_name}_plots"
        ensure_dir(output_dir)
        plot_all_for_simulation(
            sim_name,
            results,
            plotting,
            cube_reb,
            wl_axis,
            wl0,
            spt_sim,
            DET,
            SIM,
            output_dir,
        )
        print(f"Plots for {sim_name} saved in {output_dir}/")

    # Example usage for multi-file vstd vs contamination plot:
    # Uncomment and specify your files below:
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