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
import warnings
from matplotlib.colors import ListedColormap, BoundaryNorm
from astropy.coordinates import SkyCoord, SpectralCoord
from sunpy.coordinates import Helioprojective
from astropy.time import Time
import astropy.constants as const
import astropy.units as u
from ndcube import NDCube
import sunpy.map
from astropy.wcs import WCS
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def find_key_heliocentric_coords(
    cube: NDCube,
    *,
    sigma_factor: float = 1.0,
    margin_frac: float = 0.20,
) -> tuple[Tuple[u.Quantity, u.Quantity],    # mean
           Tuple[u.Quantity, u.Quantity],    # +sigma
           Tuple[u.Quantity, u.Quantity]]:   # −sigma
    """
    Find heliocentric (y, x) coordinates of pixels whose integrated intensity
    is closest to the cube mean and to mean ± sigma_factor·std.

    Parameters
    ----------
    cube : NDCube
        Specific-intensity cube with heliocentric WCS (WAVE, SOLY, SOLX).
    sigma_factor : float, optional
        Multiple of sigma around the mean to identify the ±sigma pixels.
    margin_frac : float, optional
        Fractional margin to exclude from the search (crop off edges).

    Returns
    -------
    (mean_xy, plus_xy, minus_xy) where each entry is a 2-tuple of Quantities
    (y_coord, x_coord) in the cube’s native heliocentric units.
    """

    # --- integrated intensity image -----------------------------------
    wl_step = cube.wcs.wcs.cdelt[0] * cube.wcs.wcs.cunit[0]
    ii = (cube.data.sum(axis=2) * wl_step).cgs.value          # (nx, ny)

    nx, ny = ii.shape
    m = int(margin_frac * min(nx, ny))
    sub = ii[m:nx - m, m:ny - m]

    mean_val = sub.mean()
    std_val  = sub.std()

    targets = [mean_val,
               mean_val + sigma_factor*std_val,
               mean_val - sigma_factor*std_val]

    pix_indices = []
    for t in targets:
        rel = np.abs(sub - t)
        j, i = np.unravel_index(rel.argmin(), sub.shape)      # note j,i order
        pix_indices.append((i + m, j + m))                    # global (x, y)

    # --- convert to heliocentric coordinates --------------------------
    coords = []
    for x_pix, y_pix in pix_indices:
        _, yw, xw = cube.wcs.pixel_to_world_values(           # drop wavelength
            0, y_pix, x_pix
        )
        coords.append((yw * cube.wcs.wcs.cunit[1],
                       xw * cube.wcs.wcs.cunit[2]))

    return coords[0], coords[1], coords[2]


def heliocentric_to_helioprojective_coords(
    xy_hc: Tuple[u.Quantity, u.Quantity],
    *,
    observer: str = "earth",
    obstime: Time | str = "2025-01-01T00:00:00",
) -> SkyCoord:
    """
    Convert a single (y, x) heliocentric coordinate pair to a
    Helioprojective `~astropy.coordinates.SkyCoord`.

    Parameters
    ----------
    xy_hc : tuple of (`~astropy.units.Quantity`, `~astropy.units.Quantity`)
        (y_coord, x_coord) in heliocentric coordinates.
    observer : str or SunPy observer, optional
        Observer location (default ``"earth"``).
    obstime : `~astropy.time.Time` or str, optional
        Observation time (default ``"2025-01-01T00:00:00"``).

    Returns
    -------
    hp : `~astropy.coordinates.SkyCoord`
        The position in the Helioprojective frame.
    """
    obstime = Time(obstime)

    y_hc, x_hc = xy_hc
    hc = SkyCoord(
        x_hc,
        y_hc,
        const.R_sun.to(u.Mm),
        frame="heliocentric",
        observer=observer,
        obstime=obstime,
    )

    return hc.transform_to(Helioprojective(observer=observer, obstime=obstime))


def _xyhc_to_pixel(
    xy_hc: Tuple[u.Quantity, u.Quantity], cube: NDCube
) -> tuple[int, int]:
    """
    Convert heliocentric (y, x) coordinates to integer (y, x) pixel indices for
    `cube`.  The wavelength value is set to zero - only the spatial indices are
    required for extracting the spectrum.
    """
    y_hc, x_hc = xy_hc
    # convert world → pixel; order on return: (λ, y, x)
    _, y_pix, x_pix = cube.wcs.world_to_pixel_values(
        0.0,                                       # wavelength (dummy)
        y_hc.to(cube.wcs.wcs.cunit[1]).value,
        x_hc.to(cube.wcs.wcs.cunit[2]).value,
    )
    return int(round(float(y_pix))), int(round(float(x_pix)))


def plot_key_pixel_spectra(
    total_cube: NDCube,
    line_cubes: dict[str, NDCube],
    *,
    minus_xy_hc: Tuple[u.Quantity, u.Quantity],
    mean_xy_hc: Tuple[u.Quantity, u.Quantity],
    plus_xy_hc: Tuple[u.Quantity, u.Quantity],
    sigma_factor: float = 1.0,
    save: str = "key_pixel_spectrum.png",
    # Optional keyword arguments for customization
    int_unit_label: str | None = None,           # Override intensity unit label
    log_ylim_lower: float | None = None,         # Lower y-limit for log plot
    x_lim_velocity: Tuple[float, float] | None = None,   # (v_min, v_max) in km/s
    tick_spacing_velocity: float | None = None,  # Spacing of velocity ticks
    lin_ylim_pad: float | None = 0.10,           # Fractional padding for linear y-limit
    log_ylim_pad: float | None = 10**0.5,        # Multiplicative padding for log y-limit
    total_colours: Iterable[str] = ("deeppink", "black", "mediumseagreen"),
    highlight_lines: dict[str, str] | Iterable[str] | None = None,  # Highlighted lines/colors
    highlight_lines_labels: dict[str, str] | None = None,           # Custom legend labels
    velocity_label_main_line: str | None = None,                    # Label for velocity axis
) -> None:
    """
    Plot spectra for three key pixels (mean, mean+sigma, mean-sigma) from a synthetic cube.

    This function generates a 2x3 grid of plots:
        - Top row: linear y-scale spectra for each key pixel.
        - Bottom row: log y-scale spectra for each key pixel.
    Each column corresponds to a different pixel (mean, mean+sigma, mean-sigma).
    The function supports highlighting specific spectral lines, custom axis labels,
    and overlays a multi-color legend for the total spectrum.

    Parameters
    ----------
    total_cube : NDCube
        The total intensity cube.
    line_cubes : dict[str, NDCube]
        Dictionary of line-specific cubes.
    minus_xy_hc, mean_xy_hc, plus_xy_hc : Tuple[u.Quantity, u.Quantity]
        Heliocentric coordinates for the three key pixels.
    sigma_factor : float, optional
        Sigma factor for pixel selection (default 1.0).
    save : str, optional
        Output filename for the plot.
    int_unit_label : str, optional
        Override for the intensity unit label.
    log_ylim_lower : float, optional
        Lower y-limit for log plots.
    x_lim_velocity : Tuple[float, float], optional
        Velocity limits for the x-axis (in km/s).
    tick_spacing_velocity : float, optional
        Spacing for velocity ticks (in km/s).
    lin_ylim_pad : float, optional
        Fractional padding for linear y-limit.
    log_ylim_pad : float, optional
        Multiplicative padding for log y-limit.
    total_colours : Iterable[str], optional
        Colors for the total spectrum in each column.
    highlight_lines : dict or iterable, optional
        Lines to highlight (with optional custom colors).
    highlight_lines_labels : dict, optional
        Custom labels for highlighted lines.
    velocity_label_main_line : str, optional
        Label for the velocity axis (top x-axis).
    """

    # Use provided highlight line labels or default to empty dict
    highlight_lines_labels = highlight_lines_labels or {}

    # Build mapping {line_name: color} for highlighted lines
    if highlight_lines is None:
        hl_mapping: dict[str, str] = {}
    elif isinstance(highlight_lines, dict):
        hl_mapping = dict(highlight_lines)  # User-supplied colors
    else:
        # Assign default colors to highlighted lines
        default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colour_pool = ["red", "blue"] + [c for c in default_cycle if c not in ("red", "blue")]
        hl_mapping = {name: colour_pool[i % len(colour_pool)] for i, name in enumerate(highlight_lines)}

    # Helper: get rest wavelength from cube metadata (in Angstrom)
    wl0 = u.Quantity(total_cube.meta["rest_wav"]).to(u.AA)

    # Helper: convert wavelength(s) in Angstrom to velocity (km/s)
    def wl2v_axis(wl_angstrom: np.ndarray | float) -> np.ndarray | float:
        wl_q = u.Quantity(wl_angstrom, u.AA)
        v_q = (wl_q - wl0) / wl0 * const.c
        return v_q.to(u.km/u.s).value

    # Helper: convert velocity (km/s) to wavelength(s) in Angstrom
    def v2wl_axis(v_kms: np.ndarray | float) -> np.ndarray | float:
        v_q = u.Quantity(v_kms, u.km/u.s)
        wl_q = wl0 * (1 + v_q / const.c.to(u.km/u.s))
        return wl_q.to(u.AA).value

    # Wavelength grid for total cube (in Angstrom)
    wl_grid = total_cube.axis_world_coords(2)[0].to(u.AA).value

    # Determine intensity unit string for y-axis label
    int_unit = getattr(total_cube, "unit", None)
    if int_unit_label is not None:
        int_unit_str = int_unit_label
    elif int_unit is None:
        int_unit_str = "arb."
    else:
        int_unit_str = int_unit.to_string(format="latex_inline")

    # Convert heliocentric coordinates to pixel indices for each key pixel
    pix_minus = _xyhc_to_pixel(minus_xy_hc, total_cube)
    pix_mean = _xyhc_to_pixel(mean_xy_hc, total_cube)
    pix_plus = _xyhc_to_pixel(plus_xy_hc, total_cube)
    idx_triplet = (pix_minus, pix_mean, pix_plus)

    # Titles for each column (sigma notation)
    titles = (
        r"$\mu - {:.0f}\sigma$".format(sigma_factor),
        r"$\mu$",
        r"$\mu + {:.0f}\sigma$".format(sigma_factor),
    )

    # Create 2x3 figure grid: (rows: linear/log, columns: pixels)
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 6.5), sharex="col", sharey="row",
        gridspec_kw=dict(wspace=0.0, hspace=0.0),
    )

    # Precompute y-limits for all spectra (linear and log)
    lin_maxima, log_maxima = [], []
    for pix in idx_triplet:
        y, x = pix
        tot_spec = total_cube.data[y, x, :]
        lin_maxima.append(np.nanmax(tot_spec))
        log_maxima.append(np.nanmax(tot_spec[tot_spec > 0]))
        for cube in line_cubes.values():
            spec = cube.data[y, x, :]
            lin_maxima.append(np.nanmax(spec))
            log_maxima.append(np.nanmax(spec[spec > 0]))
    lin_max = np.nanmax(lin_maxima)
    log_max = np.nanmax(log_maxima)

    # Apply padding to y-limits
    lin_ylim_upper_padded = lin_max if lin_ylim_pad is None else lin_max * (1 + lin_ylim_pad)
    if log_max > 0:
        log_ylim_upper_padded = log_max if log_ylim_pad is None else log_max * log_ylim_pad
    else:
        log_ylim_upper_padded = 1.0

    # Plotting loop for each key pixel (column)
    for col, (pix, title) in enumerate(zip(idx_triplet, titles)):
        y, x = pix
        ax_lin, ax_log = axes[0, col], axes[1, col]

        # Plot all line cubes (background and highlighted lines)
        for name, cube in line_cubes.items():
            spec = cube.data[y, x, :]
            wl_grid_i = cube.axis_world_coords(2)[0].to(u.AA).value
            if name in hl_mapping:
                # Highlighted line: use specified color and label
                colour = hl_mapping[name]
                z = 2
                lbl = highlight_lines_labels.get(name, name) if col == 0 else None
            else:
                # Background lines: grey color, only label in first column
                colour = "grey"
                z = 1
                lbl = "Background" if (col == 0) else None
            ax_lin.plot(wl_grid_i, spec, color=colour, lw=0.8, zorder=z, label=lbl)
            ax_log.plot(wl_grid_i, spec, color=colour, lw=0.8, zorder=z)

        # Plot total spectrum for this pixel (distinct color per column)
        tot_colour = list(total_colours)[col % len(total_colours)]
        tot_spec = total_cube.data[y, x, :]
        ax_lin.plot(
            wl_grid, tot_spec, color=tot_colour, lw=1.5, zorder=3,
            label="Total" if col == 0 else None,
        )
        ax_log.plot(wl_grid, tot_spec, color=tot_colour, lw=1.5, zorder=3)

        # Set x-limits based on velocity, if provided
        if x_lim_velocity is not None:
            wl_lo, wl_hi = v2wl_axis(np.asarray(x_lim_velocity))
            ax_lin.set_xlim(wl_lo, wl_hi)
            ax_log.set_xlim(wl_lo, wl_hi)

        # General axis cosmetics
        for ax in (ax_lin, ax_log):
            ax.grid(ls=":", alpha=0.4)
            ax.tick_params(direction="in", which="both", top=True, right=True)
        ax_lin.set_title(title)
        ax_log.set_yscale("log")
        ax_log.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")

        # Set y-limits for linear and log axes
        ax_lin.set_ylim(bottom=0, top=lin_ylim_upper_padded)
        if log_ylim_lower is not None:
            ax_log.set_ylim(bottom=log_ylim_lower, top=log_ylim_upper_padded)
        else:
            ax_log.set_ylim(top=log_ylim_upper_padded)

        # Set shared wavelength/velocity ticks on x-axis
        v_min, v_max = wl2v_axis(ax_lin.get_xlim())
        if tick_spacing_velocity is not None and tick_spacing_velocity > 0:
            v_start = np.floor(v_min / tick_spacing_velocity) * tick_spacing_velocity
            v_end = np.ceil(v_max / tick_spacing_velocity) * tick_spacing_velocity
            v_ticks = np.arange(v_start, v_end + 0.5 * tick_spacing_velocity, tick_spacing_velocity)
        else:
            v_ticks = MaxNLocator(nbins=5, symmetric=True).tick_values(v_min, v_max)
        wl_ticks = v2wl_axis(v_ticks)
        loc = FixedLocator(wl_ticks)
        fmt = FixedFormatter([f"{wl:.3f}" for wl in wl_ticks])
        for ax in (ax_lin, ax_log):
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(fmt)

        # Set top x-axis as velocity, with optional main line label
        if velocity_label_main_line is not None:
            velocity_label = f"Velocity ({velocity_label_main_line}) [km/s]"
        else:
            velocity_label = "Velocity [km/s]"
        sec = ax_lin.secondary_xaxis("top", functions=(wl2v_axis, v2wl_axis))
        sec.set_xlabel(velocity_label)
        sec.set_xticks(v_ticks)
        sec.set_xticklabels([f"{v:.0f}" for v in v_ticks])
        sec.tick_params(direction="in", which="both", top=True)

    # Custom legend handler for multi-color "Total" line (triple with slashes)
    class HandlerTripleWithSlash(HandlerBase):
        """
        Draw three colored strokes separated by forward slashes for the legend.
        """
        def __init__(self, pad_frac: float = 0.08, **kwargs):
            super().__init__(**kwargs)
            self.pad_frac = pad_frac

        def create_artists(self, legend, tup, x0, y0, width, height, fontsize, trans):
            l1, l2, l3 = tup  # Three Line2D objects
            slash_w = width * 0.08
            pad = width * self.pad_frac
            w_seg = (width - 2 * slash_w - 4 * pad) / 3.0
            yc = y0 + 0.5 * height
            arts = []
            # First stroke
            x_left, x_right = x0, x0 + w_seg
            arts.append(Line2D([x_left, x_right], [yc, yc],
                                color=l1.get_color(), lw=l1.get_linewidth(),
                                solid_capstyle='butt', transform=trans))
            # First slash
            x_left = x_right + pad
            x_right = x_left + slash_w
            arts.append(Line2D([x_left, x_right],
                                [y0 + 0.1 * height, y0 + 0.9 * height],
                                color="k", lw=1.0, transform=trans))
            # Second stroke
            x_left = x_right + pad
            x_right = x_left + w_seg
            arts.append(Line2D([x_left, x_right], [yc, yc],
                                color=l2.get_color(), lw=l2.get_linewidth(),
                                solid_capstyle='butt', transform=trans))
            # Second slash
            x_left = x_right + pad
            x_right = x_left + slash_w
            arts.append(Line2D([x_left, x_right],
                                [y0 + 0.1 * height, y0 + 0.9 * height],
                                color="k", lw=1.0, transform=trans))
            # Third stroke
            x_left = x_right + pad
            x_right = x_left + w_seg
            arts.append(Line2D([x_left, x_right], [yc, yc],
                                color=l3.get_color(), lw=l3.get_linewidth(),
                                solid_capstyle='butt', transform=trans))
            return arts

    # Build legend handles and labels
    handles, labels = [], []
    # Highlighted lines first (if any)
    for name, colr in hl_mapping.items():
        handles.append(Line2D([0], [0], color=colr, lw=0.8))
        labels.append(highlight_lines_labels.get(name, name))
    # Background lines
    handles.append(Line2D([0], [0], color="grey", lw=0.8))
    labels.append("Background")
    # Multi-color total spectrum
    lines_for_total = [Line2D([0], [0], color=c, lw=1.5) for c in total_colours]
    handles.append(tuple(lines_for_total))
    labels.append("Total")

    # Add legend to the top-left panel
    axes[0, 0].legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        fontsize="small",
        handlelength=3.0,
        handler_map={tuple: HandlerTripleWithSlash(pad_frac=0.10)},
    )

    # Fold scientific notation exponent into the y-axis label (top row)
    fig.canvas.draw()  # Ensure offset text is created
    offset_txt = axes[0, 0].yaxis.get_offset_text().get_text()
    axes[0, 0].yaxis.get_offset_text().set_visible(False)
    if offset_txt.startswith("1e"):
        exponent = offset_txt.replace("1e", "")
        # Use LaTeX formatting for exponent
        label_top = f"Intensity [$10^{{{exponent}}}$ {int_unit_str}]"
        try:
            exp_val = int(exponent)
        except Exception:
            exp_val = 0
        tick_base = 10 ** exp_val
        y_max = axes[0, 0].get_ylim()[1]
        max_tick = int(np.floor(y_max / tick_base))
        ticks = [i * tick_base for i in range(0, max_tick + 1)]
        for ax in axes[0, :]:
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(i) for i in range(0, max_tick + 1)])
    else:
        label_top = f"Intensity [{int_unit_str}]"
        y_max = axes[0, 0].get_ylim()[1]
        max_tick = int(np.floor(y_max))
        ticks = list(range(0, max_tick + 1))
        for ax in axes[0, :]:
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(i) for i in ticks])

    # Set y-axis labels for both rows
    axes[0, 0].set_ylabel(label_top)
    axes[1, 0].set_ylabel(f"Intensity [{int_unit_str}]")

    # Finalize layout and save figure
    plt.tight_layout(pad=0.0)
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_key_pixel_dem(
    *,
    total_cube: NDCube,
    main_cube: NDCube,
    dem_map: np.ndarray,
    em_tv: np.ndarray,
    logT_centres: np.ndarray,
    v_edges: np.ndarray,
    minus_xy_hc: Tuple[u.Quantity, u.Quantity],
    mean_xy_hc: Tuple[u.Quantity, u.Quantity],
    plus_xy_hc:  Tuple[u.Quantity, u.Quantity],
    sigma_factor: float = 1.0,
    save: str = "key_pixel_dem.png",
    goft: dict | None = None,
    main_line: str = "Fe12_195.1190",
    main_line_label: str | None = None,
    key_pixel_colors: Iterable[str] = ("deeppink", "black", "mediumseagreen"),
    xlim: tuple[float, float] = (5.5, 7.0),
    ylim_dem: tuple[float, float] = (25, 29),
    ylim_2d_dem: tuple[float, float] | None = None,
    # --- NEW: grids required to compute σ-lines of G(T,N) --------------
    logT_grid: np.ndarray | None = None,
    logN_grid: np.ndarray | None = None,
) -> None:
    """
    Plot DEM(T) and EM(T,v) for three key pixels (mean, mean+sigma, mean-sigma).

    This function creates a 2x3 panel figure:
      - Top row: DEM(T) curves for each key pixel (log10 scale).
      - Bottom row: log10 EM(T,v) images for each key pixel, with an inset showing
        the main line's intensity vs velocity ("stick plot").
      - Optionally overlays a secondary y-axis with wavelength labels.

    Parameters
    ----------
    total_cube : NDCube
        The total intensity cube (for pixel coordinate conversion).
    main_cube : NDCube
        The main line cube (not directly used, but kept for interface consistency).
    dem_map : np.ndarray
        DEM(T) map, shape (ny, nx, nT).
    em_tv : np.ndarray
        EM(T,v) map, shape (ny, nx, nT, nv).
    logT_centres : np.ndarray
        Log10 temperature bin centers (nT,).
    v_edges : np.ndarray
        Velocity bin edges (nv+1,).
    minus_xy_hc, mean_xy_hc, plus_xy_hc : Tuple[u.Quantity, u.Quantity]
        Heliocentric coordinates for the three key pixels.
    sigma_factor : float, optional
        Sigma factor for pixel selection (for plot titles).
    save : str, optional
        Output filename for the plot.
    goft : dict, optional
        Dictionary with main line intensity profiles and rest wavelength.
    main_line : str, optional
        Name of the main line for stick plot and secondary axis.
    main_line_label : str, optional
        Custom label for the main line (for stick plot and secondary axis).
    key_pixel_colors : Iterable[str], optional
        Colors for the three key pixels.
    xlim : tuple, optional
        X-axis limits for log10(T) (temperature).
    ylim_dem : tuple, optional
        Y-axis limits for DEM(T) plots.
    ylim_2d_dem : tuple, optional
        Y-axis limits for EM(T,v) images (velocity).
    """

    # Compute velocity bin centers in km/s for plotting
    v_centres = 0.5 * (v_edges[:-1] + v_edges[1:]) * (u.cm / u.s)
    v_centres_kms = v_centres.to(u.km / u.s).value

    # Define image extent for imshow: (xmin, xmax, ymin, ymax)
    extent = (
        logT_centres[0], logT_centres[-1],
        v_centres_kms[0], v_centres_kms[-1],
    )

    # Convert heliocentric coordinates to pixel indices for each key pixel
    pix_minus = _xyhc_to_pixel(minus_xy_hc, total_cube)
    pix_mean  = _xyhc_to_pixel(mean_xy_hc,  total_cube)
    pix_plus  = _xyhc_to_pixel(plus_xy_hc,  total_cube)
    pix_triplet = (pix_minus, pix_mean, pix_plus)  # Order: -sigma, mean, +sigma

    # Titles for each column (sigma notation)
    titles = (
        rf"$\mu - {sigma_factor:.0f}\sigma$",
        r"$\mu$",
        rf"$\mu + {sigma_factor:.0f}\sigma$",
    )

    # Create figure and axes: 2 rows x 3 columns
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 3, figure=fig, wspace=0.0, hspace=0.0)
    axes = np.array(gs.subplots(sharey="row"))

    # ---------------- Top row: DEM(T) curves --------------------------
    for col, (idx, title, colr) in enumerate(zip(pix_triplet, titles, key_pixel_colors)):
        if idx is None:
            # Hide axes if pixel index is missing
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue

        y, x = idx
        dem = dem_map[y, x, :]  # DEM(T) for this pixel, shape (nT,)
        # Compute log10(DEM), mask non-positive values to zero for plotting
        log_dem = np.log10(dem, where=dem > 0, out=np.zeros_like(dem))
        axes[0, col].step(
            logT_centres, log_dem, where="mid",
            color=colr, lw=1.8,
        )
        axes[0, col].set_title(title)
        axes[0, col].set_xlim(*xlim)
        axes[0, col].set_ylim(*ylim_dem)
        axes[0, col].grid(ls=":")

    # Set y-axis label for top row
    axes[0, 0].set_ylabel(r"$\log_{10}(\xi\;[1/\mathrm{cm}^5/\mathrm{dex}])$")
    # Set x-axis label and tick params for all top-row axes
    for ax in axes[0]:
        ax.set_xlabel(r"$\log_{10}\,T\,[\mathrm{K}]$")
        ax.tick_params(direction="in", top=True, right=True)

    # ------------- Bottom row: log10 EM(T,v) images -------------------
    ims = []  # Store image handles for colorbar
    vmin, vmax = None, None  # Track global min/max for color scaling

    # Define masks for temperature and velocity ranges
    mask_T = (logT_centres >= xlim[0]) & (logT_centres <= xlim[1])
    if ylim_2d_dem is None:
        mask_v = np.ones_like(v_centres_kms, dtype=bool)
    else:
        mask_v = (v_centres_kms >= ylim_2d_dem[0]) & (v_centres_kms <= ylim_2d_dem[1])

    # First pass: find global vmin/vmax for color scaling
    for idx in pix_triplet:
        if idx is None:
            continue
        y, x = idx
        # EM(T,v) for this pixel, shape (nT, nv), transpose for imshow (nv, nT)
        data = em_tv[y, x, :, :].T
        # Compute log10(EM), mask non-positive values to nan for color scaling
        log_data = np.log10(data, where=data > 0, out=np.full_like(data, np.nan))
        # Restrict to selected T and v ranges
        sub = log_data[np.ix_(mask_v, mask_T)]
        # Update global vmin/vmax
        vmin = np.nanmin(sub) if vmin is None else min(vmin, np.nanmin(sub))
        vmax = np.nanmax(sub) if vmax is None else max(vmax, np.nanmax(sub))

    # Second pass: plot each EM(T,v) image and add stick inset
    for col, (idx, colr) in enumerate(zip(pix_triplet, key_pixel_colors)):
        if idx is None:
            continue
        y, x = idx
        data = em_tv[y, x, :, :].T  # Shape (nv, nT)
        log_data = np.log10(data, where=data > 0, out=np.full_like(data, np.nan))
        # Plot EM(T,v) as image
        im = axes[1, col].imshow(
            log_data,
            origin="lower", aspect="auto",
            extent=extent, cmap="Purples",
            vmin=vmin, vmax=vmax,
        )
        ims.append(im)
        axes[1, col].set_xlim(*xlim)
        if ylim_2d_dem is not None:
            axes[1, col].set_ylim(*ylim_2d_dem)
        axes[1, col].grid(ls=":")
        axes[1, col].set_xlabel(r"$\log_{10}(T\,[\mathrm{K}])$")
        axes[1, col].tick_params(direction="in", top=True, right=True)

        # --------- Add stick inset for main line intensity vs velocity ---------
        if goft and main_line in goft:
            # Try to extract the intensity profile for this pixel
            try:
                spec = goft[main_line]["si"][y, x, :]
            except Exception:
                # Fallback for older data shape
                spec = goft[main_line]["si"][idx]
            # Create inset axis for stick plot
            stick = inset_axes(
                axes[1, col], width="50%", height="100%",
                bbox_to_anchor=(0, 0, 1, 1),
                bbox_transform=axes[1, col].transAxes,
                loc="lower left", borderpad=0,
            )
            stick.set_facecolor("none")
            stick.set_axisbelow(False)
            # Draw stick plot (intensity vs velocity)
            for spine in stick.spines.values():
                spine.set_zorder(3)
            stick.plot(spec, v_centres_kms, color="red", lw=1.3, zorder=1)
            stick.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            stick.xaxis.set_ticks_position("top")
            stick.xaxis.set_label_position("top")
            stick.tick_params(axis="x", labelsize=7, direction="in")
            fig.canvas.draw()
            # Extract scientific notation exponent for axis label
            offset_text = stick.xaxis.get_offset_text().get_text().replace("1e", "")
            stick.xaxis.get_offset_text().set_visible(False)
            # Use custom main line label if provided
            label = main_line_label if main_line_label is not None else main_line
            stick.set_xlabel(
                f"{label} intensity \n [$10^{{{offset_text}}}$ erg/s/cm$^{{2}}$/sr/cm]",
                fontsize=8, zorder=3
            )
            # Remove zero tick, set ticks and spines
            ticks = stick.get_xticks()
            ticks = ticks[ticks != 0]
            stick.set_xticks(ticks)
            stick.yaxis.set_ticks([])
            stick.spines[['right', 'bottom', 'left']].set_visible(False)
            stick.spines['top'].set_position(('axes', 0.775))
            stick.set_xlim(0, spec.max()*1.05)
            stick.set_ylim(axes[1, col].get_ylim())

    # Set y-axis label for bottom row
    axes[1, 0].set_ylabel(r"Velocity [km/s]")

    # --------- Add secondary y-axis with wavelength labels (optional) ---------
    if goft and main_line in goft:
        # Get rest wavelength in Angstroms and speed of light in km/s
        wl0 = goft[main_line]["wl0"].to_value(u.AA)
        c_kms = const.c.to_value(u.km / u.s)
        # Function to convert velocity to wavelength
        v2wl = lambda v: wl0 * (1 + v / c_kms)
        wl_ticks_cache = {}

        for ax in axes[1]:
            v_ticks = ax.get_yticks()
            vmin_ax, vmax_ax = ax.get_ylim()
            v_ticks = v_ticks[(v_ticks >= vmin_ax) & (v_ticks <= vmax_ax)]
            if not len(v_ticks):
                continue
            # Cache wavelength ticks for efficiency
            if (lo := tuple(v_ticks)) not in wl_ticks_cache:
                wl_ticks_cache[lo] = v2wl(v_ticks)
            wl_ticks = wl_ticks_cache[lo]
            ax_r = ax.secondary_yaxis("right")
            ax_r.set_yticks(v_ticks)
            ax_r.set_yticklabels([f"{wl:.3f}" for wl in wl_ticks])
            label = f"Wavelength ({main_line_label}) [$\\mathrm{{\\AA}}$]"
            ax_r.set_ylabel(label)
            ax_r.tick_params(direction="in", which="both")

    # --------- NEW: vertical “±2 σ” G(T,N) lines -----------------------
    if (
        goft is not None and main_line in goft
        and logT_grid is not None and logN_grid is not None
    ):
        g_tn = goft[main_line]["g_tn"]                   # shape (nN, nT)

        # --- optimal density: where ∫G dT is maximal ------------------
        integ_gt = np.trapz(g_tn, logT_grid, axis=1)     # (nN,)
        idx_opt  = np.argmax(integ_gt)
        opt_N    = logN_grid[idx_opt]
        g_T_opt  = g_tn[idx_opt, :]                      # (nT,)

        # --- mean & std in logT, weighted by G ------------------------
        mean_T = np.average(logT_grid, weights=g_T_opt)
        std_T  = np.sqrt(np.average((logT_grid - mean_T) ** 2,
                                    weights=g_T_opt))
        vlines = [mean_T - 2 * std_T, mean_T + 2 * std_T]

        # --- draw on ALL main panels ----------------------------------
        for row in axes:
            for ax in row:
                for v in vlines:
                    ax.axvline(
                        v, color="grey", linestyle=(0, (5, 10)),
                        linewidth=1, alpha=1.0, zorder=0
                    )

        # --- legend entry (top-right panel) ---------------------------
        custom = Line2D([0], [0], color='grey',
                        linestyle=(0, (5, 5)), linewidth=1)
        h, l = axes[0, -1].get_legend_handles_labels()

        clean_label = (main_line_label.replace('$', '').replace(' ', r'\,'))
        l_txt = rf"$G_{{\mathrm{{{clean_label}}}}}(T,N_e={opt_N:.1f})\pm2\sigma_{{G}}$"

        h.append(custom); l.append(l_txt)
        axes[0, -1].legend(
            handles=h, labels=l, loc="upper right", fontsize="small"
        )

    # --------- Add shared colorbar to the right of last panel ----------
    cax = inset_axes(
        axes[1, -1], width="3%", height="90%",
        loc="center left", bbox_to_anchor=(1.3, 0., 1, 1),
        bbox_transform=axes[1, -1].transAxes, borderpad=0,
    )
    cbar = fig.colorbar(ims[0], cax=cax, orientation="vertical", extend="min")
    cbar.set_label(r"$\log_{10}(\Xi\;[1/\mathrm{cm}^{5}/\mathrm{dex}/\Delta v]$)")

    # Hide y-tick labels for non-left columns for clarity
    for col in range(1, 3):
        axes[0, col].tick_params(labelleft=False)
        axes[1, col].tick_params(labelleft=False)

    # Finalize layout and save figure
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_g_function(
    goft: dict,
    line_name: str,
    line_name_label: str,
    logT_grid: np.ndarray,
    logN_grid: np.ndarray,
    *,
    save: str = "g_function.png",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show_vlines: bool = True,
    figsize: tuple[float, float] = (6, 5),
) -> None:
    """
    Visualize the contribution function G(log T, log N) for a given spectral line.

    This function displays the 2D contribution function G(T, N) for the specified
    line as an image, with optional overlays of vertical lines at mean +/- 2 sigma
    in log T, where the function is maximized (integrated over T) at the optimal
    density. The colorbar label includes the scientific exponent.

    Parameters
    ----------
    goft : dict
        Dictionary containing contribution function data for spectral lines.
        Must contain goft[line_name]["g_tn"] as a 2D array (nN, nT).
    line_name : str
        Name of the spectral line to plot.
    logT_grid : np.ndarray
        1D array of log10(temperature) grid points (nT,).
    logN_grid : np.ndarray
        1D array of log10(electron density) grid points (nN,).
    save : str, optional
        Output filename for the saved plot.
    xlim : tuple, optional
        (min, max) limits for log10(T) axis.
    ylim : tuple, optional
        (min, max) limits for log10(N) axis.
    show_vlines : bool, optional
        If True, overlay vertical lines at mean +/- 2 sigma in log T.
    figsize : tuple, optional
        Figure size in inches (width, height).
    """
    # --- Extract the G(T, N) data for the requested line ---
    g_data = goft[line_name]["g_tn"]  # shape (nN, nT)

    # --- Optionally restrict the log T range (columns) ---
    if xlim is not None:
        mask_T = (logT_grid >= xlim[0]) & (logT_grid <= xlim[1])
        g_data = g_data[:, mask_T]
        logT_plot = logT_grid[mask_T]
    else:
        logT_plot = logT_grid

    # --- Optionally restrict the log N range (rows) ---
    if ylim is not None:
        mask_N = (logN_grid >= ylim[0]) & (logN_grid <= ylim[1])
        g_data = g_data[mask_N, :]
        logN_plot = logN_grid[mask_N]
    else:
        logN_plot = logN_grid

    # --- Set up the figure and axis ---
    fig, ax = plt.subplots(figsize=figsize)
    # The extent defines the axis limits for imshow: (xmin, xmax, ymin, ymax)
    extent = (logT_plot[0], logT_plot[-1], logN_plot[0], logN_plot[-1])
    # Show the G(T, N) data as a grayscale image
    im = ax.imshow(
        g_data,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="Greys"
    )

    # --- Add a colorbar and fold the exponent into the label ---
    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    fig.canvas.draw()  # Needed to ensure offset text is available
    # Extract the scientific notation exponent (e.g., "6" from "1e6")
    exp_txt = cbar.ax.yaxis.get_offset_text().get_text().replace("1e", "")
    cbar.ax.yaxis.set_offset_position("left")
    cbar.ax.yaxis.get_offset_text().set_visible(False)
    # Set the colorbar label, using LaTeX for math formatting
    clean_label = (line_name_label.replace('$', '').replace(' ', r'\,'))
    l_txt = rf"$G_{{\mathrm{{{clean_label}}}}}(T,N) [10^{{{exp_txt}}}\,\mathrm{{erg\,cm^3/s}}]$"

    cbar.set_label(l_txt)

    # --- Set axis labels (math mode, with superscripts) ---
    ax.set_xlabel(r"$\log_{10}(T\,[\mathrm{K}])$")
    ax.set_ylabel(r"$\log_{10}(N_e\,[\mathrm{1/cm^{3}}])$")

    # --- Optionally overlay vertical lines at mean +/- 2 sigma in log T ---
    if show_vlines:
        # Use the full (possibly masked) G(T, N) data for statistics
        g_full = goft[line_name]["g_tn"]
        if ylim is not None:
            g_full = g_full[mask_N, :]
        if xlim is not None:
            g_full = g_full[:, mask_T]

        # Integrate G(T, N) over T to find the optimal density (row with max integral)
        integ = np.trapz(g_full, logT_plot, axis=1)
        idx_opt = np.argmax(integ)
        opt_N = logN_plot[idx_opt]
        g_T_opt = g_full[idx_opt, :]  # G(T) at optimal density

        # Compute mean and standard deviation in log T, weighted by G(T)
        mean_T = np.average(logT_plot, weights=g_T_opt)
        std_T = np.sqrt(np.average((logT_plot - mean_T) ** 2, weights=g_T_opt))

        # Draw vertical lines at mean +/- 2 sigma
        for x in (mean_T - 2 * std_T, mean_T + 2 * std_T):
            ax.axvline(
                x,
                color="grey",
                linestyle=(0, (5, 10)),
                linewidth=1,
                alpha=1.0
            )

        # Add a legend entry for the vertical lines
        label = (
            # rf"$G_{{\mathrm{{{line_name.replace('_', '\,')}}}}}"
            # rf"(T,\,N_e={opt_N:.1f})\pm2\sigma_{{G}}$"
            rf"$G_{{\mathrm{{{clean_label}}}}}(T,N_e={opt_N:.1f})\pm2\sigma_{{G}}$"
        )
        ax.legend(
            [Line2D([0], [0], color='grey', linestyle=(0, (5, 5)), linewidth=1)],
            [label],
            loc="upper right",
            fontsize="small"
        )

    # --- Set axis limits if requested ---
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # --- Save the figure and close ---
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    synthetic_spectra = Path("./run/input/synthesised_spectra.pkl")
    instr_response_dir = Path("./run/result")
    instr_responses = sorted(instr_response_dir.glob("instrument_response_*.pkl"))

    globals().update(locals());raise ValueError("Kicking back to ipython")

    print(f"Loading synthetic spectra from {synthetic_spectra}...")

    with open(synthetic_spectra, "rb") as f:
        sim_dat = dill.load(f)
    synth_cubes: dict[str, NDCube] = {"total": sim_dat["sim_si"]}
    synth_cubes.update(sim_dat.get("line_cubes", {}))
    synth_dem = {
        "dem_map": sim_dat.get("dem_map"),
        "em_tv": sim_dat.get("em_tv"),
        "logT_centres": sim_dat.get("logT_centres"),
        "v_edges": sim_dat.get("v_edges"),
        "goft": sim_dat.get("goft"),
        "logT_grid": sim_dat.get("logT_grid"),
        "logN_grid": sim_dat.get("logN_grid"),
    }
    del sim_dat

    # --- get key from synthetic spectra coordinates ------------------------------------------
    key_pixel_sigma = 1.0
    key_pixel_margin = 0.20
    mean_xy_hc, plus_xy_hc, minus_xy_hc = find_key_heliocentric_coords(
        synth_cubes['total'],
        sigma_factor=key_pixel_sigma,
        margin_frac=key_pixel_margin,
    )

    plot_key_pixel_spectra(
        synth_cubes["total"],
        {k: v for k, v in synth_cubes.items() if k != "total"},
        minus_xy_hc=minus_xy_hc,
        mean_xy_hc=mean_xy_hc,
        plus_xy_hc=plus_xy_hc,
        sigma_factor=1.0,
        save="key_pixel_spectrum.png",
        int_unit_label=r'erg/s/cm$^{2}$/sr/cm',
        log_ylim_lower=5e5,
        x_lim_velocity=(-250,250),
        tick_spacing_velocity=100,
        total_colours=("deeppink", "black", "mediumseagreen"),
        highlight_lines=['Fe12_195.1190', 'Fe12_195.1790'],
        highlight_lines_labels={
            'Fe12_195.1190': r'Fe XII 195.119 $\mathrm{\AA}$',
            'Fe12_195.1790': r'Fe XII 195.179 $\mathrm{\AA}$',
        },
        lin_ylim_pad = 0.10,           # fractional padding (top row)
        log_ylim_pad = 10**1.0,        # multiplicative padding (bottom row)
        velocity_label_main_line="Fe XII 195.119 $\mathrm{\\AA}$",
    )

    plot_key_pixel_dem(
        total_cube=synth_cubes["total"],
        main_cube=synth_cubes["Fe12_195.1190"],
        dem_map=synth_dem["dem_map"],
        em_tv=synth_dem["em_tv"],
        logT_centres=synth_dem["logT_centres"],
        v_edges=synth_dem["v_edges"],
        minus_xy_hc=minus_xy_hc,
        mean_xy_hc=mean_xy_hc,
        plus_xy_hc=plus_xy_hc,
        sigma_factor=key_pixel_sigma,
        save="key_pixel_dem.png",
        goft=synth_dem.get("goft"),
        main_line="Fe12_195.1190",
        main_line_label=r"Fe XII 195.119 $\mathrm{\AA}$",
        key_pixel_colors=("deeppink", "black", "mediumseagreen"),
        xlim=(5.5, 6.9),
        ylim_dem=(26.5, 30),
        ylim_2d_dem=(-50, 50),
        logT_grid=synth_dem.get("logT_grid"),
        logN_grid=synth_dem.get("logN_grid"),
    )

    plot_g_function(
        goft=synth_dem["goft"],
        line_name="Fe12_195.1190",
        line_name_label=r"Fe XII 195.119 $\mathrm{\AA}$",
        logT_grid=synth_dem["logT_grid"],
        logN_grid=synth_dem["logN_grid"],
        save="g_function_Fe12_195119.png",
        xlim=(5.5, 6.9),
        ylim=(7.0, 20.0),
        show_vlines=True,
        figsize=(6, 3),
    )

    mean_xy_hp = heliocentric_to_helioprojective_coords(mean_xy_hc)
    plus_xy_hp = heliocentric_to_helioprojective_coords(plus_xy_hc)
    minus_xy_hp = heliocentric_to_helioprojective_coords(minus_xy_hc)







if __name__ == "__main__":
    main()