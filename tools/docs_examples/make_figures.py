"""
Make the figures in the docs from the MURaM active region.

README.md in this folder says how to make the runs they come from. Run it in
the folder holding them, as

    python $REPO/tools/docs_examples/make_figures.py $REPO/docs/images

Names after the folder make only those figures.
"""
import sys
from functools import cache
from pathlib import Path

import astropy.units as u
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.ticker import NullFormatter

from eclipse import (Detector_SWC, Telescope_EUVST, create_sunpy_maps_from_combo,
                     get_parameter_combinations, get_results_for_combination,
                     load_instrument_response_results, read_synthesis, read_synthesis_products)
from eclipse.analysis import format_exposure_time_colorbar

OUT = Path(sys.argv[1])
DATE = "2026-01-01T00:00:00"
SYNTHESIS = "run/input/muram_ar_fe12.h5"
LINES = [("synth/fe9.h5", "Fe IX 171.07"), ("synth/fe10.h5", "Fe X 184.54"),
         (SYNTHESIS, "Fe XII 195.12"), ("synth/fe14.h5", "Fe XIV 211.32")]
# Pixels of the synthesis used as examples, as (y, x).
PIXELS = {"Core": (544, 544), "Quiet Sun": (150, 150)}
VELOCITY = dict(cmap="RdBu_r", vmin=-25, vmax=25)


@cache
def results(name):
    return load_instrument_response_results(f"run/result/{name}.h5")


def combination(name, **settings):
    return get_results_for_combination(results(name), **settings)


@cache
def main_maps():
    """The maps of the worked example's run."""
    return create_sunpy_maps_from_combo(combination("muram_ar_fe12"), date_obs=DATE)


@cache
def sweep_maps():
    """The maps of the combination that the analysis page reads."""
    return maps("muram_ar_sweep", offchip_bin_slit=2, **{
        "simulation.slit_width": 0.4 * u.arcsec, "simulation.expos": 60 * u.s})


def maps(name, **settings):
    return create_sunpy_maps_from_combo(combination(name, **settings), date_obs=DATE)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=110, bbox_inches="tight")


def extent(image):
    """The extent of a map in arcsec, with its centre at zero."""
    width = image.dimensions.x.value * image.scale.axis1.to_value(u.arcsec / u.pix)
    height = image.dimensions.y.value * image.scale.axis2.to_value(u.arcsec / u.pix)
    return [-width / 2, width / 2, -height / 2, height / 2]


def show(ax, image, title=None, **settings):
    """Draw a map on plain axes in arcsec."""
    drawn = ax.imshow(image.data, origin="lower", extent=extent(image), **settings)
    ax.set_title(title)
    ax.set_xlabel("x [arcsec]")
    return drawn


def frame_data():
    """The image columns of the frame's CCDs, in DN, and their wavelengths in Angstrom."""
    shown = {}
    with h5py.File("frame.h5") as f:
        pre, over = int(f.attrs["serial_prescan"]), int(f.attrs["serial_overscan"])
        for ccd in ("left", "right"):
            wavelength = f[ccd]["wavelength"][()]
            data = f[ccd]["dn"][()][:wavelength.size].astype(float)
            half = (data.shape[1] - 2 * pre - 2 * over) // 2
            columns = np.r_[pre:pre + half, pre + half + 2 * over:pre + 2 * half + 2 * over]
            order = np.argsort(wavelength)
            shown[ccd] = (wavelength[order], data[:, columns][order].T)
    return shown


def stages():
    """The three stages of a run, on the home page."""
    line = read_synthesis(SYNTHESIS).lines["Fe12_195.1190"]
    wavelength = line.wavelength.to_value(u.AA)
    radiance = line.intensity.to_value(u.erg / (u.s * u.cm**2 * u.sr * u.AA))
    intensity = np.trapezoid(radiance, wavelength, axis=-1)
    observed = main_maps()
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.1), sharex=True, sharey=True,
                             layout="constrained")
    box = extent(observed["total_dn"])
    drawn = axes[0].imshow(intensity, origin="lower", cmap="gray", extent=box,
                           norm=LogNorm(*np.percentile(intensity, [1, 99.9])))
    axes[0].set_title("Synthesised")
    axes[0].set_xlabel("x [arcsec]")
    fig.colorbar(drawn, ax=axes[0], location="bottom", label="erg / (s cm2 sr)", shrink=0.9)
    dn = observed["total_dn"]
    drawn = show(axes[1], dn, "Recorded", cmap="gray",
                 norm=LogNorm(*np.nanpercentile(dn.data[dn.data > 0], [1, 99.9])))
    bar = fig.colorbar(drawn, ax=axes[1], location="bottom", label="DN", shrink=0.9)
    bar.ax.xaxis.set_minor_formatter(NullFormatter())
    drawn = show(axes[2], observed["velocity_from_fit"], "Measured velocity", **VELOCITY)
    fig.colorbar(drawn, ax=axes[2], location="bottom", label="km/s", shrink=0.9, extend="both")
    drawn = show(axes[3], observed["velocity_std"], "Velocity precision", vmin=0, vmax=10)
    fig.colorbar(drawn, ax=axes[3], location="bottom", label="km/s", shrink=0.9, extend="max")
    axes[0].set_ylabel("y [arcsec]")
    save(fig, "example-stages")


def slit_and_frame():
    """The noise-free Fe XII raster with the slit, beside the full-CCD frame there."""
    det = Detector_SWC()
    raster = maps("muram_ar_psf", **{"simulation.psf": True})["total_dn"]
    pitch = det.plate_scale_angle.to_value(u.arcsec / u.pix)
    with h5py.File("frame.h5") as f:
        slit_x = f.attrs["slit_position"]
        slit_width = f.attrs["slit_width"] * u.Unit(f.attrs["slit_width_unit"])
    frame = frame_data()
    n_rows = frame["left"][0].size
    span = [-det.n_columns * pitch / 2, det.n_columns * pitch / 2]
    box = extent(raster)
    # Each CCD panel has square pixels, so its width is its rows times the columns' pitch.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, layout="constrained",
                             gridspec_kw=dict(width_ratios=[box[1] - box[0], n_rows * pitch,
                                                            n_rows * pitch]))
    ax = axes[0]
    show(ax, raster, "Fe XII 195 raster (noise-free)", cmap="gray",
         norm=PowerNorm(0.5, *np.nanpercentile(raster.data, [1, 99.9])))
    openings = Telescope_EUVST().slit_openings(slit_width).to_value(u.arcsec)
    for i, (low, high) in enumerate(openings):
        ax.plot([slit_x, slit_x], [low, high], color="tab:red", lw=1.2,
                label="slit" if i == 0 else None)
    for i, (_, high) in enumerate(openings[:-1]):
        ax.plot(slit_x, high, marker=">", color="tab:red", ms=6, ls="none",
                label="fiducials" if i == 0 else None)
    ax.set_ylim(span)
    ax.set_xlabel("raster direction [arcsec]")
    ax.set_ylabel("along the slit [arcsec]")
    ax.legend(loc="upper right", fontsize="small")
    logged = {ccd: (w, np.log10(np.clip(dn, 1, None))) for ccd, (w, dn) in frame.items()}
    # One colour range for both CCDs, from below the detector's background to the 99.9th
    # percentile, so the noise and the faint lines both show.
    top = np.percentile(np.concatenate([image.ravel() for _, image in logged.values()]), 99.9)
    low = 1.0
    for ax, (ccd, (wavelength, image)) in zip(axes[1:], logged.items()):
        step = (wavelength[-1] - wavelength[0]) / (wavelength.size - 1)
        drawn = ax.imshow(image, origin="lower", cmap="gray", vmin=low, vmax=top,
                          extent=[wavelength[0] - step / 2, wavelength[-1] + step / 2, *span])
        ax.set_aspect(step / pitch)
        ax.set_title(f"full-CCD frame, {ccd} CCD")
        ax.set_xlabel("wavelength [Angstrom]")
    fig.colorbar(drawn, ax=axes[1:], label="log10 DN", shrink=0.9, extend="max")
    save(fig, "example-slit-and-frame")


def frame_spectrum():
    """The median over the slit in each row of the frame."""
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    for ccd, (wavelength, dn) in frame_data().items():
        ax.plot(wavelength, np.median(dn, axis=0), label=f"{ccd} CCD")
    ax.set_xlim(169, 213)
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength [Angstrom]")
    ax.set_ylabel("DN per pixel")
    ax.legend()
    save(fig, "example-frame-spectrum")


def lines():
    """The four lines' intensity, seen from above, as synthesised."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, layout="constrained")
    for ax, (file, title) in zip(axes, LINES):
        synthesis = read_synthesis(file)
        line = next(iter(synthesis.lines.values()))
        intensity = np.trapezoid(line.intensity.value, line.wavelength.value, axis=-1)
        ax.imshow(intensity, origin="lower", cmap="gray", extent=[-150, 150, -150, 150],
                  norm=LogNorm(*np.percentile(intensity, [1, 99.8])))
        ax.set_title(title)
        ax.set_xlabel("x [arcsec]")
    axes[0].set_ylabel("y [arcsec]")
    save(fig, "example-lines")


def pixels():
    """The spectrum and the DEM of a pixel in the core and one in the quiet Sun."""
    synthesis = read_synthesis(SYNTHESIS)
    products = read_synthesis_products(SYNTHESIS, keys=["dem_map", "logT_grid"])
    unit = u.erg / (u.s * u.cm**2 * u.sr * u.AA)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for i, (name, (y, x)) in enumerate(PIXELS.items()):
        for line, label in zip(synthesis.lines.values(), (name, None)):
            axes[0].plot(line.wavelength.to_value(u.AA), line.intensity[y, x].to_value(unit),
                         color=f"C{i}", label=label)
        axes[1].plot(products["logT_grid"], products["dem_map"][y, x], color=f"C{i}", label=name)
    axes[0].set_xlim(195.03, 195.27)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e1, 3e5)
    axes[0].set_xlabel("Wavelength [Angstrom]")
    axes[0].set_ylabel("Radiance [erg / (s cm2 sr Angstrom)]")
    axes[0].legend()
    axes[1].set_xlim(5.5, 6.8)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e25, None)
    axes[1].set_xlabel("log10 T [K]")
    axes[1].set_ylabel("DEM [cm-5 per unit log10 T]")
    save(fig, "example-pixels")


def sweep():
    """The velocity precision for each slit and exposure, with two pixels binned."""
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.6), sharex=True, sharey=True,
                             layout="constrained")
    for row, slit in zip(axes, (0.2, 0.4)):
        for ax, exposure in zip(row, (15, 60)):
            image = maps("muram_ar_sweep", offchip_bin_slit=2, **{
                "simulation.slit_width": slit * u.arcsec,
                "simulation.expos": exposure * u.s})["velocity_std"]
            drawn = show(ax, image, f"{slit} arcsec, {exposure} s", vmin=0, vmax=10)
            ax.set_xlabel("x [arcsec]" if ax in axes[1] else None)
        row[0].set_ylabel("y [arcsec]")
    fig.colorbar(drawn, ax=axes, label="Velocity precision [km/s]", extend="max", shrink=0.8)
    save(fig, "example-sweep")


def binning():
    """The velocity precision without binning and with two pixels binned."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9), sharey=True, layout="constrained")
    for ax, binned in zip(axes, (1, 2)):
        image = maps("muram_ar_sweep", offchip_bin_slit=binned, **{
            "simulation.slit_width": 0.4 * u.arcsec, "simulation.expos": 60 * u.s})["velocity_std"]
        drawn = show(ax, image, "No binning" if binned == 1 else f"{binned} pixels binned",
                     vmin=0, vmax=10)
    axes[0].set_ylabel("y [arcsec]")
    fig.colorbar(drawn, ax=axes, label="Velocity precision [km/s]", extend="max", shrink=0.9)
    save(fig, "example-binning")


def noise():
    """The Doppler velocity of one observation, without and with noise."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9), sharey=True, layout="constrained")
    for ax, flag in zip(axes, (False, True)):
        image = maps("muram_ar_noise", **{"simulation.noise": flag})["velocity_from_fit"]
        drawn = show(ax, image, "With noise" if flag else "noise: False", **VELOCITY)
    axes[0].set_ylabel("y [arcsec]")
    for ax in axes:
        ax.set_xlim(-140, -60)
        ax.set_ylim(-140, -60)
    fig.colorbar(drawn, ax=axes, label="Doppler velocity [km/s]", extend="both", shrink=0.9)
    save(fig, "example-noise")


def psf():
    """A zoom on the core without and with the PSF, without noise."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9), sharey=True, layout="constrained")
    images = [maps("muram_ar_psf", **{"simulation.psf": flag})["total_dn"]
              for flag in (False, True)]
    norm = PowerNorm(0.5, *np.nanpercentile(images[0].data, [1, 99.9]))
    for ax, image, title in zip(axes, images, ("psf: False", "psf: True")):
        drawn = show(ax, image, title, cmap="gray", norm=norm)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
    axes[0].set_ylabel("y [arcsec]")
    fig.colorbar(drawn, ax=axes, label="DN", shrink=0.9)
    save(fig, "example-psf")


def slits():
    """A zoom on the core through each slit, without noise."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, layout="constrained")
    for ax, slit in zip(axes, (0.2, 0.4, 0.8, 1.6)):
        image = maps("muram_ar_slits", **{"simulation.slit_width": slit * u.arcsec})["total_dn"]
        show(ax, image, f"{slit} arcsec", cmap="gray",
             norm=PowerNorm(0.5, *np.nanpercentile(image.data, [1, 99.9])))
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
    axes[0].set_ylabel("y [arcsec]")
    save(fig, "example-slits")


def slit_mask():
    """The active region without and with the slit's mask."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9), sharey=True, layout="constrained")
    for ax, flag in zip(axes, (False, True)):
        image = maps("muram_ar_slit_mask", **{"simulation.slit_mask": flag})["total_dn"]
        show(ax, image, f"slit_mask: {flag}", cmap="gray",
             norm=PowerNorm(0.5, *np.nanpercentile(image.data, [1, 99.9])))
    axes[0].set_ylabel("y [arcsec]")
    save(fig, "example-slit-mask")


def blend_fit():
    """A measured spectrum and its fit, in a pixel of the core and one of the quiet Sun."""
    result = combination("muram_ar_fe12")
    spectra = result["first_dn_signal"].data
    fit = result["dn_fit_stats"]["first_fit_data"]
    wavelength = result["first_dn_signal"].axis_world_coords("em.wl")[0].to_value(u.AA)
    fine = np.linspace(wavelength[0], wavelength[-1], 400)
    ny, nx = spectra.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, (name, (y, x)) in zip(axes, PIXELS.items()):
        # The same place on the sky, in this run's pixels.
        row, column = int(y / 1088 * ny), int(x / 1088 * nx)
        ax.step(wavelength, spectra[row, column], where="mid", color="black", label="Measured")
        params = fit[row, column]
        background = params[-1]
        profiles = [amplitude * np.exp(-0.5 * ((fine - centre * 1e8) / (width * 1e8))**2)
                    for amplitude, centre, width in (params[0:3], params[3:6])]
        ax.plot(fine, background + sum(profiles), color="tab:red", lw=3, label="Fit")
        for profile, label in zip(profiles, ("Fe XII 195.119", "Fe XII 195.179")):
            ax.plot(fine, background + profile, label=label)
        ax.set_title(name)
        ax.set_xlabel("Wavelength [Angstrom]")
    axes[0].set_ylabel("DN per pixel")
    axes[0].legend()
    save(fig, "example-blend-fit")


def doppler():
    """The true velocity, and the velocity measured with each start."""
    observed = main_maps()
    default = maps("muram_ar_default_start")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True, layout="constrained")
    for ax, image, title in ((axes[0], observed["velocity_true"], "True"),
                             (axes[1], default["velocity_from_fit"], "Default start"),
                             (axes[2], observed["velocity_from_fit"], "These settings")):
        drawn = show(ax, image, title, **VELOCITY)
    axes[0].set_ylabel("y [arcsec]")
    fig.colorbar(drawn, ax=axes, label="Doppler velocity [km/s]", extend="both", shrink=0.9)
    save(fig, "example-doppler")


def precision():
    """The intensity in DN and the velocity's standard deviation over the iterations."""
    observed = sweep_maps()
    dn = observed["total_dn"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9), sharey=True, layout="constrained")
    drawn = show(axes[0], dn, "total_dn", cmap="gray",
                 norm=LogNorm(*np.nanpercentile(dn.data[dn.data > 0], [1, 99.9])))
    fig.colorbar(drawn, ax=axes[0], label="DN", shrink=0.75)
    drawn = show(axes[1], observed["velocity_std"], "velocity_std", vmin=0, vmax=10)
    fig.colorbar(drawn, ax=axes[1], label="km/s", extend="max", shrink=0.75)
    axes[0].set_ylabel("y [arcsec]")
    save(fig, "example-precision")


def velocity_error():
    """The true velocity minus the mean measured velocity."""
    fig, ax = plt.subplots(figsize=(4.8, 3.9), layout="constrained")
    drawn = show(ax, sweep_maps()["velocity_err"], "velocity_err", cmap="RdBu_r", vmin=-5, vmax=5)
    ax.set_ylabel("y [arcsec]")
    fig.colorbar(drawn, ax=ax, label="km/s", extend="both")
    save(fig, "example-velocity-error")


def exposure_time():
    """The shortest exposure that measures each pixel's velocity to within 2 km/s."""
    run = results("muram_ar_exposures")
    every = [get_results_for_combination(run, **{"simulation.expos": settings["simulation.expos"]})
             for settings in get_parameter_combinations(run)]
    image = create_sunpy_maps_from_combo(every[0], exposure_time_results=every,
                                         date_obs=DATE)["exposure_time"]
    fig = plt.figure(figsize=(4.8, 3.9), layout="constrained")
    ax = fig.add_subplot(projection=image)
    drawn = image.plot(axes=ax, title="Exposure for 2 km/s")
    format_exposure_time_colorbar(image, fig.colorbar(drawn, ax=ax))
    save(fig, "example-exposure-time")


def uniform():
    """Velocity precision against exposure time for each slit."""
    fig, ax = plt.subplots(figsize=(5, 3.6), layout="constrained")
    run = results("uniform")
    for slit in (0.2, 0.4) * u.arcsec:
        points = []
        for settings in get_parameter_combinations(run):
            if settings["simulation.slit_width"] == slit:
                stats = get_results_for_combination(run, **{
                    "simulation.slit_width": slit,
                    "simulation.expos": settings["simulation.expos"]})["dn_fit_stats"]
                velocity = stats["components"][stats["primary_component"]]["velocity"]["std"]
                points.append((settings["simulation.expos"].to_value(u.s),
                               float(np.squeeze(velocity.to_value(u.km / u.s)))))
        ax.plot(*zip(*sorted(points)), marker="o", label=str(slit))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xticks([5, 20, 80, 320], labels=["5", "20", "80", "320"])
    ax.set_yticks([0.3, 1, 3], labels=["0.3", "1", "3"])
    ax.set_xlabel("Exposure [s]")
    ax.set_ylabel("Velocity precision [km/s]")
    ax.legend()
    save(fig, "example-uniform")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    every = (stages, slit_and_frame, frame_spectrum, lines, pixels, sweep, binning, noise, psf,
             slits, slit_mask, blend_fit, doppler, precision, velocity_error, exposure_time,
             uniform)
    for make in [globals()[name] for name in sys.argv[2:]] or every:
        make()
        print(make.__name__)
