"""
Make the figures in the docs from the MURaM active region.

README.md in this folder says how to make the runs they come from. Run it
in the folder holding them, as

    python tools/docs_examples/make_figures.py . docs/images
"""
import sys
from pathlib import Path

import astropy.units as u
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm

from eclipse import (Detector_SWC, Telescope_EUVST, create_sunpy_maps_from_combo,
                     get_results_for_combination, load_instrument_response_results)
from eclipse.synthesis_file import read_synthesis
from eclipse.utils import distance_to_angle

RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DATE = "2026-01-01T00:00:00"
LINES = [("fe9", "Fe IX 171.07"), ("fe10", "Fe X 184.54"), ("fe12", "Fe XII 195.12"),
         ("fe14", "Fe XIV 211.32")]


def combination(name, **settings):
    results = load_instrument_response_results(RUN / "run" / "result" / f"{name}.h5")
    return get_results_for_combination(results, **settings) if settings else \
        next(iter(results["results"]["all_combinations"].values()))


def lines():
    """The four lines' intensity, seen from above, as synthesised."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, layout="constrained")
    for ax, (file, title) in zip(axes, LINES):
        synthesis = read_synthesis(RUN / "synth" / f"{file}.h5")
        line = next(iter(synthesis.lines.values()))
        intensity = np.trapezoid(line.intensity.value, line.wavelength.to_value(u.cm), axis=-1)
        edges = [distance_to_angle(e).to_value(u.arcsec)
                 for e in (synthesis.x_edges[[0, -1]], synthesis.y_edges[[0, -1]])]
        ax.imshow(intensity, origin="lower", cmap="gray", extent=[*edges[0], *edges[1]],
                  norm=LogNorm(*np.percentile(intensity, [1, 99.8])))
        ax.set_title(title)
        ax.set_xlabel("x [arcsec]")
    axes[0].set_ylabel("y [arcsec]")
    fig.savefig(OUT / "example-lines.png", dpi=110)


def slit_and_frame():
    """The Fe XII raster with the frame's slit, beside the full-CCD frame of both CCDs."""
    det = Detector_SWC()
    combo = combination("docs")
    image = combo["first_dn_signal_data"].sum(axis=-1)
    ny, nx = image.shape
    pitch = det.plate_scale_angle.to_value(u.arcsec / u.pix)
    row = 2 * pitch      # two pixels binned along the slit
    with h5py.File(RUN / "frame" / "frame.h5") as f:
        pre, over = f.attrs["serial_prescan"], f.attrs["serial_overscan"]
        slit_x = f.attrs["slit_position"]
        frame = {ccd: (f[ccd]["dn"][()], f[ccd]["wavelength"][()]) for ccd in ("left", "right")}
    n_rows, n_cols = frame["left"][1].size, 2048
    span = [-n_cols * pitch / 2, n_cols * pitch / 2]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, layout="constrained",
                             gridspec_kw=dict(width_ratios=[nx * 0.4, n_rows * pitch,
                                                            n_rows * pitch]))
    ax = axes[0]
    ax.imshow(image, origin="lower", cmap="gray",
              norm=PowerNorm(0.5, *np.percentile(image, [1, 99.9])),
              extent=[-nx * 0.4 / 2, nx * 0.4 / 2, -ny * row / 2, ny * row / 2])
    openings = Telescope_EUVST().slit_openings(0.4 * u.arcsec).to_value(u.arcsec)
    for i, (low, high) in enumerate(openings):
        ax.plot([slit_x, slit_x], [low, high], color="tab:red", label="slit" if i == 0 else None)
    ax.set_ylim(span)
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("Along the slit [arcsec]")
    ax.set_title("Fe XII 195.12, 60 s")
    ax.legend(loc="upper right")
    shown = {}
    for ccd, (dn, wavelength) in frame.items():
        half = (dn.shape[1] - 2 * pre - 2 * over) // 2
        columns = np.r_[pre:pre + half, pre + half + 2 * over:pre + 2 * half + 2 * over]
        order = np.argsort(wavelength)
        counts = np.clip(dn[:n_rows][:, columns][order].T, 1, None)
        shown[ccd] = (wavelength[order], np.log10(counts))
    values = np.concatenate([s.ravel() for _, s in shown.values()])
    low, high = np.percentile(values, [5, 99.9])
    for ax, (ccd, (w, s)) in zip(axes[1:], shown.items()):
        step = (w[-1] - w[0]) / (len(w) - 1)
        im = ax.imshow(s, origin="lower", cmap="gray", vmin=low, vmax=high,
                       extent=[w[0] - step / 2, w[-1] + step / 2, *span])
        ax.set_aspect(step / pitch)
        ax.set_title(f"Full-CCD frame, {ccd} CCD")
        ax.set_xlabel("Wavelength [Angstrom]")
    fig.colorbar(im, ax=axes[1:], label="log10 DN", shrink=0.9, extend="both")
    fig.savefig(OUT / "example-slit-and-frame.png", dpi=110)


def doppler():
    """The true Doppler velocity and the first iteration's measurement."""
    maps = create_sunpy_maps_from_combo(combination("docs"), date_obs=DATE)
    fig = plt.figure(figsize=(10, 4.4), layout="constrained")
    for i, (key, title) in enumerate((("velocity_true", "True"),
                                      ("velocity_from_fit", "Measured, 60 s"))):
        m = maps[key]
        ax = fig.add_subplot(1, 2, i + 1, projection=m)
        im = m.plot(axes=ax, title=title, cmap="RdBu_r", vmin=-25, vmax=25)
        fig.colorbar(im, ax=ax, shrink=0.8, label="km/s", extend="both")
    fig.savefig(OUT / "example-doppler.png", dpi=110)


def precision():
    """The intensity in DN and the velocity's standard deviation over the iterations."""
    maps = create_sunpy_maps_from_combo(combination("docs"), date_obs=DATE)
    intensity = LogNorm(*np.nanpercentile(maps["total_dn"].data, [1, 99.9]))
    fig = plt.figure(figsize=(10, 4.4), layout="constrained")
    for i, (key, title, kw, label, extend) in enumerate([
            ("total_dn", "Intensity", dict(cmap="gray", norm=intensity), "DN", "neither"),
            ("velocity_std", "Velocity precision", dict(cmap="viridis", vmin=0, vmax=10), "km/s",
             "max")]):
        m = maps[key]
        ax = fig.add_subplot(1, 2, i + 1, projection=m)
        im = m.plot(axes=ax, title=title, **kw)
        fig.colorbar(im, ax=ax, shrink=0.8, label=label, extend=extend)
    fig.savefig(OUT / "example-precision.png", dpi=110)


def psf():
    """A zoom on the core without and with the PSF, without noise."""
    fig = plt.figure(figsize=(10, 4.4), layout="constrained")
    results = load_instrument_response_results(RUN / "run" / "result" / "psf.h5")
    for i, flag in enumerate((False, True)):
        combo = get_results_for_combination(results, **{"simulation.psf": flag})
        m = create_sunpy_maps_from_combo(combo, date_obs=DATE)["total_dn"]
        ax = fig.add_subplot(1, 2, i + 1, projection=m)
        im = m.plot(axes=ax, title="PSF on" if flag else "PSF off", cmap="gray",
                    norm=PowerNorm(0.5, *np.nanpercentile(m.data, [1, 99.9])))
        # 20 by 20 arcsec of loops in the core.
        centre = m.wcs.world_to_pixel(m.center)
        for set_lim, axis, step in ((ax.set_xlim, 0, 0.4), (ax.set_ylim, 1, 0.159)):
            set_lim(centre[axis] - 10 / step, centre[axis] + 10 / step)
        fig.colorbar(im, ax=ax, shrink=0.8, label="DN")
    fig.savefig(OUT / "example-psf.png", dpi=110)


def uniform():
    """Velocity precision against exposure time, at quiet-Sun and core intensities."""
    fig, ax = plt.subplots(figsize=(5, 3.6), layout="constrained")
    for name, label in (("uniform_qs", "200 erg / (s cm2 sr)"),
                        ("uniform_ar", "2000 erg / (s cm2 sr)")):
        results = load_instrument_response_results(RUN / "run" / "result" / f"{name}.h5")
        points = []
        for combo in results["results"]["all_combinations"].values():
            expos = combo["config_objects"]["simulation"].expos.to_value(u.s)
            stats = combo["dn_fit_stats"]
            comp = stats["components"][stats["primary_component"]]
            points.append((expos, float(np.squeeze(comp["velocity"]["std"].to_value(u.km / u.s)))))
        points.sort()
        ax.plot(*zip(*points), marker="o", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([5, 10, 20, 40, 80], labels=["5", "10", "20", "40", "80"])
    ax.minorticks_off()
    ax.set_xlabel("Exposure [s]")
    ax.set_ylabel("Velocity precision [km/s]")
    ax.legend()
    fig.savefig(OUT / "example-uniform.png", dpi=110)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = [globals()[name] for name in sys.argv[3:]]
    for make in chosen or (lines, slit_and_frame, doppler, precision, psf, uniform):
        make()
        print(make.__name__)
