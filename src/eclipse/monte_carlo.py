"""
Monte Carlo simulation functions for instrument response analysis.
"""

from __future__ import annotations
import dataclasses
import warnings
from typing import Tuple
import numpy as np
import astropy.units as u
from ndcube import NDCube
from tqdm import tqdm
from .radiometric import (
    apply_exposure, sample_photon_arrivals, intensity_to_photons, add_telescope_throughput,
    photons_to_pixel_counts, apply_focusing_optics_psf, to_electrons, add_visible_stray_light, to_dn,
    add_pinhole_visible_light, dn_variance, _digitise
)
from .config import Detector_SWC
from .readout import (FocalPlane_SWC, ReadoutSequence, dark_current_time,
                      windows_from_wavelengths)
from .pinhole_diffraction import (_pinholes_on, _with_filter_transmission,
                                  apply_euv_pinhole_diffraction)
from .fitting import fit_cube_gauss, spectral_pixel_width, summarise_fits
from .utils import angle_to_distance, binned_photon_wavelengths, rebin_slit_offchip, _get_mpi_info


def _fit_results(label: str, fit_data: np.ndarray, failed: np.ndarray,
                 units: list, signal: NDCube, rest_wavelength: u.Quantity,
                 fit_config) -> dict:
    """Statistics of one signal's fits, printing how many fits failed."""
    results = summarise_fits(fit_data, failed, units,
                             spectral_pixel_width(signal), rest_wavelength,
                             fit_config)
    failed_fits = results["failed_fits"]
    n_failed = np.count_nonzero(failed)
    if n_failed:
        line = (f"  {label} fits: {n_failed} of {failed.size} failed, in "
                f"{np.count_nonzero(failed_fits)} of {failed_fits.size} pixels, "
                f"and are left out of the statistics")
        no_fit = np.count_nonzero(failed_fits == results["n_iterations"])
        if no_fit:
            line += f"; no fit succeeded in {no_fit} of them"
    else:
        line = f"  {label} fits: none of {failed.size} failed"
    print(line)
    return results


def _with_photon_wavelength(cube: NDCube) -> NDCube:
    """
    Cube with its photon wavelengths in the meta.

    `meta["photon_wavelength"]` and `meta["photon_rms_wavelength"]` are the
    wavelengths of the mean and root mean square energies of each pixel's
    photons. The cube's own are kept; otherwise each pixel's wavelength is
    used, as for photons of one energy.
    """
    meta = dict(cube.meta or {})
    if "photon_wavelength" not in meta:
        meta["photon_wavelength"] = np.broadcast_to(cube.axis_world_coords_values(2)[0],
                                                    cube.data.shape, subok=True)
    meta.setdefault("photon_rms_wavelength", meta["photon_wavelength"])
    for key in ("photon_wavelength", "photon_rms_wavelength"):
        if np.shape(meta[key]) != cube.data.shape:
            raise ValueError(f"The cube's {key} is shaped {np.shape(meta[key])}, not as its "
                             f"data, {cube.data.shape}.")
    return NDCube(cube.data, wcs=cube.wcs, unit=cube.unit, meta=meta)


def _photon_wavelengths(I_cube: NDCube, t_exp: u.Quantity, det, tel, sim,
                        offchip_bin_slit: int, uniform_mode: bool) -> tuple:
    """
    Wavelengths of the mean and root mean square photon energies in each
    pixel, binned as the DN are, for `dn_variance`.
    """
    photons = _photons_on_the_detector(I_cube, t_exp, det, tel, sim, uniform_mode)[-1]
    return binned_photon_wavelengths(photons, offchip_bin_slit)


def _weighted(fit_config) -> bool:
    """Whether to weight the fits: yes by default, otherwise as the fitting block says."""
    return fit_config is None or fit_config.weighted


def _visible_electrons(I_cube: NDCube, t_exp: u.Quantity, det, tel, sim,
                       offchip_bin_slit: int) -> np.ndarray:
    """Mean electrons from visible light in each pixel, binned as the DN are."""
    nothing = NDCube(np.zeros(I_cube.data.shape), wcs=I_cube.wcs,
                     unit=u.electron / u.pixel, meta=I_cube.meta)
    visible = add_visible_stray_light(nothing, t_exp, det, sim, tel, noise=False)
    visible = add_pinhole_visible_light(visible, t_exp, det, sim, tel, noise=False)
    return rebin_slit_offchip(visible, offchip_bin_slit).data


def expected_dn_uncertainty(I_cube: NDCube, t_exp: u.Quantity, det, tel, sim,
                            offchip_bin_slit: int = 1, uniform_mode: bool = False) -> np.ndarray:
    """
    Expected uncertainty of each pixel's DN, from the noise-free signal.

    The measured spectra are weighted by uncertainties estimated from their
    own signal (`dn_variance`). The ground truth, a fit to the noise-free
    spectra, is weighted by these instead, so it is the same fit of the
    same line and its difference from a measurement is due to the noise.

    Parameters
    ----------
    I_cube : NDCube
        Spectral radiance on the detector's pixels, per second, as given to
        the Monte Carlo.
    t_exp : u.Quantity
        Exposure time.
    det, tel, sim
        Detector, telescope and simulation settings.
    offchip_bin_slit : int, optional
        Number of pixels along the slit summed after read-out. Default 1.
    uniform_mode : bool, optional
        As for `simulate_once`. Default False.

    Returns
    -------
    np.ndarray
        Uncertainty in DN, same shape as the binned DN.
    """
    quiet = dataclasses.replace(sim, noise=False)
    steps = simulate_once(I_cube, t_exp, det, tel, quiet, uniform_mode=uniform_mode)
    electrons = steps[8]
    # Clip each pixel at the digitiser's maximum before binning, as to_dn
    # does, but do not round, as the mean DN is not rounded.
    dn = _digitise(electrons.unit.to(u.electron / u.pixel) * electrons.data, det, rounded=False)
    dn = rebin_slit_offchip(NDCube(dn, wcs=electrons.wcs, meta=electrons.meta),
                            offchip_bin_slit).data
    visible = _visible_electrons(I_cube, t_exp, det, tel, sim, offchip_bin_slit)
    mean, rms = binned_photon_wavelengths(steps[5], offchip_bin_slit)
    return np.sqrt(dn_variance(dn, mean, t_exp, det, visible, offchip_bin_slit,
                               rms_wavelength=rms, dark_time=_dark_time(I_cube, t_exp, det, sim)))


def _photons_on_the_detector(I_cube: NDCube, t_exp: u.Quantity, det, tel, sim,
                             uniform_mode: bool = False) -> Tuple[NDCube, ...]:
    """
    Noise-free cubes of `simulate_once`, up to the photons at the detector.

    Returns the radiance over the exposure, the photons, the photons
    collected by the telescope, the photons per pixel, those after the PSF,
    and those with the pinholes' light.
    """
    # The slit's ends and fiducials are cut out as the scene is laid onto the
    # pixels, so a cube laid on without them, or a uniform intensity, which
    # has no position along the slit, would silently be left unmasked.
    if getattr(sim, "slit_mask", False) and not (I_cube.meta or {}).get("slit_mask_applied"):
        raise ValueError(
            "simulation.slit_mask is on, but the slit's ends and fiducials were not cut out of "
            "this cube. They are cut out as a scene is laid onto the pixels by rebin_atmosphere "
            "or rebin_spectra, given the telescope and this simulation. A uniform intensity has "
            "no position along the slit, so it cannot have them.")
    # The blur comes before the slit, so a masked cube blurred afterwards would
    # spread light back into the fiducials and past the slit's ends.
    if (getattr(sim, "slit_mask", False) and sim.psf
            and not (I_cube.meta or {}).get("psf_applied")):
        raise ValueError(
            "simulation.slit_mask and psf are both on, but this cube was masked without the "
            "PSF, so blurring it now would put light back behind the fiducials. Lay the scene "
            "onto the pixels again with rebin_atmosphere or rebin_spectra, with both on.")

    # Apply exposure time
    intensity_exp = apply_exposure(I_cube, t_exp)

    # Convert to total photons at each pixel's wavelength, or at the
    # wavelength of the photons' mean energy if the cube gives it, as it does
    # once laid onto the pixels through the telescope.
    photons_total = _with_photon_wavelength(intensity_to_photons(intensity_exp))

    # Apply telescope optical throughput
    photons_throughput = add_telescope_throughput(photons_total, tel)

    # Convert to pixel counts
    photons_pixels = photons_to_pixel_counts(photons_throughput, det.wvl_res,
                                             det.plate_scale_length,
                                             angle_to_distance(sim.slit_width))

    # The pinholes pass the light the filter blocked from each photon, which
    # must be known before a blur mixes pixels. A cube laid onto the pixels
    # through the telescope with the pinholes on gives it at each photon's
    # own wavelength. Photons counted at their pixel's wavelength passed the
    # filter at that wavelength.
    meta = I_cube.meta or {}
    if _pinholes_on(sim) and "photon_filter_transmission" not in meta:
        if "photon_wavelength" in meta:
            raise ValueError(
                "The cube's photons were counted at their own wavelengths, but it does not say "
                "how much of their light the filter passed, which the pinholes need. Lay it "
                "onto the pixels with rebin_atmosphere or rebin_spectra, given the telescope "
                "and a simulation with the pinholes on.")
        photons_pixels = _with_filter_transmission(photons_pixels, tel)

    # Apply focusing optics PSF (primary mirror + diffraction grating),
    # unless the cube was laid onto the pixels through it (rebin_atmosphere
    # with a telescope). That is exact; blurring the pixels is not.
    if sim.psf and not (I_cube.meta or {}).get("psf_applied", False):
        photons_focused = apply_focusing_optics_psf(
            photons_pixels, tel, det, sim, convolve_spatial=not uniform_mode,
            boundary=getattr(sim, "psf_boundary", "replicate"),
        )
    else:
        photons_focused = photons_pixels

    # Apply EUV pinhole diffraction effects (after focusing optics, if enabled)
    if _pinholes_on(sim):
        photons_euv_pinholes = apply_euv_pinhole_diffraction(photons_focused, det, sim, tel)
    else:
        photons_euv_pinholes = photons_focused
    return (intensity_exp, photons_total, photons_throughput, photons_pixels, photons_focused,
            photons_euv_pinholes)


def _rows_of(focal_plane: FocalPlane_SWC, wavelength: np.ndarray) -> np.ndarray:
    """
    Row of the focal plane recording each wavelength, in Angstrom, at the
    centre of the slit. A wavelength in the gap between the CCDs, or beyond
    their outer ends, takes the nearest row of the nearer CCD.
    """
    positions = np.arange(focal_plane.n_rows + 1) - 0.5
    rows, distances = [], []
    for ccd in ("left", "right"):
        edges = focal_plane.row_edges(ccd).to_value(u.Angstrom)
        order = np.argsort(edges)
        rows.append(np.interp(wavelength, edges[order], positions[order]))
        distances.append(np.maximum(edges.min() - wavelength, wavelength - edges.max()))
    return np.where(distances[0] <= distances[1], rows[0], rows[1])


def _dark_time(I_cube: NDCube, t_exp: u.Quantity, det, sim) -> u.Quantity:
    """
    Time each wavelength pixel of `I_cube` collects dark current.

    With `sim.readout_dark_current` on and the SWC detector, this is the
    time :func:`~eclipse.readout.dark_current_time` gives the row that
    records the pixel's wavelength: the clear before the exposure, the
    exposure, and the wait to be read out. The rows of the cube's
    wavelengths are taken to be the only ones read, and the rest are
    dumped. Otherwise it is `t_exp`.
    """
    if not sim.readout_dark_current or not isinstance(det, Detector_SWC):
        return u.Quantity(t_exp)
    focal_plane = FocalPlane_SWC.from_detector(det)
    n_rows = focal_plane.n_rows
    wavelength = I_cube.axis_world_coords_values(2)[0].to_value(u.Angstrom)
    rows = _rows_of(focal_plane, wavelength)
    # The window holds every row a pixel of the cube reaches, half a pixel
    # either side of its centre. Wavelengths all in the gap reach no row, and
    # take the last, at the gap's edge.
    half = 0.5 * np.abs(np.diff(wavelength)).mean() if wavelength.size > 1 else 0.0
    try:
        windows = windows_from_wavelengths(
            focal_plane, [((wavelength.min() - half) * u.AA, (wavelength.max() + half) * u.AA)])
    except ValueError:
        windows = [(n_rows - 1, n_rows - 1)]
    sequence = ReadoutSequence.from_detector(det, windows=windows)
    if sequence.cte_parallel != 1 or sequence.cte_serial != 1:
        warnings.warn("A run does not model charge transfer, so the detector's cte_parallel "
                      "and cte_serial are taken as 1 here. Charge left behind as a frame is "
                      "read is modelled by eclipse.readout.", UserWarning, stacklevel=3)
        sequence = dataclasses.replace(sequence, cte_parallel=1.0, cte_serial=1.0)
    per_row = dark_current_time(t_exp, sequence, n_rows)[:n_rows]
    return np.interp(rows, np.arange(n_rows), per_row.to_value(u.s)) * u.s


def simulate_once(
    I_cube: NDCube,
    t_exp: u.Quantity,
    det,
    tel,
    sim,
    *,
    uniform_mode: bool = False,
    photon_shot_inverse_transform: bool = False,
    dark_current_inverse_transform: bool = False,
) -> Tuple[NDCube, ...]:
    """
    Simulate one observation of a cube of spectra, with noise, without fitting.

    The steps, in order, are the exposure, telescope, pixels, PSF, pinholes,
    photon arrivals, detector, stray light and digitiser.

    Parameters
    ----------
    I_cube : NDCube
        Spectral radiance on the detector's pixels, per second, e.g. from
        `create_uniform_intensity_cube`, with the line's rest wavelength in
        `meta["rest_wav"]`.
    t_exp : u.Quantity
        Exposure time.
    det : Detector_SWC or Detector_EIS
        Detector.
    tel : Telescope_EUVST or Telescope_EIS
        Telescope.
    sim : Simulation
        Simulation settings.
    uniform_mode : bool, optional
        If True, the cube is the same all along the slit, so the PSF blurs it
        in wavelength only. Default False.
    photon_shot_inverse_transform : bool, optional
        Draw the photons by inverse-transform sampling, for common random
        numbers between runs that differ in photon flux. Default False.
    dark_current_inverse_transform : bool, optional
        Draw the dark current by inverse-transform sampling, for common random
        numbers between runs that differ in dark current. Default False.

    Returns
    -------
    tuple of NDCube
        Cube after each step: radiance over the exposure, photons, photons
        collected by the telescope, photons per pixel, photons after the PSF,
        photons arriving, electrons, electrons with stray light, electrons
        with the pinholes' visible light, and DN.
    """
    (intensity_exp, photons_total, photons_throughput, photons_pixels, photons_focused,
     photons_euv_pinholes) = _photons_on_the_detector(I_cube, t_exp, det, tel, sim,
                                                      uniform_mode)

    # This flag controls every random draw below. With sim.noise False the
    # run returns the mean measured signal rather than one realisation.
    noise = getattr(sim, "noise", True)

    # Sample discrete photon arrivals (photon shot noise)
    photon_arrivals = sample_photon_arrivals(
        photons_euv_pinholes,
        photon_shot_inverse_transform=photon_shot_inverse_transform,
        noise=noise,
    )

    # Convert to electrons (detector response: QE, Fano noise, dark current, read noise)
    electrons = to_electrons(
        photon_arrivals, t_exp, det,
        dark_current_inverse_transform=dark_current_inverse_transform,
        noise=noise,
        photon_shot_inverse_transform=photon_shot_inverse_transform,
        dark_time=_dark_time(I_cube, t_exp, det, sim),
    )

    # Add visible stray light (with filter throughput)
    electrons_stray = add_visible_stray_light(electrons, t_exp, det, sim, tel,
                                              noise=noise)

    # Add visible light pinhole effects (if enabled)
    if sim.enable_pinholes and len(sim.pinhole_sizes) > 0:
        electrons_pinholes = add_pinhole_visible_light(electrons_stray, t_exp,
                                                       det, sim, tel, noise=noise)
    else:
        electrons_pinholes = electrons_stray
    
    # Convert to digital numbers
    dn = to_dn(electrons_pinholes, det)

    return (intensity_exp, photons_total, photons_throughput, photons_pixels, 
            photons_focused, photon_arrivals, electrons, electrons_stray, 
            electrons_pinholes, dn)


def _own_stream(rank: int) -> None:
    """
    Reseed NumPy's generator from its current state and the MPI rank.

    With the same seed on every rank, as in the docs' recipe for comparing
    runs, the ranks would repeat each other's iterations and the spread
    would come out too narrow by the square root of the number of ranks.
    Mixing in the rank avoids this, and a run still repeats with the same
    seed and number of ranks.
    """
    entropy = int(np.random.randint(0, 2**31 - 1))
    np.random.seed(np.random.SeedSequence([entropy, rank]).generate_state(8))


def monte_carlo(I_cube: NDCube, t_exp: u.Quantity, det, tel, sim, n_iter: int = 5,
                fit_config=None, offchip_bin_slit: int = 1,
                fit_signals: str = "both", uniform_mode: bool = False,
                *,
                photon_shot_inverse_transform: bool = False,
                dark_current_inverse_transform: bool = False) -> Tuple[NDCube, dict | None, NDCube, dict | None]:
    """
    Simulate an observation many times, each with new noise, and fit every one.

    `eclipse` runs this for each combination of settings. Under MPI the
    iterations are split between the ranks and the results gathered on the
    first.

    Parameters
    ----------
    I_cube : NDCube
        Spectral radiance on the detector's pixels, per second, as for
        `simulate_once`.
    t_exp : u.Quantity
        Exposure time.
    det : Detector_SWC or Detector_EIS
        Detector.
    tel : Telescope_EUVST or Telescope_EIS
        Telescope.
    sim : Simulation
        Simulation settings.
    n_iter : int, optional
        Number of iterations. Default 5.
    fit_config : FitConfig, optional
        Fit settings. Default one Gaussian.
    offchip_bin_slit : int, optional
        Number of pixels along the slit summed after read-out, each with its
        own noise. Default 1.
    fit_signals : str, optional
        Which signals to fit: `"both"` (default), `"dn"` or `"photon"`.
        Fitting only one takes about half the time.
    uniform_mode : bool, optional
        If True, the cube is a single line of known intensity, as made by
        `create_uniform_intensity_cube`, with `offchip_bin_slit` pixels along
        the slit. Default False.
    photon_shot_inverse_transform : bool, optional
        Draw the photons by inverse-transform sampling, for common random
        numbers between runs that differ in photon flux. Default False.
    dark_current_inverse_transform : bool, optional
        Draw the dark current by inverse-transform sampling, for common random
        numbers between runs that differ in dark current. Default False.

    Returns
    -------
    first_dn_signal : NDCube
        First iteration's signal, in DN.
    dn_fit_results : dict or None
        Fits to the DN signal, or None if not fitted. `"components"` holds,
        for each fitted component by name, the intensity, velocity and
        width, each as `"first"`, `"mean"` and `"std"` maps.
        `"primary_component"` names the primary component, and
        `"failed_fits"` counts the failed fits in each pixel.
    first_photon_signal : NDCube
        First iteration's photons arriving at the detector.
    photon_fit_results : dict or None
        Fits to the photons, as for the DN, or None.

    Under MPI only the first rank gets these; the other ranks get four Nones.
    """
    if fit_signals not in ("both", "dn", "photon"):
        raise ValueError(f"fit_signals must be 'both', 'dn', or 'photon', got '{fit_signals}'")

    do_dn = fit_signals in ("both", "dn")
    do_photon = fit_signals in ("both", "photon")
    rest_wavelength = I_cube.meta["rest_wav"]

    # DN spectra are weighted by uncertainties estimated from their own
    # signal, with the visible light's share of it known. Photons have no
    # read noise, so a pixel with no photons would have no uncertainty; they
    # are fitted by their Poisson likelihood instead.
    visible = (_visible_electrons(I_cube, t_exp, det, tel, sim, offchip_bin_slit)
               if do_dn and _weighted(fit_config) else None)
    # Each photon frees electrons according to its energy, so a pixel's DN
    # vary with the mean and root mean square energies of its photons.
    photon_wavelengths = (None if visible is None else _photon_wavelengths(
        I_cube, t_exp, det, tel, sim, offchip_bin_slit, uniform_mode))
    dark_time = _dark_time(I_cube, t_exp, det, sim)

    def _dn_uncertainty(dn_data: np.ndarray) -> np.ndarray | None:
        if visible is None:
            return None
        mean, rms = photon_wavelengths
        return np.sqrt(dn_variance(dn_data, mean, t_exp, det, visible, offchip_bin_slit,
                                   rms_wavelength=rms, dark_time=dark_time))

    # --- MPI distribution: split iterations across ranks -----------------
    comm, rank, world_size = _get_mpi_info()
    if world_size > 1:
        _own_stream(rank)
        base, remainder = divmod(n_iter, world_size)
        local_n_iter = base + (1 if rank < remainder else 0)
        if rank == 0:
            print(f"MPI: distributing {n_iter} MC iterations across "
                  f"{world_size} ranks ({local_n_iter} per rank)")
    else:
        local_n_iter = n_iter

    if uniform_mode:
        # Uniform intensity mode: batch all MC simulations, then fit in parallel
        first_dn_signal, first_photon_signal = None, None
        dn_data_list, photon_data_list = [], []

        show_progress = (rank == 0)
        desc = "Monte-Carlo (simulate)" if world_size == 1 else f"MC sim (rank 0/{world_size})"

        for i in tqdm(range(local_n_iter), desc=desc, unit="iter", leave=False,
                      disable=not show_progress):
            (intensity_exp, photons_total, photons_throughput, photons_pixels,
             photons_focused, photon_arrivals, electrons, electrons_stray,
             electrons_pinholes, dn) = simulate_once(
                I_cube, t_exp, det, tel, sim,
                uniform_mode=uniform_mode,
                photon_shot_inverse_transform=photon_shot_inverse_transform,
                dark_current_inverse_transform=dark_current_inverse_transform,
            )

            if i == 0 and rank == 0:
                first_dn_signal = rebin_slit_offchip(dn, offchip_bin_slit)
                first_photon_signal = rebin_slit_offchip(photon_arrivals, offchip_bin_slit)

            # Off-chip binning before batching
            dn_binned = rebin_slit_offchip(dn, offchip_bin_slit)
            photon_binned = rebin_slit_offchip(photon_arrivals, offchip_bin_slit)

            # .data shape is (n_slit, n_scan, n_lam); after binning both spatial
            # axes are length 1, so [0] leaves the one spectrum as (1, n_lam)
            if do_dn:
                dn_data_list.append(dn_binned.data[0])
            if do_photon:
                photon_data_list.append(photon_binned.data[0])

        # --- MPI gather: collect data from all ranks on root -----------
        if world_size > 1:
            if do_dn:
                all_dn_lists = comm.gather(dn_data_list, root=0)
                if rank == 0:
                    dn_data_list = [v for sublist in all_dn_lists for v in sublist]
            if do_photon:
                all_photon_lists = comm.gather(photon_data_list, root=0)
                if rank == 0:
                    photon_data_list = [v for sublist in all_photon_lists for v in sublist]

        # --- Fit (only on rank 0) --------------------------------------
        dn_fit_results = None
        photon_fit_results = None

        if rank == 0:
            if do_dn and dn_data_list:
                # Stack: each element is (1, n_lam), result is (n_iter, 1, n_lam).
                dn_stacked = np.stack(dn_data_list, axis=0)
                dn_batch = NDCube(data=dn_stacked, wcs=first_dn_signal.wcs,
                                  unit=first_dn_signal.unit)

                print(f"  Fitting {len(dn_data_list)} DN MC spectra in parallel...")
                dn_fit_values, dn_fit_units, dn_failed = fit_cube_gauss(
                    dn_batch, n_jobs=sim.ncpu, fit_config=fit_config,
                    return_failed=True, uncertainty=_dn_uncertainty(dn_stacked))
                # Reshape from (n_iter, 1, ...) to (n_iter, 1, 1, ...)
                dn_fit_results = _fit_results(
                    "DN", dn_fit_values[:, np.newaxis, :, :],
                    dn_failed[:, np.newaxis, :], dn_fit_units,
                    first_dn_signal, rest_wavelength, fit_config)

            if do_photon and photon_data_list:
                photon_stacked = np.stack(photon_data_list, axis=0)
                photon_batch = NDCube(data=photon_stacked, wcs=first_photon_signal.wcs,
                                      unit=first_photon_signal.unit)

                print(f"  Fitting {len(photon_data_list)} photon MC spectra in parallel...")
                photon_fit_values, photon_fit_units, photon_failed = fit_cube_gauss(
                    photon_batch, n_jobs=sim.ncpu, fit_config=fit_config,
                    return_failed=True, poisson=_weighted(fit_config))
                photon_fit_results = _fit_results(
                    "Photon", photon_fit_values[:, np.newaxis, :, :],
                    photon_failed[:, np.newaxis, :], photon_fit_units,
                    first_photon_signal, rest_wavelength, fit_config)

    else:
        # -- Normal mode: fit each MC iteration separately ---------------
        first_dn_signal, first_photon_signal = None, None
        dn_fit_values_list, photon_fit_values_list = [], []

        show_progress = (rank == 0)
        desc = "Monte-Carlo" if world_size == 1 else f"MC (rank 0/{world_size})"

        for i in tqdm(range(local_n_iter), desc=desc, unit="iter", leave=False,
                      disable=not show_progress):
            # Simulate one run
            (intensity_exp, photons_total, photons_throughput, photons_pixels,
             photons_focused, photon_arrivals, electrons, electrons_stray,
             electrons_pinholes, dn) = simulate_once(
                I_cube, t_exp, det, tel, sim,
                uniform_mode=uniform_mode,
                photon_shot_inverse_transform=photon_shot_inverse_transform,
                dark_current_inverse_transform=dark_current_inverse_transform,
            )

            # Store first iteration signals only on rank 0 (binned, to match fit shapes)
            if i == 0 and rank == 0:
                first_dn_signal = rebin_slit_offchip(dn, offchip_bin_slit)
                first_photon_signal = rebin_slit_offchip(photon_arrivals, offchip_bin_slit)

            # Off-chip slit binning (sum already noisy pixels)
            if do_dn:
                dn_binned = rebin_slit_offchip(dn, offchip_bin_slit)
                dn_fit_values, dn_fit_units, dn_failed = fit_cube_gauss(
                    dn_binned, n_jobs=sim.ncpu, fit_config=fit_config,
                    return_failed=True, uncertainty=_dn_uncertainty(dn_binned.data))
                dn_fit_values_list.append((dn_fit_values, dn_failed))

            if do_photon:
                photon_binned = rebin_slit_offchip(photon_arrivals, offchip_bin_slit)
                photon_fit_values, photon_fit_units, photon_failed = fit_cube_gauss(
                    photon_binned, n_jobs=sim.ncpu, fit_config=fit_config,
                    return_failed=True, poisson=_weighted(fit_config))
                photon_fit_values_list.append((photon_fit_values, photon_failed))

        # --- MPI gather: collect fit arrays from all ranks on root -----------
        if world_size > 1:
            if do_dn:
                all_dn_lists = comm.gather(dn_fit_values_list, root=0)
                if rank == 0:
                    dn_fit_values_list = [v for sublist in all_dn_lists for v in sublist]
            if do_photon:
                all_photon_lists = comm.gather(photon_fit_values_list, root=0)
                if rank == 0:
                    photon_fit_values_list = [v for sublist in all_photon_lists for v in sublist]

        # --- Compute statistics (only meaningful on rank 0) ------------------
        dn_fit_results = None
        photon_fit_results = None

        if rank == 0:
            if do_dn and dn_fit_values_list:
                dn_fit_results = _fit_results(
                    "DN", np.stack([v for v, _ in dn_fit_values_list]),
                    np.stack([f for _, f in dn_fit_values_list]),
                    dn_fit_units, first_dn_signal, rest_wavelength, fit_config)

            if do_photon and photon_fit_values_list:
                photon_fit_results = _fit_results(
                    "Photon", np.stack([v for v, _ in photon_fit_values_list]),
                    np.stack([f for _, f in photon_fit_values_list]),
                    photon_fit_units, first_photon_signal, rest_wavelength,
                    fit_config)
    
    return first_dn_signal, dn_fit_results, first_photon_signal, photon_fit_results
