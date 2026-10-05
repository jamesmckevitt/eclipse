"""
Monte Carlo simulation functions for instrument response analysis.
"""

from __future__ import annotations
import dataclasses
from typing import Tuple
import numpy as np
import astropy.units as u
from ndcube import NDCube
from tqdm import tqdm
from .radiometric import (
    apply_exposure, sample_photon_arrivals, intensity_to_photons, add_telescope_throughput,
    photons_to_pixel_counts, apply_focusing_optics_psf, to_electrons, add_visible_stray_light, to_dn,
    add_pinhole_visible_light, dn_variance
)
from .pinhole_diffraction import apply_euv_pinhole_diffraction
from .fitting import fit_cube_gauss, spectral_pixel_width, summarise_fits
from .utils import angle_to_distance, rebin_slit_offchip, _get_mpi_info


def _fit_results(label: str, fit_data: np.ndarray, failed: np.ndarray,
                 units: list, signal: NDCube, rest_wavelength: u.Quantity,
                 fit_config) -> dict:
    """Statistics of one signal's fits, and a line saying how many failed."""
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
    *cube*, with the wavelength of the mean energy of each pixel's photons in
    its ``meta["photon_wavelength"]``: its own, or each pixel's wavelength.
    """
    meta = dict(cube.meta or {})
    if "photon_wavelength" in meta:
        if np.shape(meta["photon_wavelength"]) != cube.data.shape:
            raise ValueError(
                f"The cube's photon_wavelength is shaped {np.shape(meta['photon_wavelength'])}, "
                f"not as its data, {cube.data.shape}.")
        return cube
    meta["photon_wavelength"] = np.broadcast_to(cube.axis_world_coords_values(2)[0],
                                                cube.data.shape, subok=True)
    return NDCube(cube.data, wcs=cube.wcs, unit=cube.unit, meta=meta)


def _variance_wavelength(photons: NDCube, n_bin: int) -> u.Quantity:
    """
    The wavelength `dn_variance` takes the electrons per photon at, for each pixel as the DN are binned.

    Rows binned off the chip are each read out on their own, so a binned
    pixel's EUV electrons vary as the sum of each row's: the electrons per
    photon *m*, plus the Fano factor, times the row's electrons. Over the
    rows, that is *m* weighted by the electrons, sum(N m^2) / sum(N m), which
    is the electrons per photon of the wavelength sum(N / l) / sum(N / l^2),
    for N photons in a row whose mean energy is that of wavelength l. With
    no binning, it is l.
    """
    wavelength = u.Quantity(photons.meta["photon_wavelength"])
    if n_bin == 1:
        return wavelength
    data = np.asarray(photons.data, dtype=float)
    per_wavelength = np.divide(data, wavelength.value, out=np.zeros(data.shape),
                               where=wavelength.value > 0)
    per_square = np.divide(per_wavelength, wavelength.value, out=np.zeros(data.shape),
                           where=wavelength.value > 0)
    first = rebin_slit_offchip(NDCube(per_wavelength, wcs=photons.wcs), n_bin).data
    second = rebin_slit_offchip(NDCube(per_square, wcs=photons.wcs), n_bin).data
    row = np.array(wavelength.value[:first.shape[0] * n_bin:n_bin], dtype=float)
    return np.divide(first, second, out=row, where=second > 0) * wavelength.unit


def _photon_wavelength(I_cube: NDCube, t_exp: u.Quantity, det, tel, sim,
                       offchip_bin_slit: int, uniform_mode: bool) -> u.Quantity:
    """
    The wavelength `dn_variance` takes the electrons per photon at, as the DN are binned.
    """
    quiet = dataclasses.replace(sim, noise=False)
    photons = simulate_once(I_cube, t_exp, det, tel, quiet, uniform_mode=uniform_mode)[5]
    return _variance_wavelength(photons, offchip_bin_slit)


def _weighted(fit_config) -> bool:
    """Whether the fits are weighted: by default, and unless the fitting block says not."""
    return fit_config is None or fit_config.weighted


def _visible_electrons(I_cube: NDCube, t_exp: u.Quantity, det, tel, sim,
                       offchip_bin_slit: int) -> np.ndarray:
    """The mean electrons visible light adds to each pixel, summed as the DN are binned."""
    nothing = NDCube(np.zeros(I_cube.data.shape), wcs=I_cube.wcs,
                     unit=u.electron / u.pixel, meta=I_cube.meta)
    visible = add_visible_stray_light(nothing, t_exp, det, sim, tel, noise=False)
    visible = add_pinhole_visible_light(visible, t_exp, det, sim, tel, noise=False)
    return rebin_slit_offchip(visible, offchip_bin_slit).data


def expected_dn_uncertainty(I_cube: NDCube, t_exp: u.Quantity, det, tel, sim,
                            offchip_bin_slit: int = 1, uniform_mode: bool = False) -> np.ndarray:
    """
    The uncertainty each pixel's DN has on average: that of the signal it holds with no noise.

    The measured spectra are weighted by uncertainties worked out from their
    own signal (`dn_variance`). The ground truth, fitted to the spectra with
    no noise, is weighted by these, so that it is the same fit of the same
    line, and a measurement's difference from it is the noise's doing.

    Parameters
    ----------
    I_cube : NDCube
        The spectral radiance on the detector's pixels, per second, as the
        Monte Carlo observes it.
    t_exp : u.Quantity
        The exposure time.
    det, tel, sim
        The detector, telescope and simulation settings.
    offchip_bin_slit : int, optional
        How many pixels along the slit are added together after read-out.
        Default 1.
    uniform_mode : bool, optional
        As for `simulate_once`. Default False.

    Returns
    -------
    np.ndarray
        The uncertainty in DN, shaped as the binned DN.
    """
    quiet = dataclasses.replace(sim, noise=False)
    steps = simulate_once(I_cube, t_exp, det, tel, quiet, uniform_mode=uniform_mode)
    electrons = steps[8]
    # Clipped at the digitiser's maximum in each pixel, before any are summed,
    # as to_dn clips the measured DN; not rounded, as the mean is not.
    dn = (electrons.data * electrons.unit / det.gain_e_per_dn).to_value(det.max_dn.unit)
    dn = np.minimum(dn, det.max_dn.value)
    dn = rebin_slit_offchip(NDCube(dn, wcs=electrons.wcs, meta=electrons.meta),
                            offchip_bin_slit).data
    visible = _visible_electrons(I_cube, t_exp, det, tel, sim, offchip_bin_slit)
    wavelength = _variance_wavelength(steps[5], offchip_bin_slit)
    return np.sqrt(dn_variance(dn, wavelength, t_exp, det, visible, offchip_bin_slit))


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
    Simulate one observation of a cube of spectra, with its noise, without fitting it.

    The spectra go through each step of the instrument in turn: the
    exposure, the telescope, the pixels, the PSF, the pinholes, the arrival
    of photons, the detector, the stray light and the digitiser.

    Parameters
    ----------
    I_cube : NDCube
        The spectral radiance on the detector's pixels, per second, such as
        `create_uniform_intensity_cube` makes, with the line's rest
        wavelength as ``meta["rest_wav"]``.
    t_exp : u.Quantity
        The exposure time.
    det : Detector_SWC or Detector_EIS
        The detector.
    tel : Telescope_EUVST or Telescope_EIS
        The telescope.
    sim : Simulation
        The simulation's settings.
    uniform_mode : bool, optional
        The cube is the same all along the slit, so the PSF blurs it in
        wavelength only. Default False.
    photon_shot_inverse_transform : bool, optional
        Draw the photons by inverse-transform sampling, for common random
        numbers between runs that differ in photon flux. Default False.
    dark_current_inverse_transform : bool, optional
        Draw the dark current by inverse-transform sampling, for common random
        numbers between runs that differ in dark current. Default False.

    Returns
    -------
    tuple of NDCube
        The cube after each step: the radiance over the exposure, the photons,
        the photons collected by the telescope, the photons per pixel, those
        after the PSF, the photons that arrive, the electrons, the electrons
        with the stray light, those with the pinholes' visible light, and the
        DN.
    """
    # Apply exposure time
    intensity_exp = apply_exposure(I_cube, t_exp)
    
    # Convert to total photons, each pixel's at its own wavelength unless the
    # cube gives the wavelength of the mean energy of its photons, which it
    # does once laid onto the pixels through the telescope.
    photons_total = _with_photon_wavelength(intensity_to_photons(intensity_exp))
    
    # Apply telescope optical throughput
    photons_throughput = add_telescope_throughput(photons_total, tel)
    
    # Convert to pixel counts
    photons_pixels = photons_to_pixel_counts(photons_throughput, det.wvl_res, det.plate_scale_length, angle_to_distance(sim.slit_width))

    # Apply focusing optics PSF (primary mirror + diffraction grating), unless
    # the cube was laid onto the pixels through it (rebin_atmosphere with a
    # telescope), which is exact where blurring the pixels is not.
    if sim.psf and not (I_cube.meta or {}).get("psf_applied", False):
        photons_focused = apply_focusing_optics_psf(
            photons_pixels, tel, det, sim, convolve_spatial=not uniform_mode,
            boundary=getattr(sim, "psf_boundary", "replicate"),
        )
    else:
        photons_focused = photons_pixels
    
    # Apply EUV pinhole diffraction effects (after focusing optics, if enabled)
    if sim.enable_pinholes and len(sim.pinhole_sizes) > 0:
        photons_euv_pinholes = apply_euv_pinhole_diffraction(photons_focused, det, sim, tel)
    else:
        photons_euv_pinholes = photons_focused

    # Every random draw below is controlled by this one flag, so a run with
    # sim.noise False returns the signal the instrument would measure on
    # average rather than one realisation of it.
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
    Give this MPI rank draws of its own, from the state NumPy's generator is in and the rank.

    Seeded alike on every rank, as the docs' recipe for comparing runs does,
    the ranks would repeat one another's iterations, and the spread would
    come out narrower by the square root of their number. The state each rank
    starts from is combined with its rank, so a run repeats with the same
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

    This is what `eclipse` runs for each combination of settings. Under MPI
    the iterations are shared between the ranks, and the results collected
    on the first.

    Parameters
    ----------
    I_cube : NDCube
        The spectral radiance on the detector's pixels, per second, as for
        `simulate_once`.
    t_exp : u.Quantity
        The exposure time.
    det : Detector_SWC or Detector_EIS
        The detector.
    tel : Telescope_EUVST or Telescope_EIS
        The telescope.
    sim : Simulation
        The simulation's settings.
    n_iter : int, optional
        How many iterations to run. Default 5.
    fit_config : FitConfig, optional
        How to fit. Default one Gaussian.
    offchip_bin_slit : int, optional
        How many pixels along the slit to add together after read-out, each
        with its own noise. Default 1.
    fit_signals : str, optional
        Which signals to fit: ``"both"`` (default), ``"dn"`` or ``"photon"``.
        Fitting only one takes about half the time.
    uniform_mode : bool, optional
        The cube is a single line of known intensity, as
        `create_uniform_intensity_cube` makes it, with ``offchip_bin_slit``
        pixels along the slit. Default False.
    photon_shot_inverse_transform : bool, optional
        Draw the photons by inverse-transform sampling, for common random
        numbers between runs that differ in photon flux. Default False.
    dark_current_inverse_transform : bool, optional
        Draw the dark current by inverse-transform sampling, for common random
        numbers between runs that differ in dark current. Default False.

    Returns
    -------
    first_dn_signal : NDCube
        The first iteration's signal, in DN.
    dn_fit_results : dict or None
        The fits to the DN signal, or None if they were not fitted. For each
        fitted component, by name, under ``"components"``, it holds the
        intensity, velocity and width, each as ``"first"``, ``"mean"`` and
        ``"std"`` maps. ``"primary_component"`` names the primary component,
        and ``"failed_fits"`` counts the failed fits in each pixel.
    first_photon_signal : NDCube
        The first iteration's photons arriving at the detector.
    photon_fit_results : dict or None
        The fits to the photons, as for the DN, or None.

    Under MPI, only the first rank gets these; every other rank gets four
    Nones.
    """
    if fit_signals not in ("both", "dn", "photon"):
        raise ValueError(f"fit_signals must be 'both', 'dn', or 'photon', got '{fit_signals}'")

    do_dn = fit_signals in ("both", "dn")
    do_photon = fit_signals in ("both", "photon")
    rest_wavelength = I_cube.meta["rest_wav"]

    # Each DN spectrum is weighted by its pixels' uncertainties, worked out
    # from their own signal, with the visible light's share of it known. The
    # photons, with no read noise, would give a pixel with none no
    # uncertainty; they are fitted by their Poisson likelihood instead.
    visible = (_visible_electrons(I_cube, t_exp, det, tel, sim, offchip_bin_slit)
               if do_dn and _weighted(fit_config) else None)
    # Each photon frees electrons by its own energy, so the DN of a pixel
    # vary as its photons' mean energy says.
    photon_wavelength = (None if visible is None else _photon_wavelength(
        I_cube, t_exp, det, tel, sim, offchip_bin_slit, uniform_mode))

    def _dn_uncertainty(dn_data: np.ndarray) -> np.ndarray | None:
        if visible is None:
            return None
        return np.sqrt(dn_variance(dn_data, photon_wavelength, t_exp, det, visible,
                                   offchip_bin_slit))

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
