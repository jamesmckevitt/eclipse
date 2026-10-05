# API reference

Everything on this page is imported from `euvst_response`, such as `from euvst_response import monte_carlo`, except the synthesis functions, which are in `euvst_response.synthesis` and `euvst_response.continuum`, and the lines of a band, in `euvst_response.band`. [Using ECLIPSE from Python](python.md) shows them in use.

## The instrument

::: euvst_response.Telescope_EUVST

::: euvst_response.AluminiumFilter

::: euvst_response.Detector_SWC

::: euvst_response.Telescope_EIS

::: euvst_response.Detector_EIS

::: euvst_response.Simulation

::: euvst_response.eis_effective_area

::: euvst_response.spectral_psf_fwhm

::: euvst_response.spectral_line_spread

## Running the simulation

::: euvst_response.monte_carlo

::: euvst_response.simulate_once

::: euvst_response.create_uniform_intensity_cube

::: euvst_response.load_atmosphere

::: euvst_response.main

## The steps of the simulation

These are the steps `simulate_once` takes, in order.

::: euvst_response.apply_exposure

::: euvst_response.intensity_to_photons

::: euvst_response.add_telescope_throughput

::: euvst_response.photons_to_pixel_counts

::: euvst_response.apply_focusing_optics_psf

::: euvst_response.apply_euv_pinhole_diffraction

::: euvst_response.sample_photon_arrivals

::: euvst_response.to_electrons

::: euvst_response.add_visible_stray_light

::: euvst_response.add_pinhole_visible_light

::: euvst_response.to_dn

::: euvst_response.add_poisson

::: euvst_response.airy_disk_pattern

## Fitting

::: euvst_response.FitConfig

::: euvst_response.FitComponent

::: euvst_response.fit_cube_gauss

::: euvst_response.velocity_from_fit

::: euvst_response.width_from_fit

::: euvst_response.analyse

## Analysing the results

::: euvst_response.load_instrument_response_results

::: euvst_response.summary_table

::: euvst_response.get_parameter_combinations

::: euvst_response.get_results_for_combination

::: euvst_response.list_fit_components

::: euvst_response.analyse_fit_statistics

::: euvst_response.create_sunpy_maps_from_combo

::: euvst_response.get_dem_data_from_results

## Atmosphere files

::: euvst_response.Atmosphere

::: euvst_response.write_atmosphere

::: euvst_response.read_atmosphere

::: euvst_response.describe_atmosphere_file

::: euvst_response.edges_from_centres

## Synthesis files

::: euvst_response.Synthesis

::: euvst_response.SpectralLine

::: euvst_response.write_synthesis

::: euvst_response.read_synthesis

::: euvst_response.load_synthesis

::: euvst_response.read_synthesis_products

::: euvst_response.write_line_cubes

::: euvst_response.convert_synthesis_pickle

## Synthesising spectra

::: euvst_response.synthesis.compute_goft_fiasco

::: euvst_response.synthesis.interpolate_g_on_dem

::: euvst_response.synthesis.synthesise_spectra

::: euvst_response.synthesis.create_atmosphere_ndcube

::: euvst_response.synthesis.create_line_cube

::: euvst_response.continuum.compute_continuum_fiasco

::: euvst_response.continuum.continuum_spectra

::: euvst_response.continuum.continuum_windows

## Every line in a band

A full-CCD frame holds every line in the band, not only the lines a synthesis names. These work out the contribution functions of all of them, on a synthesis's own grids, and keep them in a file.

::: euvst_response.band.band_contribution_functions

::: euvst_response.band.BandLines

::: euvst_response.band.IonLines

::: euvst_response.band.write_band_lines

::: euvst_response.band.read_band_lines

## Results files

::: euvst_response.convert_results_pickle

## Units and coordinates

::: euvst_response.wl_to_vel

::: euvst_response.vel_to_wl

::: euvst_response.angle_to_distance

::: euvst_response.distance_to_angle
