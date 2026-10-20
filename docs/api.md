# API reference

Everything on this page is imported from `eclipse`, such as `from eclipse import monte_carlo`, except the synthesis functions, which are in `eclipse.synthesis` and `eclipse.continuum`. [Using ECLIPSE from Python](python.md) shows them in use.

## The instrument

::: eclipse.Telescope_EUVST

::: eclipse.AluminiumFilter

::: eclipse.Detector_SWC

::: eclipse.Telescope_EIS

::: eclipse.Detector_EIS

::: eclipse.Simulation

::: eclipse.eis_effective_area

::: eclipse.spatial_psf_fwhm

::: eclipse.across_slit_fwhm

::: eclipse.spectral_psf_fwhm

::: eclipse.spectral_line_spread

## Running the simulation

::: eclipse.monte_carlo

::: eclipse.simulate_once

::: eclipse.create_uniform_intensity_cube

::: eclipse.load_atmosphere

::: eclipse.main

## The steps of the simulation

These are the steps `simulate_once` takes, in order.

::: eclipse.apply_exposure

::: eclipse.intensity_to_photons

::: eclipse.add_telescope_throughput

::: eclipse.photons_to_pixel_counts

::: eclipse.apply_focusing_optics_psf

::: eclipse.apply_euv_pinhole_diffraction

::: eclipse.sample_photon_arrivals

::: eclipse.to_electrons

::: eclipse.add_visible_stray_light

::: eclipse.add_pinhole_visible_light

::: eclipse.to_dn

::: eclipse.add_poisson

::: eclipse.airy_disk_pattern

## Fitting

::: eclipse.FitConfig

::: eclipse.FitComponent

::: eclipse.fit_cube_gauss

::: eclipse.velocity_from_fit

::: eclipse.width_from_fit

::: eclipse.analyse

## Analysing the results

::: eclipse.load_instrument_response_results

::: eclipse.summary_table

::: eclipse.get_parameter_combinations

::: eclipse.get_results_for_combination

::: eclipse.list_fit_components

::: eclipse.analyse_fit_statistics

::: eclipse.create_sunpy_maps_from_combo

## Atmosphere files

::: eclipse.Atmosphere

::: eclipse.write_atmosphere

::: eclipse.read_atmosphere

::: eclipse.describe_atmosphere_file

::: eclipse.edges_from_centres

## Synthesis files

::: eclipse.Synthesis

::: eclipse.SpectralLine

::: eclipse.write_synthesis

::: eclipse.read_synthesis

::: eclipse.load_synthesis

::: eclipse.read_synthesis_products

::: eclipse.write_line_cubes

::: eclipse.convert_synthesis_pickle

## Synthesising spectra

::: eclipse.synthesis.compute_goft_fiasco

::: eclipse.synthesis.interpolate_g_on_dem

::: eclipse.synthesis.synthesise_spectra

::: eclipse.synthesis.create_atmosphere_ndcube

::: eclipse.synthesis.create_line_cube

::: eclipse.continuum.compute_continuum_fiasco

::: eclipse.continuum.continuum_spectra

::: eclipse.continuum.continuum_windows

## Every line in a band

A full-CCD frame holds every line in the band, not only the lines a synthesis names. These work out the contribution functions of all of them, on a synthesis's own grids, and keep them in a file.

::: eclipse.band_contribution_functions

::: eclipse.BandLines

::: eclipse.IonLines

::: eclipse.write_band_lines

::: eclipse.read_band_lines

## Full-CCD frames

A frame of both SW CCDs at one slit position, with every line in the band and the continuum. `FocalPlane_SWC` gives the wavelength each row records, and `ReadoutSequence` how a frame is read out. `full_ccd_frame` takes the focal plane from the detector by default, and always takes the read-out from the detector. [A full-CCD frame](full-ccd-frame.md) shows how to make a frame.

::: eclipse.full_ccd_frame

::: eclipse.FullFrame

::: eclipse.write_full_frame

::: eclipse.FocalPlane_SWC

::: eclipse.ReadoutSequence

## Results files

::: eclipse.convert_results_pickle

## Units and coordinates

::: eclipse.wl_to_vel

::: eclipse.vel_to_wl

::: eclipse.angle_to_distance

::: eclipse.distance_to_angle

## Deprecated

These warn when called and will be removed in a future release. Each can still be imported from where it was before.

::: eclipse.frame.apply_spectral_psf

::: eclipse.get_dem_data_from_results
