# Where the instrument settings come from

<!-- Each setting line below is filled in with the defaults and sources in euvst_response/config.py by hooks/instrument_sources.py when the site is built. -->

ECLIPSE's default instrument settings come from SOLAR-C design documents and EIS calibrations. This page lists each setting with its default and the document it comes from, so a result can be traced to the design it reflects. The settings themselves are described in the [configuration reference](configuration.md).

Documents are named as the project names them: RSC numbers are SOLAR-C documents, SOLC-EUVST-MSSL numbers are documents for the SW camera, and TASI-SCE-SSS-0001 is the requirements for the slit assembly. RSC-2022021C is the optical design summary of March 2025, which describes optical design version 20230909. Where no source is recorded, the page says so. Where a design document gives a different value, the page gives that value too.

## EUVST telescope and optics

| Setting | Default | Source |
| --- | --- | --- |
<!-- setting SWC: telescope.D_ap -->
| Half the aperture feeding the SW channel, set in the code | 0.5 | Not recorded |
<!-- setting SWC: telescope.microroughness_sigma -->
<!-- setting SWC: telescope.pm_table -->
<!-- setting SWC: telescope.grating_table -->
<!-- setting SWC: telescope.psf_params, telescope.field_of_view -->
<!-- setting SWC: telescope.psf_slit_width -->
<!-- setting SWC: telescope.psf_across_slit -->
<!-- setting SWC: telescope.resolution_budget -->
<!-- setting SWC: telescope.pointing_jitter -->
<!-- setting SWC: simulation.slit_width -->
<!-- setting SWC: telescope.slit_length -->
<!-- setting SWC: telescope.fiducial_centres, telescope.fiducial_lengths -->

## EUVST filter

| Setting | Default | Source |
| --- | --- | --- |
<!-- setting SWC: filter.al_thickness, filter.oxide_thickness, filter.c_thickness -->
<!-- setting SWC: filter.mesh_throughput -->
<!-- setting SWC: filter.al_table, filter.oxide_table, filter.c_table -->
<!-- setting SWC: filter.table_thickness -->

## EUVST-SW detector

| Setting | Default | Source |
| --- | --- | --- |
<!-- setting SWC: detector.pix_size, detector.n_rows, detector.n_columns -->
<!-- setting SWC: detector.plate_scale_angle -->
<!-- setting SWC: detector.wvl_res -->
<!-- setting SWC: detector.ccd_gap -->
<!-- setting SWC: detector.ccd_temperature -->
<!-- setting SWC: detector.qe_euv, detector.qe_vis -->
<!-- setting SWC: detector.read_noise_rms -->
<!-- setting SWC: detector.gain_e_per_dn -->
<!-- setting SWC: detector.dark_current_293k -->
| Dark current against temperature, set in the code | the 293 K value times 122 T^3 exp(-6400/T), held at its 198 K value below 198 K | Not recorded for the law; MSSL test results for holding it below 198 K |
<!-- setting SWC: detector.full_well -->
<!-- setting SWC: detector.max_dn -->
<!-- setting SWC: detector.shutter -->
<!-- setting SWC: detector.filter_distance -->
<!-- setting SWC: detector.row_transfer_time -->
<!-- setting SWC: detector.pixel_period -->
<!-- setting SWC: detector.serial_prescan, detector.serial_overscan -->
<!-- setting SWC: detector.parallel_overscan_rows -->
<!-- setting SWC: detector.cte_parallel, detector.cte_serial -->

## EUVST-SW focal plane

These describe the focal plane, which the [full-CCD frame](full-ccd-frame.md) uses. A run does not use them yet, and the frame does not yet take the slit image tilt.

| Setting | Default | Source |
| --- | --- | --- |
<!-- setting SWC: detector.dispersion -->
<!-- setting SWC: detector.lit_band -->
<!-- setting SWC: detector.wavelength_offset -->
<!-- setting SWC: detector.slit_image_tilt -->
| Clear and read-out sequence, set in the code | dump, exposure, row-by-row read with windows | SOLC-EUVST-MSSL-SP-0001 v1.4, SOLC-EUVST-MSSL-RS-0002 v2.0, SOLC-EUVST-MSSL-ICD-0003 v3.1 |

## Detector material and visible light

| Setting | Default | Source |
| --- | --- | --- |
<!-- setting SWC: detector.fano_factor -->
<!-- setting SWC: detector.pair_energy_300k, detector.pair_energy_slope -->
<!-- setting SWC: filter.visible_decade_thickness -->
<!-- setting SWC: simulation.pinhole_visible_wavelength -->

## Hinode/EIS

| Setting | Default | Source |
| --- | --- | --- |
<!-- setting EIS: telescope.calibration -->
<!-- setting EIS: telescope.psf_params -->
<!-- setting EIS: detector.qe_euv -->
<!-- setting EIS: detector.qe_vis -->
| Dark current against temperature, set in the code | the 293 K value times 1.14e6 T^3 exp(-9080/T), held at its 198 K value below 198 K | MSSL test results |
<!-- setting EIS: detector.read_noise_rms, detector.gain_e_per_dn, detector.dark_current_293k -->
<!-- setting EIS: detector.wvl_res, detector.plate_scale_angle -->
<!-- setting EIS: detector.ccd_temperature, detector.pix_size, detector.max_dn -->
