# Where the instrument settings come from

ECLIPSE's default instrument settings come from SOLAR-C design documents and EIS calibrations. This page lists each setting with its default and the document it comes from, so a result can be traced to the design it reflects. The settings themselves are described in the [configuration reference](configuration.md).

Documents are named as the project names them: RSC numbers are SOLAR-C documents, SOLC-EUVST-MSSL numbers are documents for the SW camera, and TASI-SCE-SSS-0001 is the requirements for the slit assembly. RSC-2022021C is the optical design summary of March 2025, which describes optical design version 20230909. Where no source is recorded, the page says so. Where a design document gives a different value, the page gives that value too.

## EUVST telescope and optics

| Setting | Default | Source |
| --- | --- | --- |
| `telescope.D_ap` | 0.28 m | RSC-2022021C, Optical Design Summary: entrance pupil 280 mm |
| Half the aperture feeding the SW channel | 0.5 | Not recorded |
| `telescope.microroughness_sigma` | 0.3 nm | Not recorded |
| `telescope.pm_table` | packaged table | RSC-2022035A, Sensitivity Budget (13 Dec 2024) |
| `telescope.grating_table` | packaged table | RSC-2022035A, Sensitivity Budget (13 Dec 2024) |
| `telescope.psf_params`, through `telescope.field_of_view` | 0.423 arcsec along the slit for the default 280 x 280 arcsec field (0.195 and 0.161 for the smaller fields); 43.00 mA in wavelength with the 0.2 arcsec slit | RSC-2022021C |
| `telescope.psf_slit_width` | 0.2 arcsec, the slit the spectral PSF above is given for | RSC-2022021C, Appendix A.2 |
| `telescope.psf_across_slit` | none: across the slit only the budget terms below | Not recorded |
| `telescope.resolution_budget` | `current_estimate` | RSC-2022033B rev. B (28 Oct 2025), Tables 3-1 and 4-1 |
| `telescope.pointing_jitter` | none: the budget's 0.110 arcsec, or no jitter with `resolution_budget: none` | RSC-2022033B rev. B, Table 3-1 row 20 |
| `simulation.slit_width` | 0.2 arcsec; the slits are 0.2, 0.4, 0.8 and 1.6 arcsec wide | RSC-2022021C |
| `telescope.slit_length` | 300 arcsec | TASI-SCE-SSS-0001 issue 1, R-SA-SYS-REQ-1160 and 1170 |
| `telescope.fiducial_centres`, `telescope.fiducial_lengths` | -25 and +70 arcsec; 0.8 arcsec, 1.6 arcsec for the widest slit | TASI-SCE-SSS-0001 issue 1, R-SA-SYS-REQ-1160 and 1170 |

## EUVST filter

| Setting | Default | Source |
| --- | --- | --- |
| `filter.al_thickness`, `filter.oxide_thickness`, `filter.c_thickness` | 1485, 95 and 0 Angstrom | Not recorded. SOLC-EUVST-MSSL-RP-0007 v1.2, Table 6-2, gives a 1500 Angstrom aluminium foil |
| `filter.mesh_throughput` | 0.8 | SOLC-EUVST-MSSL-RP-0007 v1.2, Table 6-2: 80% transmission, to be confirmed. Its effective area in Sec. 7.1.3 uses 0.81 |
| `filter.al_table`, `filter.oxide_table`, `filter.c_table`, `filter.table_thickness` | packaged tables for 1000 Angstrom layers | Henke filter transmission calculator (henke.lbl.gov) |

## EUVST-SW detector

| Setting | Default | Source |
| --- | --- | --- |
| `detector.pix_size`, `detector.n_rows`, `detector.n_columns` | 13.5 micron, 2048 by 2048 | SOLC-EUVST-MSSL-SP-0001 v1.4 |
| `detector.plate_scale_angle` | 0.159 arcsec per pixel | RSC-2022021C: 0.0118 arcsec per micron |
| `detector.wvl_res` | 16.9 mA per pixel | RSC-2022021C: 1.253 mA per micron at the chief ray |
| `detector.ccd_gap` | 1 mm | SOLC-EUVST-MSSL-RP-0007 v1.2, Sec. 7.1; RSC-2022021C |
| `detector.ccd_temperature` | -60 Celsius | Not recorded. SOLC-EUVST-MSSL-RP-0007 v1.2, Sec. 5.5.2.1, gives a goal of -50 to -60 Celsius |
| `detector.qe_euv`, `detector.qe_vis` | 0.76 and 1.0 | Not recorded. SOLC-EUVST-MSSL-RP-0007 v1.2, Table 6-1, expects an EUV QE of 0.75; SOLC-EUVST-MSSL-SP-0001 v1.4 CCD-4.5.3 requires more than 0.71 |
| `detector.read_noise_rms` | 10 electrons | Not recorded. SOLC-EUVST-MSSL-SP-0001 v1.4 CCD-4.7.1 requires 6 electrons rms; SOLC-EUVST-MSSL-RP-0007 v1.2, Table 6-1, predicts 11.0 electrons at most |
| `detector.gain_e_per_dn` | 2.78 electrons per DN | MSSL engineering model test results; SOLC-EUVST-MSSL-RP-0012 v2.0, Table 5-5, gives the same design value |
| Dark current at 293 K | 20000 electrons per pixel per second | SOLC-EUVST-MSSL-SP-0001 v1.4 CCD-4.7.3 (requirement) |
| Dark current against temperature | the 293 K value times 122 T^3 exp(-6400/T), held at its 198 K value below 198 K | Not recorded for the law; MSSL test results for holding it below 198 K |
| `detector.full_well` | 150000 electrons | Teledyne e2v CCD42-40 BSI datasheet, 1B300000-A1A version 1 (January 2024); SOLC-EUVST-MSSL-SP-0001 v1.4 CCD-4.6.1 requires more than 100000, typically 150000. SOLC-EUVST-MSSL-RP-0007 v1.2, Table 6-1, gives 75000 typical for the image area |
| `detector.max_dn` | 65535 DN | SOLC-EUVST-MSSL-RP-0012 v2.0, Table 5-5: a 16-bit converter |
| `detector.shutter` | true | SOLC-EUVST-MSSL-RP-0007 v1.2, Sec. 6.1.2: the SW CCD uses a mechanical shutter |
| `detector.filter_distance` | 250 mm | Not recorded |
| `detector.row_transfer_time` | 15 us | SWC modes table; it gives the read-out times of SOLC-EUVST-MSSL-RP-0007 v1.2, Table 5-11 |
| `detector.pixel_period` | 500 ns | SOLC-EUVST-MSSL-RS-0002 v2.0 |
| `detector.serial_prescan`, `detector.serial_overscan` | 50 and 20; each settable from 0 to 200 | SOLC-EUVST-MSSL-RS-0002 v2.0 for the range. The read-out times of SOLC-EUVST-MSSL-RP-0007 v1.2, Table 5-11, follow from 50 prescan and 16 overscan |
| `detector.parallel_overscan_rows` | 20 | Not recorded |
| `detector.cte_parallel`, `detector.cte_serial` | 1, a perfect transfer | Not recorded. SOLC-EUVST-MSSL-SP-0001 v1.4 CCD-4.5.1 requires more than 0.999993; SOLC-EUVST-MSSL-RP-0007 v1.2, Table 6-1, gives 0.99999 as typical |

## EUVST-SW focal plane

These describe the focal plane in `FocalPlane_SWC`, which the [full-CCD frame](full-ccd-frame.md) uses. They are not yet settings in the configuration, and the frame does not yet take the slit image tilt.

| Quantity | Value | Source |
| --- | --- | --- |
| Wavelength of each row | quadratic in position along the dispersion | Fitted to RSC-2022021C: the detector edges, the focal plane centre, the chief ray and the ray-traced line positions |
| Band reaching the detectors | 170.0 to 212.3 Angstrom | The required range, RSC-2022021C REQ-103 and SOLC-EUVST-MSSL-RP-0007 v1.2, Table 7-2. No document gives the baffle's edges, so where the band stops to within a row or two is a modelling choice |
| Alignment of the camera to the beam | +/-0.79 Angstrom; no offset by default | SOLC-EUVST-MSSL-RP-0007 v1.2, page 45 |
| Slit image tilt, off unless asked for | a drift along the dispersion of 0.991 f + 0.605 f^2 rows, where f is the angle along the slit over 140 arcsec | No document gives it. Measured from the ray-traced spots in the SW footprint figure of RSC-2022021C, where a row is about a quarter of a pixel of the figure |
| Clear and read-out sequence | dump, exposure, row-by-row read with windows | SOLC-EUVST-MSSL-SP-0001 v1.4, SOLC-EUVST-MSSL-RS-0002 v2.0, SOLC-EUVST-MSSL-ICD-0003 v3.1 |

## Constants and assumptions in the code

These are set in the code rather than in the configuration.

| Quantity | Value | Source |
| --- | --- | --- |
| Fano factor of silicon | 0.115 | Not recorded |
| Energy to make one electron-hole pair in silicon | 3.71 - 0.0006 (T - 300 K) eV | Not recorded |
| Visible light through the aluminium filter | a factor 10 less per 170 Angstrom of aluminium, times the mesh | Not recorded |
| Wavelength of the visible light through a pinhole | 600 nm | Not recorded |

## Hinode/EIS

| Setting | Default | Source |
| --- | --- | --- |
| `telescope.calibration` | `ground` | Pre-flight MSSL calibration, April 2007 (Lang et al. 2006, Applied Optics, 45, 8689); `dz2013`, `warren2014` and `dz2025` from Del Zanna (2013), Warren, Ugarte-Urra & Landi (2014) and Del Zanna et al. (2025) |
| `telescope.psf_params` | 3 pixels FWHM | Ugarte-Urra (2016), EIS Software Note 2 |
| `detector.qe_euv` | 0.64 | EIS Software Note 2 |
| `detector.qe_vis` | 0.65 | MSSL engineering test report |
| Dark current against temperature | the 293 K value times 1.14e6 T^3 exp(-9080/T), held at its 198 K value below 198 K | MSSL test results |
| `detector.read_noise_rms`, `detector.gain_e_per_dn`, dark current at 293 K | 5 electrons, 6.3 electrons per DN, 250 electrons per pixel per second | Not recorded |
| `detector.wvl_res`, `detector.plate_scale_angle` | 22.3 mA per pixel, 1 arcsec per pixel | Not recorded |
| `detector.ccd_temperature`, `detector.pix_size`, `detector.max_dn` | -60 Celsius, 13.5 micron, 65535 DN | Not recorded |
