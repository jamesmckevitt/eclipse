# Frame files

`write_full_frame` writes a [full-CCD frame](../observe/full-ccd-frames.md). Its root has a `format` attribute of `eclipse-full-ccd-frame` and these attributes:

| Attribute | Value |
| --- | --- |
| `exposure`, `slit_position`, `slit_width` | Each with its unit in `exposure_unit`, `slit_position_unit` and `slit_width_unit` |
| `psf`, `spectral_psf`, `noise`, `psf_boundary`, `slit_mask`, `continuum` | The options the frame was made with |
| `serial_prescan`, `serial_overscan` | The prescan and overscan samples each output reads in a row |
| `parallel_overscan_rows` | The overscan rows read after the image |
| `software_version`, `git_commit_id` | The version of ECLIPSE and the git commit that wrote the file |

The root's `spectra` dataset names the entries of the synthesis file that the frame holds.

A group for each CCD, `left` and `right`, holds these datasets, each with a `unit` attribute:

| Dataset | Shape | What it holds |
| --- | --- | --- |
| `dn` | `(n_rows + parallel_overscan_rows, n_columns + 2 * (serial_prescan + serial_overscan))` | The frame as read out, in DN |
| `photons` | `(n_rows + parallel_overscan_rows, n_columns)` | The photons in each pixel over the exposure |
| `rate` | `(n_rows, n_columns)` | The photons per second reaching each image pixel |
| `wavelength` | `(n_rows,)` | The wavelength each row records |
| `field_angle` | `(n_columns,)` | The angle along the slit of each column, in arcsec from the slit's centre, positive towards solar north in nominal observations |
| `photon_wavelength`, `photon_rms_wavelength` | `(n_rows + parallel_overscan_rows, n_columns)` | The wavelengths of the mean and root mean square energies of each pixel's photons |

Each row of `dn` is, in order, the left output's prescan, the first half of the image, the left output's overscan, the right output's overscan, the second half of the image and the right output's prescan. The columns count from the left output, so on the left CCD they run from the north end of the slit, and on the right CCD from the south end.
