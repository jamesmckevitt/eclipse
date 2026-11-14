# Run file settings

Every setting of a run file, with its default. A value with units is a number and a unit, such as `0.4 arcsec`. A list of values sweeps a setting; see [The run file](../observe/simulate.md#the-run-file). [Instrument sources](../background/instrument-sources.md) gives where each instrument default comes from.

## Top level

| Key | What it sets | Default |
| --- | --- | --- |
| `instrument` | `SWC` or `EIS`; see [The instruments](../background/instrument-model.md#the-instruments) | `SWC` |
| `synthesis_file` | The synthesis file to observe | `./run/input/synthesised_spectra.h5` |
| `reference_line` | The line whose window is observed, named as in [Naming spectral lines](line-names.md); blends in the window are included | the file's only line, or `Fe12_195.1190` |
| `n_iter` | Monte Carlo iterations | `25` |
| `ncpu` | CPU cores; `-1` for all | `-1` |
| `offchip_bin_slit` | Pixels along the slit added after read-out; see [Off-chip slit binning](../observe/simulate.md#off-chip-slit-binning) | `1` |
| `fit_signals` | `dn`, `photon` or `both`; see [Which signals are fitted](../observe/fitting.md#which-signals-are-fitted) | `both` |
| `uniform_intensity` | A single line of this total intensity; see [From a single intensity](../emission/single-intensity.md) | none |
| `rest_wavelength` | That line's rest wavelength | `195.119 AA` |
| `thermal_width` | That line's 1-sigma width, as a velocity | `20 km/s` |
| `pinhole_sizes` | Pinhole diameters (SWC); see [Pinholes](#pinholes) | none |
| `pinhole_positions` | Each pinhole's place along the slit, from 0 to 1 | none |
| `pinhole_positions_spectral` | Each pinhole's place along the spectral axis, from 0 to 1 | the middle |
| `atmosphere_series` | A time series of atmosphere files; see [Rasters and sit-and-stares](../observe/rasters.md) | none |
| `synthesis_series` | A time series of synthesis files; see [Rasters and sit-and-stares](../observe/rasters.md#from-synthesis-files) | none |
| `atmosphere` | An atmosphere file to synthesise first; see [Run files](../run-files.md) | none |
| `runs` | Several runs from one file; see [Several runs in one file](../run-files.md#several-runs-in-one-file) | none |

## simulation

| Key | What it sets | Default |
| --- | --- | --- |
| `slit_width` | 0.2, 0.4, 0.8 or 1.6 arcsec for SWC, 1 or 2 arcsec for EIS | `0.2 arcsec` |
| `expos` | Exposure time | `1 s` |
| `vis_sl` | Visible stray light, in photons per second per cm2, before the filter for SWC and at the CCD for EIS | `0 photon / (s cm2)` |
| `psf` | Blur with the [PSF](../background/instrument-model.md#the-point-spread-function) | `False` |
| `psf_boundary` | What the PSF brings in from beyond the atmosphere: `replicate` or `zero` | `replicate` |
| `spectral_psf` | How the slit adds to the optics' blur: `quadrature` or `convolution` | `quadrature` |
| `slit_mask` | Block the light at the [slit's ends and fiducials](../background/instrument-model.md#the-slits-fiducials) (SWC) | `False` |
| `focal_plane_dispersion` | Take [each pixel's width in wavelength](../background/instrument-model.md#the-pixels-width-in-wavelength) from the focal plane (SWC) | `False` |
| `noise` | `False` replaces every random draw by its mean; see [Turning the noise off](../observe/simulate.md#turning-the-noise-off) | `True` |
| `readout_dark_current` | Also collect [dark current](../background/instrument-model.md#dark-current) while the CCD is cleared and read (SWC) | `True` |
| `enable_pinholes` | Model the pinholes (SWC) | `False` |
| `pinhole_visible_wavelength` | Wavelength of the visible light through the pinholes (SWC) | `600 nm` |

## detector

| Key | What it sets | SWC | EIS |
| --- | --- | --- | --- |
| `ccd_temperature` | CCD temperature, for the dark current; from 198 K to 300 K | `-60 Celsius` | `-60 Celsius` |
| `dark_current_293k` | Dark current at 293 K | `20000 electron / (pix s)` | `250 electron / (pix s)` |
| `qe_euv` | Quantum efficiency for EUV | `0.76` | `0.64` |
| `qe_vis` | Quantum efficiency for visible light | `1.0` | `0.65` |
| `read_noise_rms` | Read noise | `10 electron / pix` | `5 electron / pix` |
| `gain_e_per_dn` | Electrons per DN | `2.78 electron / DN` | `6.3 electron / DN` |
| `max_dn` | Where DN are clipped | `65535 DN / pix` | `65535 DN / pix` |
| `full_well` | Full well, for comparison only; nothing is clipped at it | `150000 electron / pix` | - |
| `pix_size` | Pixel size | `13.5 um / pix` | `13.5 um / pix` |
| `wvl_res` | Wavelength step per pixel | `16.9 mAA / pix` | `22.3 mAA / pix` |
| `plate_scale_angle` | Angle per pixel along the slit | `0.159 arcsec / pix` | `1 arcsec / pix` |
| `material` | Only `silicon` | `silicon` | `silicon` |
| `fano_factor` | Fano factor | `0.115` | `0.115` |
| `pair_energy_300k` | Energy per electron-hole pair at 300 K | `3.71 eV` | `3.71 eV` |
| `pair_energy_slope` | Change in that energy per kelvin | `-0.0006 eV / K` | `-0.0006 eV / K` |
| `filter_distance` | Filter to detector, for the pinholes' diffraction | `250 mm` | - |
| `shutter` | Frames taken with the shutter; `false` is refused | `true` | - |
| `n_rows`, `n_columns` | Pixels of each CCD along the dispersion and along the slit | `2048` | - |
| `ccd_gap` | Space between the two CCDs | `1 mm` | - |
| `row_transfer_time` | Time per row transfer | `15 us` | - |
| `pixel_period` | Time per pixel along the serial register | `500 ns` | - |
| `serial_prescan`, `serial_overscan` | Samples read before and after a row's image pixels | `50`, `20` | - |
| `parallel_overscan_rows` | Rows read after the last image row | `20` | - |
| `cte_parallel`, `cte_serial` | Charge transfer efficiency; a run refuses values below 1 | `1` | - |
| `dispersion` | Wavelength in Angstrom against mm from the middle of the CCD gap, as polynomial coefficients | `[198.885944, 1.2516577, -1.361114e-4]` | - |
| `wavelength_offset` | Shift added to every row's wavelength | `0 Angstrom` | - |
| `lit_band` | Wavelengths that reach the CCDs | `[170.0 Angstrom, 212.3 Angstrom]` | - |
| `slit_image_tilt` | Rows a line moves between the slit's middle and 140 arcsec from it: linear and quadratic terms | `[0, 0]` | - |

A run takes each pixel's wavelength from `wvl_res`, and refuses focal plane settings other than the defaults. [Full-CCD frames](../observe/full-ccd-frames.md) use the focal plane, but not `slit_image_tilt`.

## telescope

| Key | What it sets | SWC | EIS |
| --- | --- | --- | --- |
| `D_ap` | Aperture diameter; half its area feeds SW | `0.28 m` | - |
| `microroughness_sigma` | RMS roughness of the primary mirror | `0.3 nm` | - |
| `psf_type` | Only `gaussian` | `gaussian` | `gaussian` |
| `psf_params` | PSF FWHMs in pixels, along the slit and in wavelength | from `field_of_view` | `[3 pix, 3 pix]` |
| `field_of_view` | `280x280`, `100x100` or `70x70` arcsec; sets `psf_params`; see [The SW PSF along the slit](../background/instrument-model.md#the-sw-psf-along-the-slit) | `280x280` | - |
| `resolution_budget` | `current_estimate`, `allocation` or `none` | `current_estimate` | - |
| `pointing_jitter` | Pointing jitter FWHM, in place of the budget's | `0.110 arcsec` | - |
| `psf_slit_width` | Slit the wavelength FWHM was measured with | `0.2 arcsec` | none |
| `psf_across_slit` | FWHM of the blur across the slit | none | none |
| `slit_length` | Slit length | `280 arcsec` | - |
| `fiducial_centres` | Middle of each fiducial along the slit, positive towards solar north | `[-70 arcsec, 25 arcsec]` | - |
| `fiducial_lengths` | Fiducial lengths for the 0.2, 0.4, 0.8 and 1.6 arcsec slits | `[0.8, 0.8, 0.8, 1.6] arcsec` | - |
| `pm_table`, `grating_table` | Primary mirror reflectance and grating efficiency tables | packaged | - |
| `calibration` | `ground`, `dz2013`, `warren2014` or `dz2025`; see [EIS effective area](../background/instrument-model.md#eis-effective-area) | - | `ground` |
| `date` | Date of the observation, for in-flight calibrations | - | none |

`psf_params`, `psf_slit_width`, `fiducial_centres` and `fiducial_lengths` each take one value, not a sweep. A table has two columns, wavelength in nm and throughput from 0 to 1, and lines starting with `#` are skipped.

## filter

SWC's filter; ignored, with a warning, for EIS.

| Key | What it sets | Default |
| --- | --- | --- |
| `al_thickness` | Aluminium thickness | `1485 angstrom` |
| `oxide_thickness` | Aluminium oxide thickness | `95 angstrom` |
| `c_thickness` | Carbon thickness | `0 angstrom` |
| `mesh_throughput` | Fraction the mesh lets through | `0.8` |
| `al_table`, `oxide_table`, `c_table` | Transmission tables of a layer `table_thickness` thick | packaged |
| `table_thickness` | Thickness the tables are for | `1000 angstrom` |
| `visible_decade_thickness` | Aluminium that cuts visible light tenfold | `170 angstrom` |

## fitting

| Key | What it sets | Default |
| --- | --- | --- |
| `components` | The Gaussians to fit; see [Fitting the lines](../observe/fitting.md) | one Gaussian |
| `primary_component` | The component whose velocity is reported | `0` |
| `constrain_positive_intensity` | Keep amplitudes at zero or above | `False` |
| `backend` | `scipy` or `mpfit` | `scipy` |
| `max_iter` | Optimiser iterations per spectrum | `1000` |
| `bessel_correction` | Use n - 1 in standard deviations | `False` |
| `save_iterations` | Keep every iteration's fit | `False` |
| `weighted` | Weight DN fits by each pixel's uncertainty, and fit photons by their Poisson likelihood | `True` |
| `start` | `search` or `rest`; see [A faint blend](../observe/fitting.md#a-faint-blend) | `search` |

Each component also takes `velocity_limits` and `width_limits`, the lowest and highest velocity and Gaussian sigma the fit may give it.

## Pinholes

`pinhole_sizes` and `pinhole_positions` have one entry per pinhole, as does `pinhole_positions_spectral` if given. `simulation.enable_pinholes` switches them on. They model defects in SWC's filter, for filter engineering.

## atmosphere and synthesis { data-routes="mhd dem" }

`atmosphere` names the [atmosphere file](../files/atmosphere.md) to synthesise. The `synthesis:` section takes the [options of `synthesise-spectra`](../emission/mhd.md#command-line-options), with underscores in place of dashes. Two values go in a list, and an on-or-off option takes `True` or `False`.

## raster { data-routes="mhd spectra" }

[The observing plan](../observe/rasters.md#the-observing-plan) lists its keys.

## frame { data-routes="spectra mhd dem" }

[A full-CCD frame](../run-files.md#a-full-ccd-frame) lists its keys.

## runs

[Several runs in one file](../run-files.md#several-runs-in-one-file) explains them.
