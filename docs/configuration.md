# Configuration reference

This page lists every setting of `eclipse`'s configuration file, with its default. [Running a simulation](instrument-response.md#configuration-file) explains how the file works: its sections, and how giving a list of values sweeps a setting.

A value with units is written as a number and a unit, such as `0.4 arcsec`, `-60 Celsius` or `5000 erg / (s cm2 sr)`. Any unit of the right kind will do.

## Top level

| Key | What it sets | Default |
| --- | --- | --- |
| `instrument` | The instrument, `SWC` or `EIS`; see [Instruments](instruments.md) | `SWC` |
| `synthesis_file` | The synthesis file to observe | `./run/input/synthesised_spectra.h5` |
| `reference_line` | The line whose spectral window is observed, named as in [Naming spectral lines](line-names.md). Every line in the file that falls in that window is added in, so blends are included. To observe several windows, run once for each. | The file's only line. A file with several must hold `Fe12_195.1190`, or say which to observe. |
| `n_iter` | How many Monte Carlo iterations to run | `25` |
| `ncpu` | How many CPU cores to use; `-1` for all of them | `-1` |
| `offchip_bin_slit` | How many pixels along the slit to add together after read-out; see [Off-chip slit binning](instrument-response.md#off-chip-slit-binning) | `1` |
| `fit_signals` | Which signals to fit: `dn`, `photon` or `both`; see [Which signals are fitted](fitting.md#which-signals-are-fitted) | `both` |
| `uniform_intensity` | A single line of this total intensity, in place of a synthesis file; see [A single line of known intensity](uniform-intensity.md) | none |
| `rest_wavelength` | That line's rest wavelength | `195.119 AA` |
| `thermal_width` | That line's width, as a 1-sigma velocity | `20 km/s` |
| `pinhole_sizes` | The diameters of pinholes in the filter, one per pinhole (SWC only); see [Pinholes](#pinholes) | none |
| `pinhole_positions` | Where each pinhole is along the slit, as a fraction from 0 to 1 | none |
| `pinhole_positions_spectral` | Where each pinhole is along the spectral axis, as a fraction from 0 to 1 | the middle |
| `atmosphere_series` | A time series of atmosphere files, in place of a synthesis file; see [Time series](time-series.md) | none |
| `synthesis_series` | A time series of synthesis files, in place of a synthesis file; see [Time series](time-series.md#from-synthesis-files) | none |

The sections `simulation`, `detector`, `telescope`, `filter` and `fitting` are described below. A time series also has a `raster` section, the observing plan, and for atmosphere files a `synthesis` section; [Time series](time-series.md) lists their settings.

## simulation

| Key | What it sets | Default |
| --- | --- | --- |
| `slit_width` | The slit: 0.2, 0.4, 0.8 or 1.6 arcsec for SWC, 1 or 2 arcsec for EIS | `0.2 arcsec`, so EIS needs one |
| `expos` | The exposure time | `1 s` |
| `vis_sl` | Visible stray light, in photons per second per cm2: before the filter for SWC, and at the CCD for EIS | `0 photon / (s cm2)` |
| `psf` | Whether to blur the spectra with the PSF; see [The point spread function](instrument-response.md#the-point-spread-function) | `False` |
| `psf_boundary` | What the PSF brings in from beyond the edges of the atmosphere: `replicate` or `zero` | `replicate` |
| `spectral_psf` | How the slit is added to the optics' blur: `quadrature` or `convolution` | `quadrature` |
| `noise` | With `False`, every random draw in the detector is replaced by its mean; see [Turning the noise off](instrument-response.md#turning-the-noise-off) | `True` |
| `enable_pinholes` | Whether to model the pinholes (SWC only) | `False` |

## detector

| Key | What it sets | SWC | EIS |
| --- | --- | --- | --- |
| `ccd_temperature` | The CCD's temperature, from which ECLIPSE works out the dark current. Below 198 K (-75.15 Celsius) the dark current is taken as it is at 198 K, and above 300 K (26.85 Celsius) the temperature is refused. | `-60 Celsius` | `-60 Celsius` |
| `qe_euv` | The quantum efficiency for EUV: the fraction of photons detected | `0.76` | `0.64` |
| `qe_vis` | The quantum efficiency for the visible stray light | `1.0` | `0.65` |
| `read_noise_rms` | The read noise | `10 electron / pix` | `5 electron / pix` |
| `gain_e_per_dn` | Electrons per DN | `2.78 electron / DN` | `6.3 electron / DN` |
| `max_dn` | The digitiser's maximum, where DN are clipped | `65535 DN / pix` | `65535 DN / pix` |
| `full_well` | The CCD's full well. Nothing is clipped at it: it is there to compare with, to see which pixels would saturate. | `150000 electron / pix` | - |
| `pix_size` | The size of a pixel | `13.5 um / pix` | `13.5 um / pix` |
| `wvl_res` | The wavelength step from one pixel to the next | `16.9 mAA / pix` | `22.3 mAA / pix` |
| `plate_scale_angle` | The angle on the sky of one pixel along the slit | `0.159 arcsec / pix` | `1 arcsec / pix` |
| `material` | The detector's material; only `silicon` is modelled | `silicon` | `silicon` |
| `filter_distance` | The distance from the filter to the detector, for the pinholes' diffraction | `250 mm` | - |
| `shutter` | Whether frames are taken with the mechanical shutter, which keeps the CCDs dark while a frame is cleared and read. A run models frames taken with it, so `false` is refused. | `true` | - |
| `n_rows` | The pixels of each CCD along the dispersion | `2048` | - |
| `n_columns` | The pixels of each CCD along the slit | `2048` | - |
| `ccd_gap` | The space between the two CCDs' imaging areas | `1 mm` | - |
| `row_transfer_time` | The time to move every row of a CCD one step towards its serial register | `15 us` | - |
| `pixel_period` | The time to move one pixel along the serial register and digitise it | `500 ns` | - |
| `serial_prescan` | The samples each output digitises before the image pixels of a row it reads | `50` | - |
| `serial_overscan` | The samples each output digitises after them | `20` | - |
| `parallel_overscan_rows` | The rows clocked and read after the last image row | `20` | - |

## telescope

| Key | What it sets | SWC | EIS |
| --- | --- | --- | --- |
| `D_ap` | The aperture's diameter. Half of its area feeds the SW channel. | `0.28 m` | - |
| `microroughness_sigma` | The RMS roughness of the primary mirror | `0.3 nm` | - |
| `psf_type` | The shape of the PSF; only `gaussian` is modelled | `gaussian` | `gaussian` |
| `psf_params` | The PSF's FWHMs in pixels, along the slit and in wavelength. This is one value, not a sweep. | `[2.66 pix, 2.54 pix]` | `[3 pix, 3 pix]` |
| `psf_slit_width` | The slit the FWHM in wavelength was measured with. The FWHM for other slits is worked out from it. Without it, the FWHM is the same for every slit. This is one value, not a sweep. | `0.2 arcsec` | none |
| `psf_across_slit` | The FWHM of the telescope's blur across the slit, which brings in light from either side of it. Without it there is none. | none | none |
| `pm_table` | A table of the primary mirror's reflectance | packaged | - |
| `grating_table` | A table of the grating's efficiency | packaged | - |
| `calibration` | The EIS effective area: `ground`, `dz2013`, `warren2014` or `dz2025`; see [EIS effective area](instruments.md#eis-effective-area) | - | `ground` |
| `date` | The date of the observation, for the in-flight EIS calibrations, such as `"2012-06-03"` | - | none |

A table has two columns: the wavelength in nm, and the throughput, from 0 to 1. Lines starting with `#`, and any lines before the data, are skipped.

## filter

The filter is SWC's; for EIS the section is ignored, with a warning.

| Key | What it sets | Default |
| --- | --- | --- |
| `al_thickness` | The thickness of the aluminium | `1485 angstrom` |
| `oxide_thickness` | The thickness of the aluminium oxide | `95 angstrom` |
| `c_thickness` | The thickness of the carbon | `0 angstrom` |
| `mesh_throughput` | The fraction of light the filter's supporting mesh lets through | `0.8` |
| `al_table`, `oxide_table`, `c_table` | Tables of the transmission of a layer `table_thickness` thick, in the format above | packaged |
| `table_thickness` | The thickness the tables are for | `1000 angstrom` |

For visible light, the filter's transmission is 10^(-t / 170 angstrom) for a thickness t of aluminium, times `mesh_throughput`.

## fitting

| Key | What it sets | Default |
| --- | --- | --- |
| `components` | The Gaussians to fit to a blend; see [Fitting blended lines](fitting.md) | one Gaussian |
| `primary_component` | The component whose velocity is reported | `0` |
| `constrain_positive_intensity` | Whether to keep every component's amplitude at zero or above | `False` |
| `backend` | The optimiser: `scipy` or `mpfit` | `scipy` |
| `max_iter` | How many iterations the optimiser may take on one spectrum | `1000` |
| `bessel_correction` | Whether the standard deviations over the iterations use n - 1 in place of n (Bessel's correction) | `False` |
| `save_iterations` | Whether to keep every iteration's fit in the results | `False` |

## Pinholes

`pinhole_sizes` and `pinhole_positions` have one entry per pinhole, so they must be the same length. `pinhole_positions_spectral` can be left out, which puts every pinhole in the middle of the spectral window; if given, it needs an entry per pinhole too. `simulation.enable_pinholes` switches the pinholes on. They model defects in SWC's filter, which let through visible light and unattenuated EUV. They are for filter engineering rather than for science runs.
