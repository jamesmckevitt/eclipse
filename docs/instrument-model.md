# The instrument model

ECLIPSE follows the light from the Sun to the detector. For SWC, the telescope collects it with the throughput of the primary mirror, its roughness, the filter and the grating. LW has no filter, and its throughput includes the loss to contamination. For EIS, its calibration tables give the effective area. The point spread function (PSF) blurs it, and the slit selects a strip of the Sun. The detector turns the photons into electrons, with their noise, adds dark current and read noise, and digitises the result. This page describes the parts of that model that have settings of their own. The [configuration reference](configuration.md) lists every setting, and [Instrument sources](instrument-sources.md) gives the source of each default.

## The instruments

ECLIPSE models three instruments, chosen with the `instrument:` key at the top level of the configuration file:

- `SWC` - SOLAR-C/EUVST-SW, the short wavelength channel. This is the default.
- `LW` - SOLAR-C/EUVST-LW, the long wavelength channel.
- `EIS` - Hinode/EIS.

### EUVST-SW

ECLIPSE's default settings describe EUVST-SW. [McKevitt et al. (2026), PASJ 78, 1524](https://academic.oup.com/pasj/article/78/4/1524/8731000) describes the instrument model. It will be updated as EUVST is built, tested and commissioned.

Its slits are 0.2, 0.4, 0.8 and 1.6 arcsec wide. Its PSF is modelled, from simulations and some measurements of the mirror's roughness, and will be measured before launch; ECLIPSE prints a warning saying so whenever `psf: True` is set.

### EUVST-LW

The LW channel records three bands, and LW2 and LW3 in second order as well:

| Band | First order (Angstrom) | Second order (Angstrom) |
| --- | --- | --- |
| `LW1` | 709.5 - 851.4 | |
| `LW2` | 919.1 - 1049.0 | 459.5 - 524.5 |
| `LW3` | 1108.3 - 1225.4 | 554.1 - 612.7 |

These are the ranges each detector is expected to record (SSD-SPC-ET004). Give the band and order of the line you observe in the `detector:` or the `telescope:` section. The default is `LW3` in first order. A line in second order is given at its own wavelength. For example, for Si XII 499.41 in second order on LW2:

```yaml
instrument: LW
detector:
  band: LW2
  order: 2
```

The throughput comes from the sensitivity budget, RSC-2022035, which does not reach the edges of every detector. A run stops with an error if a wavelength of its window lies off the band, or on it beyond the budget's tables: above 845 Angstrom in LW1, above 1039 Angstrom in LW2, and above 520 and 609 Angstrom in second order. Mg X 609.79 is one such line.

Each band is recorded by an intensified APS. A microchannel plate (MCP) turns each detected photon into a burst of about 1000 electrons, a phosphor turns the burst into light, and an APS records the light, about 13 electrons per detected photon on average. The size of a burst varies by as much as its mean, so the noise is that of half as many photons. The read noise, dark current, temperature and full well are the values the detector specification, SSD-SPC-ET004, requires. [Instrument sources](instrument-sources.md#euvst-lw) gives the source of each setting.

The probability of detecting a photon depends on its wavelength, and is taken at each pixel's own wavelength. In LW3 it falls from 0.30 at 1208 Angstrom to 0.01 from 1212 Angstrom, where the MCP has no photocathode so that Lyman-alpha can be observed.

LW looks through the same slits as SW, 0.2, 0.4, 0.8 and 1.6 arcsec wide. Its MCP does not see visible light, so it has no filter. A `filter:` section is ignored, with a warning. A `vis_sl` other than 0, any pinhole setting, `slit_mask` and `focal_plane_dispersion` stop the run with an error.

Its PSF is a Gaussian with the band's worst resolution in the optical design, RSC-2022021C, and ECLIPSE prints a warning saying so whenever `psf: True` is set.

### Hinode/EIS

Its slits are 1 and 2 arcsec wide. ECLIPSE does not model its 40 and 266 arcsec slots.

Three settings apply to SWC only. Under `EIS`:

- `filter:` describes EUVST-SW's aluminium filter. EIS's effective area comes from its own calibration tables, which already include its filters, so for EIS the whole section is ignored, with a warning. See [EIS effective area](#eis-effective-area) below.
- `telescope.microroughness_sigma` is the roughness of EUVST's primary mirror. For EIS it is ignored, with a warning.
- Pinholes in the filter are modelled for EUVST-SW only. Any pinhole setting, `enable_pinholes` or a `pinhole_*` list, stops an EIS run with an error.

EIS's PSF is not well known. ECLIPSE uses a symmetrical Gaussian 3 pixels wide (FWHM), following Ugarte-Urra (2016), EIS Software Note 2, and prints a warning saying so whenever `psf: True` is set.

#### EIS effective area

The EIS effective area comes from the instrument's own calibration tables, so it varies with wavelength and, for the in-flight calibrations, with the date of the observation:

```yaml
instrument: EIS
telescope:
  calibration: dz2025
  date: "2012-06-03"    # quoted, so YAML keeps it a string
```

| `calibration` | Source | Date |
|---|---|---|
| `ground` (default) | Pre-flight MSSL tables, `eis_ea.pro` | not used |
| `dz2013` | Del Zanna (2013), `eis_ltds.pro` | required |
| `warren2014` | Warren, Ugarte-Urra & Landi (2014) | required |
| `dz2025` | Del Zanna et al. (2025), `interpol_eis_ea.pro` | required |

The three in-flight calibrations need a `date`, and stop with an error without one. Both keys can be swept like any other setting:

```yaml
telescope:
  calibration: dz2025
  date: ["2008-01-01", "2013-01-01", "2018-01-01"]
```

## The point spread function

With `simulation.psf: True`, ECLIPSE blurs the spectra with the instrument's point spread function (PSF), along the slit and in wavelength, before they reach the detector's pixels. It is off by default.

### What the SW PSF is made of

The SW PSF is a Gaussian. Its width is two parts added in quadrature:

- The optical design, from RSC-2022021C (March 2025). This is the worst case over a field of view, chosen with `telescope.field_of_view`: `280x280` (default, the full scan), `100x100` or `70x70` arcsec. Along the slit it is 0.423, 0.195 or 0.161 arcsec. In wavelength it is 43.00 mA with the 0.2 arcsec slit, which the design gives for the 70 x 70 arcsec field only and which is used for all three.
- Everything else, from the resolution budget RSC-2022033B rev. B (October 2025): the primary mirror, the scan mirror, focusing, alignment, wavefront measurement, pointing jitter, the spectrograph's optics and vibration of the grating. `telescope.resolution_budget` chooses `current_estimate` (default), the current best estimate, or `allocation`, what the budget allows each part. `none` leaves them out.

The budget is written for the 70 x 70 and 100 x 100 arcsec fields, and apart from the optical design it is the same for both. The full field has no budget yet, so these terms are used for every field. In wavelength the budget gives only the allocation. Each of its rows there is a row of the spatial budget, or a fraction of one, so the current best estimate in wavelength is worked out from the spatial estimate in the same way. The slit width error in the budget is a tolerance on the slit, not a blur, and is left out.

The telescope's terms, which come before the slit, also blur the scene across the slit, so some light from beside the slit falls into it. These are everything but the spectrograph's optics. The optical design's own blur across the slit is not known, so `telescope.psf_across_slit` adds it only if given.

| FWHM, 280 x 280 field | Optical design | With current estimate | With allocation |
| --- | --- | --- | --- |
| Along the slit | 0.423 arcsec | 0.516 arcsec | 0.564 arcsec |
| Across the slit | none | 0.278 arcsec | 0.336 arcsec |
| In wavelength, 0.2 arcsec slit | 43.0 mA | 45.0 mA | 47.7 mA |

```yaml
telescope:
  field_of_view: 70x70
  resolution_budget: allocation
simulation:
  psf: True
```

Setting `telescope.psf_params` gives the optical design's widths directly, in place of `field_of_view`.

The budget's pointing jitter is 0.110 arcsec FWHM. `telescope.pointing_jitter` replaces it with another FWHM, which can be swept. The jitter blurs the scene along and across the slit. The budget takes the grating's vibration as two thirds of the jitter, so that changes with it too. A jitter given as a 3-sigma amplitude is converted with FWHM = 2.3548 sigma.

```yaml
telescope:
  pointing_jitter: [0 arcsec, 0.110 arcsec, 0.2 arcsec]
```

### The spectral PSF and the slit

With `psf: True`, each line is blurred in wavelength by the optics and by the image of the slit, so the blur depends on the slit width. `telescope.psf_params` gives the optical design's FWHM in pixels, along the slit and in wavelength. For SWC, the one in wavelength is for the 0.2 arcsec slit.

`simulation.spectral_psf` sets how the slit is added to the optics' blur:

- `quadrature` (default): a Gaussian, whose FWHM is the optics' and the slit image's widths added in quadrature.
- `convolution`: the optics' Gaussian convolved with the slit's rectangular image.

```yaml
simulation:
  slit_width: [0.2 arcsec, 1.6 arcsec]
  psf: True
  spectral_psf: convolution
```

The EIS PSF is not tied to a slit, so its width in wavelength is the same for every slit, and `convolution` is refused for it. Setting `telescope.psf_slit_width`, the slit that `psf_params` was measured with, changes both.

### The edges of the atmosphere

With `psf: True`, the blur brings in light from beyond the edges of the atmosphere. `simulation.psf_boundary` sets what is there:

- `replicate` (default): the Sun beyond each edge is taken to be like the cells at the edge, so the pixels there are as bright as they would be in the middle of a larger atmosphere.
- `zero`: nothing is beyond the edges, so the light the blur carries out of the atmosphere is lost, and the pixels within a PSF width of an edge come out darker.

```yaml
simulation:
  psf: True
  psf_boundary: zero
```

## The slit's fiducials

Each SWC slit is 300 arcsec long and is interrupted in two places. These gaps, the fiducials, block the light over a short length of the slit, so they show as dark rows along the slit in every slit's data. They are there to line up observations made with different slits.

With `simulation.slit_mask: True`, no light reaches the pixels behind the fiducials or beyond the ends of the slit. A pixel partly behind one gets the part of the light that falls on its open part. It is off by default, which leaves the slit open along the whole atmosphere.

```yaml
simulation:
  slit_width: 0.4 arcsec
  psf: True
  slit_mask: True
```

- With `psf: True`, the blur along the slit happens before the slit, so the edges of a fiducial are as sharp as the pixels allow.

## Visible light

Visible light also reaches the CCD. `simulation.vis_sl` adds visible stray light, in photons per second per cm2, the same in every pixel. For SWC it is the light before the filter, which passes only a small part of it, and for EIS the light that reaches the CCD. Each visible photon is detected with the probability `detector.qe_vis` and frees one electron.

Pinholes in SWC's filter let light past it, spread out by diffraction. `simulation.enable_pinholes` switches them on, and [Pinholes](configuration.md#pinholes) describes their settings.

## Dark current

The CCD collects dark current over the exposure. It also collects it while it is cleared before the exposure, and while each row waits to be read out afterwards. For SWC, ECLIPSE counts all of this, as a [full-CCD frame](full-ccd-frame.md) does. Each wavelength pixel takes the time of the CCD row that records its wavelength.

The wait depends on which rows are read. A row that is read takes about 0.56 ms, and one that is not read is dumped in 15 us. A run takes only the rows of its own wavelengths to be read, as one window. With the default timing, a window of 32 rows adds 0.03 to 0.05 s to the exposure, wherever it is. The clear takes longest for the rows far from the gap between the CCDs, and the wait to be read takes longest for the rows near it. A real observation reads a window around each of its lines, so its rows wait longer than a run gives. A full-CCD frame reads every row, and its last row waits 1.15 s. A run also leaves out charge transfer, as the rest of a run does.

```yaml
simulation:
  readout_dark_current: False   # dark current over the exposure only
```

With `readout_dark_current: False`, dark current is counted over the exposure only, as in ECLIPSE 0.12 and earlier. EIS runs count it over the exposure only.

## The pixels' width in wavelength

The grating does not spread the wavelengths quite evenly over the CCD, so the width in wavelength of a pixel changes slowly along it. It is 16.98 mA at 171 Angstrom, 16.91 mA at 195 Angstrom and 16.86 mA at 211 Angstrom. By default a run gives every pixel the same width, `detector.wvl_res`, 16.9 mA.

```yaml
simulation:
  focal_plane_dispersion: True   # the focal plane's pixel width where the line lands
```

With `focal_plane_dispersion: True`, a run takes the width of the pixel the observed line lands on, from the SW focal plane, and gives it to every pixel of its window. The pixels then collect as much of the spectrum, and span as many km/s, as on the CCD. Across a window of 60 pixels the true width changes by about 2 parts in 10,000, which a run leaves out. A run does not place its pixels on the CCD's rows either, so a line sits at the same place within its pixel as without the setting. It cannot be combined with `detector.wvl_res`, and works for SWC, with synthesis files or a uniform intensity, for a line that reaches the CCDs within the lit band.
