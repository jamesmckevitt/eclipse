# Instruments

ECLIPSE models two instruments, chosen with the `instrument:` key at the top level of the configuration file:

- `SWC` - SOLAR-C/EUVST-SW, the short wavelength channel. This is the default.
- `EIS` - Hinode/EIS.

The EUVST long wavelength channel (LW) is coming soon, and will complete EUVST.

## EUVST-SW

ECLIPSE's default settings describe EUVST-SW. The [configuration reference](configuration.md) lists them, and [McKevitt et al. (2026), PASJ 78, 1524](https://academic.oup.com/pasj/article/78/4/1524/8731000) describes the instrument model. It will be updated as EUVST is built, tested and commissioned.

Its slits are 0.2, 0.4, 0.8 and 1.6 arcsec wide. Its PSF is modelled, from simulations and some measurements of the mirror's roughness, and will be measured before launch; ECLIPSE prints a warning saying so whenever `psf: True` is set.

## Hinode/EIS

Its slits are 1 and 2 arcsec wide. ECLIPSE does not model its 40 and 266 arcsec slots.

Three settings apply to SWC only. Under `EIS`:

- `filter:` describes EUVST-SW's aluminium filter. EIS's effective area comes from its own calibration tables, which already include its filters, so for EIS the whole section is ignored, with a warning. See [EIS effective area](#eis-effective-area) below.
- `telescope.microroughness_sigma` is the roughness of EUVST's primary mirror. For EIS it is ignored, with a warning.
- Pinholes in the filter are modelled for EUVST-SW only. Any pinhole setting, `enable_pinholes` or a `pinhole_*` list, stops an EIS run with an error.

EIS's point spread function (PSF) is not well known. ECLIPSE uses a symmetrical Gaussian 3 pixels wide (FWHM), following Ugarte-Urra (2016), EIS Software Note 2, and prints a warning saying so whenever `psf: True` is set.

`simulation.vis_sl`, the visible stray light, is given before the filter for SWC. EIS's filters are not modelled for visible light, so for EIS give the light that reaches its CCD.

### EIS effective area

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

