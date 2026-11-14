# Band lines files

A band lines file holds the contribution function of every CHIANTI line in a band, on a grid of temperature and density. A [full-CCD frame](../observe/full-ccd-frames.md) of the whole band needs them, and they take a few hours to work out, so `synthesise-spectra --band` keeps them in a file and reads them back for later syntheses. `write_band_lines` writes one and `read_band_lines` reads it.

The file is read back for a synthesis if it covers the synthesis's band, is on the same temperatures and abundances, and has every density the synthesis needs.

## What one holds

The root has a `format` attribute of `eclipse-band-lines` and a `version` of `1`, and these attributes:

| Attribute | Value |
| --- | --- |
| `band` | The band's edges, in Angstrom |
| `abundance` | The CHIANTI abundance set |
| `hdf5_dbase_root` | The CHIANTI database the lines came from |

The datasets `logT_grid` and `logN_grid` hold the grids, as log10(T / K) and log10(n_e / cm-3). A group for each ion with lines in the band, under `ions`, holds:

| Name | Kind | What it holds |
| --- | --- | --- |
| `name`, `atom`, `stage`, `mass` | attributes | The ion, such as `Fe 12`, its atomic number and ionisation stage, and its mass in atomic mass units |
| `wavelength` | dataset, `(n_lines,)` | Each line's wavelength in Angstrom: CHIANTI's observed value where it has one, and the theoretical value otherwise |
| `observed` | dataset, `(n_lines,)` | Whether each wavelength is observed |
| `temperatures` | dataset | The indices of the grid temperatures where the ion exists |
| `g` | dataset, `(n_lines, n_densities, n_temperatures)` | The contribution function at those temperatures and every grid density, in `erg cm3 / s` |
