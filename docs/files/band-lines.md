# Band lines files

A band lines file holds the contribution function of every CHIANTI line in a band, for a [full-CCD frame](../observe/full-ccd-frames.md) of the whole band. `synthesise-spectra --band` keeps one and reads it back for later syntheses. `write_band_lines` writes one and `read_band_lines` reads it.

The root has a `format` attribute of `eclipse-band-lines`, a `version` of `1`, and the attributes `band` (in Angstrom), `abundance` and `hdf5_dbase_root`. The datasets `logT_grid` and `logN_grid` hold the grids. A group for each ion under `ions` holds its `name`, `atom`, `stage` and `mass` as attributes, and the datasets `wavelength` (Angstrom), `observed`, `temperatures` (indices into the grid) and `g`, shaped `(n_lines, n_densities, n_temperatures)` in `erg cm3 / s`.
