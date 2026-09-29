# Naming spectral lines

Lines are named `<Element><Stage>_<Wavelength>`, for example `Fe12_195.1190`:

- `Fe` - element symbol, capitalised as usual (`Fe`, `Si`, `S`, `O`).
- `12` - ionisation stage as an **arabic** numeral, in spectroscopic notation, so `Fe12` is Fe XII, not Fe XI or Fe XIII.
- `195.1190` - rest wavelength in Angstrom.

The same names are used by `--lines` in `synthesise-spectra`, by `lines` in the `synthesis:` section of a [time series](time-series.md), by `reference_line` in the instrument configuration, and for the lines in a [synthesis file](files.md#synthesis-files).

## How a name is matched to a line

ECLIPSE picks the line of that ion whose CHIANTI wavelength, rounded to as many decimals as the name has, equals the name's wavelength. So `Fe12_195.119` and `Fe12_195.12` both name Fe XII 195.119. Lines that CHIANTI has only a theoretical wavelength for are named the same way, by that wavelength. If an observed and a theoretical line both match, the observed one is taken. If several transitions match, the brightest is taken. The line is synthesised at CHIANTI's wavelength, whatever the digits of the name.

If no line of the ion matches, as can happen with a name from another line list, ECLIPSE takes the nearest line with an observed wavelength, and warns. It skips theoretical wavelengths here, because many of them are weak transitions within a few mA of strong lines. It prints the requested and matched wavelength of every line:

```text
  Fe12_195.1190: requested 195.1190 Angstrom, matched 195.1190 Angstrom, observed (delta=0.0000 Angstrom)
```

Check these. A large difference means the line you meant is not in CHIANTI for that ion, and a neighbouring one was taken instead. Two names that match the same line are refused, since the line would be synthesised twice. A name that does not follow the pattern is refused straight away.
