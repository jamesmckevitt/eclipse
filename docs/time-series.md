# Synthesising from a time series

A slit spectrograph sees one strip of the Sun at a time, so in a raster each exposure sees a different strip at a later time, and in a sit-and-stare it sees the same strip over and over. ECLIPSE can observe a time series of [atmosphere files](synthesis.md#atmosphere-files) in the same way: for each exposure it synthesises only the columns under the slit, from the snapshots that overlap the exposure in time.

This happens in the instrument run rather than in `synthesise-spectra`, because what the slit sees depends on the slit width and the exposure time. The configuration names the files, says how to synthesise them, and gives the observing plan:

```yaml
instrument: SWC
atmosphere_series: ./data/bifrost/en024048_hion_*.h5
reference_line: Fe09_171.0730
n_iter: 100

synthesis:
  lines: [Fe09_171.0730]
  crop_z: [0 Mm, 20 Mm]

raster:
  start: 3850 s
  steps: 20

simulation:
  slit_width: 0.4 arcsec
  expos: [10 s, 40 s]
  psf: True
```

The rest of the configuration is as for a [synthesis file](instrument-response.md). `synthesis_file` and `uniform_intensity` can't be given with `atmosphere_series`.

## The files

`atmosphere_series` is a glob pattern or a list of atmosphere files. Each file needs a `time` and a `velocity_z`, as the view is from above, and all of them must be on the same grid. The files are read a few columns at a time, so a long series of large files needs little memory.

Each snapshot stands for the atmosphere from its own time until the next snapshot's, and the last one for as long again as the gap before it. An exposure outside that range is refused.

## The synthesis

The `synthesis:` section takes the same settings as `synthesise-spectra`:

| Key | Meaning | Default |
| --- | --- | --- |
| `lines` | The lines to synthesise, named as on the [synthesis page](synthesis.md#naming-spectral-lines) | required |
| `abundance` | The CHIANTI abundance set | `sun_coronal_2021_chianti` |
| `vel_res`, `vel_lim` | The velocity grid's spacing and half range | `5 km/s`, `300 km/s` |
| `crop_y`, `crop_z` | Ranges to keep along y and z, as `[low, high]` with units | the whole box |
| `precision` | `float32` or `float64` | `float64` |
| `mass_per_electron` | Atomic mass units per free electron, for files with only a mass density | calculated from the abundances |
| `hdf5_dbase_root` | The CHIANTI database for fiasco | fiasco's default |
| `n_workers` | Workers for the contribution functions | every CPU |
| `goft_temperature_chunk` | Temperatures to compute the contribution functions for at a time, to use less memory | the whole grid |

`reference_line` picks the line whose wavelength grid the others are summed onto, as for a synthesis file, and defaults to the first line.

## The observing plan

| Key | Meaning | Default |
| --- | --- | --- |
| `start` | The simulation time at which the first exposure starts | required |
| `steps` | Slit positions in one raster; `1` is a sit-and-stare | `1` |
| `step` | The angle between neighbouring slit positions | the slit width |
| `repeats` | How many rasters follow one another | `1` |
| `cadence` | The time between the starts of consecutive exposures | the exposure time |
| `centre` | The heliocentric x, as a length, the raster is centred on | the middle of the box |

Exposure *i* starts at `start + i * cadence` and lasts the exposure time. The slit steps from left to right, and then the next raster begins.

With `repeats` above 1, each raster is a separate result: `raster.repeat` is added as a sweep dimension, so a raster is picked out like any swept parameter, for example `get_results_for_combination(results, **{"raster.repeat": 2, ...})`.

## What each exposure sees

An exposure averages the columns under the slit, weighted by how much of the slit each covers. When it spans more than one snapshot, it averages their spectra, weighted by the time each covers; the atmosphere itself is never interpolated between snapshots.

Each column of each snapshot is synthesised once and then reused, so sweeping the exposure time or slit width costs little after the first combination.

The cube for each combination has one column per exposure, at the slit positions. The results file records the plan, the synthesis settings, the files and their times, and each combination's cube, under `raster`.

## From the old dynamic mode

Dynamic mode in `synthesise-spectra` (`--slit-rest-time`) is deprecated and will be removed in a future release; see the note at the end of the [synthesis page](synthesis.md). To move a dynamic-mode run over, write each snapshot as an atmosphere file with its time, as in the [MURaM example](synthesis.md#worked-example-a-muram-flare), and give the files as `atmosphere_series`.
