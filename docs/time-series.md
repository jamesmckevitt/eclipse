# Simulating a time series

A slit spectrograph sees one strip of the Sun at a time. In a raster, each exposure sees the next strip along, a little later than the one before. In a sit-and-stare, it sees the same strip again and again. ECLIPSE can observe a time series of [atmosphere files](synthesis.md#atmosphere-files) in the same way. For each exposure it synthesises only the columns under the slit, from the snapshots that overlap the exposure in time. A series of spectra already synthesised, by ECLIPSE or [another code](other-codes.md), can be observed too; see [From synthesis files](#from-synthesis-files).

What the slit sees depends on the slit width and the exposure time, so this all happens in the `eclipse` run rather than in `synthesise-spectra`. The configuration names the files, says how to synthesise them, and gives the observing plan:

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

The rest of the configuration is as for [a single snapshot](instrument-response.md). `synthesis_file` and `uniform_intensity` can't be given with a series.

## The files

`atmosphere_series` is a glob pattern or a list of atmosphere files. Each file needs a `time`, and a `velocity_z` since the view is from above. All of them must be on the same grid. The files are read a few columns at a time, so even a long series of large files needs little memory.

Each snapshot stands for the atmosphere from its own time until the next snapshot's. The last one lasts as long as the gap before it. An exposure outside that time range is refused.

## The synthesis

The `synthesis:` section takes these settings, which mean what they do in `synthesise-spectra`. The view is always from above and the slit picks the columns, so `integration_axis`, `crop_x` and `downsample` aren't among them.

| Key | Meaning | Default |
| --- | --- | --- |
| `lines` | The lines to synthesise, named as in [Naming spectral lines](line-names.md) | required |
| `abundance` | The CHIANTI abundance set | `sun_coronal_2021_chianti` |
| `vel_res`, `vel_lim` | The velocity grid's spacing and half range | `5 km/s`, `300 km/s` |
| `crop_y`, `crop_z` | Ranges to keep along y and z, as `[low, high]` with units | the whole box |
| `precision` | `float32` or `float64` | `float64` |
| `mass_per_electron` | Atomic mass units per free electron, for files with only a mass density | worked out from the abundances |
| `hdf5_dbase_root` | The CHIANTI database for fiasco | fiasco's default |
| `n_workers` | Processes computing the contribution functions, one ion each at a time | one per CPU, but no more than one per ion |
| `goft_temperature_chunk` | How many temperatures to compute the contribution functions for at once; fewer uses less memory | the whole grid |

`reference_line` chooses the spectral window to observe, as for a single snapshot. It defaults to the first of `lines`.

## From synthesis files

A series can also be given as [synthesis files](files.md#synthesis-files), one per snapshot, with `synthesis_series` in place of `atmosphere_series`. They can come from ECLIPSE's own synthesis or from another code. The slit then reads the columns under it rather than synthesising them, so there is no `synthesis:` section:

```yaml
instrument: SWC
synthesis_series: ./my_code/snapshot_*.h5
n_iter: 100

raster:
  start: 3850 s
  steps: 20

simulation:
  slit_width: 0.4 arcsec
  expos: [10 s, 40 s]
  psf: True
```

Each file needs a `time`. All of them must be seen from the same side, on the same grid of pixels, and hold the same lines on the same wavelengths within the window observed. Lines outside that window are not read. ECLIPSE's own synthesis copies the atmosphere file's time into the synthesis file. `reference_line` works as it does for a [single snapshot](instrument-response.md), and defaults to the files' only line. The observing plan, and what each exposure sees, are the same as for atmosphere files.

Synthesising every snapshot with `synthesise-spectra`, then observing the synthesis files with the same `reference_line`, gives the same result as observing the atmosphere files. It just takes longer, since it synthesises every column of every snapshot rather than only those under the slit.

## The observing plan

| Key | Meaning | Default |
| --- | --- | --- |
| `start` | The simulation time at which the first exposure starts | required |
| `steps` | Slit positions in one raster; `1` is a sit-and-stare | `1` |
| `step` | The angle between neighbouring slit positions | the slit width |
| `repeats` | How many rasters follow one another | `1` |
| `cadence` | The time between the starts of consecutive exposures | the exposure time |
| `centre` | The heliocentric x, as a length, that the raster is centred on | the middle of the box, or of the image |
| `direction` | The way the slit steps: `increasing` x or `decreasing` x | `increasing` |

Exposure *i* starts at `start + i * cadence` and lasts the exposure time. The slit steps across in the chosen direction, and then the next raster begins.

With `repeats` above 1, each raster is kept as a separate result, and is picked out like any setting the run swept through, for example `get_results_for_combination(results, **{"raster.repeat": 2, ...})`.

## What each exposure sees

An exposure averages the columns under the slit, weighted by how much of the slit each one covers. When an exposure spans more than one snapshot, it averages their spectra, weighted by how long each one lasts within the exposure. The atmosphere itself is never interpolated between snapshots.

Each column of each snapshot is synthesised, or read, only once and then reused, so sweeping the exposure time or slit width costs little after the first combination.

The cube for each combination has one column per exposure, at the slit positions. The results file keeps the plan, the synthesis settings for atmosphere files, the files and their times, and each combination's cube, under `raster`. If the synthesis files' wavelengths are not evenly spaced, there is no cube, because its coordinates can't describe them.

## From the old dynamic mode

Dynamic mode in `synthesise-spectra` (`--slit-rest-time`) is deprecated. [Older versions](older-versions.md#dynamic-mode) says how to move a dynamic-mode run over to a time series of atmosphere files.
