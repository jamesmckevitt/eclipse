# Observing a time series

A slit spectrograph sees one strip of the Sun at a time. Over a raster each exposure sees a different strip at a later time; over a sit-and-stare it sees the same strip again and again. Given a time series of [atmosphere files](atmosphere-files.md), ECLIPSE observes it the way the instrument would: for each exposure it synthesises only the columns under the slit, from the snapshots that overlap the exposure, so an observation of a long series costs about one snapshot's worth of columns rather than every snapshot in full.

This happens inside the instrument run, not in `synthesise-spectra`, because the slit width and the exposure time decide what the slit sees, and they are instrument settings that can be swept. The configuration therefore names the files, says how to synthesise them, and gives the observing plan:

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
  repeats: 1

simulation:
  slit_width: 0.4 arcsec
  expos: [10 s, 40 s]
  psf: True
```

Everything else is as for a [synthesis file](instrument-response.md): the detector, telescope and filter sections, the fitting block, the Monte Carlo iterations. `synthesis_file` and `uniform_intensity` are not given alongside `atmosphere_series`.

## The files

`atmosphere_series` is a glob pattern or a list of atmosphere files. Every file must record its `time`, and all must share one grid, since the columns of one snapshot stand in for those of another within an exposure. The view is from above (the top-down view, integrating along z), so each file needs `velocity_z`. The files are read a few columns at a time, so a series of large files costs no more memory than one strip of one of them.

A snapshot stands for the atmosphere from its time until the next snapshot's time, and the last one for as long again as the gap before it. An exposure that starts before the first snapshot or ends after the last one's span is refused rather than filled from the nearest snapshot.

## The synthesis

The `synthesis:` section takes what `synthesise-spectra` takes on the command line:

| Key | Meaning | Default |
| --- | --- | --- |
| `lines` | The lines to synthesise, named as on the [synthesis page](synthesis.md#naming-spectral-lines) | required |
| `abundance` | The CHIANTI abundance set | `sun_coronal_2021_chianti` |
| `vel_res`, `vel_lim` | The velocity grid's spacing and half range | `5 km/s`, `300 km/s` |
| `crop_y`, `crop_z` | Ranges to keep along y and z, as `[low, high]` with units | the whole box |
| `precision` | `float32` or `float64` | `float64` |
| `mass_per_electron` | Atomic mass units per free electron, for files with only a mass density | derived from the abundances |
| `hdf5_dbase_root` | The CHIANTI database for fiasco | fiasco's default |
| `n_workers` | Workers for the contribution functions | every CPU |

The contribution functions are computed once for the run. `reference_line` chooses which line's wavelength grid the others are summed onto, as it does for a synthesis file, and defaults to the first line.

## The plan

The `raster:` section is the observing plan:

| Key | Meaning | Default |
| --- | --- | --- |
| `start` | The simulation time at which the first exposure starts | required |
| `steps` | Slit positions in one raster; `1` is a sit-and-stare | `1` |
| `step` | The angle between neighbouring slit positions | the slit width, so positions abut |
| `repeats` | How many rasters follow one another | `1` |
| `cadence` | The time between the starts of consecutive exposures | the exposure time, so one exposure starts as the last ends |
| `centre` | The heliocentric x, as a length, on which the raster is centred | the middle of the box |

Exposures run in order: the slit steps from the leftmost position to the rightmost, then the next raster begins. Exposure *i* starts at `start + i * cadence` and lasts the exposure time. A cadence shorter than the exposure is refused.

## What an exposure collects

The columns under the slit are averaged over the slit, each weighted by how much of the slit it covers. When an exposure spans more than one snapshot, the spectra from each are averaged, weighted by the time each covers. Emission is averaged, not the atmosphere: nothing is interpolated between snapshots, so no plasma is invented that neither snapshot holds.

Every column of every snapshot is synthesised the first time an exposure needs it and kept for the rest of the run, so a sweep over exposure times or slit widths, which changes which snapshots and columns each exposure uses, reuses most of the work.

## The cube the instrument sees

Each combination of slit width and exposure gets its own cube, with one column per exposure. Along a raster the columns sit at the slit positions; the instrument run keeps that scan axis as it is and puts only the slit axis on the plate scale. In a sit-and-stare every column is the same strip at a later time, and the maps' scan axis is then exposure number rather than position; the cube's metadata records each exposure's position, start and end.

The results file records the plan, the synthesis settings, the files and their times, and the cube each combination saw, under `raster`.

## From the old dynamic mode

Earlier versions synthesised a time series in `synthesise-spectra` with `--slit-rest-time`, for one slit width and one exposure fixed at synthesis, along x only, one snapshot per exposure, from MURaM's files. That mode is gone. Its synthesis files are refused by the instrument run with a message pointing here; write the snapshots as atmosphere files with `eclipse-atmosphere from-muram`, one per snapshot with its time, and observe them as above.
