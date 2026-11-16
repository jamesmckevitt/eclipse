# Rasters and sit-and-stares

<div class="glance" markdown>
<div markdown>
**You need**

A series of [atmosphere files](../files/atmosphere.md) or [synthesis files](../files/synthesis.md), each with its time
</div>
<div markdown>
**You run**

`eclipse run`
</div>
<div markdown>
**You get**

A [results file](../files/results.md), with a cube of the raster
</div>
<div markdown>
**Next**

[Analysing an observation](../analyse/observation.md)
</div>
</div>

In a raster each exposure sees the next strip along, a little later; in a sit-and-stare it sees the same strip each time. ECLIPSE observes a time series of atmosphere files this way, synthesising only the columns under the slit from the snapshots each exposure overlaps. A series of [synthesis files](#from-synthesis-files) works too.

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

The rest is as for [a single snapshot](simulate.md).

## The files { data-routes="mhd" }

`atmosphere_series` is a glob pattern or a list of files on the same grid, each with a `time` and a `velocity_z`. Each snapshot lasts until the next one; the last lasts as long as the gap before it.

## The synthesis { data-routes="mhd" }

The view is from above, and the slit picks the columns. The `synthesis:` section takes:

| Key | Meaning | Default |
| --- | --- | --- |
| `lines` | The lines to synthesise, named as in [Naming spectral lines](../reference/line-names.md) | required |
| `abundance` | The CHIANTI abundance set | `sun_coronal_2021_chianti` |
| `vel_res`, `vel_lim` | The velocity grid's spacing and half range | `5 km/s`, `300 km/s` |
| `crop_y`, `crop_z` | Ranges to keep along y and z, as `[low, high]` with units | the whole box |
| `precision` | `float32` or `float64` | `float64` |
| `mass_per_electron`, `hdf5_dbase_root`, `n_workers`, `goft_temperature_chunk` | As in [synthesise-spectra](../emission/mhd.md#command-line-options) | |

## From synthesis files

`synthesis_series` gives one [synthesis file](../files/synthesis.md) per snapshot, from ECLIPSE or another code, in place of `atmosphere_series`:

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

Each file needs a `time`, and all must share the view, the pixels and the observed window's wavelengths. The result is the same as from the atmosphere files.

## The observing plan

| Key | Meaning | Default |
| --- | --- | --- |
| `start` | Simulation time of the first exposure | required |
| `steps` | Slit positions per raster; `1` is a sit-and-stare | `1` |
| `step` | Angle between slit positions | the slit width |
| `repeats` | Rasters, one after another | `1` |
| `cadence` | Time between exposure starts | the exposure time |
| `centre` | Heliocentric x of the raster's centre, as a length | the middle |
| `direction` | `increasing` or `decreasing` x | `increasing` |

Each repeat is a separate result, picked out with `get_results_for_combination(results, **{"raster.repeat": 2, ...})`.

## What each exposure sees

An exposure averages the columns under the slit, weighted by how much of the slit each covers, and the snapshots it spans, weighted by time. Snapshots are not interpolated. The results file keeps one column per exposure, under `raster`.
