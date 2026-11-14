# Run files

A run file is a YAML file that describes a whole job: the simulation to synthesise, how to observe it, and the full-CCD frame to make, if any. One command does every step it describes:

```bash
eclipse run job.yaml
```

## Installing

ECLIPSE needs Python 3.11 or later. Install it from PyPI with `pip install eclipse`, or the latest version from GitHub with:

```bash
pip install git+https://github.com/jamesmckevitt/eclipse.git
```

ECLIPSE computes line emission with [fiasco](https://fiasco.readthedocs.io/), which needs its own copy of the CHIANTI atomic database. The first time ECLIPSE needs the database, fiasco asks whether to download and build it, so the first run is slower than later ones.

## What a run file holds

A run file is the [configuration file](observe/simulate.md#the-run-file) of the instrument run, with an atmosphere to synthesise in place of a synthesis file:

```yaml
instrument: SWC
atmosphere: ./data/muram_ar.h5        # the simulation to synthesise
n_iter: 100

synthesis:
  lines: [Fe12_195.1190]              # the first is the reference line, unless one is given
  crop_z: [0 Mm, 20 Mm]

simulation:
  slit_width: 0.4 arcsec
  expos: [5 s, 20 s]
  psf: True
```

The `synthesis:` section takes the options of [`synthesise-spectra`](emission/mhd.md), with underscores in place of the dashes: `lines`, `band`, `band_step`, `band_lines`, `abundance`, `continuum`, `integration_axis`, `crop_x`, `crop_y`, `crop_z`, `image_x`, `downsample`, `vel_res`, `vel_lim`, `precision`, `mass_per_electron`, `hdf5_dbase_root`, `n_workers` and `goft_temperature_chunk`. An option that takes two values, such as `crop_z`, takes a list, and an option that is on or off, such as `continuum`, takes `True` or `False`. The [run file settings](reference/run-file-settings.md) lists the rest of the file's settings.

A run file can name an `emission_measure`, a [DEM or VDEM](emission/dem.md), in place of the `atmosphere`. Its `synthesis:` section then also takes `electron_density` and `pressure`, and does not take the options that describe an atmosphere: `integration_axis`, the crops, `downsample` and `mass_per_electron`.

A run file without `atmosphere:` observes what it names, a `synthesis_file`, a `uniform_intensity` or a [time series](observe/rasters.md), as before.

## The commands

| Command | What it does |
| --- | --- |
| `eclipse run job.yaml` | Every step: synthesises the atmosphere, unless the synthesis is made already, observes it, and makes the frame if the file has a `frame:` section |
| `eclipse synthesise job.yaml` | Synthesises the atmosphere only |
| `eclipse observe job.yaml` | Observes the synthesis only, which must be made already |
| `eclipse frame job.yaml` | Makes the [full-CCD frame](#a-full-ccd-frame), synthesising the band under the slit first if the file has an `atmosphere` |
| `eclipse atmosphere ...` | Writes or inspects [atmosphere files](files/atmosphere.md), as `eclipse-atmosphere` |
| `eclipse science-cases ...` | Writes run files for the Concept Study Report's science cases, as `eclipse-science-cases` |

The first four take `--set` and `--only`, below, and `--debug`, which opens an IPython debugger where an error happens. `eclipse --help` and `eclipse run --help` list the options.

The synthesis is written to `synthesis_file`, if the run file gives one, or to `run/input/<run file name>.h5`. The results are written to `run/result/<run file name>.h5`, as before.

## Sweeping a setting

A setting given as a list of more than one value is swept: ECLIPSE runs every combination of the swept values, and keeps a result for each. [The run file](observe/simulate.md#the-run-file) on Simulating an observation says which lists are sweeps, and shows one.

## Changing a setting from the command line

`--set` changes one setting for one run of the command, without editing the file. The key names the setting through its sections, and the value is written as in the file:

```bash
eclipse run job.yaml --set n_iter=10 --set simulation.expos="[5 s, 60 s]"
```

In a file with `runs:`, a setting changed with `--set` changes every run, and wins over a run's own. The results are written under the run file's name as usual, so a run with other settings overwrites the results of the last one. Copy the file to keep both.

## Several runs in one file

A list of `runs:` makes several runs from one file. Each run is the rest of the file with its own settings laid over it, section by section, and its `name` names its files:

```yaml
instrument: SWC
atmosphere: ./data/muram_ar.h5
n_iter: 100
simulation:
  expos: [5 s, 20 s]

runs:
  - name: fe12
    synthesis:
      lines: [Fe12_195.1190]
  - name: fe09
    synthesis:
      lines: [Fe09_171.0730]
    simulation:
      expos: [20 s, 60 s]
```

`eclipse run job.yaml` does both runs in turn. The first writes `run/input/job_fe12.h5` and `run/result/job_fe12.h5`. `--only fe09` does one run only, which suits a job array on a cluster. Every run is checked before any is started, and runs that would write different syntheses or frames to the same file are refused.

A setting given as a list of values is still [swept](observe/simulate.md#the-run-file) within a run. Runs are for what a sweep cannot do, such as observing different lines or synthesising with different settings.

`eclipse science-cases --one-file` writes the science cases as one run file, with a run for each line.

## Reusing the synthesis { data-routes="mhd" }

A synthesis can take hours, so a run reuses the one it made before if the atmosphere file and the `synthesis:` section are the same. The atmosphere file counts as the same if its path, its size and the time it was last changed are. A run made with another version of ECLIPSE synthesises again.

So changing the exposure times, or any other setting of the instrument, and running the file again only observes. So does changing `n_workers` or `goft_temperature_chunk`, which change only how the synthesis is computed. Changing the lines, or remaking the atmosphere file, synthesises again and overwrites the old synthesis. `eclipse observe` refuses a synthesis made from other settings, and says to make it again. A file at `synthesis_file` that a run file did not make is never overwritten: the run stops and says so.

## A full-CCD frame { data-routes="spectra mhd dem" }

A `frame:` section describes the [full-CCD frame](observe/full-ccd-frames.md) to make. A frame holds the spectra of a synthesis file. A run with an `atmosphere` synthesises every line in the band, in the part of the image under the slit, into `run/input/<run file name>_band.h5`, and makes the frame from that. A run with a `synthesis_file` makes the frame from the spectra in that file as they are.

```yaml
frame:
  slit_position: 0 arcsec
  continuum: True
```

| Key | What it sets | Default |
| --- | --- | --- |
| `slit_position` | Where the slit's centre is across the slit, in the synthesis's x coordinates | required |
| `exposure` | The exposure time | `simulation.expos`, if it has one value |
| `slit_width` | The slit | `simulation.slit_width`, or 0.2 arcsec |
| `output` | The frame file | `run/result/<run file name>_frame.h5` |
| `continuum` | Whether to synthesise the continuum under the band | `False` |
| `band` | The wavelengths to synthesise between, as `[low, high]` | `[169 Angstrom, 213.5 Angstrom]` |
| `band_step` | The spacing of the band's wavelengths | `0.002 Angstrom` |
| `band_lines` | Where to keep the contribution functions of the band's lines, worked out once and reused | `run/input/band_lines.h5` |
| `band_file` | Where the band's synthesis goes. Runs that synthesise the same band, such as runs that differ only in exposure, can name one file and share it. | `run/input/<run file name>_band.h5` |
| `n_workers` | Processes working out the lines and the band's spectrum | one per CPU |

The last six are for a run with an `atmosphere`, and are refused with a `synthesis_file`. The band is synthesised with the run's `synthesis:` settings, apart from its `lines` and `continuum`.

The detector, telescope and filter are those of the run's `detector:`, `telescope:` and `filter:` sections, and `simulation:` sets `psf`, `spectral_psf`, `psf_boundary`, `noise` and `slit_mask`. A frame takes one value of each setting, so a swept setting is refused. A frame is of the SW camera, so the instrument is SWC.

## Running it elsewhere

[On a cluster](cluster.md) runs the steps on different nodes, or on many at once. [From Python](python.md#running-a-run-file) runs a run file from a script.
