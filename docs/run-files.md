# Run files

A run file is a YAML file that describes a whole job: the simulation to synthesise, how to observe it, and the full-CCD frame to make. One command does every step:

```bash
eclipse run job.yaml
```

## Installing

```bash
pip install eclipse
```

ECLIPSE needs Python 3.11 or later. The first synthesis offers to download and build the CHIANTI atomic database for [fiasco](https://fiasco.readthedocs.io/).

## What a run file holds

A run file is the [configuration file](observe/simulate.md#the-run-file) of a run, with an atmosphere to synthesise in place of a synthesis file:

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

The `synthesis:` section takes the options of [`synthesise-spectra`](emission/mhd.md), with underscores in place of dashes, such as `lines`, `crop_z` or `position`. The [run file settings](reference/run-file-settings.md) lists the other settings.

A run file can name an `emission_measure`, a [DEM or VDEM](emission/dem.md), in place of the `atmosphere`, with `electron_density` or `pressure` in its `synthesis:` section.

A run file can name an `emission_measure`, a [DEM or VDEM](dem-synthesis.md), in place of the `atmosphere`, with `electron_density` or `pressure` in its `synthesis:` section.

## The commands

| Command | What it does |
| --- | --- |
| `eclipse run job.yaml` | Every step the file describes |
| `eclipse synthesise job.yaml` | Synthesises the atmosphere only |
| `eclipse observe job.yaml` | Observes the synthesis only |
| `eclipse frame job.yaml` | Makes the [full-CCD frame](#a-full-ccd-frame) |
| `eclipse atmosphere ...` | Writes or inspects [atmosphere files](files/atmosphere.md) |
| `eclipse science-cases ...` | Writes run files for the science cases |

The synthesis goes to `run/input/<run file name>.h5`, unless `synthesis_file` names another, and the results to `run/result/<run file name>.h5`. `eclipse --help` lists the options.

## Sweeping a setting

A setting given as a list of values is swept: ECLIPSE runs every combination; see [The run file](observe/simulate.md#the-run-file).

## Changing a setting from the command line

```bash
eclipse run job.yaml --set n_iter=10 --set simulation.expos="[5 s, 60 s]"
```

## Several runs in one file

A `runs:` list makes several runs, each the rest of the file with its own settings laid over it:

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
```

Each run's files take its name, such as `run/result/job_fe12.h5`. `--only fe09` does one run.

## Reusing the synthesis { data-routes="mhd dem" }

A run reuses its synthesis while the atmosphere file and the `synthesis:` section are unchanged, so changing an exposure time only observes again.

## A full-CCD frame { data-routes="spectra mhd dem" }

A `frame:` section describes the [full-CCD frame](observe/full-ccd-frames.md) to make. With an `atmosphere`, the run synthesises the band under the slit for it. With a `synthesis_file`, the frame holds that file's spectra.

```yaml
frame:
  slit_position: 0 arcsec
  continuum: True
```

| Key | What it sets | Default |
| --- | --- | --- |
| `slit_position` | The slit's centre, in the synthesis's x coordinates | required |
| `exposure` | The exposure time | `simulation.expos` |
| `slit_width` | The slit width | `simulation.slit_width`, or 0.2 arcsec |
| `output` | The frame file | `run/result/<run file name>_frame.h5` |
| `continuum` | Whether to add the continuum | `False` |
| `band` | The wavelengths of the lines to add | `[169 Angstrom, 213.5 Angstrom]` |
| `band_step` | The spacing of the band's wavelengths | `0.002 Angstrom` |
| `band_lines` | Where to keep the band's lines for later frames | `run/input/band_lines.h5` |
| `band_file` | Where the band's synthesis goes, which runs can share | `run/input/<run file name>_band.h5` |
| `n_workers` | Processes to use | one per CPU |

The last six need an `atmosphere`. The instrument settings come from the run's other sections.

## Running it elsewhere

[On a cluster](cluster.md) runs the steps on different nodes, or on many at once. [From Python](python.md#running-a-run-file) runs a run file from a script.
