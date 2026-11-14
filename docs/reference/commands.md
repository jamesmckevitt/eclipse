# Commands

`eclipse` runs a [run file](../run-files.md), every step it describes or one at a time. `eclipse --help` lists the commands, and `eclipse <command> --help` their options.

## eclipse run

Every step the run file describes: it synthesises the atmosphere, unless the synthesis is made already, observes it, and makes the frame if the file has a `frame:` section.

## eclipse synthesise { data-routes="mhd dem" }

Synthesises the atmosphere only; see [From an MHD simulation](../emission/mhd.md).

## eclipse observe

Observes the synthesis only, which must be made already; see [Simulating an observation](../observe/simulate.md).

## eclipse frame { data-routes="spectra mhd dem" }

Makes the full-CCD frame, synthesising the band under the slit first if the file has an atmosphere or an emission measure; see [Full-CCD frames](../observe/full-ccd-frames.md).

These four take `--set`, which changes a setting for one run of the command ([Changing a setting from the command line](../run-files.md#changing-a-setting-from-the-command-line)), `--only`, which does one of [several runs in one file](../run-files.md#several-runs-in-one-file), and `--debug`, which opens an IPython debugger where an error happens.

## eclipse atmosphere { data-routes="mhd" }

Writes or inspects [atmosphere files](../files/atmosphere.md), as `eclipse-atmosphere` does. `eclipse atmosphere info atmosphere.h5` shows what a file holds without loading the cubes.

## eclipse science-cases

Writes run files for the Concept Study Report's science cases, as `eclipse-science-cases` does. `--one-file` writes them as one run file, with a run for each line.

## synthesise-spectra { data-routes="mhd dem" }

`synthesise-spectra --config job.yaml` reads the atmosphere, the `synthesis:` section and `synthesis_file` of a run file. Options given on the command line override the file's.

[From an MHD simulation](../emission/mhd.md#command-line-options) lists its options.
