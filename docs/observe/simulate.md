# Simulating an observation

<div class="glance" markdown>
<div markdown>
**You need**

A [synthesis file](../files/synthesis.md), or a [single intensity](../emission/single-intensity.md)
</div>
<div markdown>
**You run**

`eclipse run`, or `eclipse observe` for this step alone
</div>
<div markdown>
**You get**

A [results file](../files/results.md) in `run/result/`
</div>
<div markdown>
**Next**

[Analysing an observation](../analyse/observation.md)
</div>
</div>

This is the second stage of a run. `eclipse` takes the spectra you made when you [synthesised them](../index.md), passes them through the telescope and detector, adds the noise, and fits the result as you would fit real data. The noise is random, so ECLIPSE repeats this many times (a Monte Carlo simulation). That gives the spread of the measured intensities, velocities and line widths, to compare with the known truth.

This page covers a run on a synthesis file. The same settings apply to [a single line of known intensity](../emission/single-intensity.md) and to a [time series](rasters.md).

## Running it

Run the simulation with:

```bash
eclipse run ./run/input/muram_ar_sweep.yaml
```

`eclipse run` does every step the configuration file describes, and `eclipse observe` does only this one. A configuration file can also name an atmosphere to synthesise first; [Run files](../run-files.md) explains this and the commands' options, such as `--set`, which changes a setting without editing the file. `eclipse --config FILE`, the old form, still works for now.

The results are written to `run/result/<config name>.h5`, named after the configuration file. [Analysing an observation](../analyse/observation.md) shows how to read them.

## The run file

The configuration file is YAML. Its top level says which instrument to simulate, what to observe and how many Monte Carlo iterations to run. The instrument's settings go in four sections, `simulation`, `detector`, `telescope` and `filter`, and the fits' in `fitting`. The [run file settings](../reference/run-file-settings.md) lists every setting with its default, and [The instrument model](../background/instrument-model.md) describes what they model.

A complete example, which observes the active region of the [worked example](../worked-example.ipynb):

```yaml
# Input
instrument: SWC
synthesis_file: ./run/input/muram_ar_fe12.h5
reference_line: Fe12_195.1190

# Global settings (apply to all combinations)
n_iter: 10
ncpu: -1
fit_signals: dn
offchip_bin_slit: [1, 2]  # sweep no binning and 2-pixel off-chip binning

# Simulation parameters
# Any field listed as a list is swept over; all combinations are run.
simulation:
  slit_width: [0.2 arcsec, 0.4 arcsec]  # sweep over two slit widths
  expos: [15 s, 60 s]                   # sweep over two exposure times
  psf: True
  slit_mask: True

# Detector parameters
detector:
  ccd_temperature: -60 Celsius  # used to compute dark current via the CCD model

# Telescope parameters
telescope:
  microroughness_sigma: 0.3 nm  # RMS surface roughness

# Aluminium filter parameters (SWC only)
filter:
  al_thickness: 1485 angstrom
  oxide_thickness: 95 angstrom
  mesh_throughput: 0.8

# The fit, as in the worked example
fitting:
  start: rest
  constrain_positive_intensity: true
  components:
    - wavelength: 195.119 angstrom
      name: Fe XII 195.119
      velocity_limits: [-61.5 km/s, 61.5 km/s]
      width_limits: [0.0191 angstrom, 0.0510 angstrom]
    - wavelength: 195.179 angstrom
      name: Fe XII 195.179
      tie_center: 0
      tie_width: 0
```

**A setting given as a list of more than one value is swept**: ECLIPSE runs every combination of the swept values, as for `offchip_bin_slit`, `slit_width` and `expos` above, which make eight combinations. A list of one value is just that value. At the top level, only `offchip_bin_slit` is swept. The other lists there, such as the pinhole lists and a time series' files, are lists of things rather than sweeps, as are the fitting components. A few settings are lists themselves, and take one value: `telescope.psf_params`, `telescope.fiducial_centres` and `telescope.fiducial_lengths`. So does `telescope.psf_slit_width`, the slit `psf_params` was measured with.

Here are four of the eight combinations, with two pixels binned along the slit. Each map shows how precisely the velocity is measured in each pixel (its standard deviation over the ten iterations):

![The velocity precision over the active region for the 0.2 and 0.4 arcsec slits, in 15 and 60 s exposures](../images/example-sweep.png)

[Comparing combinations](../analyse/observation.md#comparing-combinations) gives these as numbers.

Any setting in the [run file settings](../reference/run-file-settings.md) can be added to its section in the same way. For example, to sweep over the detector's quantum efficiency:

```yaml
detector:
  ccd_temperature: -60 Celsius
  qe_euv: [0.5, 0.65, 0.76]   # sweep over three QE values
```

For recommended values, see [McKevitt et al. (2026), PASJ 78, 1524](https://academic.oup.com/pasj/article/78/4/1524/8731000).

!!! warning "Parameters must go inside their section"

    A setting written at the top level instead, such as `expos:` or `ccd_temperature:` directly under the document root, stops the run with an error naming the section it belongs in. Any other key ECLIPSE does not read, such as a misspelt name, stops the run in the same way.

## Off-chip slit binning

`offchip_bin_slit` adds together neighbouring pixels along the slit after read-out, so the binning is done on the ground rather than on the detector. Adding `n` pixels multiplies the signal by `n`, while the noise only grows as `sqrt(n)`. The signal to noise therefore goes up as `sqrt(n)`, at the cost of resolution along the slit.

```yaml
offchip_bin_slit: [1, 2, 4]   # swept like any other list
```

In the example above, with the 0.4 arcsec slit and 60 s exposures, binning two pixels takes the median precision from 4.0 to 2.8 km/s. Each pixel along the slit then covers 0.32 arcsec rather than 0.16:

![The velocity precision over the active region without binning and with two pixels binned](../images/example-binning.png)

## Turning the noise off

`noise: False` turns the noise off: every random draw in the detector is replaced by its mean. Every iteration is then the same, so one is enough:

```yaml
simulation:
  noise: False

n_iter: 1     # every iteration would be identical
```

Like any setting, it can be swept. Here is the fitted velocity in a quiet part of the [worked example](../worked-example.ipynb)'s active region, in a 60 s exposure, from a run with `noise: [False, True]`:

![The fitted Doppler velocity in a quiet part of the active region, without and with noise](../images/example-noise.png)

## Going further

[Rasters and sit-and-stares](rasters.md) observes a Sun that changes as the slit moves. [Fitting the lines](fitting.md) sets up the fit for blended lines. [The instrument model](../background/instrument-model.md) describes what each setting models, and [On a cluster](../cluster.md#many-cores-with-mpi) shares the iterations between nodes.
