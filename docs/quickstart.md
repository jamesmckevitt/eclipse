# Getting started

## Installation

ECLIPSE needs Python 3.11 or later.

### From PyPI (recommended)

```bash
pip install eclipse
```

### From source

```bash
pip install git+https://github.com/jamesmckevitt/eclipse.git
```

ECLIPSE computes line emission with [fiasco](https://fiasco.readthedocs.io/), which needs its own copy of the CHIANTI atomic database. The first time ECLIPSE needs the database, fiasco asks whether to download and build it, so the first run is slower than later ones.

## A first run

The quickest way to see ECLIPSE work is to simulate one spectral line of known brightness. It needs no simulation and no atomic data. Write this configuration to `first_run.yaml`:

```yaml
instrument: SWC                            # EUVST's short wavelength channel
uniform_intensity: 5000 erg / (s cm2 sr)   # the line's total intensity
rest_wavelength: 195.119 AA                # Fe XII 195.119
n_iter: 100                                # repeat each exposure 100 times, each with new noise

simulation:
  expos: [5 s, 20 s]                       # exposure times
  psf: True
```

and run it:

```bash
eclipse run first_run.yaml
```

This observes the line in exposures of 5 s and 20 s, and writes the results to `run/result/first_run.h5`. Then, in Python:

```python
import astropy.units as u
from eclipse import (analyse_fit_statistics, get_results_for_combination,
                     load_instrument_response_results, summary_table)

results = load_instrument_response_results("run/result/first_run.h5")
summary_table(results)

for exposure in [5, 20] * u.s:
    combination = get_results_for_combination(results, **{"simulation.expos": exposure})
    stats = analyse_fit_statistics(combination)
    print(f"{exposure}: velocity precision {stats['v_std'].squeeze():.2f}")
```

`summary_table` lists the settings of each exposure time's run, and the loop prints the scatter of the measured velocity, its standard deviation over the 100 repeats:

```text
5.0 s: velocity precision 3.07 km / s
20.0 s: velocity precision 1.62 km / s
```

Your numbers will differ a little, since the noise is random. [A single line of known intensity](uniform-intensity.md) says more about this kind of run, and [Analysing the results](analysis.md) about the results.

## A full run

With a simulation, ECLIPSE first synthesises the spectra it emits, then observes them. A run file gives the settings of both steps. Write this to `fe12.yaml`:

```yaml
instrument: SWC                    # EUVST's short wavelength channel
atmosphere: ./data/atmosphere.h5   # the simulation to synthesise
n_iter: 100                        # repeat each exposure 100 times, each with new noise

synthesis:
  lines: [Fe12_195.1190]           # the lines to synthesise; the first is observed

simulation:
  expos: [10 s, 40 s]              # exposure times
```

and run it:

```bash
eclipse run fe12.yaml
```

The spectra go to `run/input/fe12.h5` and the results to `run/result/fe12.h5`. [From an MHD simulation](synthesis.md#atmosphere-files) explains how to write an atmosphere file.

[Run files](run-files.md) explains the commands, and the [configuration reference](configuration.md) lists every setting.

The [worked example](worked-example.ipynb) goes through a full run with a real simulation, from downloading it to maps of the measured intensity and Doppler velocity. [Where to start](index.md#where-to-start) lists the other starting points.
