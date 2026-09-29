# Getting started

## Installation

ECLIPSE needs Python 3.11 or later.

### From PyPI (recommended)

```bash
pip install solarc-eclipse
```

### From source

```bash
pip install git+https://github.com/jamesmckevitt/eclipse.git
```

ECLIPSE computes line emission with [fiasco](https://fiasco.readthedocs.io/), which needs its own copy of the CHIANTI atomic database. The first time ECLIPSE needs the database, fiasco asks whether to download and build it. The download is about 600 MB and the build takes several minutes, so the first run is slower than later ones.

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
eclipse --config first_run.yaml
```

This observes the line in exposures of 5 s and 20 s, and writes the results to `run/result/first_run.h5`. It takes about half a minute on 8 cores. Then, in Python:

```python
import astropy.units as u
from euvst_response import (analyse_fit_statistics, get_results_for_combination,
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

With a simulation, a run has two steps. First `synthesise-spectra` turns the simulation into the spectra it emits. Then `eclipse` passes those spectra through the instrument, adds noise, and fits the lines as you would fit real data:

```bash
# Make the spectrum of Fe XII 195.119 Angstrom from an atmosphere file
synthesise-spectra --atmosphere ./data/atmosphere.h5 --lines Fe12_195.1190 --output-dir ./run/input

# Pass it through the instrument
eclipse --config ./run/input/config.yaml
```

An atmosphere file holds a simulation's temperature, density and velocity. [From an MHD simulation](synthesis.md#atmosphere-files) explains how to write one from your simulation's output. `synthesise-spectra` writes the spectra to `./run/input/synthesised_spectra.h5`.

`eclipse` reads its settings from a configuration file. There is no default one, so write `config.yaml` yourself before running it:

```yaml
instrument: SWC                                      # EUVST's short wavelength channel
synthesis_file: ./run/input/synthesised_spectra.h5
reference_line: Fe12_195.1190                        # observe the window around this line
n_iter: 100                                          # repeat each exposure 100 times, each with new noise

simulation:
  expos: [10 s, 40 s]                                # exposure times
```

Both commands list their options with `--help`. [Running a simulation](instrument-response.md) explains the configuration file, and the [configuration reference](configuration.md) lists every setting.

The [worked example](worked-example.ipynb) goes through a full run with a real simulation, from downloading it to a map of the measured intensity. [Where to start](index.md#where-to-start) lists the other starting points.
