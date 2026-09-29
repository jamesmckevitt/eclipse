# Quick start

ECLIPSE works in two steps. First `synthesise-spectra` turns a model of the solar atmosphere into the spectra it emits. Then `eclipse` passes those spectra through the instrument, adds noise, and fits the lines as you would fit real data.

## Command line interface

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

Both commands list their options with `--help`, and [Simulating a single snapshot](instrument-response.md) describes every setting in the configuration file.

You don't need an MHD simulation. You can start from an observed DEM instead ([From a DEM](dem-synthesis.md)). If you only want to know how precisely a line of a given brightness can be measured, start from a single intensity ([From a single intensity](uniform-intensity.md)), which skips `synthesise-spectra` altogether.

## Python API

Everything is also available as a Python package, `euvst_response`. For example, the effective area at Fe XII 195.119 Angstrom:

```python
import astropy.units as u
from euvst_response import Detector_SWC, Telescope_EUVST

telescope = Telescope_EUVST()
detector = Detector_SWC()
wavelength = 195.119 * u.AA

# Collecting area, times the throughput of the mirror (with its roughness), grating and filter, times the detector's quantum efficiency
effective_area = telescope.ea_and_throughput(wavelength) * detector.qe_euv
print(effective_area.to(u.cm**2))

# Each part on its own
mirror = telescope.primary_mirror_efficiency(wavelength)
roughness = telescope.microroughness_efficiency(wavelength)
grating = telescope.grating_efficiency(wavelength)
filter_throughput = telescope.filter.total_throughput(wavelength)
```

## Working with results

`eclipse` writes its results to `run/result/<config name>.h5`, so here `run/result/config.h5`. To load them:

```python
import astropy.units as u
from euvst_response import load_instrument_response_results, get_results_for_combination, summary_table

results = load_instrument_response_results("run/result/config.h5")

# One row for each combination of settings in the run, and the version of ECLIPSE that made it
summary_table(results)

# The results for the 40 s exposures. Settings are named by their section and key in the configuration file.
combo = get_results_for_combination(results, **{"simulation.expos": 40 * u.s})
```

The [worked example](worked-example.ipynb) goes the whole way, from an MHD snapshot to a map of the measured intensity.
