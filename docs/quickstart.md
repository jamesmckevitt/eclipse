# Quick start

## Command line interface

After installation, you can run ECLIPSE from the command line:

```bash
# Run synthesis script (convert 3D MHD data to synthetic spectra)
synthesise-spectra --data-dir ./data/atmosphere --lines Fe12_195.1190 --output-dir ./run/input

# Run instrument response simulation
eclipse --config ./run/input/config.yaml

# Help can be accessed with
synthesise-spectra --help
eclipse --help
```

That is the two-stage pipeline: `synthesise-spectra` builds the atmosphere's
spectra, `eclipse` puts them through the instrument. See
[From an MHD simulation](synthesis.md) and
[Simulating the instrument](instrument-response.md) for the full set of options.

Starting from an observed DEM instead of an MHD cube? See
[From a DEM](dem-synthesis.md). Only want the precision on a line of known
brightness, with no atmosphere at all? See
[From a single intensity](uniform-intensity.md), which skips
`synthesise-spectra` entirely.

## Python API

You can also use ECLIPSE as a Python library:

```python
import astropy.units as u
import euvst_response
from euvst_response import AluminiumFilter, Detector_SWC, Telescope_EUVST

telescope = Telescope_EUVST()
detector = Detector_SWC()

print(f"Telescope collecting area: {telescope.collecting_area:.4f}")
print(f"Detector QE (EUV): {detector.qe_euv:.2f}")

# Calculate effective area at Fe XII 195.119 Angstrom
fe12_wl = 195.119 * u.AA
effective_area = telescope.collecting_area * telescope.throughput(fe12_wl) * detector.qe_euv

# Get breakdown of throughput by component
pm_eff = telescope.primary_mirror_efficiency(fe12_wl)
grating_eff = telescope.grating_efficiency(fe12_wl)
micro_eff = telescope.microroughness_efficiency(fe12_wl)
filter_eff = telescope.filter.total_throughput(fe12_wl)
```

## Working with results

For analyzing simulation results, see the [analysis tutorial](tutorial.ipynb), which demonstrates how to:

- Load simulation results
- Explore parameter combinations
- Analyze fit statistics and compute velocity/line width errors
- Create SunPy maps for visualization

The analysis functions are available directly from the package:

```python
from euvst_response import (
    load_instrument_response_results,
    get_results_for_combination,
    analyse_fit_statistics,
    create_sunpy_maps_from_combo,
    summary_table
)

# Load results
results = load_instrument_response_results("run/result/my_run.pkl")

# Print a summary table (auto-discovers all parameters and shows git commit)
summary_table(results)

# Retrieve a specific combination using full section.attribute names
combo = get_results_for_combination(results, **{"simulation.expos": 40*u.s, "simulation.psf": True})
```
