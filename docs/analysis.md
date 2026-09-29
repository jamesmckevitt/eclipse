# Analysing the results

`eclipse` writes its results to `run/result/<config name>.h5`. ECLIPSE's analysis functions read them, and pick out the measurements you need.

## Loading the results

```python
from euvst_response import load_instrument_response_results, summary_table

results = load_instrument_response_results("run/result/config.h5")
summary_table(results)
```

`summary_table` prints the version of ECLIPSE and the git commit that made the results, the settings of every combination, which settings were swept, and the names of the fitted components.

## Picking a combination

A run makes one set of results for each combination of the settings it swept through. `get_results_for_combination` picks one out, by the names `summary_table` prints: the section and key, such as `simulation.expos`. Python keyword arguments can't have a dot in them, so they are given as a dictionary:

```python
import astropy.units as u
from euvst_response import get_results_for_combination

combination = get_results_for_combination(results, **{
    "simulation.expos": 20 * u.s,
    "simulation.slit_width": 0.2 * u.arcsec,
})
```

Values with units need their units. If what you give matches more than one combination, or none, it stops with an error saying so. `get_parameter_combinations(results)` lists the settings of every combination.

## How precisely each line was measured

```python
from euvst_response import analyse_fit_statistics

stats = analyse_fit_statistics(combination)
print(stats["v_std"])
```

`analyse_fit_statistics` gives maps, with one value per pixel, of the fitted line's:

| Key | What it is |
| --- | --- |
| `v_first`, `w_first`, `i_first` | The velocity, width and intensity from the first Monte Carlo iteration |
| `v_mean`, `w_mean`, `i_mean` | Their means over the iterations |
| `v_std`, `w_std`, `i_std` | Their standard deviations over the iterations: how precisely they were measured |
| `v_true`, `w_true` | The true velocity and width |
| `v_err` | The true velocity minus the mean measured velocity |
| `failed_fits`, `n_iterations` | The number of fits that failed, and the number of iterations |

The width is the Gaussian's sigma, and the intensity is the fitted line's counts. Failed fits are left out of the means and standard deviations.

By default these are the fits to the signal in DN. `data_type="photon"` gives the fits to the photons arriving at the detector instead, if the run fitted them (see [Which signals are fitted](fitting.md#which-signals-are-fitted)). For a [blend](fitting.md), `component=` chooses the component, by a name from `list_fit_components(combination)`, and defaults to the primary component.

## The truth

The truth is what ECLIPSE fits to the spectra on the detector's pixels, with no noise and no PSF. `v_true` and `w_true` come from that fit. `v_err` then says how far the measured velocity is off on average, where `v_std` says how much it scatters.

## Maps

`create_sunpy_maps_from_combo` makes SunPy maps of a combination's results:

```python
from euvst_response import create_sunpy_maps_from_combo

maps = create_sunpy_maps_from_combo(combination, date_obs="2024-03-20T00:00:00")
maps["velocity_std"].peek()
```

A synthesised scene has no date of its own, so the maps need one from you, as `date_obs`. The only exception is an EIS run with a dated calibration, whose date is used. The maps are:

- `total_dn` and `total_photons`: the first iteration's signal, summed over wavelength
- `velocity_from_fit`, `line_width_from_fit` and `intensity_from_fit`: the first iteration's fit
- `velocity_mean`, `line_width_mean` and `intensity_mean`: the means over the iterations
- `velocity_std`, `line_width_std` and `intensity_std`: the standard deviations over the iterations
- `velocity_true` and `velocity_err`: the true velocity, and the true velocity minus the mean
- `failed_fits`: the number of fits that failed in each pixel

It takes `data_type` and `component` as `analyse_fit_statistics` does. Given the combinations at each exposure time as `exposure_time_results`, it also makes an `exposure_time` map: the shortest of those exposures at which each pixel's velocity is measured to within `precision_requirement`, 2 km/s by default.

The [worked example](worked-example.ipynb) plots one of these maps.

## Every iteration

With `save_iterations: true` in the [fitting section](fitting.md#the-fit), the results also keep every iteration's fit. `combination["dn_fit_stats"]["iterations"]` holds `fit_data`, every fitted value of every iteration, and `failed`, which marks the fits that failed.

## The synthesis behind the results

The DEM and the emission measure by temperature and velocity are in the synthesis file, not in the results. `read_synthesis_products` reads them; see [Files](files.md#synthesis-files).

For a [time series](time-series.md), `results["raster"]` holds the observing plan, the files and their times, and the cube each combination observed. With more than one raster, `raster.repeat` picks one out like any other setting.

Results files from ECLIPSE 0.11.0 and earlier were pickles; see [Older versions](older-versions.md).
