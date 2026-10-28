# Fitting blended lines

ECLIPSE fits each measured spectrum as you would fit real data, with a Gaussian on a flat background, and reports the line's intensity, velocity and width. When other lines fall in the same spectral window, it can fit several Gaussians together instead. The settings go in a `fitting` section of the configuration file.

## Several components

To fit a blend, list its lines under `components`:

```yaml
fitting:
  primary_component: 0           # index of the component the analysis reports by default
  constrain_positive_intensity: true  # keep every amplitude at zero or above during the fit
  backend: scipy                 # optimiser: "scipy" (default) or "mpfit"
  max_iter: 1000                 # optimiser iterations before it gives up
  components:
    - wavelength: 195.119 angstrom     # component 0: free centre, width, amplitude
      name: Fe XII 195.119
    - wavelength: 195.179 angstrom     # component 1: centre & width tied to component 0
      name: Fe XII 195.179
      tie_center: 0
      tie_width: 0
```

Each entry in `components` is one Gaussian, and needs a `wavelength`, its rest wavelength. It can also have:

- `tie_center: <i>`: fit this component at the same velocity as component *i*. Its centre is the other's scaled by the ratio of their rest wavelengths, so one velocity holds across the whole window.
- `tie_width: <i>`: fit this component with the same line width as component *i*.
- `amplitude_greater_than: <i>`: keep this component's amplitude above that of component *i*.
- `name: <text>`: the component's name in the results. It defaults to its rest wavelength, such as `195.1190 Angstrom`.

Without `components`, a single Gaussian is fitted. A `components` list needs at least two entries, since one on its own is not a blend.

The results hold the intensity, velocity and width of every component, by its name. `primary_component` is the one the analysis functions give unless asked for another. Results files from older versions of ECLIPSE may hold the primary component alone, and `list_fit_components` then says so. [Every component of a blend](analysis.md#every-component-of-a-blend) shows how to get each.

!!! warning "The primary component must be present in the data"

    The analysis reports `primary_component` unless asked for another. The fit starts with all of the components moved together, to where they best match the profile. So if that line is not in your data, the fit can settle a whole component spacing away: 92 km/s for Fe XII 195.119 and 195.179. In a noisy spectrum this can also happen when the line is there but the others are faint; see [A faint blend](#a-faint-blend).

    Also, two lines of similar brightness closer than about three line widths are often fitted as one broad component instead of two. The dip between them never falls below half the peak, so the fit starts with a width that covers the whole blend. `constrain_positive_intensity` does not help here, and can make it worse.

## A faint blend

In a noisy spectrum, a faint line of a blend says little about where the blend is. Fe XII 195.179, for example, can be a twentieth as bright as 195.119 beside it. Some fits then put 195.119 on nothing and 195.179 on the bright line, a whole line spacing from the truth, or put the two either side of it. eispac's templates avoid this by starting every fit at the lines' rest wavelengths, and by holding each line's centre and width within limits. `start: rest` and each component's `velocity_limits` and `width_limits` do the same. These are the settings of eispac's `fe_12_195_119.2c` template:

```yaml
fitting:
  start: rest
  constrain_positive_intensity: true
  components:
    - wavelength: 195.119 angstrom
      velocity_limits: [-61.5 km/s, 61.5 km/s]
      width_limits: [0.0191 angstrom, 0.0510 angstrom]
    - wavelength: 195.179 angstrom
      tie_center: 0
      tie_width: 0
```

- `start: rest` starts each component at its rest wavelength, with its width measured from the spectrum. The default, `start: search`, moves the components together to where they best match the spectrum first.
- `velocity_limits`: keep the component's velocity, measured from its own rest wavelength, between the two values.
- `width_limits`: keep the component's Gaussian sigma between the two values.

A component tied to another follows that one's limits, so give the limits to the free one. A fit that would start outside the limits starts at the nearest limit. A line moving faster than the limits is fitted at a limit, so set them wider than the flows you expect.

## The fit

`max_iter` limits how many iterations the optimiser may take on one spectrum. It defaults to 1000. A fit that runs out, or fails for any other reason, is left out of the mean and standard deviation, and the run prints how many fits failed and in how many pixels.

`backend` chooses the optimiser, `scipy` or `mpfit`. Without it, ECLIPSE uses `scipy`.

`bessel_correction: true` computes the standard deviation over the Monte Carlo iterations with n - 1 in place of n (Bessel's correction).

`save_iterations: true` keeps every iteration's fitted values in the results, as well as their statistics. The results are then about `n_iter` times larger.

## Which signals are fitted

By default ECLIPSE fits two signals at every Monte Carlo iteration: the detector's output in DN, and the photons arriving at the detector. If you only need one, `fit_signals`, at the top level of the configuration file, saves time:

```yaml
fit_signals: dn   # "dn" fits only the DN signal
                  # "photon" fits only the photon signal
                  # "both" fits both, and is the default
```
