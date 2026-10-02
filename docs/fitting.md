# Fitting blended lines

ECLIPSE fits each measured spectrum as you would fit real data, with a Gaussian on a flat background, and reports the line's intensity, velocity and width. When other lines fall in the same spectral window, it can fit several Gaussians together instead. The settings go in a `fitting` section of the configuration file.

## Several components

To fit a blend, list its lines under `components`:

```yaml
fitting:
  primary_component: 0           # index of the component whose velocity is reported
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

!!! warning "The primary component must be present in the data"

    ECLIPSE reports the velocity of `primary_component`. The fit starts with all of the components moved together, to where they best match the profile. So if the line you asked for is not in your data, the fit can settle a whole component spacing away: 92 km/s for Fe XII 195.119 and 195.179. A few per cent of the blend is enough to place it correctly.

    Also, two lines of similar brightness closer than about three line widths are often fitted as one broad component instead of two. The dip between them never falls below half the peak, so the fit starts with a width that covers the whole blend. `constrain_positive_intensity` does not help here, and can make it worse.

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
