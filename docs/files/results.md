# Results files

`eclipse` writes its results to `run/result/<config name>.h5`. They include:

- The first iteration's signals, in DN and in photons
- For each fitted component, by name, in each pixel: its first fit, the mean and standard deviation of its intensity, velocity and width over the iterations, and the number of fits that failed
- The truth to compare against: the fit to the spectra on the detector's pixels, with no noise and no PSF, weighted as the DN fits are
- Whether the fits were weighted, as `fit_weighted`
- The settings (`Detector`, `Telescope`, `Simulation`) of each combination
- The version of ECLIPSE and the git commit that made them

Read them with `load_instrument_response_results`, as described in [Analysing an observation](../analyse/observation.md).
