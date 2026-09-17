# Science cases

ECLIPSE includes the science cases from the Concept Study Report (CSR), each with every line it uses, as a fixed set of observations to test changes to the instrument, or to ECLIPSE, against. `eclipse-science-cases` writes ECLIPSE configurations for them:

```bash
eclipse-science-cases --out run/input
eclipse --config run/input/1.1.1-nanoflares_events_fe12_195119.yaml
```

That writes one configuration for each case and line ECLIPSE can simulate, in [uniform intensity mode](uniform-intensity.md). The intensity is the line's intensity in that case times the case's filling factor, the thermal width is the one at the line's formation temperature, and the case sets the exposure time and slit width. Every configuration has the PSF on, runs 512 Monte Carlo iterations, and compares off-chip binning of 1 and 2 slit pixels, with the instrument at the ECLIPSE defaults.

To choose cases and lines, or change the settings:

```bash
eclipse-science-cases --list                                   # every case and line, and which can be simulated
eclipse-science-cases --case I-1-1 --case I-4-1                # only these cases
eclipse-science-cases --line "Fe XII" --line "Fe IX 171.073"   # only these lines, in every case that has them
eclipse-science-cases --base settings.yaml                     # add these settings to every configuration
```

The `--base` file is a configuration without a line:

```yaml
n_iter: 100
telescope:
  microroughness_sigma: 0.6 nm
filter:
  c_thickness: 40 angstrom
```

Its settings replace the defaults. A `simulation` section in it can also replace a case's exposure time or slit width, for example to sweep exposure times.

| Case | Science | Target | Slit (arcsec) | Exposure (s) | Filling factor |
|---|---|---|---|---|---|
| I-1-1 | Observe small scale heating events | Active region | 0.4 | 5 | 4 |
| I-1-2 | Observe evaporative upflows | Active region | 0.4 | 5 | 4 |
| I-1-3 | Search for evidence of braiding | Active region | 0.4 | 5 | 4 |
| I-2-1 | Detect Alfven waves | Quiet off limb | 0.8 | 2.5 | 1 |
| I-2-2 | Observe wave dissipation | Active region | 0.8 | 2.5 | 1 |
| I-3-1 | Observe spicule heating | Active region plage | 0.4 | 2 | 1 |
| I-4-1 | Observe solar wind source regions | Active region outflows | 0.8 | 6 | 0.2 |
| I-4-2 | Detect coronal Alfven waves | Coronal hole limb | 0.8 | 100 | 1 |
| II-1-1 | Probe flare reconnection | Flare | 0.4 | 3 | 1 |
| II-1-2 | Probe flare ribbons in reconnection events | Flare | 0.4 | 0.5 | 1 |
| II-1-3 | Probe reconnection in the chromosphere | Active region | 0.4 | 0.5 | 4 |
| II-2-1 | Energy buildup | Active region | 1.6 | 0.5 | 4 |
| II-2-2 | Eruption physics | Active region | 0.4 | 0.5 | 4 |

The cases and their lines are in `euvst_response/data/science_cases.yaml`.
