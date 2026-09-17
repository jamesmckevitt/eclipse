# Science cases

The `science_cases` folder in the repository has a configuration for each science case in the Concept Study Report (CSR), one for each of its lines in the short wavelength channel. They are a fixed set of observations to test changes to the instrument, or to ECLIPSE, against. Run one from the top folder of the repository:

```bash
eclipse --config science_cases/1.1.1-nanoflares_events_fe12_195119.yaml
```

Each configuration uses [uniform intensity mode](uniform-intensity.md). The intensity is the line's intensity in that case multiplied by the case's filling factor, and the thermal width is the one at the line's formation temperature. The case sets the slit width and exposure time. Every configuration has the PSF on, runs 512 Monte Carlo iterations, and compares off-chip binning of 1 and 2 slit pixels. The instrument is left at the ECLIPSE defaults, so add a `detector`, `telescope` or `filter` section to try a different one.

| Case | Science | Target | Slit (arcsec) | Exposure (s) | Filling factor | Lines (Angstrom) |
|---|---|---|---|---|---|---|
| I-1-1 | Observe small scale heating events | Active region | 0.4 | 5 | 4 | Fe IX 171.073, Ca XIV 193.874, Fe XII 195.119 |
| I-1-2 | Observe evaporative upflows | Active region | 0.4 | 5 | 4 | Fe IX 171.073, Ca XIV 193.874, Fe XII 195.119 |
| I-1-3 | Search for evidence of braiding | Active region | 0.4 | 5 | 4 | Fe IX 171.073, Ca XIV 193.874, Fe XII 195.119 |
| I-2-1 | Detect Alfven waves | Quiet off limb | 0.8 | 2.5 | 1 | Fe IX 171.07, Fe X 174.53, Fe X 177.24, Fe XI 180.40, Fe XII 195.12 |
| I-2-2 | Observe wave dissipation | Active region | 0.8 | 2.5 | 1 | Fe IX 171.073, Ca XIV 193.874 |
| I-3-1 | Observe spicule heating | Active region plage | 0.4 | 2 | 1 | Fe IX 171.073 |
| I-4-1 | Observe solar wind source regions | Active region outflows | 0.8 | 6 | 0.2 | Fe IX 171.073, Fe X 174.531, Fe XI 180.408, Fe XII 195.119 |
| I-4-2 | Detect coronal Alfven waves | Coronal hole limb | 0.8 | 100 | 1 | Fe IX 171.07, Fe X 174.53, Fe X 177.24, Fe IX 177.59, Fe VIII 185.21 |
| II-1-1 | Probe flare reconnection | Flare | 0.4 | 3 | 1 | Fe XXIV 192.03 |
| II-1-2 | Probe flare ribbons in reconnection events | Flare | 0.4 | 0.5 | 1 | Fe XXIV 192.03, Ca XVII 192.85 |
| II-1-3 | Probe reconnection in the chromosphere | Active region | 0.4 | 0.5 | 4 | Fe IX 171.073 |
| II-2-1 | Energy buildup | Active region | 1.6 | 0.5 | 4 | Fe IX 171.073, Fe XII 195.119 |
| II-2-2 | Eruption physics | Active region | 0.4 | 0.5 | 4 | Fe IX 171.073 |

The cases' lines outside the short wavelength channel, 170 to 210 Angstrom, have no configuration.
