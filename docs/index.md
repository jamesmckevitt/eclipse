# ECLIPSE

**E**mission **C**alculation and **Li**ne **P**rediction for **S**OLAR-C **E**UVST

ECLIPSE predicts what the EUV spectrograph EUVST on SOLAR-C will measure, and how precisely, starting from a model of the solar atmosphere. It also models Hinode/EIS.

Contact: James McKevitt (jm2@mssl.ucl.ac.uk). See [LICENSE](https://github.com/jamesmckevitt/eclipse/blob/master/LICENSE) for usage terms.

The instrument model will be updated as EUVST is built, tested and commissioned.

[![PyPI](https://img.shields.io/pypi/v/solarc-eclipse.svg)](https://pypi.org/project/solarc-eclipse/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17543844-blue.svg)](https://doi.org/10.5281/zenodo.17543844)

## How ECLIPSE works

A run has three stages:

```mermaid
flowchart LR
    MHD["MHD simulation"]
    DEM["Observed DEM"]
    UNI["Single intensity"]

    MHD --> VDEM["VDEM"]

    subgraph SYN["1. Synthesise an atmosphere"]
        ECL["ECLIPSE synthesis<br>optically thin, G(T, n_e)"]
        EXT["Any other synthesis code<br>Lightweaver, RH1.5D, ...<br>optically thin or thick"]
    end

    subgraph SIM["2. Simulate the instrument"]
        INS["Optics, detector and noise,<br>then line fitting"]
    end

    subgraph ANL["3. Analyse the results"]
        ANA["Measured against truth"]
    end

    MHD --> ECL
    MHD --> EXT
    VDEM --> ECL
    DEM --> ECL

    ECL -- "synthesis file" --> INS
    EXT -- "synthesis file" --> INS
    UNI --> INS
    INS -- "results file" --> ANA

    classDef node fill:#ECECFF,fill-opacity:1,stroke:#9370DB,stroke-opacity:1,color:#1a1a1a;
    class MHD,DEM,UNI,VDEM,ECL,EXT,INS,ANA node
    style SYN fill:#fdf6d8,fill-opacity:1,stroke:#c9b46a
    style SIM fill:#fdf6d8,fill-opacity:1,stroke:#c9b46a
    style ANL fill:#fdf6d8,fill-opacity:1,stroke:#c9b46a
```

**1. Synthesise an atmosphere.** Turn a model of the emitting plasma into the spectra it emits, as they leave the Sun. There are several ways to do this, depending on what you start from:

| Starting point | Page | Use it when |
| --- | --- | --- |
| A 3D MHD simulation | [From an MHD simulation](synthesis.md) | You have a numerical model of the atmosphere and want realistic spatial structure and Doppler shifts. The simulation goes in as an [atmosphere file](synthesis.md#atmosphere-files), which you write from your code's output. |
| A VDEM from an MHD simulation | [From a VDEM](vdem-synthesis.md) | The simulation has already been reduced to its emission measure as a function of temperature and line-of-sight velocity, a VDEM. |
| An observed DEM | [From a DEM](dem-synthesis.md) | You have a differential emission measure (DEM) inferred from real data. A DEM has no velocities, so the lines are not Doppler shifted. |
| Spectra from any other code | [From another code](other-codes.md) | You have already synthesised the spectra with another code, such as the optically thick codes Lightweaver or RH1.5D, and want only the instrument simulation. |
| A single intensity | [From a single intensity](uniform-intensity.md) | You only want to know how precisely a line of a given brightness can be measured. |

**2. Simulate the instrument.** Pass those spectra through the telescope, filter, grating and detector, add the noise, and fit the lines as you would fit real data. Repeating this many times with new noise each time (a Monte Carlo simulation) shows how much the measured intensities, velocities and widths scatter. See [Simulating a single snapshot](instrument-response.md).

To observe a series of snapshots as a raster or a sit-and-stare, so that each exposure sees the atmosphere at its own time, see [Simulating a time series](time-series.md). The snapshots can be synthesis files, or atmosphere files, which are then synthesised one exposure at a time.

**3. Analyse the results.** Find how precisely the instrument measured each line, compare the measurements with the truth, compare the settings the run swept through, and make maps. The [worked example](worked-example.ipynb) goes from an MHD snapshot to a map of the measured intensity.

## Instruments

- `SWC` - SOLAR-C/EUVST-SW, the short wavelength channel. This is the default.
- `EIS` - Hinode/EIS.
- The EUVST long wavelength channel (LW) is coming soon, and will complete EUVST.

The instrument is chosen with the `instrument:` key at the top level of the configuration file.

## Installation

### From PyPI (recommended)

```bash
pip install solarc-eclipse
```

### From source

```bash
pip install git+https://github.com/jamesmckevitt/eclipse.git
```

ECLIPSE computes line emission with [fiasco](https://fiasco.readthedocs.io/), which needs its own copy of the CHIANTI atomic database. The first time ECLIPSE needs the database, fiasco asks whether to download and build it. The download is about 600 MB and the build takes several minutes, so the first run is slower than later ones.

## Citing ECLIPSE

If ECLIPSE contributed to a publication, please cite both the paper describing the instrument model and the version of ECLIPSE you ran:

- [McKevitt et al. (2026), PASJ 78, 1524](https://academic.oup.com/pasj/article/78/4/1524/8731000)
- The Zenodo DOI for the release. [10.5281/zenodo.17543844](https://doi.org/10.5281/zenodo.17543844) always points to the latest version, and each release has its own DOI.

Every results file records the version and git commit that made it, and `summary_table(results)` prints them.

## Where to go next

- [Quick start](quickstart.md) - the command line and Python basics
- [From an MHD simulation](synthesis.md) - every option of `synthesise-spectra`
- [From a DEM](dem-synthesis.md) - start from an observed DEM instead of an MHD simulation
- [From a VDEM](vdem-synthesis.md) - start from a simulation already reduced in temperature and velocity
- [From a single intensity](uniform-intensity.md) - no atmosphere, just one line
- [Simulating a single snapshot](instrument-response.md) - every setting in the configuration file, and the `eclipse` command
- [Simulating a time series](time-series.md) - a raster or sit-and-stare over a series of atmosphere or synthesis files
- [From another code](other-codes.md) - the instrument on spectra another code synthesised
- [Worked example](worked-example.ipynb) - a notebook taking an MHD snapshot through to a map of the measured intensity
