# From another code

To simulate the instrument response to spectra that another code has synthesised, write them as a synthesis file first. These can then be observed as a [single snapshot](instrument-response.md) or as a [time series](time-series.md#from-synthesis-files).

## The synthesis file

The file is laid out much like an [atmosphere file](synthesis.md#atmosphere-files). The root has a `format` attribute of `eclipse-synthesis`, a `version` of `1`, and optionally a `source` saying where the spectra came from. Each dataset has a `unit` attribute that astropy can read.

| Dataset | Shape | What it holds |
| --- | --- | --- |
| `x_edges` | `(nx + 1,)` | The pixel boundaries across the slit, evenly spaced |
| `y_edges` | `(ny + 1,)` | The pixel boundaries along the slit, evenly spaced |
| `time` | scalar | The time of the snapshot, which only a time series needs |
| `lines/<name>/intensity` | `(ny, nx, n_wavelength)` | The spectral radiance at each pixel and wavelength |
| `lines/<name>/wavelength` | `(n_wavelength,)` | The wavelengths, increasing |
| `lines/<name>/rest_wavelength` | scalar | The wavelength the line's Doppler shifts are measured from |

Each group under `lines` holds a line, named as `reference_line` names it in the instrument configuration. It can equally hold a whole spectral window with its blends, as most codes give it, and the blends are fitted with a `fitting` block as on the [single snapshot](instrument-response.md) page. ECLIPSE's own synthesis writes a group for each line, with its `atom` and `ion` as attributes, an `integration_axis` attribute on the root for the axis it looked along, and a `synthesis` group of what it worked out on the way, all of which a file from another code can leave out.

The intensity can be in any unit of spectral radiance, per wavelength or per frequency, in energy or in photons (e.g. `erg / (s cm2 sr Angstrom)`, `W / (m2 sr Hz)` and `ph / (s cm2 sr nm)`). The wavelengths don't have to be evenly spaced, so a grid that is denser in the line cores can be input. Each wavelength stands for the interval halfway to its neighbours, so the spacing should change gradually.

x runs across the slit, the direction a raster steps in, and y runs along it. The edges can be lengths on the Sun, such as `Mm`, or angles as seen from 1 AU, such as `arcsec`. If your code gives pixel centres, `edges_from_centres` places the edges halfway between them.

## Writing one

```python
import astropy.units as u
from euvst_response import SpectralLine, Synthesis, edges_from_centres, write_synthesis

# intensity[y, x, wavelength] from your code, with its wavelengths and pixel centres
synthesis = Synthesis(
    lines={"Fe12_195.1190": SpectralLine(
        intensity=intensity * u.erg / (u.s * u.cm**2 * u.sr * u.AA),
        wavelength=wavelength * u.AA,
        rest_wavelength=195.119 * u.AA,
    )},
    x_edges=edges_from_centres(x * u.Mm),
    y_edges=edges_from_centres(y * u.Mm),
    source="My code, snapshot 1200",
    time=1200 * u.s,  # needed only for a time series
)
write_synthesis(synthesis, "my_code.h5")
```

## Worked example: FoMo

[FoMo](https://github.com/TomVeeDee/FoMo) ([Van Doorsselaere et al. 2016](https://doi.org/10.3389/fspas.2016.00004)) synthesises optically thin lines from an MHD simulation, from any viewing angle.

Make the FoMo synthesis:

```cpp
Object.setchiantifile("chiantitables/goft_table_fe_12_0195_abco.dat");
Object.setobservationtype(FoMo::Spectroscopic);
// pixels in x and y, points along the line of sight, wavelengths, and the window in m/s
Object.setresolution(128, 128, 2000, 121, 600000);
Object.render(0, M_PI);  // the viewing angles l and b: down from above, for a box with z up
```

A window of 600000 m/s reaches 300 km/s on both sides of the line, and 121 wavelengths put the points 5 km/s apart.

Then format the synthesis into a format ECLIPSE will accept:

```python
import gzip

import astropy.units as u
import numpy as np
from euvst_response import SpectralLine, Synthesis, edges_from_centres, write_synthesis


def read_fomo(path):
    """The points of a FoMo-C rendering: x, y, wavelength and intensity, one row each."""
    if ".txt" in str(path):
        # Seven header lines, then one line per point.
        return np.loadtxt(path, skiprows=7).T
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rb") as f:
        raw = f.read()
    # The version string ends at the first "#". Then come three ints, then
    # the units, the CHIANTI table and the abundance file, each after its
    # length, and then each column in single precision.
    position = raw.index(b"#") + 1
    dim, n_points, n_vars = np.frombuffer(raw, np.int32, 3, position)
    position += 12
    for _ in range(dim + n_vars + 2):
        position += 8 + int(np.frombuffer(raw, np.int64, 1, position)[0])
    return np.frombuffer(raw, np.float32, (dim + n_vars) * n_points,
                         position).reshape(dim + n_vars, n_points).astype(float)


def fomo_synthesis(path, x_pixel, y_pixel, lambda_pixel, line, rest_wavelength):
    """A FoMo-C rendering of *line* as a Synthesis, given the resolution it was rendered with."""
    x, y, wavelength, intensity = read_fomo(path)
    shape = (y_pixel, x_pixel, lambda_pixel)  # FoMo's own order: y, then x, then wavelength
    # FoMo's image is the mirror image of the view along its line of sight,
    # so x is reversed, and negated to keep it increasing.
    x = -x.reshape(shape)[0, ::-1, 0]
    y = y.reshape(shape)[:, 0, 0]
    wavelength = wavelength.reshape(shape)[0, 0, :]
    spectra = SpectralLine(
        # FoMo labels its spectra erg cm^-2 s^-1 A^-1, but they are per
        # steradian: its CHIANTI tables include the 1/(4 pi).
        intensity=intensity.reshape(shape)[:, ::-1, :] * u.erg / (u.s * u.cm**2 * u.sr * u.AA),
        # FoMo writes single precision, so its even grids come back slightly
        # uneven; they are rebuilt from their ends.
        wavelength=np.linspace(wavelength[0], wavelength[-1], lambda_pixel) * u.AA,
        rest_wavelength=rest_wavelength,
    )
    return Synthesis(
        lines={line: spectra},
        x_edges=edges_from_centres(np.linspace(x[0], x[-1], x_pixel) * u.Mm),
        y_edges=edges_from_centres(np.linspace(y[0], y[-1], y_pixel) * u.Mm),
        source=f"FoMo, {line}",
    )


# The rest wavelength is the one on the second line of the FoMo table.
synthesis = fomo_synthesis("fomo-output.txt", x_pixel=128, y_pixel=128, lambda_pixel=121,
                           line="Fe12_195.1190", rest_wavelength=195.119 * u.AA)
write_synthesis(synthesis, "fomo.h5")
```

FoMo writes text unless told otherwise, and binary with `setwriteoutbinary()`, as in its own example; `read_fomo` reads either, zipped or not. The file then goes into the instrument configuration as `synthesis_file: ./fomo.h5`.

## Worked example: PINTofALE

[PINTofALE](https://hea-www.harvard.edu/PINTofALE/) ([Kashyap & Drake 2000](https://ui.adsabs.harvard.edu/abs/2000BASI...28..475K)) computes line intensities for a DEM. It gives one intensity per line, with no line profile and nothing on the sky. If you just want your DEM observed, the [DEM route](dem-synthesis.md) does that with ECLIPSE's own atomic data, and is simpler. To keep PINTofALE's line list and atomic data, turn its lines into a synthesis file.

In IDL, with PINTofALE set up as usual:

```idl
; Every line in the window, with its ion balance, at n_e = 1e9 cm^-3
ff = rd_line(wrange=[194.8, 195.45], n_e=1e9, wvl=wvl, logT=logT, Z=Z, ion=ion, jon=jon)
ff = fold_ioneq(ff, Z, jon, logT=logT)

; Your DEM per kelvin [cm^-5 K^-1] at each temperature of logT
dem = 1d21 * exp(-0.5d * ((logT - 6.2d) / 0.15d)^2)

flx = lineflx(ff, 10d^logT, abs(wvl), Z, DEM=dem, /temp, abund=getabund('schmelz'), /noph)
save, file='pintofale_lines.sav', wvl, flx, Z, logT, ff, dem
```

Give the DEM per kelvin with `/temp`, as here. Without it, LINEFLX takes the DEM per unit natural log of T, which is T times the DEM per kelvin.

Then in Python:

```python
import astropy.constants as const
import astropy.units as u
import numpy as np
from mendeleev import element
from scipy.io import readsav
from scipy.special import erf
from euvst_response import SpectralLine, Synthesis, write_synthesis

poa = readsav("pintofale_lines.sav")
rest = np.abs(poa["wvl"]) * u.AA  # PINTofALE marks theoretical wavelengths negative
# LINEFLX gives what each line emits into all directions: a radiance is
# that over the 4 pi steradians.
radiance = poa["flx"] / (4 * np.pi) * u.erg / (u.s * u.cm**2 * u.sr)

# Where each line forms: its emissivity times the emission measure at each
# temperature, on the grid in log T.
weight = poa["ff"] * poa["dem"] * 10.0 ** poa["logt"]  # (line, temperature)
temperature = 10.0 ** poa["logt"] * u.K
mass = np.array([element(int(z)).atomic_weight for z in poa["z"]]) * u.u

# Each line is thermally broadened at every temperature it forms at, as in
# ECLIPSE's own synthesis, and integrated over each bin of a grid much finer
# than EUVST's pixels.
edges = np.arange(194.8, 195.45, 0.002) * u.AA
spectrum = np.zeros(edges.size - 1) * u.erg / (u.s * u.cm**2 * u.sr * u.AA)
for line in np.flatnonzero(poa["flx"] > 0):
    sigma = rest[line] * np.sqrt(const.k_B * temperature / mass[line]) / const.c
    z = ((edges[:, None] - rest[line]) / (np.sqrt(2) * sigma)).decompose().value
    share = weight[line] / weight[line].sum()
    profile = (0.5 * np.diff(erf(z), axis=0) * share).sum(axis=1) / np.diff(edges)
    spectrum += radiance[line] * profile

# The line to measure, at the wavelength PINTofALE has it, to all its digits:
# the line is placed there, so a rounded one would read as a Doppler shift.
measured = np.argmin(np.abs(rest - 195.119 * u.AA))

# A DEM has no structure on the sky, so the spectrum is laid over a patch
# a few slit widths across.
n = 20
synthesis = Synthesis(
    lines={"Fe12_195.1190": SpectralLine(
        intensity=np.tile(spectrum, (n, n, 1)),
        wavelength=0.5 * (edges[1:] + edges[:-1]),
        rest_wavelength=rest[measured],
    )},
    x_edges=np.arange(n + 1) * 0.1 * u.arcsec,
    y_edges=np.arange(n + 1) * 0.1 * u.arcsec,
    source="PINTofALE, Gaussian DEM",
)
write_synthesis(synthesis, "pintofale.h5")
```

The file then goes into the instrument configuration as `synthesis_file: ./pintofale.h5`.
