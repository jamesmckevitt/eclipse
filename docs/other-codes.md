# Spectra from another code

If another code has already synthesised the spectra, optically thick or thin, ECLIPSE can take them straight into the instrument simulation without a synthesis step of its own. The spectra go in as a spectra file, which holds the spectral radiance leaving the Sun at each pixel and wavelength, and ECLIPSE works out what EUVST would record from it.

## The spectra file

This is an HDF5 file laid out much like an [atmosphere file](synthesis.md#atmosphere-files). The root has a `format` attribute of `eclipse-spectra`, a `version` of `1`, and optionally a `source` saying where the spectra came from. Each dataset has a `unit` attribute that astropy can read.

| Dataset | Shape | What it holds |
| --- | --- | --- |
| `intensity` | `(ny, nx, n_wavelength)` | The spectral radiance at each pixel and wavelength |
| `wavelength` | `(n_wavelength,)` | The wavelengths, increasing |
| `x_edges` | `(nx + 1,)` | The pixel boundaries across the slit, evenly spaced |
| `y_edges` | `(ny + 1,)` | The pixel boundaries along the slit, evenly spaced |

The intensity can be in any unit of spectral radiance, per wavelength or per frequency, in energy or in photons: `erg / (s cm2 sr Angstrom)`, `W / (m2 sr Hz)` and `ph / (s cm2 sr nm)` all work. The wavelengths don't have to be evenly spaced, so a grid that is denser in the line cores, as Lightweaver and RH1.5D use, can go in as it is.

x runs across the slit, the direction a raster steps in, and y runs along it. The edges can be lengths on the Sun, such as `Mm`, or angles as seen from 1 AU, such as `arcsec`. If your code gives pixel centres, `edges_from_centres` places the edges halfway between them.

## Writing one

```python
import astropy.units as u
from euvst_response import Spectra, edges_from_centres, write_spectra

# intensity[y, x, wavelength] from your code, with its wavelengths and pixel centres
spectra = Spectra(
    intensity=intensity * u.erg / (u.s * u.cm**2 * u.sr * u.AA),
    wavelength=wavelength * u.AA,
    x_edges=edges_from_centres(x * u.Mm),
    y_edges=edges_from_centres(y * u.Mm),
    source="My code, snapshot 1200",
)
write_spectra(spectra, "spectra.h5")
```

Any other HDF5 writer works too, as long as the attributes and units are there.

## Observing it

In the instrument configuration, `spectra_file` takes the place of `synthesis_file`, and `rest_wavelength` gives the rest wavelength of the line whose velocity is measured:

```yaml
instrument: SWC
spectra_file: ./spectra.h5
rest_wavelength: 195.119 AA
n_iter: 100

simulation:
  slit_width: 0.4 arcsec
  expos: [10 s, 40 s]
  psf: True
```

```bash
eclipse --config spectra.yaml
```

The rest of the configuration is as for [a single snapshot](instrument-response.md). ECLIPSE resamples the spectra onto the detector's wavelength pixels, keeping the total intensity, and lays the pixels onto the slit and the plate scale as it does with its own synthesis.

The fit covers the whole wavelength range of the file, so give one spectral window per file, with room around the line for its Doppler shifts and the instrument's blurring. ECLIPSE's own windows reach 300 km/s either side of the line. Blends in the window are fitted with a `fitting` block, as for a synthesis file.

The Doppler shifts are taken as they are in the spectra, so a redshift should be a motion away from the observer.

## Worked example: FoMo

[FoMo](https://github.com/TomVeeDee/FoMo) ([Van Doorsselaere et al. 2016](https://doi.org/10.3389/fspas.2016.00004)) synthesises optically thin lines from an MHD simulation, from any viewing angle, and stops at the spectra leaving the Sun. This example takes a FoMo-C rendering the rest of the way.

Render the line as spectra, with a window that leaves room for the Doppler shifts and the instrument's blurring:

```cpp
Object.setchiantifile("chiantitables/goft_table_fe_12_0195_abco.dat");
Object.setobservationtype(FoMo::Spectroscopic);
// pixels in x and y, points along the line of sight, wavelengths, and the window in m/s
Object.setresolution(128, 128, 2000, 121, 600000);
Object.render(0, M_PI);  // the viewing angles l and b; see below
```

A window of 600000 m/s reaches 300 km/s either side of the line, as ECLIPSE's own windows do, and 121 wavelengths put the points 5 km/s apart, much finer than EUVST's pixels.

Then read the rendering into a spectra file:

```python
import gzip

import astropy.units as u
import numpy as np
from euvst_response import Spectra, edges_from_centres, write_spectra


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


def fomo_spectra(path, x_pixel, y_pixel, lambda_pixel):
    """A FoMo-C rendering as Spectra, given the resolution it was rendered with."""
    x, y, wavelength, intensity = read_fomo(path)
    shape = (y_pixel, x_pixel, lambda_pixel)  # FoMo's own order: y, then x, then wavelength
    # FoMo's image is the mirror image of the view along its line of sight,
    # so x is reversed, and negated to keep it increasing.
    x = -x.reshape(shape)[0, ::-1, 0]
    y = y.reshape(shape)[:, 0, 0]
    wavelength = wavelength.reshape(shape)[0, 0, :]
    return Spectra(
        # FoMo labels its spectra erg cm^-2 s^-1 A^-1, but they are per
        # steradian: its CHIANTI tables include the 1/(4 pi).
        intensity=intensity.reshape(shape)[:, ::-1, :] * u.erg / (u.s * u.cm**2 * u.sr * u.AA),
        # FoMo writes single precision, so its even grids come back slightly
        # uneven; they are rebuilt from their ends.
        wavelength=np.linspace(wavelength[0], wavelength[-1], lambda_pixel) * u.AA,
        x_edges=edges_from_centres(np.linspace(x[0], x[-1], x_pixel) * u.Mm),
        y_edges=edges_from_centres(np.linspace(y[0], y[-1], y_pixel) * u.Mm),
        source="FoMo, Fe XII 195.119",
    )


spectra = fomo_spectra("fomo-output.txt", x_pixel=128, y_pixel=128, lambda_pixel=121)
write_spectra(spectra, "fomo.h5")
```

FoMo writes text unless told otherwise, and binary with `setwriteoutbinary()`, as in its own example; `read_fomo` reads either, zipped or not.

In the configuration, `rest_wavelength` is the wavelength on the second line of the FoMo table the line was rendered with, 195.119 Angstrom for `goft_table_fe_12_0195_abco.dat`:

```yaml
spectra_file: ./fomo.h5
rest_wavelength: 195.119 AA
```

A few things to know about FoMo's side:

- FoMo shifts a line to the red for a flow along its line of sight, which points away from the observer. For a simulation with height along z, `render(0, 0)` looks up from below, and `render(0, M_PI)` looks down from above, with upflows blueshifted.
- FoMo renders one line at a time. For a window with blends, render each line on the same window and add them before writing the file.
- FoMo's tables come from an older CHIANTI than ECLIPSE's own synthesis uses. On a block of the [Bifrost snapshot](synthesis.md#worked-example-bifrost-quiet-sun) from the synthesis page, FoMo's Fe XII 195 came out about 25 per cent fainter than ECLIPSE's synthesis of the same cells with the same abundances, with the same Doppler shifts to within about 0.5 km/s.

## Worked example: PINTofALE

[PINTofALE](https://hea-www.harvard.edu/PINTofALE/) ([Kashyap & Drake 2000](https://ui.adsabs.harvard.edu/abs/2000BASI...28..475K)) computes line intensities for a DEM. It gives one intensity per line, with no line profile and nothing on the sky. If you just want your DEM observed, the [DEM route](dem-synthesis.md) does that with ECLIPSE's own atomic data, and is simpler. To keep PINTofALE's line list and atomic data, turn its lines into a spectra file.

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

Give the DEM per kelvin with `/temp`, as here. Without it, LINEFLX takes the DEM per unit natural log of T, which is T times the DEM per kelvin, so a DEM per unit log10 T would come out 2.3 times too bright.

Then in Python:

```python
import astropy.constants as const
import astropy.units as u
import numpy as np
from mendeleev import element
from scipy.io import readsav
from scipy.special import erf
from euvst_response import Spectra, write_spectra

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

# A DEM has no structure on the sky, so the spectrum is laid over a patch
# a few slit widths across.
n = 20
spectra = Spectra(
    intensity=np.tile(spectrum, (n, n, 1)),
    wavelength=0.5 * (edges[1:] + edges[:-1]),
    x_edges=np.arange(n + 1) * 0.1 * u.arcsec,
    y_edges=np.arange(n + 1) * 0.1 * u.arcsec,
    source="PINTofALE, Gaussian DEM",
)
write_spectra(spectra, "pintofale.h5")
```

In the configuration, `rest_wavelength` is the wavelength PINTofALE lists for the line, `abs(wvl)`, to all its digits. The line is placed there, so a rounded value reads as a Doppler shift: 195.119 instead of 195.1193 is 0.5 km/s.

```yaml
spectra_file: ./pintofale.h5
rest_wavelength: 195.1193 AA
```

PINTofALE's CHIANTI line database is from CHIANTI 7.1.2. With the same DEM and abundances, its intensities for thirteen lines from O V to Fe XXIV came out within about 5 per cent of ECLIPSE's own for most lines, and within 20 per cent for all of them.
