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
