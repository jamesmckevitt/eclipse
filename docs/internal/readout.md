---
search:
  exclude: true
---

# Reading out without a shutter

The SW camera has a mechanical shutter that keeps the CCDs dark while a frame is cleared and read. If it is not there, or does not close, the chip stays illuminated throughout, and every charge packet collects light from each row it is clocked through on the way to the serial register. A bright line therefore appears again, faintly, in every row between it and the register. `euvst_response.readout` models that.

This is a Python API rather than a configuration key: the read-out depends on the window layout of a particular observation, which the YAML does not describe.

## The focal plane

The two CCDs are butted along the dispersion with their serial registers on the outer edges, so a row is one wavelength, the columns of a row run along the slit, and charge is clocked along the dispersion. Rows count from the register, so row 0 is the outermost row of a device and row 2047 sits at the butted edge.

```python
import astropy.units as u
from euvst_response.readout import FocalPlane_SWC

fp = FocalPlane_SWC()

fp.wavelength(0, "left")                        # 163.55 A, the outermost row
fp.row_of_wavelength(195.119 * u.Angstrom)      # ('left', 1861.7)
fp.lit_rows("right")                            # (1290, 2047)
fp.row_edges("left")                            # 2049 wavelengths, for binning a spectrum onto rows
```

The wavelength of each row comes from a quadratic fitted to the focal plane positions in RSC-2022021C. The spacing is not the same everywhere: it runs from 17.0 mA per row at the short-wavelength end to 16.8 at the long one, so a single 16.9 would misplace a line by up to nine rows at the gap.

- `wavelength_offset`: added to every row, for an as-built or in-flight wavelength calibration. The design positions are good to about a row, but the camera's alignment to the beam is quoted at +/-0.79 Angstrom, some 47 rows.
- `lit_band`: the wavelengths between which light reaches the chip. Outside them a baffle vignettes the beam, which leaves 380 dark rows on the left CCD and 1290 on the right. Charge from the lit rows is clocked across all of them.
- `slit_image_tilt`: how far a line drifts along the dispersion between the centre of the slit and its ends. By default a line lands on one row for the whole slit, and the `column` argument these methods take makes no difference. Pass `MEASURED_SLIT_IMAGE_TILT` to model the drift the optical design shows, which is about two rows end to end:

```python
from euvst_response.readout import MEASURED_SLIT_IMAGE_TILT

tilted = FocalPlane_SWC(slit_image_tilt=MEASURED_SLIT_IMAGE_TILT)
tilted.row_of_wavelength(195.119 * u.Angstrom, column=1904)   # ('left', 1863.3)
```

!!! warning "The dark rows are an assumption, not a measurement"

    The drawings do not give the baffle's dimensions, only that its edge is within a few rows of the band limits. `lit_rows` is therefore uncertain by about four rows on the left CCD and seven on the right, and the real vignetting is gradual rather than a step.

## The read-out

```python
from euvst_response.readout import ReadoutSequence, windows_from_wavelengths

windows = windows_from_wavelengths(fp, [(194.9 * u.Angstrom, 195.3 * u.Angstrom)])
sequence = ReadoutSequence(shutter=False, windows=windows)

sequence.line_read_time       # 547 us, the register read of one row
sequence.readout_duration(2048)
```

A frame is cleared, exposed, then read row by row. Rows inside a window go through the serial register; the rest are dumped through the dump drain, which costs only the row transfer.

- `shutter`: `True` keeps the chip dark outside the exposure, which is what ECLIPSE assumes everywhere else. `False` is the case this module exists for.
- `row_transfer_time`: 15 us by default, and configurable in flight.
- `pixel_period`, `serial_prescan`, `serial_image_pixels`, `serial_overscan`: the register read, 50 + 1024 + 20 samples per output at 500 ns. Both scans are configurable in flight between 0 and 200.
- `parallel_overscan_rows`: rows clocked and read after the last image row. Without a shutter they hold pure smear, since their charge crosses the whole illuminated area on the way out, which makes them a direct measurement of it.
- `dump_rows`: transfers used to clear the image area before the exposure. The default clears a whole CCD.
- `windows`: inclusive row ranges to read. An empty list reads everything.

!!! warning "One row timeline covers both CCDs"

    The FEE clocks the two devices together, and a row wanted on either of them is read on both. Window rows are therefore row numbers without a CCD attached, and a window placed for a line on one CCD slows the read-out of the other.

## Putting light on it

`euvst_response.frame` turns a spectrum into the photons per second each row receives. It is the radiometric equation `radiometric` applies to a synthesis cube, with the radiance integrated between the row boundaries instead of multiplied by one pixel bandwidth, which matters here because the rows are not evenly spaced and a frame spans the whole band.

```python
from euvst_response.config import Detector_SWC, Telescope_EUVST
from euvst_response.frame import apply_spectral_psf, photons_from_lines, thermal_width
from euvst_response.radiometric import spectral_psf_reach

telescope, det, slit = Telescope_EUVST(), Detector_SWC(), 0.4 * u.arcsec
width = thermal_width(192.030 * u.Angstrom, 1.8e7 * u.K, 55.845 * u.u)     # Fe XXIV where it forms
margin = spectral_psf_reach(telescope, det, slit)       # rows past each end of the chip, for the blur

rows = photons_from_lines(fp, "left", telescope, slit,
                          [192.030] * u.Angstrom,
                          [5.3e4] * u.erg / (u.s * u.cm**2 * u.sr), [width],
                          lit_only=False, margin=margin)
rows = apply_spectral_psf(rows, telescope, det, slit, margin=margin).to_value(1 / u.s)
first, last = fp.lit_rows("left")                       # the baffle comes after the grating
rows[:first] = 0.0
rows[last + 1:] = 0.0
```

- `photons_from_lines`: a list of lines, each a Gaussian of the given 1-sigma width as the Sun emits it, integrated between the row boundaries so that its flux is conserved wherever it falls.
- `photons_from_spectrum`: a spectrum already on a wavelength grid, such as a continuum, integrated between the row boundaries by trapezium rule.
- Both take a `column` for a focal plane with the slit image tilt switched on. By default they zero the rows the baffle keeps dark themselves, but the baffle is after the grating, so a spectrum that is to be blurred is laid with `lit_only=False` and cut after the blur, as above.
- `apply_spectral_psf`: blurs the rows with the spectral response a synthesis through the same slit gets from `radiometric.apply_focusing_optics_psf`. The slit's image is part of it, so it widens with the slit: 2.54 rows of FWHM for the 0.2 arcsec slit and 3.35 for the 0.4 arcsec one. `spectral_psf` is `"quadrature"` (the default) or `"convolution"`, as in the configuration.
- Nothing is assumed beyond the rows `apply_spectral_psf` is given, and the chip's edge rows do receive light from just off it, from across the gap at the butted edge in particular. So the spectrum is laid with a `margin` of rows past each end, at least `spectral_psf_reach`, and `apply_spectral_psf` given the same margin returns the chip's own rows.

`expose` then takes the photon rate reaching each pixel and returns the photons a frame records, exposure and smear together.

```python
import numpy as np
from euvst_response.readout import expose

rate = np.repeat(rows[:, np.newaxis], fp.n_columns, axis=1)   # the same along the slit

frame = expose(rate, 1.0 * u.s, sequence)       # photons, including the parallel overscan rows
```

The frame has `parallel_overscan_rows` more rows than the image area. With a shutter, the smear is zero and those rows are empty.

`dark_current_time` gives how long each packet collects dark current, which is how long it spends in the image area: its part of the clear, the exposure, and the wait while the rows before it are read, or for a parallel overscan packet its crossing of the chip during the read-out. It is the same with a shutter as without one, and longest for the rows read last.

`detect` and `digitise` in `euvst_response.frame` take the frame through the detector stages of `radiometric`, to electrons and then DN, with two things a full-band frame needs: a dark current time for each row, and a photon energy for each pixel, since a 170 A photon liberates a fifth more electrons than a 212 A one. Without a shutter a pixel holds photons from every row its charge crossed, so its energy is the mean of what it holds. `expose` is linear in the rate, so `expose_with_wavelength` exposes the energy-weighted rate alongside the photons and gives each pixel the wavelength of that mean.

```python
from euvst_response.frame import detect, digitise, expose_with_wavelength
from euvst_response.readout import dark_current_time

wavelength = fp.wavelength(np.arange(fp.n_rows), "left")
frame, pixel_wavelength = expose_with_wavelength(rate, wavelength, 1.0 * u.s, sequence)

photons = np.random.poisson(frame)
electrons = detect(photons, pixel_wavelength, dark_current_time(1.0 * u.s, sequence, fp.n_rows), det)
dn = digitise(electrons, det)
```

## What is not modelled

- **Blooming.** A saturated line spills along the column, which is the same axis as the smear. The CCDs have no anti-blooming, and the full well, `Detector_SWC.full_well`, is 150 ke- (typical in non-inverted mode; 80 ke- at the least), below the 182 ke- the FEE accepts, so in a flare a pixel fills before the digitiser does. Nothing is clipped or spilled at the full well: it says which pixels a frame would saturate.
- **The shutter in motion.** The shutter takes about 30 ms to open and the same to close, and light falls during both.
- **Vignetting shape.** The edge of the illuminated area is treated as a step.
- **Dark current in the serial register.** A packet collects dark current while it is in the image area, not while it is read.
