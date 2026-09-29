---
search:
  exclude: true
---

# Reading out without a shutter

The SW camera has a mechanical shutter that keeps the CCDs dark while a frame is cleared and read. Without it, or if it does not close, light keeps falling on the chip the whole time. Each packet of charge then picks up light from every row it is moved through on its way to the serial register, so a bright line also shows up faintly in other rows. During the read-out, the packets of the rows beyond the line pass through it on their way to the register. During the clear before the exposure, the packets that end up in the rows between the line and the register pass through it too. `euvst_response.readout` models this smear.

There is no configuration key for it, only a Python API, because the read-out depends on the window layout of a particular observation, which the configuration file does not describe.

## The focal plane

The two CCDs sit side by side along the dispersion, with their serial registers on the outer edges. So a row is one wavelength, the columns of a row run along the slit, and charge is moved along the dispersion. Rows are counted from the register: row 0 is the outermost row of each CCD, and row 2047 is at the edge where the two meet.

```python
import astropy.units as u
from euvst_response.readout import FocalPlane_SWC

fp = FocalPlane_SWC()

fp.wavelength(0, "left")                        # 163.55 A, the outermost row
fp.row_of_wavelength(195.119 * u.Angstrom)      # ('left', 1861.7)
fp.lit_rows("right")                            # (1290, 2047)
fp.row_edges("left")                            # 2049 wavelengths, for binning a spectrum onto rows
```

The wavelength of each row comes from a quadratic fitted to the focal plane positions in RSC-2022021C. The spacing changes along the chip, from 17.0 mA per row at the short-wavelength end to 16.8 at the long one. A single spacing of 16.9, from the middle, would put a line up to about seven rows out at the ends.

- `wavelength_offset`: added to every row's wavelength, for a wavelength calibration of the camera as built or in flight. The design positions are good to about a row, but the camera's alignment to the beam is quoted at +/-0.79 Angstrom, some 47 rows.
- `lit_band`: the wavelengths between which light reaches the chip. Outside them a baffle blocks the beam, which leaves 380 dark rows on the left CCD and 1290 on the right. Charge from the lit rows is moved across all of them.
- `slit_image_tilt`: how far a line moves along the dispersion between the middle of the slit and its ends. By default a line falls on the same row all along the slit, and the `column` argument these methods take makes no difference. Pass `MEASURED_SLIT_IMAGE_TILT` to model the shift the optical design shows, about two rows from end to end:

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

A frame is cleared, exposed, then read row by row. Rows inside a window are read through the serial register. The rest are dumped through the dump drain, which only takes the time of a row transfer.

- `shutter`: `True` keeps the chip dark outside the exposure, which is what ECLIPSE assumes everywhere else. `False` is the case this module is for.
- `row_transfer_time`: 15 us by default, and can be changed in flight.
- `pixel_period`, `serial_prescan`, `serial_image_pixels`, `serial_overscan`: the register read, 50 + 1024 + 20 samples per output at 500 ns. Both scans can be set between 0 and 200 in flight.
- `parallel_overscan_rows`: rows moved and read after the last image row. Without a shutter they hold only smear, since their charge crosses the whole lit area on the way out, so they measure it directly.
- `dump_rows`: the row transfers that clear the image area before the exposure. The default clears a whole CCD. Here a frame always starts from an empty chip, so with fewer transfers the charge that a partial clear would leave behind is missing, and ECLIPSE warns about it.
- `windows`: the ranges of rows to read, including both ends. An empty list reads every row.

!!! warning "One row timeline covers both CCDs"

    The front-end electronics (FEE) clock the two CCDs together, so a row wanted on either of them is read on both. Window rows are therefore just row numbers, with no CCD, and a window placed for a line on one CCD slows the read-out of the other.

## Putting light on it

`euvst_response.frame` turns a spectrum into the photons per second each row receives. It uses the same radiometric equation as the rest of ECLIPSE, but integrates the radiance between each row's edges rather than multiplying it by one pixel's bandwidth. That matters here, because the rows are not evenly spaced and a frame spans the whole band.

```python
from euvst_response.config import Detector_SWC, Telescope_EUVST
from euvst_response.frame import photons_from_lines, thermal_width

telescope, det, slit = Telescope_EUVST(), Detector_SWC(), 0.4 * u.arcsec
width = thermal_width(192.030 * u.Angstrom, 1.8e7 * u.K, 55.845 * u.u)     # Fe XXIV where it forms

rows = photons_from_lines(fp, "left", telescope, slit,
                          [192.030] * u.Angstrom,
                          [5.3e4] * u.erg / (u.s * u.cm**2 * u.sr), [width],
                          det=det, spectral_psf="quadrature").to_value(1 / u.s)
```

- `photons_from_lines`: a list of lines, each a Gaussian with the given 1-sigma width, as the Sun emits it. Each line is integrated between the rows' edges, so its total is kept wherever it falls.
- `photons_from_spectrum`: a spectrum already on a wavelength grid, such as a continuum, integrated between the rows' edges by the trapezium rule.
- Both take a `column`, for a focal plane with the slit image tilt switched on, and give zero in the rows the baffle keeps dark, unless given `lit_only=False`.
- Given `spectral_psf` and `det`, both blur the light as they lay it onto the rows, with the same spectral PSF that a synthesis through the same slit gets. A line narrower than a row then keeps its place within the row. The PSF includes the slit's image, so it widens with the slit: 2.54 rows (FWHM) for the 0.2 arcsec slit and 3.35 for the 0.4 arcsec one. `spectral_psf` is `"quadrature"` or `"convolution"`, as in the configuration file. A line just off the chip, such as one in the gap between the two CCDs, still reaches the edge rows through the PSF.
- `apply_spectral_psf` is deprecated. It blurred the rows after the light was laid on them, which moved a narrow line toward the middle of its row.

`expose` then takes the photon rate reaching each pixel, and returns the photons a frame records, from the exposure and the smear together.

```python
import numpy as np
from euvst_response.readout import expose

rate = np.repeat(rows[:, np.newaxis], fp.n_columns, axis=1)   # the same along the slit

frame = expose(rate, 1.0 * u.s, sequence)       # photons, including the parallel overscan rows
```

The frame has `parallel_overscan_rows` more rows than the image area. With a shutter, the smear is zero and those rows are empty.

`dark_current_time` gives how long each packet collects dark current, which is the time it spends in the image area. For an image row, that is its part of the clear, the exposure, and the wait while the rows before it are read. For a parallel overscan row, it is the time it takes to cross the chip during the read-out. It is the same with a shutter as without one, and longest for the rows read last.

`detect` and `digitise` in `euvst_response.frame` take the frame through the same detector stages as the rest of ECLIPSE, to electrons and then to DN. A full-band frame needs two things more: a dark current time for each row, and a photon energy for each pixel, since a 170 A photon frees a quarter more electrons than a 212 A one. Without a shutter, a pixel holds photons from every row its charge crossed, so its photon energy is the mean over what it holds. `expose_with_wavelength` works this out: it exposes the energy-weighted rate alongside the photons, and gives each pixel the wavelength of that mean energy.

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

- **Blooming.** A saturated pixel's charge spills along the column, the same direction as the smear. The CCDs have no anti-blooming. Their full well, `Detector_SWC.full_well`, is 150 ke- (typical in non-inverted mode; 80 ke- at the least), below the 182 ke- the FEE accepts, so in a flare a pixel fills before the digitiser does. ECLIPSE does not clip or spill charge at the full well; use it to find which pixels a frame would saturate.
- **The shutter in motion.** The shutter takes about 30 ms to open and the same to close, and light falls during both.
- **Vignetting shape.** The edge of the illuminated area is treated as a step.
- **Dark current in the serial register.** A packet collects dark current while it is in the image area, not while it is read.
