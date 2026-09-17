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

    No drawing dimensions the baffle. The documents put the edge within a few rows of the band limits, so `lit_rows` is good to about four rows on the left CCD and seven on the right, and the real vignetting is gradual rather than a step.

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

`expose` takes the photon rate reaching each pixel and returns the photons a frame records, exposure and smear together. Feed the result to the detector stages in `radiometric` in place of the exposure-only photon count.

```python
import numpy as np
from euvst_response.readout import expose

rate = np.zeros((fp.n_rows, fp.n_columns))      # photons per second per pixel, one CCD
rate[1679] = 3.0e3                              # Fe XXIV 192.030 in a flare

frame = expose(rate, 1.0 * u.s, sequence)       # photons, including the parallel overscan rows
```

The frame has `parallel_overscan_rows` more rows than the image area. With a shutter, the smear is zero and those rows are empty.

`dark_current_time` gives how long each row accumulates dark current, from the clear to its own read-out. That is longer than the exposure whether or not there is a shutter, and longest for the rows read last.

## What is not modelled

- **Blooming.** A saturated line spills along the column, which is the same axis as the smear. The full well is above 100 ke- and the CCDs have no anti-blooming, so this matters in a flare.
- **The shutter in motion.** The shutter takes about 30 ms to open and the same to close, and light falls during both.
- **Vignetting shape.** The edge of the illuminated area is treated as a step.
