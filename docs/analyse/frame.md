# Analysing a full-CCD frame

<div class="glance" markdown>
<div markdown>
**You need**

A [full-CCD frame](../observe/full-ccd-frames.md)
</div>
<div markdown>
**You run**

Python
</div>
<div markdown>
**You get**

The spectrum of the whole band, as the camera records it
</div>
</div>

`dn` holds the frame as it is read out, with the prescan and overscan of each output, and `rate` holds the photons reaching each pixel. A [frame file](../files/frame.md) keeps both.

## What a frame holds

For each CCD, `"left"` and `"right"`:

- `rate`: the photons per second reaching each image pixel;
- `photons`: the photons in each pixel over the exposure;
- `dn`: the frame as it is read out, in DN;
- `wavelength`: the wavelength each row records;
- `field_angle`: the angle along the slit of each column, from the slit's centre, positive towards solar north in nominal observations;
- `photon_wavelength`: the wavelength of the mean energy of each pixel's photons;
- `photon_rms_wavelength`: the wavelength of the root mean square energy of each pixel's photons, which sets how far their electrons spread.

`dn` holds each row as the two outputs read it: the left output's prescan, the first half of the image, the left output's overscan, the right output's overscan, the second half of the image and the right output's prescan. The parallel overscan rows follow the image's rows.

`settings["spectra"]`, and the `spectra` dataset of a frame file, name the entries of the synthesis file that the frame holds.

## The spectrum of the band

The median of the image columns in each row gives the spectrum of the whole band:

```python
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pre, over = frame.settings["serial_prescan"], frame.settings["serial_overscan"]
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    for ccd in ("left", "right"):
        rows = frame.wavelength[ccd].to_value(u.AA)
        half = frame.rate[ccd].shape[1] // 2
        columns = np.r_[pre:pre + half, pre + half + 2 * over:pre + 2 * half + 2 * over]
        order = np.argsort(rows)
        spectrum = np.median(frame.dn[ccd][:rows.size, columns], axis=1)
        ax.plot(rows[order], spectrum[order], label=f"{ccd} CCD")
    ax.set_xlim(169, 213)
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength [Angstrom]")
    ax.set_ylabel("DN per pixel")
    ax.legend()
    plt.show()
```

![The spectrum of the band in the frame, from 170 to 212 Angstrom, with the median over the slit in each row](../images/example-frame-spectrum.png)
