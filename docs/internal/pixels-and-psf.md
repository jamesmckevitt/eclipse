---
search:
  exclude: true
---

# Pixels and the PSF

How the instrument run lays a synthesis onto the detector's pixels, and where the PSF comes in.

Each detector pixel holds the mean of the scene over its footprint: along the slit over the plate scale, across it over the slit width, and along the dispersion over the pixel's wavelengths, with each cell of the synthesis taken to be uniform. With `psf: True` the scene is blurred by the PSF on the synthesis's own grids before the pixels average it, so a line narrower than a pixel keeps its place within the pixel. Light the blur moves to another wavelength keeps the photon count of its own wavelength. The ground truth is the scene without the PSF.

The PSF along the slit and along the dispersion is measured at the detector, after the slit. The telescope also blurs the image it forms on the slit, which brings in light from beside the slit, but that blur across the slit is not in `psf_params`. `telescope.psf_across_slit` gives it, as a FWHM, and leaves it out when not set (the default):

```yaml
telescope:
  psf_across_slit: 3 arcsec
simulation:
  psf: True
```

`simulation.psf_boundary` decides what the blur brings in from beyond the edges of the atmosphere, along and across the slit:

- `replicate` (default): the Sun beyond each edge is taken to be like the cells at the edge.
- `zero`: nothing is beyond the edges, so the pixels within a PSF width of an edge come out darker.
