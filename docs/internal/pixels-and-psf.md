---
search:
  exclude: true
---

# Pixels and the PSF

How `eclipse` lays a synthesis onto the detector's pixels, and where the PSF comes in.

Each detector pixel holds the mean of the scene over the area it covers: along the slit over the plate scale, across it over the slit width, and in wavelength over the pixel's range of wavelengths. Each cell of the synthesis is taken to be uniform. With `psf: True`, the scene is blurred by the PSF on the synthesis's own grids before the pixels average it, so a line narrower than a pixel keeps its place within the pixel. Each photon is counted at its own wavelength, with the telescope's throughput there, before the blur moves it, with the PSF on or off, and frees the electrons a photon of that wavelength frees. The ground truth is the scene without the PSF.

The PSF along the slit and in wavelength is measured at the detector, after the slit. The telescope also blurs the image it forms on the slit, which brings in light from either side of the slit, but that blur across the slit is not in `psf_params`. `telescope.psf_across_slit` sets it, as a FWHM. Without it, which is the default, there is no blur across the slit:

```yaml
telescope:
  psf_across_slit: 3 arcsec
simulation:
  psf: True
```

`simulation.psf_boundary` decides what the blur brings in from beyond the edges of the atmosphere, along and across the slit:

- `replicate` (default): the Sun beyond each edge is taken to be like the cells at the edge.
- `zero`: nothing is beyond the edges, so the pixels within a PSF width of an edge come out darker.
