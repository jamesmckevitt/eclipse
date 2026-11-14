# Older versions

What older versions used still works, with a warning, until a future release. This page says how to move over.

## The name on PyPI

ECLIPSE 0.12 and earlier were `solarc-eclipse` on PyPI (`pip install solarc-eclipse==0.12.1` still works). To move over:

```bash
pip uninstall solarc-eclipse
pip install eclipse
```

Uninstall first, or the commands go too; `pip install --force-reinstall --no-deps eclipse` puts them back. A git install of a later commit is `eclipse @ git+...`.

## The package's name

`import euvst_response` is now `import eclipse`. The code is in the clone's `src` folder: rerun `pip install -e .` after updating, or put `<clone>/src` on `PYTHONPATH`. Rename any `eclipse.py` of your own, or it is imported instead.

## Synthesis files

ECLIPSE 0.11.0 and earlier wrote pickles. `eclipse.convert_synthesis_pickle("old.pkl", "new.h5")` converts one.

## Results files

ECLIPSE 0.11.0 and earlier wrote pickles. `eclipse.convert_results_pickle("run.pkl")` converts one; rerunning renames the old pickle to `run.pkl.old`.

## Frame files

In frame files from ECLIPSE 0.13.0 to 0.14.0, with no `field_angle`, both CCDs' columns count from the slit's south end; later ones count from each CCD's left output. Frame files from 0.15.0 and earlier have no `spectra` dataset.

## Frames from a band's lines

`full_ccd_frame`'s `lines` and `continuum` arguments are replaced by `synthesise-spectra --band`, or a [run file](../observe/full-ccd-frames.md#from-an-mhd-simulation).

## Reading MURaM's own files

Command lines from ECLIPSE 0.11.0 and earlier that read MURaM's binary files still run:

```bash
synthesise-spectra \
  --data-dir ./data/atmosphere \
  --temp-file temp/eosT.0270000 \
  --rho-file rho/result_prim_0.0270000 \
  --vz-file vz/result_prim_2.0270000 \
  --cube-shape 512 768 256 \
  --voxel-dx "0.192 Mm" --voxel-dy "0.192 Mm" --voxel-dz "0.064 Mm" \
  --lines Fe12_195.1190 \
  --output-dir ./run/input
```

Here x and y are centred on zero, and z = 0 is the bottom cell's centre, for `--crop-x`, `--crop-y` and `--crop-z`. To move over, write each snapshot as an [atmosphere file](../files/atmosphere.md), as in the [Bifrost example](../emission/mhd.md#a-bifrost-quiet-sun).

## Dynamic mode

Dynamic mode, a raster of MURaM snapshots made by `synthesise-spectra` with `--slit-rest-time`, still runs:

```bash
synthesise-spectra \
  --data-dir ./data/atmosphere \
  --lines Fe12_195.1190 \
  --slit-rest-time "40 s" \
  --slit-width "0.2 arcsec" \
  --cube-shape 512 768 256 \
  --voxel-dx "0.192 Mm" --voxel-dy "0.192 Mm" --voxel-dz "0.064 Mm" \
  --output-dir ./run/input
```

The run on its output must use that slit width, and an exposure equal to `--slit-rest-time`. To move over, write each snapshot as an atmosphere file with its time and give them as `atmosphere_series`; see [Rasters and sit-and-stares](../observe/rasters.md).

## Renamed options

`eclipse --config FILE` is now `eclipse run FILE`. `--mean-mol-wt` is now `--mass-per-electron`, and its default, 1.29 up to 0.11.0, now comes from the [abundances](../emission/mhd.md#electron-density).
