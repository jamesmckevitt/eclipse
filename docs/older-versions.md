# Older versions

ECLIPSE 0.11.0 and earlier wrote pickles rather than HDF5 files, and read MURaM's own files directly. These still work, with a warning, until a future release stops them. This page says how to move over.

## Synthesis files

ECLIPSE 0.11.0 and earlier wrote the synthesis as a pickle. `eclipse` still reads one, and `synthesise-spectra` still writes one if `--output-name` ends in `.pkl`, both with a warning, until a future release stops them. `euvst_response.convert_synthesis_pickle("old.pkl", "new.h5")` rewrites a pickle as a synthesis file, keeping everything it held.

## Results files

ECLIPSE 0.11.0 and earlier wrote the results as a pickle. `load_instrument_response_results` still reads one, with a warning, until a future release stops it. Rerunning `run.yaml` writes a new `run.h5`, and renames the old `run.pkl` to `run.pkl.old`, or `run.pkl.old.1` and so on if that name is taken. A script that loads `run.pkl` with ECLIPSE's functions then reads `run.h5` instead, with a warning. `euvst_response.convert_results_pickle("run.pkl")` converts an old pickle to the new format without rerunning.

## Reading MURaM's own files

Command lines from ECLIPSE 0.11.0 and earlier, which read MURaM's binary files directly, still run without `--atmosphere`, with a warning, until a future release removes them:

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

- `--data-dir`: Directory the file names are relative to (default: `data/atmosphere`)
- `--temp-file`, `--rho-file`: Temperature and density files (default: `temp/eosT.0270000`, `rho/result_prim_0.0270000`)
- `--vx-file`, `--vy-file`, `--vz-file`: Velocity files; only the one along `--integration-axis` is read (default: `vx/result_prim_1.0270000`, `vy/result_prim_3.0270000`, `vz/result_prim_2.0270000`)
- `--cube-shape`: Cube dimensions in the order the files store them, `(nx nz ny)` (default: `512 768 256`)
- `--voxel-dx`, `--voxel-dy`, `--voxel-dz`: Cell sizes (default: `"0.192 Mm"`, `"0.192 Mm"`, `"0.064 Mm"`)

In this mode x and y are centred on zero, and z = 0 is the centre of the bottom cell. `--crop-x`, `--crop-y` and `--crop-z` are measured from there.

To move over, write each snapshot as an [atmosphere file](synthesis.md#atmosphere-files), as in the [MURaM example](synthesis.md#worked-example-a-muram-flare).

## Dynamic mode

Dynamic mode synthesises a raster over a time series of MURaM snapshots in `synthesise-spectra`, with the slit width and exposure fixed at synthesis. It still runs, with a warning, until a future release removes it. A time series of atmosphere files is now observed by the instrument run instead, as described in [Simulating a time series](time-series.md).

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

- `--slit-rest-time`: Time the slit rests at each position, which turns dynamic mode on
- `--slit-width`: Slit width
- `--temp-dir`, `--rho-dir`, `--vx-dir`, `--vy-dir`, `--vz-dir`, `--time-dir`: Directory of each quantity's files, relative to `--data-dir` (default: `temp`, `rho`, `vx`, `vy`, `vz` and `header`)
- `--temp-filename`, `--rho-filename`, `--vx-filename`, `--vy-filename`, `--vz-filename`, `--time-filename`: File name before the snapshot suffix (default: `eosT`, `result_prim_0`, `result_prim_1`, `result_prim_3`, `result_prim_2` and `Header`)
- `--cube-shape`, `--voxel-dx`, `--voxel-dy`, `--voxel-dz`: As for reading MURaM's own files above

The instrument run on the synthesis file must have exactly one slit width, the one used in the synthesis, and exactly one exposure time, equal to `--slit-rest-time`.

To move a dynamic-mode run over, write each snapshot as an atmosphere file with its time, as in the [MURaM example](synthesis.md#worked-example-a-muram-flare), and give the files as `atmosphere_series`; see [Time series](time-series.md).

## Renamed options

`--mass-per-electron` in `synthesise-spectra` was called `--mean-mol-wt`, and defaulted to 1.29 up to ECLIPSE 0.11.0. The old name still works. Its default is now worked out from the abundances; see [Electron density](synthesis.md#electron-density).
