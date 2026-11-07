# The docs' examples

The figures in `docs/images` whose names start with `example-` come from the MURaM active region of the [worked example](../../docs/worked-example.ipynb), [McKevitt (2026)](https://doi.org/10.5522/04/34806267). To make them again, run the worked example, then work in its folder, with `REPO` set to your copy of this repository.

1. Synthesise three more lines:

    ```bash
    for spec in "fe9 Fe09_171.0730" "fe10 Fe10_184.5370" "fe14 Fe14_211.3170"; do
      set -- $spec
      synthesise-spectra --atmosphere muram_ar_1160000.h5 --lines $2 --mass-per-electron 1.19 \
        --goft-temperature-chunk 20 --precision float32 --output-dir synth --output-name $1.h5
    done
    ```

2. Run the instrument with each configuration in this folder:

    ```bash
    for config in $REPO/tools/docs_examples/*.yaml; do eclipse --config $config; done
    ```

3. Work out every line in the band and make the full-CCD frame:

    ```bash
    python $REPO/tools/docs_examples/band.py
    python $REPO/tools/docs_examples/frame.py
    ```

4. Make the figures:

    ```bash
    python $REPO/tools/docs_examples/make_figures.py $REPO/docs/images
    ```

On 128 cores, each synthesis takes about 15 minutes and 210 GB of memory, `band.py` about two hours and 210 GB, `muram_ar_sweep.yaml` about three hours and 120 GB, and `muram_ar_exposures.yaml` about two hours and 100 GB. `frame.py` takes about 8 minutes, and the other configurations up to half an hour each.
