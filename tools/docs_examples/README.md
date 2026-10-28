# The docs' examples

The figures in `docs/images` whose names start with `example-` come from the MURaM active region of the [worked example](../../docs/worked-example.ipynb). To make them again, work in a folder holding the atmosphere file the worked example writes, `muram_ar_1160000.h5`.

1. Synthesise four lines:

    ```bash
    for spec in "fe9 Fe9_171.0730" "fe10 Fe10_184.5370" "fe12 Fe12_195.1190 Fe12_195.1790" "fe14 Fe14_211.3170"; do
      set -- $spec; name=$1; shift
      synthesise-spectra --atmosphere muram_ar_1160000.h5 --lines "$@" --mass-per-electron 1.19 \
        --goft-temperature-chunk 20 --precision float32 --output-dir synth --output-name $name.h5
    done
    ```

2. Run the instrument with each configuration in this folder:

    ```bash
    for config in docs psf uniform_qs uniform_ar; do eclipse --config tools/docs_examples/$config.yaml; done
    ```

3. Work out every line in the band and make the full-CCD frame:

    ```bash
    python tools/docs_examples/band.py
    python tools/docs_examples/frame.py
    ```

4. Make the figures:

    ```bash
    python tools/docs_examples/make_figures.py . docs/images
    ```

Each synthesis takes about 15 minutes and 330 GB of memory on 128 cores, `band.py` about two hours, and each of the rest a few minutes.
