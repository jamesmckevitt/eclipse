# The docs' examples

The figures in `docs/images` whose names start with `example-` come from the MURaM active region of the [worked example](../../docs/worked-example.ipynb), [McKevitt (2026)](https://doi.org/10.5522/04/34806267). To make them again, run the worked example, then work in its folder, with `REPO` set to your copy of this repository.

1. Run the instrument with each configuration in this folder:

    ```bash
    for config in $REPO/tools/docs_examples/*.yaml; do eclipse --config $config; done
    ```

2. Work out every line in the band and make the full-CCD frame:

    ```bash
    python $REPO/tools/docs_examples/band.py
    python $REPO/tools/docs_examples/frame.py
    ```

3. Make the figures:

    ```bash
    python $REPO/tools/docs_examples/make_figures.py $REPO/docs/images
    ```
