"""Every line in the SW band, on the Fe XII synthesis's grids, for the docs' full-CCD frame."""
from pathlib import Path

from eclipse import band_contribution_functions, read_synthesis_products, write_band_lines

if __name__ == "__main__":
    products = read_synthesis_products("synth/fe12.h5", keys=["logT_grid", "logN_grid", "config"])
    lines = band_contribution_functions((169.0, 213.5), products["logT_grid"],
                                        products["logN_grid"],
                                        abundance=products["config"]["abundance"])
    Path("frame").mkdir(exist_ok=True)
    write_band_lines(lines, "frame/band_lines.h5")
