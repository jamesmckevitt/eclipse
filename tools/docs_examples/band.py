"""Every line in the SW band, as in step 1 of the full-CCD frame page."""
from eclipse import band_contribution_functions, read_synthesis_products, write_band_lines

if __name__ == "__main__":
    products = read_synthesis_products("run/input/muram_ar_fe12.h5",
                                       keys=["logT_grid", "logN_grid", "config"])
    lines = band_contribution_functions((169.0, 213.5), products["logT_grid"],
                                        products["logN_grid"],
                                        abundance=products["config"]["abundance"])
    write_band_lines(lines, "band_lines.h5")
