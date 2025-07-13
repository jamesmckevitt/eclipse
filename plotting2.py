import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import glob
import dill
from instr_response import angle_to_distance, velocity_from_fit
import matplotlib
from typing import Tuple, Iterable
from tqdm import tqdm
import warnings
from matplotlib.colors import ListedColormap, BoundaryNorm
from astropy.coordinates import SkyCoord, SpectralCoord
from sunpy.coordinates import Helioprojective
from astropy.time import Time
import astropy.constants as const
import astropy.units as u
from ndcube import NDCube
import sunpy.map
from astropy.wcs import WCS
from matplotlib.gridspec import GridSpec

def mean_sigma_heliocentric_coords(
    cube: NDCube,
    *,
    sigma_factor: float = 1.0,
    margin_frac: float = 0.20,
) -> tuple[Tuple[u.Quantity, u.Quantity],    # mean
           Tuple[u.Quantity, u.Quantity],    # +σ
           Tuple[u.Quantity, u.Quantity]]:   # −σ
    """
    Find heliocentric (y, x) coordinates of pixels whose integrated intensity
    is closest to the cube mean and to mean ± sigma_factor·std.

    Parameters
    ----------
    cube : NDCube
        Specific-intensity cube with heliocentric WCS (WAVE, SOLY, SOLX).
    sigma_factor : float, optional
        Multiple of σ around the mean to identify the ±σ pixels.
    margin_frac : float, optional
        Fractional margin to exclude from the search (crop off edges).

    Returns
    -------
    (mean_xy, plus_xy, minus_xy) where each entry is a 2-tuple of Quantities
    (y_coord, x_coord) in the cube’s native heliocentric units.
    """

    # --- integrated intensity image -----------------------------------
    wl_step = cube.wcs.wcs.cdelt[0] * cube.wcs.wcs.cunit[0]
    ii = (cube.data.sum(axis=2) * wl_step).cgs.value          # (nx, ny)

    nx, ny = ii.shape
    m = int(margin_frac * min(nx, ny))
    sub = ii[m:nx - m, m:ny - m]

    mean_val = sub.mean()
    std_val  = sub.std()

    targets = [mean_val,
               mean_val + sigma_factor*std_val,
               mean_val - sigma_factor*std_val]

    pix_indices = []
    for t in targets:
        rel = np.abs(sub - t)
        j, i = np.unravel_index(rel.argmin(), sub.shape)      # note j,i order
        pix_indices.append((i + m, j + m))                    # global (x, y)

    # --- convert to heliocentric coordinates --------------------------
    coords = []
    for x_pix, y_pix in pix_indices:
        _, yw, xw = cube.wcs.pixel_to_world_values(           # drop wavelength
            0, y_pix, x_pix
        )
        coords.append((yw * cube.wcs.wcs.cunit[1],
                       xw * cube.wcs.wcs.cunit[2]))

    return coords[0], coords[1], coords[2]


def main():
    synthetic_spectra = Path("./run/input/synthesised_spectra.pkl")
    instr_response_dir = Path("./run/result")
    instr_responses = sorted(instr_response_dir.glob("instrument_response_*.pkl"))


    if not synthetic_spectra.exists():
        raise FileNotFoundError(synthetic_spectra)

    with open(synthetic_spectra, "rb") as f:
        sim_dat = dill.load(f)
    cube: NDCube = sim_dat["sim_si"]              # heliocentric specific-intensity cube

    # --- get key coordinates ------------------------------------------
    mean_xy, plus_xy, minus_xy = mean_sigma_heliocentric_coords(
        cube,
        sigma_factor=1.0,      # change as needed
        margin_frac=0.20,      # change as needed
    )

    print("Mean intensity pixel (heliocentric):      ", mean_xy)
    print("+1σ intensity pixel (heliocentric):       ", plus_xy)
    print("−1σ intensity pixel (heliocentric):       ", minus_xy)


if __name__ == "__main__":
    main()