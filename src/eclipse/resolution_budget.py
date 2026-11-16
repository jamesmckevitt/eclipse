"""
EUVST-SW resolution budget beyond the optical design.

RSC-2022033B rev. B (28 Oct 2025), "EUVST Spatial and Spectral Resolution
Budget", lists each source of blur as a FWHM in arcsec, added in
quadrature. Its first row, the optical design, is the RSC-2022021C PSF
already in `Telescope_EUVST.psf_params`. This module holds the other rows,
for the current best estimate (CBE) and the allocation.

The budget covers the 70 x 70 and 100 x 100 arcsec fields, with the same
rows after the first for both. There is no 280 x 280 arcsec budget yet, so
these rows are used for every field.
"""
import numpy as np

BUDGETS = ("current_estimate", "allocation", "none")

# Table 3-1, SW spatial resolution budget: FWHM in arcsec as (current best
# estimate, allocation, before the slit). Rows before the slit also blur the
# scene across the slit. Row 10, the spectrograph optics, is after the slit.
SPATIAL_ROWS = {
    "2: primary mirror": (0.206, 0.247, True),
    "3: scan and tip-tilt mirror": (0.122, 0.122, True),
    "4: focus mechanism": (0.033, 0.073, True),
    "5: telescope alignment": (0.055, 0.125, True),
    "6: measurement error": (0.063, 0.063, True),
    "10: spectrograph optical elements": (0.100, 0.163, False),
    "20: pointing jitter": (0.110, 0.110, True),
}

# Table 4-1, SW spectral resolution budget, rows 3 to 5: FWHM in arcsec
# along the dispersion as (current best estimate, allocation). The table
# gives only the allocation. Each row is a row of Table 3-1, or a fraction
# of one, so the estimate applies the same rule to the Table 3-1 estimates.
SPECTRAL_ROWS = {
    "3-1: alignment of the spectrograph (#10-3)": (0.029, 0.100),
    "3-2: residual of 1G compensation (2/3 of #5-2)": (2 / 3 * 0.036, 0.040),
    "3-3: thermal deformation, ground to space (2/3 of #5-3)": (2 / 3 * 0.012, 0.053),
    "3-4: thermal deformation, orbital (1/3 of 3-3)": (1 / 3 * (2 / 3 * 0.012), 0.018),
    "3-5: launch load (#5-5)": (0.018, 0.031),
    "3-6: CFRP moisture desorption (#5-6)": (0.003, 0.015),
    "3-7: BUS I/F thermal deformation (#5-7)": (0.001, 0.010),
    "3-8: BUS I/F flatness (#5-8)": (0.001, 0.010),
    "4-1: grating surface figure and mounting stress (#10-1)": (0.090, 0.120),
    "4-2: slit assembly (#10-2)": (0.033, 0.046),
    "5: grating micro-vibrations (2/3 of #20)": (2 / 3 * 0.110, 0.073),
}


def _column(budget: str) -> int | None:
    if budget not in BUDGETS:
        raise ValueError(f"resolution_budget must be one of {', '.join(BUDGETS)}, got "
                         f"{budget!r}.")
    return {"current_estimate": 0, "allocation": 1}.get(budget)


def _sum_in_quadrature(values) -> float:
    return float(np.sqrt(np.sum(np.square(list(values)))))


JITTER_ROW = "20: pointing jitter"
VIBRATION_ROW = "5: grating micro-vibrations (2/3 of #20)"


def _spatial_rows(budget: str, pointing_jitter: float | None) -> list:
    """(FWHM, before the slit) of each spatial row, row 20 replaced by `pointing_jitter`."""
    column = _column(budget)
    rows = [] if column is None else [(row[column], row[2]) for name, row in SPATIAL_ROWS.items()
                                      if name != JITTER_ROW or pointing_jitter is None]
    if pointing_jitter is not None:
        rows.append((pointing_jitter, True))
    return rows


def along_slit_fwhm(budget: str, pointing_jitter: float | None = None) -> float:
    """
    FWHM in arcsec of the budget's blur along the slit, all rows but the optical design.

    `pointing_jitter`, a FWHM in arcsec, replaces row 20, and is added even
    when `budget` is `"none"`.
    """
    return _sum_in_quadrature(fwhm for fwhm, _ in _spatial_rows(budget, pointing_jitter))


def across_slit_fwhm(budget: str, pointing_jitter: float | None = None) -> float:
    """FWHM in arcsec of the budget's blur across the slit, from rows before the slit."""
    return _sum_in_quadrature(fwhm for fwhm, before in _spatial_rows(budget, pointing_jitter)
                              if before)


def spectral_fwhm(budget: str, pointing_jitter: float | None = None) -> float:
    """
    FWHM in arcsec of the budget's blur along the dispersion.

    Rows 3 to 5 of Table 4-1. Rows 1 and 2, the optical design and the
    slit, are already in `Telescope_EUVST.psf_params`. Row 2-2, the slit
    width error, is a tolerance rather than a blur and is left out. Row 5,
    the grating micro-vibration, is 2/3 of the pointing jitter, so
    `pointing_jitter` sets it too.
    """
    column = _column(budget)
    rows = [] if column is None else [row[column] for name, row in SPECTRAL_ROWS.items()
                                      if name != VIBRATION_ROW or pointing_jitter is None]
    if pointing_jitter is not None:
        rows.append(2 / 3 * pointing_jitter)
    return _sum_in_quadrature(rows)
