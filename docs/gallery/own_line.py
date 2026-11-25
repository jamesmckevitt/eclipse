import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

import eclipse
from eclipse import SpectralLine, Synthesis, write_synthesis

# A Gaussian line at 192.5 Angstrom, with the same spectrum in every pixel.
rest = 192.5 * u.AA
wavelength = np.linspace(192.3, 192.7, 201) * u.AA
sigma = 0.02 * u.AA
total = 1000 * u.erg / (u.s * u.cm**2 * u.sr)
profile = total / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((wavelength - rest) / sigma) ** 2)
intensity = np.ones((40, 10, wavelength.size)) * profile  # [y, x, wavelength]

# 40 pixels along the slit and 10 across it, each 0.2 arcsec.
write_synthesis(Synthesis(
    lines={"My_192.500": SpectralLine(intensity=intensity, wavelength=wavelength,
                                      rest_wavelength=rest)},
    x_edges=np.arange(11) * 0.2 * u.arcsec,
    y_edges=np.arange(41) * 0.2 * u.arcsec,
), "my_line.h5")

# Observe it, as `eclipse run` would a run file.
[output] = eclipse.observe({"synthesis_file": "my_line.h5", "n_iter": 10,
                            "simulation": {"expos": "20 s"}}, name="my_line")
results = eclipse.load_instrument_response_results(output.results)
dn = eclipse.get_results_for_combination(results)["first_dn_signal"]

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
ny, nx, _ = dn.data.shape
ax.step(dn.axis_world_coords("em.wl")[0].to_value(u.AA), dn.data[ny // 2, nx // 2],
        where="mid")
ax.set_xlabel("Wavelength [Angstrom]")
ax.set_ylabel("Signal [DN / pixel]")
ax.set_title("A line at 192.5 Angstrom, 20 s, one pixel")
plt.show()
