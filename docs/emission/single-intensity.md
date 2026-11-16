# A single intensity

<div class="glance" markdown>
<div markdown>
**You need**

A line's total intensity, rest wavelength and width
</div>
<div markdown>
**You run**

Nothing yet. The line goes in the run file.
</div>
<div markdown>
**You get**

No synthesis file, as the run file holds the line
</div>
<div markdown>
**Next**

[Simulating an observation](../observe/simulate.md#a-single-intensity)
</div>
</div>

ECLIPSE can observe one Gaussian line of known total intensity, given in the [run file](../run-files.md). All the scatter in the results then comes from the instrument, which suits uncertainty budgets, and there is no synthesis, so it is fast.

```yaml
instrument: SWC
uniform_intensity: 5000 erg / (s cm2 sr)   # units are required
rest_wavelength: 195.119 AA
thermal_width: 20 km/s                     # 1-sigma width, as a velocity
```

[Simulating an observation](../observe/simulate.md#a-single-intensity) shows how to observe it.
