"""Settings every test runs with."""
import matplotlib

# The plotting tests draw without a display, whatever backend the
# environment asks for, since the package imports pyplot.
matplotlib.use("Agg", force=True)
