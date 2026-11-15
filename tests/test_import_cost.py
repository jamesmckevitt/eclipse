"""`import eclipse` leaves out the packages only a few functions need."""
import os
import subprocess
import sys
from pathlib import Path

import eclipse


def test_importing_the_package_leaves_out_mendeleev_and_sunpy_maps():
    source = str(Path(eclipse.__file__).resolve().parents[1])
    path = os.pathsep.join(filter(None, [source, os.environ.get("PYTHONPATH")]))
    code = ("import sys, eclipse\n"
            "print(sorted(name for name in ('mendeleev', 'sqlalchemy', 'sunpy.map') "
            "if name in sys.modules))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=True, env={**os.environ, "PYTHONPATH": path})
    assert result.stdout.strip() == "[]"
