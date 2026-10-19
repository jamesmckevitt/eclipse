"""
The former name of the `eclipse` package, kept so existing scripts still run.

Importing it warns that it will be removed in a future release. Each of its
modules, such as `euvst_response.frame`, is the `eclipse` module of the same
name, so older pickles that name these modules also still load.
"""
import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import eclipse
from eclipse import *  # noqa: F401,F403
from eclipse import __all__, __version__  # noqa: F401

MESSAGE = ("The euvst_response package is now called eclipse: use `import eclipse`. "
           "euvst_response will be removed in a future release.")


def _caller_level() -> int:
    """Stack level of the code that imported this package, past the import machinery."""
    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_code.co_filename.startswith("<frozen importlib"):
        frame, level = frame.f_back, level + 1
    return level + 1


warnings.warn(MESSAGE, FutureWarning, stacklevel=_caller_level())


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import `euvst_response.<name>` as the module `eclipse.<name>`."""

    euvst_response_alias = True

    def find_spec(self, name, path=None, target=None):
        if name.startswith("euvst_response."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        module = importlib.import_module("eclipse" + spec.name[len("euvst_response"):])
        spec.loader_state = module.__spec__
        return module

    def exec_module(self, module):
        # The import machinery gave the module this alias's spec; give it back
        # its own, so reloading it and its relative imports still work.
        module.__spec__ = module.__spec__.loader_state


if not any(getattr(finder, "euvst_response_alias", False) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Alias())


def __getattr__(name):
    return getattr(eclipse, name)
