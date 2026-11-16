"""
Former name of the `eclipse` package, kept so existing scripts still run.

Importing it warns that it will be removed in a future release. Each module,
such as `euvst_response.frame`, is the `eclipse` module of the same name, so
old pickles still load. Package data are those of `eclipse`.
"""
import importlib
import importlib.abc
import importlib.util
import sys
import types
import warnings

import eclipse
from eclipse import *  # noqa: F401,F403
from eclipse import __all__, __version__  # noqa: F401
from eclipse.utils import FormerNameWarning

MESSAGE = ("The euvst_response package is now called eclipse: use `import eclipse`. "
           "euvst_response will be removed in a future release.")

warnings.warn(MESSAGE, FormerNameWarning, stacklevel=2)


def _new_name(name):
    return "eclipse" + name[len("euvst_response"):]


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import `euvst_response.<name>` as the module `eclipse.<name>`."""

    euvst_response_alias = True

    def find_spec(self, name, path=None, target=None):
        if not name.startswith("euvst_response."):
            return None
        real = importlib.util.find_spec(_new_name(name))
        if real is None:
            return None
        spec = importlib.util.spec_from_loader(
            name, self, origin=real.origin,
            is_package=real.submodule_search_locations is not None)
        spec.has_location = real.has_location
        return spec

    def create_module(self, spec):
        module = importlib.import_module(_new_name(spec.name))
        spec.loader_state = module.__spec__
        return module

    def exec_module(self, module):
        # Restore the module's own spec, so reloading and relative imports work.
        module.__spec__ = module.__spec__.loader_state

    # For `python -m euvst_response.<name>`, which runs a module's code
    # without importing it.
    def _loader(self, name):
        spec = importlib.util.find_spec(_new_name(name))
        if spec is None:
            raise ImportError(f"No module named {_new_name(name)!r}", name=name)
        return spec.loader, spec.name

    def get_code(self, name):
        loader, name = self._loader(name)
        return loader.get_code(name)

    def get_source(self, name):
        loader, name = self._loader(name)
        return loader.get_source(name)

    def is_package(self, name):
        loader, name = self._loader(name)
        return loader.is_package(name)


class _Loader:
    """This package's loader, reading package data from `eclipse`."""

    def __init__(self, loader):
        self._loader = loader

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def get_resource_reader(self, name):
        return eclipse.__spec__.loader.get_resource_reader(eclipse.__name__)


class _Package(types.ModuleType):
    """This package, whose attributes are those of `eclipse`."""

    def __setattr__(self, name, value):
        # Importing `euvst_response.<name>` sets the submodule here; keep the
        # attribute from `eclipse` instead, such as the function `main`.
        if (isinstance(value, types.ModuleType) and value.__name__ == f"eclipse.{name}"
                and getattr(self, name, value) is not value):
            return
        super().__setattr__(name, value)


# Modules run with `python -m`, left out of `_register`, as `-m` runs a
# module's code only if it is not yet imported under the name given.
_COMMANDS = {"atmosphere", "band", "cli", "continuum", "main", "synthesis", "synthesis_cli"}


def _register():
    """Put each loaded `eclipse` module in `sys.modules` under the old name too."""
    for name, value in list(sys.modules.items()):
        if name.startswith("eclipse.") and name.split(".")[1] not in _COMMANDS:
            sys.modules.setdefault("euvst_response" + name[len("eclipse"):], value)


if not any(getattr(finder, "euvst_response_alias", False) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Alias())
_register()
sys.modules[__name__].__class__ = _Package

# Package data, from importlib.resources or the folder of `__file__`, come
# from `eclipse`. `__path__` is left alone, so `eclipse` modules are only
# imported under their own names.
__spec__.loader = __loader__ = _Loader(__spec__.loader)
__file__ = eclipse.__file__


def __getattr__(name):
    return getattr(eclipse, name)
