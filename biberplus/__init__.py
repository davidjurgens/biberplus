from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("biberplus")
except PackageNotFoundError:  # package not installed (e.g. running from source)
    __version__ = "0.4.0"

__author__ = 'Kenan Alkiek'
__credits__ = 'University of Michigan - The Blablablab'

from . import tagger
from . import reducer
from . import neurobiber

__all__ = ['tagger', 'reducer', 'neurobiber']
