"""Module with classes that facilitate building a Bayesian network.

Examples are borrowed from Koller and Friedmand's "Probabilistic
Graphical Models: Principles and Techniques"
"""
import logging

from . import util  # noqa
from .util import get_pkg_filename  # noqa
from ._version import __version__  # noqa
from ._options import options # noqa

from . import factors  # noqa
from . import models  # noqa

# Convenience imports
from .factors.factor import Factor  # noqa
from .factors.cpt import CPT  # noqa
from .factors.jpt import JPT  # noqa

from .models.bag import Bag  # noqa
from .models.bn import BayesianNetwork  # noqa

# ...
log = logging.getLogger('thomas')

