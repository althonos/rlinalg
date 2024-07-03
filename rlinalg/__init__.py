"""Linear Algebra routines as implemented in the ``R`` language.
"""

__version__ = "0.0.1"
__author__ = "Martin Larralde <martin.larralde@embl.de>"
__license__ = "GPL-3.0-or-later"

from . import linpack
from ._decomp_qr import qr
from ._kappa import kappa

__all__ = [
    "qr",
    "kappa",
    "linpack"
]