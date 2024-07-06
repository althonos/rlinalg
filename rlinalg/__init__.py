"""Linear Algebra routines as implemented in the ``R`` language.
"""

__version__ = "0.0.1"
__author__ = "Martin Larralde <martin.larralde@embl.de>"
__license__ = "GPL-3.0-or-later"

from . import linpack
from ._decomp_qr import qr, qr_multiply
from ._cond import cond
from ._lstsq import lstsq

__all__ = ["qr", "cond", "linpack", "lstsq", "qr_multiply"]
__test__ = {
    "cond": cond,
    "lstsq": lstsq,
    "linpack": lstsq,
    "qr": qr,
    "qr_multiply": qr_multiply,
}
