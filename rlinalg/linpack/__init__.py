import numpy
from numpy import atleast_2d

from .._misc import _asarray_validated, _datacopied
from ._linpack import (
    dqrdc2 as dqrdc2,
    dqrqy as dqrqy,
    dtrco as dtrco
)

__all__ = [
    "dqrdc2",
    "dqrqy",
    "dtrco"
]
