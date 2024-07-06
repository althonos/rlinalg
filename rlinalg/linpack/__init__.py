import functools

from .._misc import set_module
from ._linpack import (
    dqrdc2,
    dtrco,
    dqrls,
    dqrsl,
    dqrqy,
    dqrqty,
)

__all__ = ["dqrdc2", "dqrqy", "dqrqty", "dtrco", "dqrsl", "dqrls"]
