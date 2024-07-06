import numpy
from numpy import atleast_2d

from .._misc import _asarray_validated, _datacopied
from ._linpack import (
    dqrdc2,
    dtrco,
    dqrls,
    dqrsl,
)

__all__ = ["dqrdc2", "dqrqy", "dqrqty", "dtrco", "dqrsl", "dqrls"]


def dqrqy(x, qraux, y, k=None, check_finite=True, overwrite_x=False, overwrite_y=False):
    """
    Implementation of `dqrqty.`
    """
    asarray = numpy.asarray_chkfinite if check_finite else numpy.asarray

    x1 = asarray(x, order="F", dtype=numpy.double)
    overwrite_x = overwrite_x or _datacopied(x1, x)
    if len(x1.shape) != 2:
        raise ValueError("expected a 2-D array")

    y1 = asarray(y, order="F", dtype=numpy.double)
    overwrite_y = overwrite_y or _datacopied(y1, y)
    if not 1 <= len(y1.shape) <= 2:
        raise ValueError("expected a 1-D or 2-D array")

    ldx = x1.shape[0]
    ny = y1.shape[-1]
    dummy = numpy.array(0, dtype=numpy.double)

    if k is None:
        k = x1.shape[1]

    if y1.ndim == 1:
        qy = y1.reshape(-1, 1)
        if not overwrite_y:
            qy = qy.copy()
        dqrsl(
            x1[:, :],
            k=k,
            qraux=qraux,
            y=qy[:, 0],
            qy=qy[:, 0],
            qty=dummy,
            b=dummy,
            rsd=dummy,
            xb=dummy,
            job=10000,
            overwrite_x=overwrite_x,
        )

    else:
        qy = y1
        if not overwrite_y:
            qy = numpy.copy(y1, order="F")
        for j in range(ny):
            dqrsl(
                x[:, :],
                k=k,
                qraux=qraux,
                y=qy[:, j],
                qy=qy[:, j],
                qty=dummy,
                b=dummy,
                rsd=dummy,
                xb=dummy,
                job=10000,
                overwrite_x=overwrite_x,
            )

    return qy


def dqrqty(x, qraux, y, k=None, overwrite_x=True):
    """
    Implementation of `dqrqty.`
    """
    ldx = x.shape[0]
    ny = y.shape[-1]
    dummy = numpy.array(0, dtype=numpy.double)

    if k is None:
        k = x.shape[1]

    if y.ndim == 1:
        qty = numpy.zeros((y.shape[0], 1), order="F")
        info = dqrsl(
            x[:, :],
            k=k,
            qraux=qraux,
            y=y[:],
            qy=dummy,
            qty=qty[:, 0],
            b=dummy,
            rsd=dummy,
            xb=dummy,
            job=1000,
            overwrite_x=overwrite_x,
        )

    else:
        qty = numpy.zeros_like(y, order="F")
        for j in range(ny):
            info = dqrsl(
                x[:, :],
                k=k,
                qraux=qraux,
                y=y[:, j],
                qy=dummy,
                qty=qty[:, j],
                b=dummy,
                rsd=dummy,
                xb=dummy,
                job=1000,
                overwrite_x=overwrite_x,
            )

    return qty
