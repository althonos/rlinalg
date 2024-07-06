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


def dqrqy(x, qraux, y, k=None, overwrite_x=True):
    """Implementation of `dqrqty.`"""
    ldx = x.shape[0]
    ny = y.shape[-1]
    dummy = numpy.array(0)

    if k is None:
        k = x.shape[1]

    if y.ndim == 1:
        qy = numpy.zeros((y.shape[0], 1), order="F")
        info = dqrsl(
            x[:, :],
            k=k,
            qraux=qraux,
            y=y[:],
            qy=qy[:, 0],
            qty=dummy,
            b=dummy,
            rsd=dummy,
            xb=dummy,
            job=10000,
            overwrite_x=overwrite_x,
        )

    else:
        qy = numpy.zeros_like(y, order="F")
        for j in range(ny):
            info = dqrsl(
                x[:, :],
                k=k,
                qraux=qraux,
                y=y[:, j],
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
    """Implementation of `dqrqty.`"""
    ldx = x.shape[0]
    ny = y.shape[-1]
    dummy = numpy.array(0)

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
