import functools

import numpy
from numpy import atleast_2d

from .._misc import _asarray_validated, _datacopied
from ._linpack import (
    dqrdc2,
    dtrco,
    dqrls,
    dqrsl,
    dqrqy as _dqrqy,
)

__all__ = ["dqrdc2", "dqrqy", "dqrqty", "dtrco", "dqrsl", "dqrls"]


def dqrqy(x, qraux, y, k=None, check_finite=True, overwrite_x=False, overwrite_y=False):
    """
    Wrapper for ``dqrqy``.

    Parameters
    ----------
    x : input rank-2 array('d') with bounds (ldx,p)
    qraux : input rank-1 array('d') with bounds (p)
    y : input rank-2 array('d') with bounds (n,ny)

    Other Parameters
    ----------------
    k : input int, optional
        Default: shape(x,1)
    overwrite_x : input int, optional
        Default: 1
    overwrite_y : input int, optional
        Default: 1

    Returns
    -------
    qy : rank-2 array('d') with bounds (n,ny)

    """
    asarray = numpy.asarray_chkfinite if check_finite else numpy.asarray

    y1 = asarray(y, order="F", dtype=numpy.double)
    overwrite_y = overwrite_y or _datacopied(y1, y)
    if not 1 <= len(y1.shape) <= 2:
        raise ValueError("expected a 1-D or 2-D array")

    qy = y1.reshape(-1, 1) if y.ndim == 1 else y1
    if not overwrite_y:
        qy = qy.copy()
    return _dqrqy(
        x=x,
        qraux=qraux,
        y=y.reshape(-1, 1) if y.ndim == 1 else y,
        ldx=x.shape[0],
        n=y.shape[0],
        p=qraux.shape[0],
        k=k,
        ny=1 if y.ndim == 1 else y.shape[1],
        qy=qy,
        overwrite_x=overwrite_x,
    )


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
