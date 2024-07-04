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

# def dqrdc2(
#     a,
#     tol=1e-12,
#     overwrite_a=False,
#     check_finite=True,
# ):
#     """
#     Wrapper for ``dqrdc2``.

#     Parameters
#     ----------
#     a : input rank-2 array('d') with bound (m,n)
#         The array-like input.

#     Returns
#     -------
#     qr: rank-2 array('d') with bounds (m,n) and a storage
#     k: int
#     jpvt: rank-1 array('i') with bounds (n)
#     tau: rank-1 array('d') with bounds (MIN(m,n))
#     work: rank-1 array('d') with bounds (n,2)
#     info: int

#     """

#     a1 = atleast_2d(
#         _asarray_validated(a, check_finite=check_finite, order="F", dtype=numpy.double)
#     )
#     overwrite_a = overwrite_a or _datacopied(a1, a)

#     n, p = a1.shape

#     tau = numpy.zeros(p, dtype=numpy.double, order="F")
#     work = numpy.zeros((p, 2), dtype=numpy.double, order="F")
#     jpvt = numpy.arange(1, p + 1, dtype=numpy.int32)

#     if not overwrite_a:
#         a1 = numpy.array(a1, order="F", dtype=numpy.double)

#     qr, k, tau, jpvt =  _dqrdc2(a1, tol=tol, jpvt=jpvt, work=work, qraux=tau)
#     return (qr, k, tau, jpvt)


# def dqrqy(
#     qr,
#     tau,
#     a,
#     overwrite_a=False,
#     check_finite=True,
# ):
#     """
#     Wrapper for ``dqrqy``.
#     """
#     qr1 = atleast_2d(
#         _asarray_validated(qr, check_finite=check_finite, order="F", dtype=numpy.double)
#     )
#     a1 = atleast_2d(
#         _asarray_validated(a, check_finite=check_finite, order="F", dtype=numpy.double)
#     )

#     overwrite_a = overwrite_a or _datacopied(a1, a)
#     if not overwrite_a:
#         a1 = numpy.array(a1, order="F", dtype=numpy.double)

#     _dqrqy(qr1, qr1.shape[0], qr1.shape[1], tau, a1, a1.shape[1], a1)
#     return a1


# def dtrco(
#     a,
#     lower=False,
#     check_finite=True,
# ):
#     """
#     Wrapper for ``drtco``.
#     """
#     a1 = atleast_2d(
#         _asarray_validated(a, check_finite=check_finite, order="F", dtype=numpy.double)
#     )

#     m, n = a1.shape
#     z = numpy.zeros(n, order="F", dtype=numpy.double)

#     _, _, _, rcond, z, _ = _dtrco(a1, m, n, 0, z, 1 - bool(lower))
#     return rcond, z
 