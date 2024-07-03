import collections
import numpy

from . import linpack
from ._misc import _datacopied


QRResult = collections.namedtuple("QRResult", "Q R P")


def qr(
    a,
    overwrite_a=False,
    mode='full',
    check_finite=True,
    tol=1e-12
):
    """
    Compute QR decomposition of a matrix.

    Calculate the decomposition ``A[:, P] = Q R`` where Q is
    unitary/orthogonal, R upper triangular, and P is a permutation
    such that columns with near-zero norm are moved towards the
    right-hand edge of A.


    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be decomposed
    overwrite_a : bool, optional
        Whether data in `a` is overwritten (may improve performance if
        `overwrite_a` is set to True by reusing the existing input data
        structure rather than creating a new one.)
    mode : {'full', 'r', 'economic', 'raw'}, optional
        Determines what information is to be returned: either both Q and R
        ('full', default), only R ('r') or both Q and R but computed in
        economy-size ('economic', see Notes). The final option 'raw'
        makes the function return two matrices (Q, TAU) in the internal
        format used by LAPACK and LINPACK.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    tol : float
        The absolute tolerance to which each column norm is required.
        An column is considered negligible when its norm falls under
        this value.

    """
    if mode not in {'full', 'qr', 'r', 'economic', 'raw'}:
        raise ValueError(
            "Mode argument should be one of ['full', 'r', 'economic', 'raw']"
        )

    asarray = numpy.asarray_chkfinite if check_finite else numpy.asarray
    a1 = asarray(a, order='F', dtype=numpy.double)
    if len(a1.shape) != 2:
        raise ValueError("expected a 2-D array")

    M, N = a1.shape

    # accomodate empty arrays
    if a1.size == 0:
        raise NotImplementedError

    overwrite_a = overwrite_a or _datacopied(a1, a)

    qr, rank, jpvt, tau, _, _ = linpack.dqrdc2(a1, tol=tol, overwrite_a=overwrite_a)
    jpvt -= 1

    if mode not in {'economic', 'raw'} or M < N:
        R = numpy.triu(qr)
    else:
        R = numpy.triu(qr[:N, :])

    if mode == 'r':
        return R, jpvt
    elif mode == 'raw':
        return ((qr, tau), R, jpvt)
    
    if M < N:
        D = numpy.eye(M, dtype=numpy.double, order='F')
        Q = linpack.dqrqy(qr[:, :M], tau, D, overwrite_a=True)
    elif mode == 'economic':
        D = numpy.eye(M, N, dtype=numpy.double, order='F')
        Q = linpack.dqrqy(qr, tau, D, overwrite_a=True)
    else:
        D = numpy.eye(M, dtype=numpy.double, order='F')
        Q = linpack.dqrqy(qr, tau, D, overwrite_a=True)

    return QRResult(Q, R, jpvt)