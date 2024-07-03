import numpy

from . import linpack
from ._misc import _datacopied


def qr(
    a, 
    overwrite_a=False, 
    lwork=None, 
    mode='full', 
    pivoting=False, 
    check_finite=True, 
    tol=1e-12
):
    """
    Compute QR decomposition of a matrix.

    Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
    and R upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be decomposed
    overwrite_a : bool, optional
        Whether data in `a` is overwritten (may improve performance if
        `overwrite_a` is set to True by reusing the existing input data
        structure rather than creating a new one.)
    lwork : int, optional
        Ignored, left for compatibility with ``scipy.linalg.qr``.
    mode : {'full', 'r', 'economic', 'raw'}, optional
        Determines what information is to be returned: either both Q and R
        ('full', default), only R ('r') or both Q and R but computed in
        economy-size ('economic', see Notes). The final option 'raw'
        (added in SciPy 0.11) makes the function return two matrices
        (Q, TAU) in the internal format used by LINPACK.
    pivoting : bool, optional
        Whether or not factorization should include pivoting for rank-revealing
        qr decomposition. If pivoting, compute the decomposition
        ``A[:, P] = Q @ R`` as above, but where P is chosen such that the 
        diagonal of R is non-increasing.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

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

    if pivoting:
        qr, rank, jpvt, tau, _, _ = linpack.dqrdc2(a1, tol=tol, overwrite_a=overwrite_a)
    else:
        raise NotImplementedError

    if mode not in {'economic', 'raw'} or M < N:
        R = numpy.triu(qr)
    else:
        R = numpy.triu(qr[:N, :])
    
    if pivoting:
        Rj = R, jpvt
    else:
        Rj = R,

    if mode == 'r':
        return Rj
    elif mode == 'raw':
        return ((qr, tau), *Rj)

    if mode == 'economic':
        raise NotImplementedError
    else:
        D = numpy.eye(M, dtype=numpy.double, order='F')
        Q = linpack.dqrqy(qr, tau, D, overwrite_a=True)

    return (Q, *Rj)