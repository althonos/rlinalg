import collections
import numpy

from . import linpack
from ._misc import _datacopied, set_module


QRResult = collections.namedtuple("QRResult", "Q R P")


@set_module("rlinalg")
def qr(a, overwrite_a=False, mode="full", tol=1e-7, check_finite=True):
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

    Returns
    -------
    Q : float or complex ndarray
        Of shape (M, M), or (M, K) for ``mode='economic'``. Not returned
        if ``mode='r'``. Replaced by tuple ``(Q, TAU)`` if ``mode='raw'``.
    R : float or complex ndarray
        Of shape (M, N), or (K, N) for ``mode in ['economic', 'raw']``.
        ``K = min(M, N)``.
    P : int ndarray
        Of shape (N,).

    Raises
    ------
    LinAlgError
        Raised if decomposition fails

    Notes
    -----
    This is an interface to the LAPACK routines dgeqrf, zgeqrf,
    dorgqr, zungqr, dgeqp3, and zgeqp3.

    If ``mode=economic``, the shapes of Q and R are (M, K) and (K, N) instead
    of (M,M) and (M,N), with ``K=min(M,N)``.

    Examples
    --------
    >>> import numpy
    >>> import rlinalg
    >>> rng = numpy.random.default_rng()
    >>> a = rng.standard_normal((9, 6))

    >>> q, r, p = rlinalg.qr(a)
    >>> numpy.allclose(a, numpy.dot(q, r))
    True
    >>> q.shape, r.shape
    ((9, 9), (9, 6))

    >>> r2, p2 = rlinalg.qr(a, mode='r')
    >>> numpy.allclose(r, r2)
    True

    >>> q3, r3, p3 = rlinalg.qr(a, mode='economic')
    >>> q3.shape, r3.shape
    ((9, 6), (6, 6))

    >>> q4, r4, p4 = rlinalg.qr(a)
    >>> d = numpy.abs(numpy.diag(r4))

    >>> numpy.allclose(a[:, p4], numpy.dot(q4, r4))
    True
    >>> q4.shape, r4.shape, p4.shape
    ((9, 9), (9, 6), (6,))

    >>> q5, r5, p5 = rlinalg.qr(a, mode='economic')
    >>> q5.shape, r5.shape, p5.shape
    ((9, 6), (6, 6), (6,))

    """
    if mode not in {"full", "qr", "r", "economic", "raw"}:
        raise ValueError(
            "Mode argument should be one of ['full', 'r', 'economic', 'raw']"
        )

    asarray = numpy.asarray_chkfinite if check_finite else numpy.asarray
    a1 = asarray(a, order="F", dtype=numpy.double)
    if len(a1.shape) != 2:
        raise ValueError("expected a 2-D array")

    M, N = a1.shape

    # accomodate empty arrays
    if a1.size == 0:
        raise NotImplementedError

    overwrite_a = overwrite_a or _datacopied(a1, a)

    QR, k, tau, P = linpack.dqrdc2(a1, tol=tol, overwrite_x=overwrite_a)
    P -= 1  # <-- dqrdc2 returns indices that are one-based

    # TODO: Avoid copying data (`numpy.triu` does) if possible?
    if mode not in {"economic", "raw"} or M < N:
        R = numpy.triu(QR)
    else:
        R = numpy.triu(QR[:N, :])

    if mode == "r":
        return R, P
    elif mode == "raw":
        return ((QR, tau), R, P)

    if M < N:
        D = numpy.eye(M, dtype=numpy.double, order="F")
        Q = linpack.dqrqy(QR[:, :M], tau[:M], D)
    elif mode == "economic":
        D = numpy.eye(M, N, dtype=numpy.double, order="F")
        Q = linpack.dqrqy(QR, tau, D)
    else:
        D = numpy.eye(M, dtype=numpy.double, order="F")
        Q = linpack.dqrqy(QR, tau, D)

    return QRResult(Q, R, P)
