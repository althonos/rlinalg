import collections
import numpy

from . import linpack
from ._misc import _datacopied, set_module


QRResult = collections.namedtuple("QRResult", "Q R P rank")


@set_module("rlinalg")
def qr(a, mode="full", tol=1e-7, check_finite=True, overwrite_a=False):
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
        Of shape (N,). The column permutation.
    rank : int
        The rank of the matrix.

    Raises
    ------
    ValueError
        When parameters are not compatible.

    Notes
    -----
    This is an interface to the R-modified LINPACK routine ``dqrdc2``.

    If ``mode=economic``, the shapes of Q and R are (M, K) and (K, N) instead
    of (M,M) and (M,N), with ``K=min(M,N)``.

    See Also
    --------
    `numpy.linalg.qr`
        The NumPy implementation based on LAPACK.
    `scipy.linalg.qr`
        The SciPy implementation based on LAPACK.
    `qr <https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/qr>`_
        Documentation of the equivalent R function ``qr``.

    Examples
    --------
    >>> import numpy
    >>> import rlinalg
    >>> rng = numpy.random.default_rng()
    >>> a = rng.standard_normal((9, 6))

    >>> q, r, p, k = rlinalg.qr(a)
    >>> numpy.allclose(a, numpy.dot(q, r))
    True
    >>> q.shape, r.shape
    ((9, 9), (9, 6))

    >>> r2, p2, rank = rlinalg.qr(a, mode='r')
    >>> numpy.allclose(r, r2)
    True

    >>> q3, r3, p3, rank = rlinalg.qr(a, mode='economic')
    >>> q3.shape, r3.shape
    ((9, 6), (6, 6))

    >>> q4, r4, p4, rank = rlinalg.qr(a)
    >>> d = numpy.abs(numpy.diag(r4))

    >>> numpy.allclose(a[:, p4], numpy.dot(q4, r4))
    True
    >>> q4.shape, r4.shape, p4.shape
    ((9, 9), (9, 6), (6,))

    >>> q5, r5, p5, rank = rlinalg.qr(a, mode='economic')
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
        K = min(M, N)
        if mode not in ["economic", "raw"]:
            Q = numpy.empty_like(a1, shape=(M, M))
            Q[...] = numpy.identity(M)
            R = numpy.empty_like(a1)
        else:
            Q = numpy.empty_like(a1, shape=(M, K))
            R = numpy.empty_like(a1, shape=(K, N))
        R = R
        P = numpy.arange(N, dtype=numpy.int32)
        if mode == "r":
            return R, P, 0
        elif mode == "raw":
            QR = numpy.empty_like(a1, shape=(M, N))
            tau = numpy.zeros_like(a1, shape=(K,))
            return ((QR, tau), R, P, 0)
        return QRResult(Q, R, P, 0)

    overwrite_a = overwrite_a or _datacopied(a1, a)

    QR, k, tau, P = linpack.dqrdc2(a1, tol=tol, overwrite_x=overwrite_a)
    P -= 1  # <-- dqrdc2 returns indices that are one-based

    # TODO: Avoid copying data (`numpy.triu` does) if possible?
    if mode not in {"economic", "raw"} or M < N:
        R = numpy.triu(QR)
    else:
        R = numpy.triu(QR[:N, :])

    if mode == "r":
        return R, P, k
    elif mode == "raw":
        return ((QR, tau), R, P, k)

    if M < N:
        D = numpy.eye(M, dtype=numpy.double, order="F")
        Q = linpack.dqrqy(
            QR[:, :M],
            tau[:k],
            D,
            k=k,
            overwrite_x=True,
            overwrite_y=True,
        )
    elif mode == "economic":
        D = numpy.eye(M, N, dtype=numpy.double, order="F")
        Q = linpack.dqrqy(
            QR[:, :k],
            tau[:k],
            D,
            k=k,
            overwrite_x=True,
            overwrite_y=True,
        )
    else:
        D = numpy.eye(M, dtype=numpy.double, order="F")
        Q = linpack.dqrqy(
            QR[:, :k],
            tau[:k],
            D,
            k=k,
            overwrite_x=True,
            overwrite_y=True,
        )

    return QRResult(Q, R, P, k)


def qr_multiply(
    a,
    c,
    mode="right",
    tol=1e-7,
    check_finite=True,
    overwrite_a=False,
    overwrite_c=False,
):
    """
    Calculate the QR decomposition and multiply Q with a matrix.

    Calculate the decomposition ``A[:, P] = Q R`` where Q is
    unitary/orthogonal and R upper triangular. Multiply Q with a vector
    or a matrix c.

    Parameters
    ----------
    a : (M, N), array_like
        Input array
    c : array_like
        Input array to be multiplied by ``Q``.
    mode : {'left', 'right'}, optional
        ``Q @ c`` is returned if mode is 'left', ``c @ Q`` is returned if
        mode is 'right'.
        The shape of c must be appropriate for the matrix multiplications,
        if mode is 'left', ``min(a.shape) == c.shape[0]``,
        if mode is 'right', ``a.shape[0] == c.shape[1]``.
    overwrite_a : bool, optional
        Whether data in a is overwritten (may improve performance)
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
    CQ : ndarray
        The product of ``Q`` and ``c``.
    R : (K, N), ndarray
        R array of the resulting QR factorization where ``K = min(M, N)``.
    P : (N,) ndarray
        Integer pivot array.
    rank : int
        The rank of the matrix.

    Raises
    ------
    ValueError
        When parameters are not compatible.

    Notes
    -----
    This is an interface to the R-modified LINPACK routine ``dqrdc2``
    and ``dqrsl``.

    See Also
    --------
    `scipy.linalg.qr`
        The SciPy implementation based on LAPACK.

    Examples
    --------
    >>> import numpy as np
    >>> from rlinalg import qr, qr_multiply
    >>> A = np.array([[1, 3, 3], [2, 3, 2], [2, 3, 3], [1, 3, 2]])
    >>> qc, r1, piv1, rank = qr_multiply(A, 2*np.eye(4))
    >>> qc
    array([[-0.63245553,  1.26491106, -1.        ],
           [-1.26491106, -0.63245553,  1.        ],
           [-1.26491106, -0.63245553, -1.        ],
           [-0.63245553,  1.26491106,  1.        ]])
    >>> r1
    array([[-3.16227766, -5.69209979, -4.74341649],
           [ 0.        ,  1.8973666 ,  1.58113883],
           [ 0.        ,  0.        , -1.        ]])
    >>> piv1
    array([0, 1, 2], dtype=int32)
    >>> q2, r2, piv2, rank = qr(A, mode='economic')
    >>> np.allclose(2*q2 - qc, np.zeros((4, 3)))
    True

    """
    if mode not in {"right", "left"}:
        raise ValueError("Mode argument should be one of ['right', 'left']")

    (QR, tau), R, P, k = qr(
        a, mode="raw", tol=tol, check_finite=check_finite, overwrite_a=overwrite_a
    )
    M, N = QR.shape

    asarray = numpy.asarray_chkfinite if check_finite else numpy.asarray
    c1 = asarray(c, order="F" if mode == "left" else "C", dtype=numpy.double)
    vector = c1.ndim == 1

    if QR.size == 0:
        return numpy.empty_like(c1), R, P, k

    if mode == "left":
        # FIXME: check if padding is really necessary here?
        #        maybe calling dqrsl directly would be sufficient
        if c1.shape[0] < M:
            c1 = numpy.pad(c1, (0, M - c1.shape[0]))
        # compute QC = Q @ c with c of dim (M,1)
        if vector:
            QC = linpack.dqrqy(
                QR[:, :],
                tau[:k],
                c1[:M],
                k=k,
                overwrite_x=True,
                overwrite_y=overwrite_c,
            )
        # compute QC = Q @ c with c of dim (M,*)
        else:
            QC = linpack.dqrqy(
                QR[:, :],
                tau[:k],
                c1[:M],
                k=k,
                overwrite_x=True,
                overwrite_y=overwrite_c,
            )
            # make sure the resulting dimension is at most
            if c1.shape[0] > QR.shape[1]:
                QC = QC[:, : QR.shape[1]]

    elif vector:
        # compute QC = c @ Q with c of dim (1,M)
        if c1.shape[0] != QR.shape[0]:
            raise ValueError(
                f"Array shapes are not compatible for c @ Q operation: {(1, c1.shape[0])} vs {QR.shape}"
            )
        QC = linpack.dqrqty(
            QR[:, :k],
            tau[:k],
            c1[:M],
            k=k,
            overwrite_y=overwrite_c,
            overwrite_x=True,
        )[:k]
    else:
        # compute QC = c @ Q with c of dim (*,M)
        if c1.shape[1] != QR.shape[0]:
            raise ValueError(
                f"Array shapes are not compatible for c @ Q operation: {c1.shape} vs {QR.shape}"
            )
        QC = linpack.dqrqty(
            QR[:, :k],
            tau[:k],
            c1.T[:M],
            k=k,
            overwrite_x=True,
            overwrite_y=overwrite_c,
        ).T[:, :k]

    return QC, R, P, k
