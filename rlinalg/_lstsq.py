import numpy

from . import linpack
from ._misc import _datacopied, set_module


@set_module("rlinalg")
def lstsq(a, b, tol=1e-7, overwrite_a=False, check_finite=True):
    """
    Compute least-squares solution to equation Ax = b.

    Compute a vector x such that the 2-norm ``|b - A x|`` is minimized.

    Parameters
    ----------
    a : (M, N) array_like
        Left-hand side array
    b : (M,) or (M, K) array_like
        Right hand side array
    tol : float
        The absolute tolerance to use for QR decomposition.
    overwrite_a : bool, optional
        Discard data in `a` (may enhance performance). Default is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (N,) or (N, K) ndarray
        Least-squares solution. For underdetermined systems, ``x[i]`` will
        be zero for all ``i >= rank(A)``.
    residues : (K,) ndarray or float
        Square of the 2-norm for each column in ``b - a x``, if ``M > N`` and
        ``rank(A) == n`` (returns a scalar if ``b`` is 1-D). Otherwise a
        (0,)-shaped array is returned.
    rank : int
        Effective rank of `a`.

    Raises
    ------
    ValueError
        When parameters are not compatible.

    Notes
    -----
    This function uses a QR decomposition rather than a SVD decomposition
    in the case of `numpy.linalg.lstsq` or `scipy.linalg.lstsq`. This is
    the same behaviour as :r:func:`lm.fit`.

    See Also
    --------
    `numpy.linalg.lstsq`
        The NumPy implementation using LAPACK.
    `scipy.linalg.lstsq`
        The NumPy implementation using LAPACK.
    `lm.fit <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm.fit>`_
        Documentation of the equivalent R function ``lm.fit``.

    Examples
    --------
    >>> import numpy
    >>> from rlinalg import lstsq

    Suppose we have the following data:

    >>> x = numpy.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
    >>> y = numpy.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])

    We want to fit a quadratic polynomial of the form ``y = a + b*x**2``
    to this data.  We first form the "design matrix" M, with a constant
    column of 1s and a column containing ``x**2``:

    >>> M = x[:, numpy.newaxis]**[0, 2]
    >>> M
    array([[ 1.  ,  1.  ],
           [ 1.  ,  6.25],
           [ 1.  , 12.25],
           [ 1.  , 16.  ],
           [ 1.  , 25.  ],
           [ 1.  , 49.  ],
           [ 1.  , 72.25]])

    We want to find the least-squares solution to ``M.dot(p) = y``,
    where ``p`` is a vector with length 2 that holds the parameters
    ``a`` and ``b``.

    >>> p, res, rank = lstsq(M, y)
    >>> p
    array([0.20925829, 0.12013861])

    """

    asarray = numpy.asarray_chkfinite if check_finite else numpy.asarray

    a1 = asarray(a, order="F", dtype=numpy.double)
    if len(a1.shape) != 2:
        raise ValueError("expected a 2-D array")
    m, n = a1.shape

    b1 = asarray(b, order="F", dtype=numpy.double)
    vector = len(b1.shape) == 1
    if not 1 <= len(b1.shape) <= 2:
        raise ValueError("expected a 1-D or 2-D array")

    if m != b1.shape[0]:
        raise ValueError(
            "Shape mismatch: a and b should have the same number "
            f"of rows ({m} != {b1.shape[0]})."
        )
    if m == 0 or n == 0:  # Zero-sized problem, confuses LAPACK
        x = numpy.zeros((n,) + b1.shape[1:], dtype=numpy.common_type(a1, b1))
        if n == 0:
            residues = numpy.linalg.norm(b1, axis=0) ** 2
        else:
            residues = numpy.empty((0,))
        return x, residues, 0

    qr, x, residues, _, rank, jpvt, _ = linpack.dqrls(
        x=a1, y=b, tol=tol, overwrite_x=overwrite_a
    )

    # x is sorted after the permutation of the QR decomposition, so it needs
    # to be permuted back before being returned
    rp = jpvt.argsort()
    x = x[rp]

    # return a vector is given a vector
    if vector:
        x = x[:, 0]
        residues = residues[:, 0]

    return x, residues, rank
