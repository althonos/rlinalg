import numpy
from numpy import atleast_2d

from ._dqrdc2 import dqrdc2 as _dqrdc2


def _datacopied(arr, original):
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = asarray(original)

    """
    if arr is original:
        return False
    if not isinstance(original, numpy.ndarray) and hasattr(original, "__array__"):
        return False
    return arr.base is None


def _asarray_validated(
    a, 
    check_finite=True,
    sparse_ok=False, 
    mask_ok=False,
    dtype=numpy.double,
    order='F',
):
    """
    Helper function for argument validation (adapted from SciPy).

    Many SciPy linear algebra functions do support arbitrary array-like
    input arguments. Examples of commonly unsupported inputs include
    matrices containing inf/nan, sparse matrix representations, and
    matrices with complicated elements.

    Parameters
    ----------
    a : array_like
        The array-like input.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    sparse_ok : bool, optional
        True if scipy sparse matrices are allowed.
    mask_ok : bool, optional
        True if masked arrays are allowed.
    dtype: numpy.dtype, optional
        The expected datatype of the array.
        Default: numpy.double
    order: {'C', 'F', 'A'}, optional
        The expected order of the array.

    Returns
    -------
    ret : ndarray
        The converted validated array.

    """
    if not sparse_ok:
        import scipy.sparse
        if scipy.sparse.issparse(a):
            msg = ('Sparse matrices are not supported by this function. '
                   'Perhaps one of the scipy.sparse.linalg functions '
                   'would work instead.')
            raise ValueError(msg)
    if not mask_ok:
        if numpy.ma.isMaskedArray(a):
            raise ValueError('masked arrays are not supported')
    toarray = numpy.asarray_chkfinite if check_finite else numpy.asarray
    a = toarray(a, dtype=dtype, order=order)
    return a


def dqrdc2(a, tol=1e-12, overwrite_a=False):
    """
    Wrapper for ``dqrdc2``.

    Parameters
    ----------
    a : input rank-2 array('d') with bound (m,n)
        The array-like input.

    Returns
    -------
    qr: rank-2 array('d') with bounds (m,n) and a storage
    k: int
    jpvt: rank-1 array('i') with bounds (n)
    tau: rank-1 array('d') with bounds (MIN(m,n))
    work: rank-1 array('d') with bounds (n,2)
    info: int

    """

    a1 = atleast_2d(_asarray_validated(a, check_finite=False, order='F', dtype=numpy.double))
    overwrite_a = overwrite_a or _datacopied(a1, a)

    n, p = a1.shape

    tau = numpy.zeros(p, dtype=numpy.double, order='F')
    work = numpy.zeros((p, 2), dtype=numpy.double, order='F')
    jpvt = numpy.arange(1, p + 1, dtype=numpy.int32)

    if not overwrite_a:
        a1 = numpy.array(a1, order='F', dtype=numpy.double)

    try:
        x, _, k, *_ = _dqrdc2(a1, p, n, p, tol, 0, tau, jpvt, work)
    except ValueError as err:
        info = -getattr(err, "errno", 1)
    else:
        info = 0

    return (a1, k, jpvt, tau, work, info)