import numpy


def set_module(module):
    """Private decorator for overriding __module__ on a function or class.

    Note
    ----
    Taken from NumPy.

    """

    def decorator(func):
        func.__module__ = module
        return func

    return decorator


def _asarray_validated(
    a,
    check_finite=True,
    sparse_ok=False,
    mask_ok=False,
    dtype=numpy.double,
    order="F",
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
        if a.__class__.__module__.startswith("scipy.sparse"):
            import scipy.sparse

            if scipy.sparse.issparse(a):
                msg = (
                    "Sparse matrices are not supported by this function. "
                    "Perhaps one of the scipy.sparse.linalg functions "
                    "would work instead."
                )
                raise ValueError(msg)
    if not mask_ok:
        if numpy.ma.isMaskedArray(a):
            raise ValueError("masked arrays are not supported")
    toarray = numpy.asarray_chkfinite if check_finite else numpy.asarray
    a = toarray(a, dtype=dtype, order=order)
    return a


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
