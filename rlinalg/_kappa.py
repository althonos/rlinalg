import math

import numpy

from . import linpack
from ._misc import _asarray_validated, set_module
from ._decomp_qr import qr


def _kappa_tri(a, lower=False, norm=None, check_finite=True):

    a1 = numpy.atleast_2d(_asarray_validated(a, check_finite=check_finite))
    m, n = a1.shape

    if m != n:
        raise ValueError("matrix must be square")

    if norm is None:
        norm = 1
    elif norm == numpy.inf:
        a1 = a1.T

    rcond, _ = linpack.dtrco(a1, lower=lower, check_finite=False)
    return 1 / rcond


@set_module("rlinalg")
def kappa(a, exact=False, norm=None, method="qr", check_finite=True, lower=False):
    """
    Compute or estimate the condition number of a matrix.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to compute the condition number for
    exact : bool, optional
        Whether to compute an exact result, or an approximation.
        ``exact=True`` requires ``norm=None`` or ``norm=2``.
    norm : {None, 1, 2, inf}, optional
        Order of the norm used in the condition number computation:
    method : {'qr', 'raw'}, optional
        Determines whether the input matrix should be first decomposed
        using QR decomposition ('qr'), or if the given matrix is already
        triangular ('raw').
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    lower : bool, optional
        Whether `a` is a lower or upper triangular matrix when running in
        'raw' mode. Ignored in 'qr' mode.

    Returns
    -------
    kappa : float
        The condition number, or an approximation if ``exact=False``.

    Examples
    --------
    >>> import numpy
    >>> import rlinalg

    >>> a = numpy.stack([numpy.ones(10), numpy.arange(1, 11)])
    >>> rlinalg.kappa(a)
    15.7059...
    >>> rlinalg.kappa(a, exact=True)
    13.6790...

    >>> b = numpy.concat([a, numpy.arange(2, 12).reshape(1, -1)], axis=0)
    >>> rlinalg.kappa(b)
    4.45702...e+16

    """
    if norm not in {None, 1, 2, numpy.inf}:
        raise ValueError("Mode argument should be one of [None, 1, 2, numpy.inf]")

    if method not in {"qr", "raw"}:
        raise ValueError("Method argument should be one of ['qr', 'raw']")

    a1 = numpy.atleast_2d(_asarray_validated(a, check_finite=check_finite))

    if exact and norm is None or norm == 2:
        s = numpy.linalg.svd(a1, compute_uv=False)
        return (s.max() / s.min(where=s > 0, initial=numpy.inf)).item()

    M, N = a1.shape

    if method == "qr" or M != N:
        R, P = qr(a1.T if M < N else a1, mode="r")
        R = R[: R.shape[1]]
        lower = False
    else:
        R = a1

    return _kappa_tri(R, lower=lower, norm=norm, check_finite=False)
