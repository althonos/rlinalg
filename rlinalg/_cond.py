import math

import numpy

from . import linpack
from ._misc import _asarray_validated, set_module, _datacopied
from ._decomp_qr import qr


def _kappa_tri(a, lower=False, norm=None, check_finite=True):
    a1 = numpy.atleast_2d(_asarray_validated(a, check_finite=check_finite))
    m, n = a1.shape

    if m != n:
        raise ValueError("matrix must be square")

    if norm == numpy.inf:
        a1 = a1.T

    rcond = linpack.dtrco(a1, job=1 - bool(lower))
    return 1 / rcond


@set_module("rlinalg")
def cond(
    a,
    overwrite_a=False,
    exact=False,
    norm=None,
    method="qr",
    check_finite=True,
    lower=False,
):
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

    Raises
    ------
    ValueError
        When parameters are not compatible.

    Notes
    -----
    This is an interface to the R-modified LINPACK routine ``dqrdc2`` and
    ``dtrco``.

    When ``exact=False``, this function uses a QR decomposition rather than
    a SVD decomposition in the case of `numpy.linalg.cond`. This is the same
    behaviour as the R function ``kappa``.

    See Also
    --------
    `numpy.linalg.cond`
        The NumPy implementation using an SVD decomposition.
    `kappa <https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/kappa>`_
        Documentation of the equivalent R function ``kappa``.

    Examples
    --------
    >>> import numpy
    >>> import rlinalg

    >>> a = numpy.stack([numpy.ones(10), numpy.arange(1, 11)])
    >>> rlinalg.cond(a)
    15.7059...
    >>> rlinalg.cond(a, exact=True)
    13.6790...
    >>> float(numpy.linalg.cond(a))
    13.6790...

    """
    if norm not in {None, 1, 2, numpy.inf}:
        raise ValueError("Mode argument should be one of [None, 1, 2, numpy.inf]")

    if method not in {"qr", "raw"}:
        raise ValueError("Method argument should be one of ['qr', 'raw']")

    a1 = numpy.atleast_2d(_asarray_validated(a, check_finite=check_finite))
    overwrite_a = overwrite_a or _datacopied(a1, a)

    if exact and norm is None or norm == 2:
        s = numpy.linalg.svd(a1, compute_uv=False)
        return (s.max() / s.min(where=s > 0, initial=numpy.inf)).item()

    M, N = a1.shape

    if method == "qr" or M != N:
        R, P, rank = qr(
            a1.T if M < N else a1, mode="r", overwrite_a=overwrite_a, check_finite=False
        )
        R = R[: min(M, N)]
        lower = False
    else:
        R = a1

    return _kappa_tri(R, lower=lower, norm=norm, check_finite=False)
