import numpy

from . import linpack
from ._misc import _datacopied, set_module


def lstsq(a, b, tol=1e-7, overwrite_a=False, check_finite=True):
    """
    Compute least-squares solution to equation Ax = b.
    """

    asarray = numpy.asarray_chkfinite if check_finite else numpy.asarray

    a1 = asarray(a, order="F", dtype=numpy.double)
    if len(a1.shape) != 2:
        raise ValueError("expected a 2-D array")
    m, n = a1.shape

    b1 = asarray(b, order="F", dtype=numpy.double)
    vector = False
    if len(b1.shape) == 1:
        # b1 = b1.reshape(-1, 1)
        vector = True
    elif len(b1.shape) != 2:
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
