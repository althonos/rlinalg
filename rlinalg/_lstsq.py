# File rlinalg/_lstsq.py
# Part of the `rlinalg` package, https://rlinalg.readthedocs.io
#
# The `rlinalg` library is developed and released under the GNU General
# Public License version 3 or later:
#
#    Copyright (C) 2024 Martin Larralde <martin.larralde@embl.de>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# The `rlinalg` library distributes, builds and links to code from the
# R project, redistributed and modified under the GNU General Public License
# version 2 or later:
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    A copy of the GNU General Public License is available at
#    https://www.R-project.org/Licenses/
#
# The `rlinalg` library contains code adapted from SciPy, redistributed
# and modified under the BSD-3-clause license, see original SciPy
# copyright below:
#
#    Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions
#    are met:
#
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#    3. Neither the name of the copyright holder nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
from __future__ import annotations

import typing

import numpy

from . import linpack
from ._misc import _datacopied, set_module, _asarray_validated

if typing.TYPE_CHECKING:
    import numpy.typing


class LSTSQResult(typing.NamedTuple):
    x: numpy.ndarray
    residues: numpy.ndarray
    rank: int


@set_module("rlinalg")
def lstsq(
    a: numpy.typing.ArrayLike,
    b: numpy.typing.ArrayLike,
    tol: float = 1e-7,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> LSTSQResult:
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
    the same behaviour as ``lm.fit``.

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

    References
    ----------
    Chambers, J. M. (1992) *Linear models.* Chapter 4 of *Statistical Models
    in S* eds J. M. Chambers and T. J. Hastie, Wadsworth & Brooks/Cole.

    """
    a1 = _asarray_validated(a, order="F", dtype=numpy.double, check_finite=check_finite)
    if len(a1.shape) != 2:
        raise ValueError("expected a 2-D array")
    m, n = a1.shape

    b1 = _asarray_validated(b, order="F", dtype=numpy.double, check_finite=check_finite)
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
        return LSTSQResult(x, residues, 0)

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

    return LSTSQResult(x, residues, rank)
