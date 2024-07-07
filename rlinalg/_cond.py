# File rlinalg/_cond.py
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

from __future__ import annotations

import typing
import math

import numpy

from . import linpack
from ._misc import _asarray_validated, set_module, _datacopied
from ._decomp_qr import qr

if typing.TYPE_CHECKING:
    from typing import Literal
    import numpy.typing


def _kappa_tri(
    a: numpy.typing.ArrayLike,
    lower: bool = False,
    norm: typing.Union[Literal[1], Literal[numpy.inf]] = 1,  # type: ignore
    check_finite: bool = True,
) -> float:
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
    a: numpy.typing.ArrayLike,
    overwrite_a: bool = False,
    exact: bool = False,
    norm: typing.Union[Literal[1], Literal[2], Literal[numpy.inf], None] = None,  # type: ignore
    method: typing.Union[Literal["qr"], Literal["raw"]] = "qr",
    check_finite: bool = True,
    lower: bool = False,
) -> float:
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

    References
    ----------
    Anderson., E., et al. (1999) *LAPACK Users' Guide*. Third Edition.
    SIAM. Available on-line at https://netlib.org/lapack/lug/lapack_lug.html.

    Chambers, J. M. (1992) *Linear models.* Chapter 4 of *Statistical Models
    in S* eds J. M. Chambers and T. J. Hastie, Wadsworth & Brooks/Cole.

    Dongarra, J. J., Bunch, J. R., Moler, C. B. and Stewart, G. W.
    (1978) *LINPACK Users Guide.* Philadelphia: SIAM Publications.

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
