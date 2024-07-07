# File rlinalg/_misc.py
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

if typing.TYPE_CHECKING:
    import numpy.typing
    from typing import Literal


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
    a: numpy.typing.ArrayLike,
    check_finite: bool = True,
    sparse_ok: bool = False,
    mask_ok: bool = False,
    dtype: numpy.typing.DTypeLike = numpy.double,
    order: Literal["C", "F", "A"] = "F",
) -> numpy.ndarray:
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
    return toarray(a, dtype=dtype, order=order)  # type: ignore


def _datacopied(arr: numpy.ndarray, original: numpy.typing.ArrayLike) -> bool:
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = asarray(original)

    """
    if arr is original:
        return False
    if not isinstance(original, numpy.ndarray) and hasattr(original, "__array__"):
        return False
    return arr.base is None
