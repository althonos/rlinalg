# File rlinalg/linpack/__init__.py
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
"""Low-level LAPACK functions (:mod:`rlinalg.lapack`)

This module contains low-level functions from the LINPACK library.

.. note::

    The common ``overwrite_<>`` option in many routines, allows the
    input arrays to be overwritten to avoid extra memory allocation.
    However this requires the array to satisfy two conditions
    which are memory order and the data type to match exactly the
    order and the type expected by the routine.

    As an example, if you pass a double precision float array to any
    ``S....`` routine which expects single precision arguments, f2py
    will create an intermediate array to match the argument types and
    overwriting will be performed on that intermediate array.

    Similarly, if a C-contiguous array is passed, f2py will pass a
    FORTRAN-contiguous array internally. Please make sure that these
    details are satisfied. More information can be found in the f2py
    documentation.

.. warning::

    These functions do little to no error checking.
    It is possible to cause crashes by mis-using them,
    so prefer using the higher-level routines in `rlinalg`.

"""
import functools

from .._misc import set_module
from ._linpack import (
    dqrdc2,
    dtrco,
    dqrls,
    dqrsl,
    dqrqy,
    dqrqty,
)

__all__ = ["dqrdc2", "dqrqy", "dqrqty", "dtrco", "dqrsl", "dqrls"]
