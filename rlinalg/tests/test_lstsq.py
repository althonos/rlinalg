# File rlinalg/tests/test_lstsq.py
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
# The `linalg` library contains code adapted from SciPy, redistributed
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
import itertools
import unittest

import numpy
from numpy.testing import assert_allclose, assert_equal

from rlinalg import lstsq


class TestLstsq(unittest.TestCase):

    def test_simple_exact(self):
        a = numpy.array([[1, 20], [-30, 4]], dtype=numpy.double)
        for overwrite in (True, False):
            for bt in (((1, 0), (0, 1)), (1, 0), ((2, 1), (-30, 4))):
                # Store values in case they are overwritten
                # later
                a1 = a.copy()
                b = numpy.array(bt, dtype=numpy.double)
                b1 = b.copy()
                out = lstsq(a1, b1, overwrite_a=overwrite)
                x = out[0]
                r = out[2]
                self.assertEqual(r, 2, f"expected efficient rank 2, got {r}")
                assert_allclose(numpy.dot(a, x), b, atol=1e-15)

    def test_simple_overdet(self):
        a = numpy.array([[1, 2], [4, 5], [3, 4]], dtype=numpy.double)
        b = numpy.array([1, 2, 3], dtype=numpy.double)
        for overwrite in (True, False):
            # Store values in case they are overwritten later
            a1 = a.copy()
            b1 = b.copy()
            out = lstsq(a1, b1, overwrite_a=overwrite)
            x = out[0]
            # if lapack_driver == 'gelsy':
            residuals = numpy.sum((b - a.dot(x)) ** 2)
            # else:
            #     residuals = out[1]
            r = out[2]
            self.assertEqual(r, 2, f"expected efficient rank 2, got {r}")
            assert_allclose(
                abs((numpy.dot(a, x) - b) ** 2).sum(axis=0), residuals, atol=1e-15
            )
            assert_allclose(x, (-0.428571428571429, 0.85714285714285), atol=1e-15)

    def test_simple_underdet(self):
        a = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=numpy.double)
        b = numpy.array([1, 2], dtype=numpy.double)
        for overwrite in (True, False):
            # Store values in case they are overwritten later
            a1 = a.copy()
            b1 = b.copy()
            out = lstsq(a1, b1, overwrite_a=overwrite)
            x = out[0]
            r = out[2]
            self.assertEqual(r, 2, f"expected efficient rank 2, got {r}")
            assert_allclose(x, (-0.3333333, 0.6666667, 0.0), atol=1e-7)

    def test_random_exact(self):
        rng = numpy.random.RandomState(1234)
        for n in (20, 200):
            for overwrite in (True, False):
                a = numpy.asarray(rng.random([n, n]), dtype=numpy.double)
                for i in range(n):
                    a[i, i] = 20 * (0.1 + a[i, i])
                for i in range(4):
                    b = numpy.asarray(rng.random([n, 3]), dtype=numpy.double)
                    # Store values in case they are overwritten later
                    a1 = a.copy()
                    b1 = b.copy()
                    out = lstsq(a1, b1, overwrite_a=overwrite)
                    x = out[0]
                    r = out[2]
                    self.assertEqual(r, n, f"expected efficient rank {n}, got {r}")
                    assert_allclose(numpy.dot(a, x), b, atol=1e-7)

    def test_random_overdet(self):
        rng = numpy.random.RandomState(1234)
        for n, m in ((20, 15), (200, 2)):
            for overwrite in (True, False):
                a = numpy.asarray(rng.random([n, m]), dtype=numpy.double)
                for i in range(m):
                    a[i, i] = 20 * (0.1 + a[i, i])
                for i in range(4):
                    b = numpy.asarray(rng.random([n, 3]), dtype=numpy.double)
                    # Store values in case they are overwritten later
                    a1 = a.copy()
                    b1 = b.copy()
                    out = lstsq(a1, b1, overwrite_a=overwrite)
                    x = out[0]
                    r = out[2]
                    self.assertEqual(r, m, f"expected efficient rank {m}, got {r}")
                    assert_allclose(
                        x, numpy.linalg.lstsq(a, b, rcond=None)[0], atol=1e-7
                    )

    def test_random_underdet(self):
        rng = numpy.random.RandomState(1234)
        for n, m in ((15, 20), (2, 200)):
            for overwrite in (True, False):
                a = numpy.asarray(rng.random([n, m]), dtype=numpy.double)
                for i in range(n):
                    a[i, i] = 20 * (0.1 + a[i, i])
                for i in range(4):
                    b = numpy.asarray(rng.random([n, 3]), dtype=numpy.double)
                    # Store values in case they are overwritten later
                    a1 = a.copy()
                    b1 = b.copy()
                    out = lstsq(a1, b1, overwrite_a=overwrite)
                    x = out[0]
                    r = out[2]
                    self.assertEqual(r, n, f"expected efficient rank {m}, got {r}")
                    assert_allclose(x[r:], 0.0, atol=1e-7)
                    assert_allclose(numpy.dot(a, x), b, atol=1e-7)

    def test_check_finite(self):
        # with suppress_warnings() as sup:
        #     # On (some) OSX this tests triggers a warning (gh-7538)
        #     sup.filter(RuntimeWarning,
        #                "internal gelsd driver lwork query error,.*"
        #                "Falling back to 'gelss' driver.")

        at = numpy.array(((1, 20), (-30, 4)))
        for bt, overwrite, check_finite in itertools.product(
            (((1, 0), (0, 1)), (1, 0), ((2, 1), (-30, 4))), (True, False), (True, False)
        ):

            a = at.astype(numpy.double)
            b = numpy.array(bt, dtype=numpy.double)
            # Store values in case they are overwritten
            # later
            a1 = a.copy()
            b1 = b.copy()
            out = lstsq(a1, b1, check_finite=check_finite, overwrite_a=overwrite)
            x = out[0]
            r = out[2]
            self.assertEqual(r, 2, f"expected efficient rank 2, got {r}")
            assert_allclose(numpy.dot(a, x), b, atol=1e-7)

    def test_empty(self):
        for a_shape, b_shape in (
            ((0, 2), (0,)),
            ((0, 4), (0, 2)),
            ((4, 0), (4,)),
            ((4, 0), (4, 2)),
        ):
            b = numpy.ones(b_shape)
            x, residues, rank = lstsq(numpy.zeros(a_shape), b)
            assert_equal(x, numpy.zeros((a_shape[1],) + b_shape[1:]))
            residues_should_be = (
                numpy.empty((0,)) if a_shape[1] else numpy.linalg.norm(b, axis=0) ** 2
            )
            assert_equal(residues, residues_should_be)
            self.assertEqual(rank, 0, "expected rank 0")

    # @pytest.mark.parametrize('dt_a', [int, float, np.float32, complex, np.complex64])
    # @pytest.mark.parametrize('dt_b', [int, float, np.float32, complex, np.complex64])
    # def test_empty_dtype(self, dt_a, dt_b):
    #     a = np.empty((0, 0), dtype=dt_a)
    #     b = np.empty(0, dtype=dt_b)
    #     x, residues, rank, s = lstsq(a, b)

    #     assert x.size == 0
    #     dt_nonempty = lstsq(np.eye(2, dtype=dt_a), np.ones(2, dtype=dt_b))[0].dtype
    #     assert x.dtype == dt_nonempty
