# File rlinalg/tests/test_qr.py
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
import unittest

import numpy
import numpy.testing

from numpy.testing import (
    assert_equal,
    assert_allclose,
    assert_array_equal,
    assert_array_almost_equal,
    assert_,
)

from rlinalg import qr, qr_multiply


class TestQR(unittest.TestCase):

    def test_simple(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, _, _ = qr(a)
        assert_array_almost_equal(q.T @ q, numpy.eye(3))
        assert_array_almost_equal(q @ r, a)
        qexp = [[-0.83, 0.27, -0.49], [-0.21, -0.96, -0.18], [-0.52, -0.04, 0.85]]
        assert_allclose(q, qexp, atol=1e-2)

    def test_simple_left(self):
        # from scipy.linalg import qr, qr_multiply
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]

        q, r, *_ = qr(a)
        c = [1, 2, 3]
        qc, r2, *_ = qr_multiply(a, c, "left")
        assert_array_almost_equal(q @ c, qc)
        assert_array_almost_equal(r, r2)
        qc, r2, *_ = qr_multiply(a, numpy.eye(3), "left")
        assert_array_almost_equal(q, qc)
        c = numpy.arange(15).reshape(3, 5)
        cq, r, *_ = qr_multiply(a, c, "left")
        assert_array_almost_equal(cq, q @ c)

    def test_simple_right(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, *_ = qr(a)
        c = [1, 2, 3]
        qc, r2, *_ = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, qc)
        assert_array_almost_equal(r, r2)
        qc, r, *_ = qr_multiply(a, numpy.eye(3))
        assert_array_almost_equal(q, qc)

        c = numpy.arange(15).reshape(5, 3)
        qc, r, *_ = qr_multiply(a, c)
        self.assertEqual(qc.shape, (5, 3))
        assert_array_almost_equal(qc, c @ q)

        with self.assertRaises(ValueError):
            c = numpy.arange(20).reshape(5, 4)
            qc, r, *_ = qr_multiply(a, c)

        with self.assertRaises(ValueError):
            c = numpy.arange(4)
            qc, r, *_ = qr_multiply(a, c)

    def test_simple_pivoting(self):
        a = numpy.asarray([[8, 2, 3], [2, 9, 3], [5, 3, 6]])
        q, r, p, _ = qr(a)
        d = abs(numpy.diag(r))
        assert_(numpy.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, numpy.eye(3))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2, p2, _ = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_left_pivoting(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, p, _ = qr(a)
        c = [1, 2, 3]
        qc, r, jpvt, rank = qr_multiply(a, c, "left")
        assert_array_almost_equal(q @ c, qc)

    def test_simple_right_pivoting(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, p, rank = qr(a)
        c = [1, 2, 3]
        qc, r, jpvt, rank = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, qc)

    def test_simple_trap(self):
        a = [[8, 2, 3], [2, 9, 3]]
        q, r, p, _ = qr(a)
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a)

    def test_simple_trap_pivoting(self):
        a = numpy.asarray([[8, 2, 3], [2, 9, 3]])
        q, r, p, _ = qr(a)
        d = abs(numpy.diag(r))
        assert_(numpy.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2, p2, _ = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_tall(self):
        # full version
        a = [[8, 2], [2, 9], [5, 3]]
        q, r, p, _ = qr(a)
        assert_array_almost_equal(q.T @ q, numpy.eye(3))
        assert_array_almost_equal(q @ r, a)

    def test_simple_tall_pivoting(self):
        # full version pivoting
        a = numpy.asarray([[8, 2], [2, 9], [5, 3]])
        q, r, p, _ = qr(a)
        d = abs(numpy.diag(r))
        assert_(numpy.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, numpy.eye(3))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2, p2, _ = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_tall_e(self):
        # economy version
        a = [[8, 2], [2, 9], [5, 3]]
        q, r, p, _ = qr(a, mode="economic")
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (3, 2))
        assert_equal(r.shape, (2, 2))

    def test_simple_tall_e_pivoting(self):
        # economy version pivoting
        a = numpy.asarray([[8, 2], [2, 9], [5, 3]])
        q, r, p, _ = qr(a, mode="economic")
        # d = abs(numpy.diag(r))
        # assert_(numpy.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2, p2, _ = qr(a[:, p], mode="economic")
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_tall_left(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r, p, _ = qr(a, mode="economic")
        c = [1, 2]
        qc, r2, p2, _ = qr_multiply(a, c, "left")
        assert_array_almost_equal(q @ c, qc)
        assert_array_almost_equal(r, r2)
        c = numpy.array([1, 2, 0])
        qc, r2, p2, _ = qr_multiply(a, c, "left")
        assert_array_almost_equal(q @ c[:2], qc)
        qc, r, p, rank = qr_multiply(a, numpy.eye(2), "left")
        assert_array_almost_equal(qc, q)

    def test_simple_tall_left_pivoting(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r, jpvt, rank = qr(a, mode="economic")
        c = [1, 2]
        qc, r, kpvt, rank = qr_multiply(a, c, "left")
        assert_array_equal(jpvt, kpvt)
        assert_array_almost_equal(q @ c, qc)
        qc, r, *_ = qr_multiply(a, numpy.eye(2), "left")
        assert_array_almost_equal(qc, q)

    def test_shapes(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q2, _, _, _ = qr(a, mode="full")
        self.assertEqual(q2.shape, (3, 3))
        q1, _, _, _ = qr(a, mode="economic")
        self.assertEqual(q1.shape, (3, 2))

    def test_simple_tall_right(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r, p, _ = qr(a, mode="economic")
        c = [1, 2, 3]
        cq, r2, p2, _ = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, cq)
        assert_array_almost_equal(r, r2)
        cq, r, p, _ = qr_multiply(a, numpy.eye(3))
        assert_array_almost_equal(cq, q)

    def test_simple_tall_right_pivoting(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r, jpvt, rank = qr(a, mode="economic")
        c = [1, 2, 3]
        cq, r, jpvt, rank = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, cq)
        cq, r, jpvt, rank = qr_multiply(a, numpy.eye(3))
        assert_array_almost_equal(cq, q)

    def test_simple_fat(self):
        # full version
        a = [[8, 2, 5], [2, 9, 3]]
        q, r, p, _ = qr(a)
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))

    def test_simple_fat_pivoting(self):
        # full version pivoting
        a = numpy.asarray([[8, 2, 5], [2, 9, 3]])
        q, r, p, _ = qr(a)
        d = abs(numpy.diag(r))
        assert_(numpy.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))
        q2, r2, p2, _ = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_fat_e(self):
        # economy version
        a = [[8, 2, 3], [2, 9, 5]]
        q, r, p, _ = qr(a, mode="economic")
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))

    def test_simple_fat_e_pivoting(self):
        # economy version pivoting
        a = numpy.asarray([[8, 2, 3], [2, 9, 5]])
        q, r, p, _ = qr(a, mode="economic")
        # d = abs(numpy.diag(r))
        # assert_(numpy.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))
        q2, r2, p2, _ = qr(a[:, p], mode="economic")
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_fat_left(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r, p, _ = qr(a, mode="economic")
        c = [1, 2]
        qc, r2, p2, _ = qr_multiply(a, c, "left")
        assert_array_almost_equal(q @ c, qc)
        assert_array_almost_equal(r, r2)
        qc, r, p, rank = qr_multiply(a, numpy.eye(2), "left")
        assert_array_almost_equal(qc, q)

    def test_simple_fat_left_pivoting(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r, jpvt, rank = qr(a, mode="economic")
        c = [1, 2]
        qc, r, jpvt, rank = qr_multiply(a, c, "left")
        assert_array_almost_equal(q @ c, qc)
        qc, r, jpvt, rank = qr_multiply(a, numpy.eye(2), "left")
        assert_array_almost_equal(qc, q)

    def test_simple_fat_right(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r, p, _ = qr(a, mode="economic")
        c = [1, 2]
        cq, r2, p2, _ = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, cq)
        assert_array_almost_equal(r, r2)
        cq, r, p, _ = qr_multiply(a, numpy.eye(2))
        assert_array_almost_equal(cq, q)

    def test_simple_fat_right_pivoting(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r, jpvt, rank = qr(a, mode="economic")
        c = [1, 2]
        cq, r, jpvt, rank = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, cq)
        cq, r, jpvt, rank = qr_multiply(a, numpy.eye(2))
        assert_array_almost_equal(cq, q)

    def test_random(self):
        rng = numpy.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            q, r, p, rank = qr(a)
            assert_array_almost_equal(q.T @ q, numpy.eye(n))
            assert_array_almost_equal(q @ r, a)

    def test_random_left(self):
        rng = numpy.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            q, r, p, rank = qr(a)
            c = rng.random([n])
            qc, r, p, rank = qr_multiply(a, c, "left")
            assert_array_almost_equal(q @ c, qc)
            qc, r, p, rank = qr_multiply(a, numpy.eye(n), "left")
            assert_array_almost_equal(q, qc)

    def test_random_right(self):
        rng = numpy.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            q, r, p, rank = qr(a)
            c = rng.random([n])
            cq, r, p, rank = qr_multiply(a, c)
            assert_array_almost_equal(c @ q, cq)
            cq, r, p, rank = qr_multiply(a, numpy.eye(n))
            assert_array_almost_equal(q, cq)

    def test_random_pivoting(self):
        rng = numpy.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            q, r, p, rank = qr(a)
            # d = abs(numpy.diag(r))
            # assert_(numpy.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, numpy.eye(n))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2, p2, rank = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_tall(self):
        rng = numpy.random.RandomState(1234)
        # full version
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r, p, _ = qr(a)
            assert_array_almost_equal(q.T @ q, numpy.eye(m))
            assert_array_almost_equal(q @ r, a)

    def test_random_tall_left(self):
        rng = numpy.random.RandomState(1234)
        # full version
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r, p, _ = qr(a, mode="economic")
            c = rng.random([n])
            qc, r, p, rank = qr_multiply(a, c, "left")
            assert_array_almost_equal(q @ c, qc)
            qc, r, p, rank = qr_multiply(a, numpy.eye(n), "left")
            assert_array_almost_equal(qc, q)

    def test_random_tall_right(self):
        rng = numpy.random.RandomState(1234)
        # full version
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r, p, _ = qr(a, mode="economic")
            c = rng.random([m])
            cq, r, p, _ = qr_multiply(a, c)
            assert_array_almost_equal(c @ q, cq)
            cq, r, p, _ = qr_multiply(a, numpy.eye(m))
            assert_array_almost_equal(cq, q)

    def test_random_tall_pivoting(self):
        rng = numpy.random.RandomState(1234)
        # full version pivoting
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r, p, _ = qr(a)
            # d = abs(numpy.diag(r))
            # assert_(numpy.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, numpy.eye(m))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2, p2, _ = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_tall_e(self):
        rng = numpy.random.RandomState(1234)
        # economy version
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r, p, _ = qr(a, mode="economic")
            assert_array_almost_equal(q.T @ q, numpy.eye(n))
            assert_array_almost_equal(q @ r, a)
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (n, n))

    def test_random_tall_e_pivoting(self):
        rng = numpy.random.RandomState(1234)
        # economy version pivoting
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r, p, _ = qr(a, mode="economic")
            assert_array_almost_equal(q.T @ q, numpy.eye(n))
            assert_array_almost_equal(q @ r, a[:, p])
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (n, n))
            q2, r2, p2, _ = qr(a[:, p], mode="economic")
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_trap(self):
        rng = numpy.random.RandomState(1234)
        m = 100
        n = 200
        for k in range(2):
            a = rng.random([m, n])
            q, r, p, _ = qr(a)
            assert_array_almost_equal(q.T @ q, numpy.eye(m))
            assert_array_almost_equal(q @ r, a)

    def test_random_trap_pivoting(self):
        rng = numpy.random.RandomState(1234)
        m = 100
        n = 200
        for k in range(2):
            a = rng.random([m, n])
            q, r, p, _ = qr(a)
            # d = abs(numpy.diag(r)) # <-- not true with R's strategy
            # assert_(numpy.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, numpy.eye(m))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2, p2, _ = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_check_finite(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, p, _ = qr(a, check_finite=False)
        assert_array_almost_equal(q.T @ q, numpy.eye(3))
        assert_array_almost_equal(q @ r, a)

    # @pytest.mark.parametrize("m", [0, 1, 2])
    # @pytest.mark.parametrize("n", [0, 1, 2])
    # @pytest.mark.parametrize("pivoting", [False, True])
    # @pytest.mark.parametrize('dtype', DTYPES)
    # def test_shape_dtype(self, m, n, pivoting, dtype):
    #     k = min(m, n)

    #     a = numpy.zeros((m, n), dtype=dtype)
    #     q, r, *other = qr(a, pivoting=pivoting)
    #     assert_equal(q.shape, (m, m))
    #     assert_equal(q.dtype, dtype)
    #     assert_equal(r.shape, (m, n))
    #     assert_equal(r.dtype, dtype)
    #     assert len(other) == (1 if pivoting else 0)
    #     if pivoting:
    #         p, = other
    #         assert_equal(p.shape, (n,))
    #         assert_equal(p.dtype, numpy.int32)

    #     r, *other = qr(a, mode='r', pivoting=pivoting)
    #     assert_equal(r.shape, (m, n))
    #     assert_equal(r.dtype, dtype)
    #     assert len(other) == (1 if pivoting else 0)
    #     if pivoting:
    #         p, = other
    #         assert_equal(p.shape, (n,))
    #         assert_equal(p.dtype, numpy.int32)

    #     q, r, *other = qr(a, mode='economic', pivoting=pivoting)
    #     assert_equal(q.shape, (m, k))
    #     assert_equal(q.dtype, dtype)
    #     assert_equal(r.shape, (k, n))
    #     assert_equal(r.dtype, dtype)
    #     assert len(other) == (1 if pivoting else 0)
    #     if pivoting:
    #         p, = other
    #         assert_equal(p.shape, (n,))
    #         assert_equal(p.dtype, numpy.int32)

    #     (raw, tau), r, *other = qr(a, mode='raw', pivoting=pivoting)
    #     assert_equal(raw.shape, (m, n))
    #     assert_equal(raw.dtype, dtype)
    #     assert_equal(tau.shape, (k,))
    #     assert_equal(tau.dtype, dtype)
    #     assert_equal(r.shape, (k, n))
    #     assert_equal(r.dtype, dtype)
    #     assert len(other) == (1 if pivoting else 0)
    #     if pivoting:
    #         p, = other
    #         assert_equal(p.shape, (n,))
    #         assert_equal(p.dtype, numpy.int32)

    def test_empty(self):
        for m, n in [(0, 0), (0, 2), (2, 0)]:
            k = min(m, n)

            a = numpy.empty((m, n))
            q, r, p, rank = qr(a)
            assert_allclose(q, numpy.identity(m))
            assert_allclose(r, numpy.empty((m, n)))
            self.assertEqual(rank, 0)

            q, r, p, rank = qr(a)
            assert_allclose(q, numpy.identity(m))
            assert_allclose(r, numpy.empty((m, n)))
            assert_allclose(p, numpy.arange(n))
            self.assertEqual(rank, 0)

            r, p, rank = qr(a, mode="r")
            assert_allclose(r, numpy.empty((m, n)))
            self.assertEqual(rank, 0)

            q, r, p, rank = qr(a, mode="economic")
            assert_allclose(q, numpy.empty((m, k)))
            assert_allclose(r, numpy.empty((k, n)))
            self.assertEqual(rank, 0)

            (raw, tau), r, p, rank = qr(a, mode="raw")
            assert_allclose(raw, numpy.empty((m, n)))
            assert_allclose(tau, numpy.empty((k,)))
            assert_allclose(r, numpy.empty((k, n)))
            self.assertEqual(rank, 0)

    def test_multiply_empty(self):
        a = numpy.empty((0, 0))
        c = numpy.empty((0, 0))
        cq, r, p, rank = qr_multiply(a, c)
        assert_allclose(cq, numpy.empty((0, 0)))

        a = numpy.empty((0, 2))
        c = numpy.empty((2, 0))
        cq, r, p, rank = qr_multiply(a, c)
        assert_allclose(cq, numpy.empty((2, 0)))

        a = numpy.empty((2, 0))
        c = numpy.empty((0, 2))
        cq, r, p, rank = qr_multiply(a, c)
        assert_allclose(cq, numpy.empty((0, 2)))
