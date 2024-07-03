import unittest

import numpy
import numpy.testing

from numpy.testing import (
    assert_equal,
    assert_array_almost_equal,
    assert_,
)

from rlinalg import qr


class TestQR(unittest.TestCase):

    def test_simple(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, _ = qr(a)
        assert_array_almost_equal(q.T @ q, numpy.eye(3))
        assert_array_almost_equal(q @ r, a)

    def test_simple_left(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, _ = qr(a)
        c = [1, 2, 3]
        # qc, r2 = qr_multiply(a, c, "left")
        # assert_array_almost_equal(q @ c, qc)
        # assert_array_almost_equal(r, r2)
        # qc, r2 = qr_multiply(a, numpy.eye(3), "left")
        # assert_array_almost_equal(q, qc)

    def test_simple_right(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, _ = qr(a)
        c = [1, 2, 3]
        # qc, r2 = qr_multiply(a, c)
        # assert_array_almost_equal(c @ q, qc)
        # assert_array_almost_equal(r, r2)
        # qc, r = qr_multiply(a, numpy.eye(3))
        # assert_array_almost_equal(q, qc)

    def test_simple_pivoting(self):
        a = numpy.asarray([[8, 2, 3], [2, 9, 3], [5, 3, 6]])
        q, r, p = qr(a)
        d = abs(numpy.diag(r))
        assert_(numpy.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, numpy.eye(3))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2, p2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_left_pivoting(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, p = qr(a)
        c = [1, 2, 3]
        # qc, r, jpvt = qr_multiply(a, c, "left", True)
        # assert_array_almost_equal(q @ c, qc)

    def test_simple_right_pivoting(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, p = qr(a)
        # c = [1, 2, 3]
        # qc, r, jpvt = qr_multiply(a, c)
        # assert_array_almost_equal(c @ q, qc)

    def test_simple_trap(self):
        a = [[8, 2, 3], [2, 9, 3]]
        q, r, p = qr(a)
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a)

    def test_simple_trap_pivoting(self):
        a = numpy.asarray([[8, 2, 3], [2, 9, 3]])
        q, r, p = qr(a)
        d = abs(numpy.diag(r))
        assert_(numpy.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2, p2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_tall(self):
        # full version
        a = [[8, 2], [2, 9], [5, 3]]
        q, r, p = qr(a)
        assert_array_almost_equal(q.T @ q, numpy.eye(3))
        assert_array_almost_equal(q @ r, a)

    def test_simple_tall_pivoting(self):
        # full version pivoting
        a = numpy.asarray([[8, 2], [2, 9], [5, 3]])
        q, r, p = qr(a)
        d = abs(numpy.diag(r))
        assert_(numpy.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, numpy.eye(3))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2, p2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_tall_e(self):
        # economy version
        a = [[8, 2], [2, 9], [5, 3]]
        q, r, p = qr(a, mode="economic")
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (3, 2))
        assert_equal(r.shape, (2, 2))

    def test_simple_tall_e_pivoting(self):
        # economy version pivoting
        a = numpy.asarray([[8, 2], [2, 9], [5, 3]])
        q, r, p = qr(a, mode="economic")
        # d = abs(numpy.diag(r))
        # assert_(numpy.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2, p2 = qr(a[:, p], mode="economic")
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    # def test_simple_tall_left(self):
    #     a = [[8, 2], [2, 9], [5, 3]]
    #     q, r, p = qr(a, mode="economic")
    #     c = [1, 2]
    #     qc, r2 = qr_multiply(a, c, "left")
    #     assert_array_almost_equal(q @ c, qc)
    #     assert_array_almost_equal(r, r2)
    #     c = array([1, 2, 0])
    #     qc, r2 = qr_multiply(a, c, "left", overwrite_c=True)
    #     assert_array_almost_equal(q @ c[:2], qc)
    #     qc, r = qr_multiply(a, numpy.eye(2), "left")
    #     assert_array_almost_equal(qc, q)

    # def test_simple_tall_left_pivoting(self):
    #     a = [[8, 2], [2, 9], [5, 3]]
    #     q, r, jpvt = qr(a, mode="economic")
    #     c = [1, 2]
    #     qc, r, kpvt = qr_multiply(a, c, "left", True)
    #     assert_array_equal(jpvt, kpvt)
    #     assert_array_almost_equal(q @ c, qc)
    #     qc, r, jpvt = qr_multiply(a, numpy.eye(2), "left", True)
    #     assert_array_almost_equal(qc, q)

    # def test_simple_tall_right(self):
    #     a = [[8, 2], [2, 9], [5, 3]]
    #     q, r, p = qr(a, mode="economic")
    #     c = [1, 2, 3]
    #     cq, r2 = qr_multiply(a, c)
    #     assert_array_almost_equal(c @ q, cq)
    #     assert_array_almost_equal(r, r2)
    #     cq, r, p = qr_multiply(a, numpy.eye(3))
    #     assert_array_almost_equal(cq, q)

    # def test_simple_tall_right_pivoting(self):
    #     a = [[8, 2], [2, 9], [5, 3]]
    #     q, r, jpvt = qr(a, mode="economic")
    #     c = [1, 2, 3]
    #     cq, r, jpvt = qr_multiply(a, c)
    #     assert_array_almost_equal(c @ q, cq)
    #     cq, r, jpvt = qr_multiply(a, numpy.eye(3))
    #     assert_array_almost_equal(cq, q)

    def test_simple_fat(self):
        # full version
        a = [[8, 2, 5], [2, 9, 3]]
        q, r, p = qr(a)
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))

    def test_simple_fat_pivoting(self):
        # full version pivoting
        a = numpy.asarray([[8, 2, 5], [2, 9, 3]])
        q, r, p = qr(a)
        d = abs(numpy.diag(r))
        assert_(numpy.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))
        q2, r2, p2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_fat_e(self):
        # economy version
        a = [[8, 2, 3], [2, 9, 5]]
        q, r, p = qr(a, mode="economic")
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))

    def test_simple_fat_e_pivoting(self):
        # economy version pivoting
        a = numpy.asarray([[8, 2, 3], [2, 9, 5]])
        q, r, p = qr(a, mode="economic")
        # d = abs(numpy.diag(r))
        # assert_(numpy.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, numpy.eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))
        q2, r2, p2 = qr(a[:, p], mode="economic")
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    # def test_simple_fat_left(self):
    #     a = [[8, 2, 3], [2, 9, 5]]
    #     q, r, p = qr(a, mode="economic")
    #     c = [1, 2]
    #     qc, r2 = qr_multiply(a, c, "left")
    #     assert_array_almost_equal(q @ c, qc)
    #     assert_array_almost_equal(r, r2)
    #     qc, r = qr_multiply(a, numpy.eye(2), "left")
    #     assert_array_almost_equal(qc, q)

    # def test_simple_fat_left_pivoting(self):
    #     a = [[8, 2, 3], [2, 9, 5]]
    #     q, r, jpvt = qr(a, mode="economic")
    #     c = [1, 2]
    #     qc, r, jpvt = qr_multiply(a, c, "left", True)
    #     assert_array_almost_equal(q @ c, qc)
    #     qc, r, jpvt = qr_multiply(a, numpy.eye(2), "left", True)
    #     assert_array_almost_equal(qc, q)

    # def test_simple_fat_right(self):
    #     a = [[8, 2, 3], [2, 9, 5]]
    #     q, r, p = qr(a, mode="economic")
    #     c = [1, 2]
    #     cq, r2 = qr_multiply(a, c)
    #     assert_array_almost_equal(c @ q, cq)
    #     assert_array_almost_equal(r, r2)
    #     cq, r, p = qr_multiply(a, numpy.eye(2))
    #     assert_array_almost_equal(cq, q)

    # def test_simple_fat_right_pivoting(self):
    #     a = [[8, 2, 3], [2, 9, 5]]
    #     q, r, jpvt = qr(a, mode="economic")
    #     c = [1, 2]
    #     cq, r, jpvt = qr_multiply(a, c)
    #     assert_array_almost_equal(c @ q, cq)
    #     cq, r, jpvt = qr_multiply(a, numpy.eye(2))
    #     assert_array_almost_equal(cq, q)

    # def test_simple_complex(self):
    #     a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
    #     q, r, p = qr(a)
    #     assert_array_almost_equal(q.conj().T @ q, numpy.eye(3))
    #     assert_array_almost_equal(q @ r, a)

    # def test_simple_complex_left(self):
    #     a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
    #     q, r, p = qr(a)
    #     c = [1, 2, 3+4j]
    #     qc, r = qr_multiply(a, c, "left")
    #     assert_array_almost_equal(q @ c, qc)
    #     qc, r = qr_multiply(a, numpy.eye(3), "left")
    #     assert_array_almost_equal(q, qc)

    # def test_simple_complex_right(self):
    #     a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
    #     q, r, p = qr(a)
    #     c = [1, 2, 3+4j]
    #     qc, r = qr_multiply(a, c)
    #     assert_array_almost_equal(c @ q, qc)
    #     qc, r = qr_multiply(a, numpy.eye(3))
    #     assert_array_almost_equal(q, qc)

    # def test_simple_tall_complex_left(self):
    #     a = [[8, 2+3j], [2, 9], [5+7j, 3]]
    #     q, r, p = qr(a, mode="economic")
    #     c = [1, 2+2j]
    #     qc, r2 = qr_multiply(a, c, "left")
    #     assert_array_almost_equal(q @ c, qc)
    #     assert_array_almost_equal(r, r2)
    #     c = array([1, 2, 0])
    #     qc, r2 = qr_multiply(a, c, "left", overwrite_c=True)
    #     assert_array_almost_equal(q @ c[:2], qc)
    #     qc, r = qr_multiply(a, numpy.eye(2), "left")
    #     assert_array_almost_equal(qc, q)

    # def test_simple_complex_left_conjugate(self):
    #     a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
    #     q, r, p = qr(a)
    #     c = [1, 2, 3+4j]
    #     qc, r = qr_multiply(a, c, "left", conjugate=True)
    #     assert_array_almost_equal(q.conj() @ c, qc)

    # def test_simple_complex_tall_left_conjugate(self):
    #     a = [[3, 3+4j], [5, 2+2j], [3, 2]]
    #     q, r, p = qr(a, mode='economic')
    #     c = [1, 3+4j]
    #     qc, r = qr_multiply(a, c, "left", conjugate=True)
    #     assert_array_almost_equal(q.conj() @ c, qc)

    # def test_simple_complex_right_conjugate(self):
    #     a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
    #     q, r, p = qr(a)
    #     c = numpy.array([1, 2, 3+4j])
    #     qc, r = qr_multiply(a, c, conjugate=True)
    #     assert_array_almost_equal(c @ q.conj(), qc)

    # def test_simple_complex_pivoting(self):
    #     a = array([[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]])
    #     q, r, p = qr(a)
    #     d = abs(numpy.diag(r))
    #     assert_(numpy.all(d[1:] <= d[:-1]))
    #     assert_array_almost_equal(q.conj().T @ q, numpy.eye(3))
    #     assert_array_almost_equal(q @ r, a[:, p])
    #     q2, r2, p2 = qr(a[:, p])
    #     assert_array_almost_equal(q, q2)
    #     assert_array_almost_equal(r, r2)

    # def test_simple_complex_left_pivoting(self):
    #     a = array([[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]])
    #     q, r, jpvt = qr(a)
    #     c = [1, 2, 3+4j]
    #     qc, r, jpvt = qr_multiply(a, c, "left", True)
    #     assert_array_almost_equal(q @ c, qc)

    # def test_simple_complex_right_pivoting(self):
    #     a = array([[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]])
    #     q, r, jpvt = qr(a)
    #     c = [1, 2, 3+4j]
    #     qc, r, jpvt = qr_multiply(a, c)
    #     assert_array_almost_equal(c @ q, qc)

    def test_random(self):
        rng = numpy.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            q, r, p = qr(a)
            assert_array_almost_equal(q.T @ q, numpy.eye(n))
            assert_array_almost_equal(q @ r, a)

    # def test_random_left(self):
    #     rng = numpy.random.RandomState(1234)
    #     n = 20
    #     for k in range(2):
    #         a = rng.random([n, n])
    #         q, r, p = qr(a)
    #         c = rng.random([n])
    #         qc, r = qr_multiply(a, c, "left")
    #         assert_array_almost_equal(q @ c, qc)
    #         qc, r = qr_multiply(a, numpy.eye(n), "left")
    #         assert_array_almost_equal(q, qc)

    # def test_random_right(self):
    #     rng = numpy.random.RandomState(1234)
    #     n = 20
    #     for k in range(2):
    #         a = rng.random([n, n])
    #         q, r, p = qr(a)
    #         c = rng.random([n])
    #         cq, r, p = qr_multiply(a, c)
    #         assert_array_almost_equal(c @ q, cq)
    #         cq, r, p = qr_multiply(a, numpy.eye(n))
    #         assert_array_almost_equal(q, cq)

    def test_random_pivoting(self):
        rng = numpy.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            q, r, p = qr(a)
            # d = abs(numpy.diag(r))
            # assert_(numpy.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, numpy.eye(n))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2, p2 = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_tall(self):
        rng = numpy.random.RandomState(1234)
        # full version
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r, p = qr(a)
            assert_array_almost_equal(q.T @ q, numpy.eye(m))
            assert_array_almost_equal(q @ r, a)

    # def test_random_tall_left(self):
    #     rng = numpy.random.RandomState(1234)
    #     # full version
    #     m = 200
    #     n = 100
    #     for k in range(2):
    #         a = rng.random([m, n])
    #         q, r, p = qr(a, mode="economic")
    #         c = rng.random([n])
    #         qc, r = qr_multiply(a, c, "left")
    #         assert_array_almost_equal(q @ c, qc)
    #         qc, r = qr_multiply(a, numpy.eye(n), "left")
    #         assert_array_almost_equal(qc, q)

    # def test_random_tall_right(self):
    #     rng = numpy.random.RandomState(1234)
    #     # full version
    #     m = 200
    #     n = 100
    #     for k in range(2):
    #         a = rng.random([m, n])
    #         q, r, p = qr(a, mode="economic")
    #         c = rng.random([m])
    #         cq, r, p = qr_multiply(a, c)
    #         assert_array_almost_equal(c @ q, cq)
    #         cq, r, p = qr_multiply(a, numpy.eye(m))
    #         assert_array_almost_equal(cq, q)

    def test_random_tall_pivoting(self):
        rng = numpy.random.RandomState(1234)
        # full version pivoting
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r, p = qr(a)
            # d = abs(numpy.diag(r))
            # assert_(numpy.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, numpy.eye(m))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2, p2 = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    # def test_random_tall_e(self):
    #     rng = numpy.random.RandomState(1234)
    #     # economy version
    #     m = 200
    #     n = 100
    #     for k in range(2):
    #         a = rng.random([m, n])
    #         q, r, p = qr(a, mode='economic')
    #         assert_array_almost_equal(q.T @ q, numpy.eye(n))
    #         assert_array_almost_equal(q @ r, a)
    #         assert_equal(q.shape, (m, n))
    #         assert_equal(r.shape, (n, n))

    # def test_random_tall_e_pivoting(self):
    #     rng = numpy.random.RandomState(1234)
    #     # economy version pivoting
    #     m = 200
    #     n = 100
    #     for k in range(2):
    #         a = rng.random([m, n])
    #         q, r, p = qr(a, mode='economic')
    #         d = abs(numpy.diag(r))
    #         assert_(numpy.all(d[1:] <= d[:-1]))
    #         assert_array_almost_equal(q.T @ q, numpy.eye(n))
    #         assert_array_almost_equal(q @ r, a[:, p])
    #         assert_equal(q.shape, (m, n))
    #         assert_equal(r.shape, (n, n))
    #         q2, r2, p2 = qr(a[:, p], mode='economic')
    #         assert_array_almost_equal(q, q2)
    #         assert_array_almost_equal(r, r2)

    def test_random_trap(self):
        rng = numpy.random.RandomState(1234)
        m = 100
        n = 200
        for k in range(2):
            a = rng.random([m, n])
            q, r, p = qr(a)
            assert_array_almost_equal(q.T @ q, numpy.eye(m))
            assert_array_almost_equal(q @ r, a)

    def test_random_trap_pivoting(self):
        rng = numpy.random.RandomState(1234)
        m = 100
        n = 200
        for k in range(2):
            a = rng.random([m, n])
            q, r, p = qr(a)
            # d = abs(numpy.diag(r)) # <-- not true with R's strategy
            # assert_(numpy.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, numpy.eye(m))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2, p2 = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    # def test_random_complex(self):
    #     rng = numpy.random.RandomState(1234)
    #     n = 20
    #     for k in range(2):
    #         a = rng.random([n, n]) + 1j*rng.random([n, n])
    #         q, r, p = qr(a)
    #         assert_array_almost_equal(q.conj().T @ q, numpy.eye(n))
    #         assert_array_almost_equal(q @ r, a)

    # def test_random_complex_left(self):
    #     rng = numpy.random.RandomState(1234)
    #     n = 20
    #     for k in range(2):
    #         a = rng.random([n, n]) + 1j*rng.random([n, n])
    #         q, r, p = qr(a)
    #         c = rng.random([n]) + 1j*rng.random([n])
    #         qc, r = qr_multiply(a, c, "left")
    #         assert_array_almost_equal(q @ c, qc)
    #         qc, r = qr_multiply(a, numpy.eye(n), "left")
    #         assert_array_almost_equal(q, qc)

    # def test_random_complex_right(self):
    #     rng = numpy.random.RandomState(1234)
    #     n = 20
    #     for k in range(2):
    #         a = rng.random([n, n]) + 1j*rng.random([n, n])
    #         q, r, p = qr(a)
    #         c = rng.random([n]) + 1j*rng.random([n])
    #         cq, r, p = qr_multiply(a, c)
    #         assert_array_almost_equal(c @ q, cq)
    #         cq, r, p = qr_multiply(a, numpy.eye(n))
    #         assert_array_almost_equal(q, cq)

    # def test_random_complex_pivoting(self):
    #     rng = numpy.random.RandomState(1234)
    #     n = 20
    #     for k in range(2):
    #         a = rng.random([n, n]) + 1j*rng.random([n, n])
    #         q, r, p = qr(a)
    #         d = abs(numpy.diag(r))
    #         assert_(numpy.all(d[1:] <= d[:-1]))
    #         assert_array_almost_equal(q.conj().T @ q, numpy.eye(n))
    #         assert_array_almost_equal(q @ r, a[:, p])
    #         q2, r2, p2 = qr(a[:, p])
    #         assert_array_almost_equal(q, q2)
    #         assert_array_almost_equal(r, r2)

    def test_check_finite(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, p = qr(a, check_finite=False)
        assert_array_almost_equal(q.T @ q, numpy.eye(3))
        assert_array_almost_equal(q @ r, a)

    # def test_lwork(self):
    #     a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
    #     # Get comparison values
    #     q, r, p = qr(a, lwork=None)

    #     # Test against minimum valid lwork
    #     q2, r2, p2 = qr(a, lwork=3)
    #     assert_array_almost_equal(q2, q)
    #     assert_array_almost_equal(r2, r)

    #     # Test against larger lwork
    #     q3, r3 = qr(a, lwork=10)
    #     assert_array_almost_equal(q3, q)
    #     assert_array_almost_equal(r3, r)

    #     # Test against explicit lwork=-1
    #     q4, r4 = qr(a, lwork=-1)
    #     assert_array_almost_equal(q4, q)
    #     assert_array_almost_equal(r4, r)

    #     # Test against invalid lwork
    #     assert_raises(Exception, qr, (a,), {'lwork': 0})
    #     assert_raises(Exception, qr, (a,), {'lwork': 2})

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

    # @pytest.mark.parametrize(("m", "n"), [(0, 0), (0, 2), (2, 0)])
    # def test_empty(self, m, n):
    #     k = min(m, n)

    #     a = numpy.empty((m, n))
    #     q, r, p = qr(a)
    #     assert_allclose(q, numpy.identity(m))
    #     assert_allclose(r, numpy.empty((m, n)))

    #     q, r, p = qr(a)
    #     assert_allclose(q, numpy.identity(m))
    #     assert_allclose(r, numpy.empty((m, n)))
    #     assert_allclose(p, numpy.arange(n))

    #     r, = qr(a, mode='r')
    #     assert_allclose(r, numpy.empty((m, n)))

    #     q, r, p = qr(a, mode='economic')
    #     assert_allclose(q, numpy.empty((m, k)))
    #     assert_allclose(r, numpy.empty((k, n)))

    #     (raw, tau), r = qr(a, mode='raw')
    #     assert_allclose(raw, numpy.empty((m, n)))
    #     assert_allclose(tau, numpy.empty((k,)))
    #     assert_allclose(r, numpy.empty((k, n)))

    # def test_multiply_empty(self):
    #     a = numpy.empty((0, 0))
    #     c = numpy.empty((0, 0))
    #     cq, r, p = qr_multiply(a, c)
    #     assert_allclose(cq, numpy.empty((0, 0)))

    #     a = numpy.empty((0, 2))
    #     c = numpy.empty((2, 0))
    #     cq, r, p = qr_multiply(a, c)
    #     assert_allclose(cq, numpy.empty((2, 0)))

    #     a = numpy.empty((2, 0))
    #     c = numpy.empty((0, 2))
    #     cq, r, p = qr_multiply(a, c)
    #     assert_allclose(cq, numpy.empty((0, 2)))
