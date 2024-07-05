import unittest

import numpy
from numpy.testing import assert_allclose

from rlinalg import lstsq


class TestLstsq(unittest.TestCase):

    def test_simple_exact(self):
        a = numpy.array([[1, 20], [-30, 4]], dtype=numpy.double)
        for overwrite in (True, False):
            for bt in (((1, 0), (0, 1)), (1, 0),
                        ((2, 1), (-30, 4))):
                # Store values in case they are overwritten
                # later
                a1 = a.copy()
                b = numpy.array(bt, dtype=numpy.double)
                b1 = b.copy()
                out = lstsq(a1, b1, overwrite_a=overwrite)
                x = out[0]
                r = out[2]
                self.assertEqual(r, 2, f'expected efficient rank 2, got {r}')
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
            residuals = numpy.sum((b - a.dot(x))**2)
            # else:
            #     residuals = out[1]
            r = out[2]
            self.assertEqual(r, 2, f'expected efficient rank 2, got {r}')
            assert_allclose(abs((numpy.dot(a, x) - b)**2).sum(axis=0),
                            residuals, atol=1e-15)
            assert_allclose(x, (-0.428571428571429, 0.85714285714285),
                            atol=1e-15)

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
            self.assertEqual(r, 2, f'expected efficient rank 2, got {r}')
            assert_allclose(x, (-0.3333333, 0.6666667, 0.0), atol=1e-7)

    # @pytest.mark.parametrize("dtype", REAL_DTYPES)
    # @pytest.mark.parametrize("n", (20, 200))
    # @pytest.mark.parametrize("lapack_driver", lapack_drivers)
    # @pytest.mark.parametrize("overwrite", (True, False))
    # def test_random_exact(self, dtype, n, lapack_driver, overwrite):
    #     rng = np.random.RandomState(1234)

    #     a = np.asarray(rng.random([n, n]), dtype=dtype)
    #     for i in range(n):
    #         a[i, i] = 20 * (0.1 + a[i, i])
    #     for i in range(4):
    #         b = np.asarray(rng.random([n, 3]), dtype=dtype)
    #         # Store values in case they are overwritten later
    #         a1 = a.copy()
    #         b1 = b.copy()
    #         out = lstsq(a1, b1,
    #                     lapack_driver=lapack_driver,
    #                     overwrite_a=overwrite,
    #                     overwrite_b=overwrite)
    #         x = out[0]
    #         r = out[2]
    #         assert_(r == n, f'expected efficient rank {n}, '
    #                 f'got {r}')
    #         if dtype is np.float32:
    #             assert_allclose(
    #                       dot(a, x), b,
    #                       rtol=500 * _eps_cast(a1.dtype),
    #                       atol=500 * _eps_cast(a1.dtype),
    #                       err_msg=f"driver: {lapack_driver}")
    #         else:
    #             assert_allclose(
    #                       dot(a, x), b,
    #                       rtol=1000 * _eps_cast(a1.dtype),
    #                       atol=1000 * _eps_cast(a1.dtype),
    #                       err_msg=f"driver: {lapack_driver}")

    # @pytest.mark.skipif(IS_MUSL, reason="may segfault on Alpine, see gh-17630")
    # @pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
    # @pytest.mark.parametrize("n", (20, 200))
    # @pytest.mark.parametrize("lapack_driver", lapack_drivers)
    # @pytest.mark.parametrize("overwrite", (True, False))
    # def test_random_complex_exact(self, dtype, n, lapack_driver, overwrite):
    #     rng = np.random.RandomState(1234)

    #     a = np.asarray(rng.random([n, n]) + 1j*rng.random([n, n]),
    #                    dtype=dtype)
    #     for i in range(n):
    #         a[i, i] = 20 * (0.1 + a[i, i])
    #     for i in range(2):
    #         b = np.asarray(rng.random([n, 3]), dtype=dtype)
    #         # Store values in case they are overwritten later
    #         a1 = a.copy()
    #         b1 = b.copy()
    #         out = lstsq(a1, b1, lapack_driver=lapack_driver,
    #                     overwrite_a=overwrite,
    #                     overwrite_b=overwrite)
    #         x = out[0]
    #         r = out[2]
    #         assert_(r == n, f'expected efficient rank {n}, '
    #                 f'got {r}')
    #         if dtype is np.complex64:
    #             assert_allclose(
    #                       dot(a, x), b,
    #                       rtol=400 * _eps_cast(a1.dtype),
    #                       atol=400 * _eps_cast(a1.dtype),
    #                       err_msg=f"driver: {lapack_driver}")
    #         else:
    #             assert_allclose(
    #                       dot(a, x), b,
    #                       rtol=1000 * _eps_cast(a1.dtype),
    #                       atol=1000 * _eps_cast(a1.dtype),
    #                       err_msg=f"driver: {lapack_driver}")

    # def test_random_overdet(self):
    #     rng = np.random.RandomState(1234)
    #     for dtype in REAL_DTYPES:
    #         for (n, m) in ((20, 15), (200, 2)):
    #             for lapack_driver in TestLstsq.lapack_drivers:
    #                 for overwrite in (True, False):
    #                     a = np.asarray(rng.random([n, m]), dtype=dtype)
    #                     for i in range(m):
    #                         a[i, i] = 20 * (0.1 + a[i, i])
    #                     for i in range(4):
    #                         b = np.asarray(rng.random([n, 3]), dtype=dtype)
    #                         # Store values in case they are overwritten later
    #                         a1 = a.copy()
    #                         b1 = b.copy()
    #                         out = lstsq(a1, b1,
    #                                     lapack_driver=lapack_driver,
    #                                     overwrite_a=overwrite,
    #                                     overwrite_b=overwrite)
    #                         x = out[0]
    #                         r = out[2]
    #                         assert_(r == m, f'expected efficient rank {m}, '
    #                                 f'got {r}')
    #                         assert_allclose(
    #                                       x, direct_lstsq(a, b, cmplx=0),
    #                                       rtol=25 * _eps_cast(a1.dtype),
    #                                       atol=25 * _eps_cast(a1.dtype),
    #                                       err_msg=f"driver: {lapack_driver}")

    # def test_random_complex_overdet(self):
    #     rng = np.random.RandomState(1234)
    #     for dtype in COMPLEX_DTYPES:
    #         for (n, m) in ((20, 15), (200, 2)):
    #             for lapack_driver in TestLstsq.lapack_drivers:
    #                 for overwrite in (True, False):
    #                     a = np.asarray(rng.random([n, m]) + 1j*rng.random([n, m]),
    #                                    dtype=dtype)
    #                     for i in range(m):
    #                         a[i, i] = 20 * (0.1 + a[i, i])
    #                     for i in range(2):
    #                         b = np.asarray(rng.random([n, 3]), dtype=dtype)
    #                         # Store values in case they are overwritten
    #                         # later
    #                         a1 = a.copy()
    #                         b1 = b.copy()
    #                         out = lstsq(a1, b1,
    #                                     lapack_driver=lapack_driver,
    #                                     overwrite_a=overwrite,
    #                                     overwrite_b=overwrite)
    #                         x = out[0]
    #                         r = out[2]
    #                         assert_(r == m, f'expected efficient rank {m}, '
    #                                 f'got {r}')
    #                         assert_allclose(
    #                                   x, direct_lstsq(a, b, cmplx=1),
    #                                   rtol=25 * _eps_cast(a1.dtype),
    #                                   atol=25 * _eps_cast(a1.dtype),
    #                                   err_msg=f"driver: {lapack_driver}")

    # def test_check_finite(self):
    #     with suppress_warnings() as sup:
    #         # On (some) OSX this tests triggers a warning (gh-7538)
    #         sup.filter(RuntimeWarning,
    #                    "internal gelsd driver lwork query error,.*"
    #                    "Falling back to 'gelss' driver.")

    #     at = np.array(((1, 20), (-30, 4)))
    #     for dtype, bt, lapack_driver, overwrite, check_finite in \
    #         itertools.product(REAL_DTYPES,
    #                           (((1, 0), (0, 1)), (1, 0), ((2, 1), (-30, 4))),
    #                           TestLstsq.lapack_drivers,
    #                           (True, False),
    #                           (True, False)):

    #         a = at.astype(dtype)
    #         b = np.array(bt, dtype=dtype)
    #         # Store values in case they are overwritten
    #         # later
    #         a1 = a.copy()
    #         b1 = b.copy()
    #         out = lstsq(a1, b1, lapack_driver=lapack_driver,
    #                     check_finite=check_finite, overwrite_a=overwrite,
    #                     overwrite_b=overwrite)
    #         x = out[0]
    #         r = out[2]
    #         assert_(r == 2, f'expected efficient rank 2, got {r}')
    #         assert_allclose(dot(a, x), b,
    #                         rtol=25 * _eps_cast(a.dtype),
    #                         atol=25 * _eps_cast(a.dtype),
    #                         err_msg=f"driver: {lapack_driver}")

    # def test_empty(self):
    #     for a_shape, b_shape in (((0, 2), (0,)),
    #                              ((0, 4), (0, 2)),
    #                              ((4, 0), (4,)),
    #                              ((4, 0), (4, 2))):
    #         b = np.ones(b_shape)
    #         x, residues, rank, s = lstsq(np.zeros(a_shape), b)
    #         assert_equal(x, np.zeros((a_shape[1],) + b_shape[1:]))
    #         residues_should_be = (np.empty((0,)) if a_shape[1]
    #                               else np.linalg.norm(b, axis=0)**2)
    #         assert_equal(residues, residues_should_be)
    #         assert_(rank == 0, 'expected rank 0')
    #         assert_equal(s, np.empty((0,)))

    # @pytest.mark.parametrize('dt_a', [int, float, np.float32, complex, np.complex64])
    # @pytest.mark.parametrize('dt_b', [int, float, np.float32, complex, np.complex64])
    # def test_empty_dtype(self, dt_a, dt_b):
    #     a = np.empty((0, 0), dtype=dt_a)
    #     b = np.empty(0, dtype=dt_b)
    #     x, residues, rank, s = lstsq(a, b)

    #     assert x.size == 0
    #     dt_nonempty = lstsq(np.eye(2, dtype=dt_a), np.ones(2, dtype=dt_b))[0].dtype
    #     assert x.dtype == dt_nonempty
