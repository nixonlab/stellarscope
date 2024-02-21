# -*- coding: utf-8 -*-
import pytest

import numpy as np
from stellarscope.utils.sparse_plus import csr_matrix_plus

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2023 Matthew L. Bendall"

def sparse_equal(m1, m2):
    if m1.shape != m2.shape:
        return False
    return (m1!=m2).nnz == 0


class TestSparsePlusSimple:
    # Simple 3x3 sparse matrix
    m1 = csr_matrix_plus([[1, 0, 2],
                          [0, 0, 3],
                          [4, 5, 6]])

    # 3x3 sparse matrix where row 2 is all zero
    m1z = csr_matrix_plus([[1, 0, 2],
                           [0, 0, 0],
                           [4, 5, 6]])

    # Simple 3x3 sparse matrix
    m2 = csr_matrix_plus([[6, 0, 2],
                          [0, 0, 3],
                          [4, 5, 6]])
    def test_mplus_identity(self):
        assert self.m1[0, 0] == 1
        assert self.m1[0, 2] == 2
        assert self.m1[1, 2] == 3
        assert self.m1[2, 0] == 4
        assert self.m1[2, 1] == 5
        assert self.m1[2, 2] == 6

    def test_mplus_norm(self):
        assert sparse_equal(
            self.m1.norm(),
            csr_matrix_plus([[(1. / 21),         0, (2. / 21)],
                             [        0,         0, (3. / 21)],
                             [(4. / 21), (5. / 21), (6. / 21)]]
                            )
        )

    def test_mplus_norm_row(self):
        assert sparse_equal(
            self.m1.norm(1),
            csr_matrix_plus([[   (1./3),        0,   (2./3)],
                             [        0,        0,       1.],
                             [  (4./15),  (5./15),  (6./15)]]
                            )
        )
    def test_mplus_norm_row_withzero(self):
        assert sparse_equal(
            self.m1z.norm(1),
            csr_matrix_plus([[   (1./3),        0,   (2./3)],
                             [        0,        0,        0],
                             [  (4./15),  (5./15),  (6./15)]]
                            )
        )

    def test_binmax_noaxis(self):
        assert sparse_equal(
            self.m2.binmax(),
            csr_matrix_plus([[ 1, 0, 0],
                             [ 0, 0, 0],
                             [ 0, 0, 1]]
                            )
        )

    def test_binmax_row(self):
        assert sparse_equal(
            self.m2.binmax(1),
            csr_matrix_plus([[ 1, 0, 0],
                             [ 0, 0, 1],
                             [ 0, 0, 1]]
                            )
        )


@pytest.fixture
def float_mins():
    _preverr = np.seterr(under='ignore')
    d = {
        'float32': np.nextafter(np.float32(0), np.float32(1)),
        'float64': np.nextafter(np.float64(0), np.float64(1)),
        'float128': np.nextafter(np.float128(0), np.float128(1)),
    }
    np.seterr(**_preverr)
    return d

class TestSparsePlusFloatingPoint:
    def test_underflow(self, float_mins):
        _preverr = np.seterr(under='raise')

        with pytest.raises(FloatingPointError):
            float_mins['float32'] * float_mins['float32']

        with pytest.raises(FloatingPointError):
            float_mins['float64'] * float_mins['float64']

        with pytest.raises(FloatingPointError):
            float_mins['float128'] * float_mins['float128']

        np.seterr(**_preverr)

    def test_recast(self, float_mins):
        _preverr = np.seterr(under='raise')

        # Should not raise FloatingPointError
        np.float64(float_mins['float32']) * np.float64(float_mins['float32'])
        np.float128(float_mins['float64']) * np.float128(float_mins['float64'])

        # Will still raise FloatingPointError
        with pytest.raises(FloatingPointError):
            float_mins['float128'] * float_mins['float128']

        np.seterr(**_preverr)


