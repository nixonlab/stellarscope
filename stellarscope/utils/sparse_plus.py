# -*- coding: utf-8 -*-
""" Provides sparse matrix classes augmented with additional functions
"""
from __future__ import division
from __future__ import annotations

import numpy.random
from future import standard_library
standard_library.install_aliases()
from builtins import range
from typing import Optional

import numpy as np
import scipy.sparse
from numpy.random import default_rng

import logging as lg

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2019 Matthew L. Bendall"

def _recip0(v):
    ''' Return the reciprocal of a vector '''
    old_settings = np.seterr(divide='ignore')
    ret = 1. / v
    ret[np.isinf(ret)] = 0
    np.seterr(**old_settings)
    return ret




class csr_matrix_plus(scipy.sparse.csr_matrix):

    def norm(self, axis=None):
        """ Normalize matrix along axis

        Args:
            axis:

        Returns:
        Examples:
            >>> row = np.array([0, 0, 1, 2, 2, 2])
            >>> col = np.array([0, 2, 2, 0, 1, 2])
            >>> data = np.array([1, 2, 3, 4, 5, 6])
            >>> M = csr_matrix_plus((data, (row, col)), shape=(3, 3))
            >>> print(M.norm(1).toarray())
            [[ 0.33333333  0.          0.66666667]
             [ 0.          0.          1.        ]
             [ 0.26666667  0.33333333  0.4       ]]
        """
        # return self._norm_loop(axis)
        return self._norm(axis)

    def _norm(self, axis=None):
        if axis is None:
            return self.multiply(1. / self.sum())
        elif axis == 0:
            raise NotImplementedError
        elif axis == 1:
            return self.multiply(_recip0(self.sum(1)))

    def _norm_loop(self, axis=None):
        if axis is None:
            ret = self.copy().astype(np.float)
            ret.data /= sum(ret)
            return ret
        elif axis == 0:
            raise NotImplementedError
        elif axis == 1:
            ret = self.copy().astype(np.float)
            rowiter = zip(ret.indptr[:-1], ret.indptr[1:], ret.sum(1).A1)
            for d_start, d_end, d_sum in rowiter:
                if d_sum != 0:
                    ret.data[d_start:d_end] /= d_sum
            return ret

    def scale(self, axis=None):
        """ Scale matrix so values are between 0 and 1

        Args:
            axis:

        Returns:
        Examples:
            >>> M = csr_matrix_plus([[10, 0, 20],[0, 0, 30],[40, 50, 60]])
            >>> print(M.scale().toarray())
            [[ 0.1  0.   0.2]
             [ 0.   0.   0.3]
             [ 0.4  0.5  1. ]]
            >>> print(M.scale(1).toarray())
            [[ 0.5  0.   1. ]
             [ 0.   0.   1. ]
             [ 0.4  0.5  1. ]]
        """
        return self._scale(axis)

    def _scale(self, axis=None):
        if self.nnz == 0: return self
        if axis is None:
            return self.multiply(1. / self.max())
            # ret = self.copy().astype(np.float)
            # return ret / ret.max()
        elif axis == 0:
            raise NotImplementedError
        elif axis == 1:
            return self.multiply(_recip0(self.max(1).toarray()))

    def binmax(self, axis=None):
        """ Set max values to 1 and others to 0

        Args:
            axis:

        Returns:
        Examples:
            >>> M = csr_matrix_plus([[6, 0, 2],[0, 0, 3],[4, 5, 6]])
            >>> print(M.binmax().toarray())
            [[1 0 0]
             [0 0 0]
             [0 0 1]]
            >>> print(M.binmax(1).toarray())
            [[1 0 0]
             [0 0 1]
             [0 0 1]]
        """
        if axis is None:
            raise NotImplementedError
        elif axis == 0:
            raise NotImplementedError
        elif axis == 1:
            _data = np.zeros(self.data.shape, dtype=np.int8)
            rit = zip(self.indptr[:-1], self.indptr[1:], self.max(1).toarray())
            for d_start, d_end, d_max in rit:
                _data[d_start:d_end] = (self.data[d_start:d_end] == d_max)
            ret = type(self)((_data, self.indices.copy(), self.indptr.copy()),
                              shape=self.shape)
            ret.eliminate_zeros()
            return ret

    def count(self, axis=None):
        if axis is None:
            raise NotImplementedError
        elif axis == 0:
            raise NotImplementedError
        elif axis == 1:
            ret = self.indptr[1:] - self.indptr[:-1]
            return np.array(ret, ndmin=2).T

    def choose_random(
            self,
            axis: Optional[int] = None,
            rng: numpy.random.Generator = default_rng()
    ) -> csr_matrix_plus:
        """ Select random elements from matrix

        Random elements are selected from the nonzero elements of the matrix.
        All nonzero elements have a uniform chance of being selected using the
        `numpy.random.Generator.choice()` function of `rng`. If generator must
        have a set seed, the generator should be initialized outside and passed
        to this function, otherwise a new generator will be created.

        Parameters
        ----------
        axis : Optional[int], default=None
            Axis along which random elements are chosen. `axis = 1` will choose
            a one element randomly
        rng : numpy.random.Generator, default=default_rng()
            A random number generator. If not provided, creates a new random
            generator using numpy.random.default_rng()

        Returns
        -------
        csr_matrix_plus
            Sparse CSR matrix with randomly selected values, unselected values
            are set to 0.
        """
        if axis is None:
            raise NotImplementedError
        elif axis == 0:
            raise NotImplementedError
        elif axis == 1:
            ret = self.copy()
            for d_start, d_end in zip(ret.indptr[:-1], ret.indptr[1:]):
                if d_end - d_start > 1:
                    chosen = rng.choice(range(d_start, d_end))
                    for j in range(d_start, d_end):
                        if j != chosen:
                            ret.data[j] = 0
            ret.eliminate_zeros()
            return ret

    def check_equal(self, other):
        if self.shape != other.shape:
            return False
        return (self != other).nnz == 0

    def apply_func(self, func):
        ret = self.copy()
        ret.data = np.fromiter((func(v) for v in self.data),
                               self.data.dtype, count=len(self.data))
        return ret

    def multiply(self, other):
        try:
            return type(self)(super().multiply(other))
        except FloatingPointError:
            lg.debug('using extended precision')
            longcopy = scipy.sparse.csr_matrix(self).astype(np.float128)
            return type(self)(longcopy.multiply(other))


    def colsums(self) -> csr_matrix_plus:
        """ Sum columns and return as `csr_matrix_plus`

        Returns
        -------

        """
        _colsums = self.sum(0)
        return type(self)(_colsums)

    def colsums_nodense(self) -> csr_matrix_plus:
        """ Sum columns and return as `csr_matrix_plus`

        This (might) save memory by avoiding `scipy.sparse.spmatrix.sum`,
        which returns a dense np.matrix. However, initial testing shows that
        this implementation may be more than 10 times slower than using the
        `spmatrix.sum` function.

        Returns
        -------

        """
        _csc = self.tocsc()
        _dok = scipy.sparse.dok_matrix((1, self.shape[1]), dtype=self.dtype)

        for i in range(self.shape[1]):
            if _csc.indptr[i] ==_csc.indptr[i + 1]:
                continue
            _dok[0, i] = sum(_csc.data[_csc.indptr[i]:_csc.indptr[i + 1]])
        return type(self)(_dok)

    def save(self, filename):
        np.savez(filename, data=self.data, indices=self.indices,
                 indptr=self.indptr, shape=self.shape)
    @classmethod
    def load(cls, filename):
        loader = np.load(filename)
        return cls((loader['data'], loader['indices'], loader['indptr']),
                   shape = loader['shape'])


def row_identity_matrix(selected, nrows):
    """ Create identity matrix from list of selected rows

    The identity matrix I is a matrix of shape = (nrows, 1) where I[i, 0] = 1
    if the row is selected (i ∈ selected) and I[i, 0] = 0 otherwise.

    The identity matrix is useful for efficiently subsetting sparse matrices.
    For example, M.multiply(I) returns a sparse matrix with the same shape
    as M with rows not in `selected` set to 0.

    The matrix is created by first constructing a sparse matrix in COOrdinate
    format (coo_matrix) using the constructor:

        `coo_matrix((data, (i, j)), [shape=(M, N)])`

    Parameters
    ----------
    selected
    nrows

    Returns
    -------
    csr_matrix_plus
        Sparse matrix with shape = (nrows, 1)

    """
    _nnz = len(selected)
    _data = [1] * _nnz
    _i = selected
    _j = [0] * _nnz
    _M = scipy.sparse.coo_matrix(
        (_data, (_i, _j)),
        shape=(nrows, 1),
        dtype=np.uint8
    )
    return csr_matrix_plus(_M)

