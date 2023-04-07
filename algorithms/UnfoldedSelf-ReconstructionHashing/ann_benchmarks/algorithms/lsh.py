from __future__ import absolute_import
import sys
import numpy
import sklearn.preprocessing
import ctypes
import faiss
from ann_benchmarks.algorithms.base import BaseANN
from sklearn.preprocessing import StandardScaler

import torch
import math
import jax
import time
jax.config.update('jax_platform_name', 'cpu')


class Lsh():

    def __init__(self, hash_bits_per_dim, dimension):
        self.dim=dimension
        self.hash_bits_per_dim = hash_bits_per_dim
        self.hash_size = int(self.dim * self.hash_bits_per_dim)
        self.packed_hash_size = -int(self.hash_size // -8)

        self.W = numpy.random.standard_normal(size=(self.hash_size, self.dim)).astype(numpy.float32)

        W = jax.numpy.array(self.W, dtype=jax.numpy.float32)

        def hash_func(x):
            return x@W.T

        jfn = jax.jit(hash_func)
        self.hash_func = jfn


    def fit(self, X):

        batch_size = 100000
        batch_num = -int(len(X)//-batch_size)
        self.hash = numpy.zeros((len(X),  self.packed_hash_size), dtype=numpy.uint8)

        i = 0
        for x in numpy.array_split(X, batch_num):
            hf = self.hash_func(x)
            hf = numpy.array(hf)
            faiss.fvecs2bitvecs(
                faiss.swig_ptr(hf),
                faiss.swig_ptr(self.hash[i]),
                self.hash_size,
                len(x)
            )
            i += len(x)
        self.faiss_index = faiss.IndexBinaryFlat(self.packed_hash_size*8)
        self.faiss_index.add(self.hash)

    def query(self, q, k=1):
        
        q = jax.numpy.array(q)
        q_hashf = numpy.array(self.hash_func(q))
        b = numpy.zeros((len(q),self.packed_hash_size), dtype=numpy.uint8)
        faiss.fvecs2bitvecs(
            faiss.swig_ptr(q_hashf),
            faiss.swig_ptr(b),
            self.hash_size,
            len(q),
        )
        _, I = self.faiss_index.search(b, k)

        return I

    def delete(self, id):
        self.faiss_index.remove_ids(numpy.asarray([id]))

class LshANN(BaseANN):

    def __init__(self, metric, hash_bits_per_dim):
        self.index = None
        self._metric = metric

        self.hash_bits_per_dim = hash_bits_per_dim

    def __str__(self):
        return f"Lsh(m={self.hash_bits_per_dim})"

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        
        D = X.shape[1]

        self.index = Lsh(
            dimension=D,
            hash_bits_per_dim=self.hash_bits_per_dim,
        )
        self.index.fit(X)

    def query(self, q, k):
        if q.dtype != numpy.float32:
            q = q.astype(numpy.float32)
        if self._metric == 'angular':
            q /= numpy.linalg.norm(q)

        ids = self.index.query(numpy.expand_dims(q,0), k)
        return ids[0]

    def delete(self, id, X):
        X = numpy.delete(X, id, axis=0)
        self.index.delete(id)

    def batch_query(self, Q, n):
        if Q.dtype != numpy.float32:
            Q = Q.astype(numpy.float32)
        if self._metric == 'angular':
            Q /= numpy.linalg.norm(Q)

        self.batch_result = self.index.query(Q,n)

    def get_batch_results(self):
        return self.batch_result