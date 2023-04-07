from __future__ import absolute_import
import sys
import numpy
import sklearn.preprocessing
import ctypes
import faiss
from ann_benchmarks.algorithms.base import BaseANN
from sklearn.preprocessing import StandardScaler

from sympy.ntheory.residue_ntheory import primitive_root
from sympy.ntheory.generate import nextprime
import heapq
import torch
import math
import jax
import time

jax.config.update('jax_platform_name', 'cpu')

class UsrFaiss():

    def __init__(self, iterations, hash_bits_per_dim, dimension, hash_func_constant):
        self.iterations = iterations
        self.dim=dimension
        self.hash_bits_per_dim = hash_bits_per_dim
        self.hash_func_constant = hash_func_constant
        self.hash_size = int(self.dim * self.hash_bits_per_dim)
        self.packed_hash_size = -int(self.hash_size // -8)
        # self.packed_hash_size = -int(self.hash_size // -8)
        self.scaler = StandardScaler(with_std=False)

        self.I = numpy.eye(self.hash_size, dtype=numpy.float32)
        self.W = numpy.zeros((self.hash_size, self.dim), dtype=numpy.float32)

        self.packed_q_holder = numpy.zeros((1,self.packed_hash_size), dtype=numpy.uint8)

        Z = numpy.random.standard_normal(size=(self.hash_bits_per_dim, self.dim, self.dim))
        (U, S, V) = numpy.linalg.svd(Z)
        R = ( U@V.transpose((0,2,1)) ).reshape(self.hash_bits_per_dim*self.dim,self.dim)
        self.W[:] = math.sqrt(self.dim)/math.sqrt(self.hash_size) * R

        W = jax.numpy.array(self.W, dtype=jax.numpy.float32)
        # cW = jax.numpy.array(self.hash_func_constant*self.W, dtype=jax.numpy.float32)
        I_WW = self.I - W@W.T
        I_WW = jax.numpy.array(I_WW, dtype=jax.numpy.float32)

        def hash_func(c, x, t):
            b = jax.numpy.tanh(c * 0.5 * math.sqrt(self.hash_bits_per_dim*self.dim) * x@W.T)
            def fn(_,v):
                return \
                    jax.numpy.tanh( \
                        c * 0.5 * math.sqrt(self.hash_bits_per_dim*self.dim) * x@W.T + \
                        jax.numpy.matmul(v, I_WW)
                    )
                
            b = jax.lax.fori_loop(0, t, fn, b)

            return b

        jfn = jax.jit(hash_func)
        self.hash_func = jfn

    def fft_index_generator(self, dim, minsize):

        size = minsize
        while True:
            size = nextprime(size)
            if (size - 1) % (dim) == 0: break
        g = primitive_root(size)
        m = pow(g, (size-1)//(dim), size)
        return numpy.int64(size), numpy.array([pow(m, i, size) for i in range(dim)], dtype=numpy.int64)

    def fft_hash(self, x, b):

        c = self.hash_func_constant
        half_hash_size = int(self.hash_size/2)
        half_dim = int(self.dim/2)

        # X_hat = torch.complex(x.T[half_dim:,:], x.T[:half_dim,:]) 
        X_hat = c * torch.complex(x.T[half_dim:,:], x.T[:half_dim,:]) / (2*math.sqrt(half_dim))
        y = torch.zeros(half_hash_size, len(x), dtype=torch.cfloat).to(self.device)
        y[self.fft_index] = X_hat
        y = torch.fft.fft(y, dim=0)

        # b_hat = torch.complex(b.T[half_hash_size:,:], b.T[:half_hash_size,:])
        b_hat = torch.complex(b.T[half_hash_size:,:], b.T[:half_hash_size,:]) / math.sqrt(self.dim)
        y_b = torch.zeros(half_hash_size, len(x), dtype=torch.cfloat).to(self.device)
        y_b[self.fft_index] = torch.fft.ifft(b_hat, dim=0)[self.fft_index]
        y_b_H = torch.fft.fft(y_b, dim=0)

        r = torch.cat((torch.real(y), torch.imag(y)), dim=0).T + \
            b - torch.cat((torch.real(y_b_H), torch.imag(y_b_H)), dim=0).T
        
        if self.sparse_n == -1:
            return torch.tanh(r)
        else:
            (_, m) = torch.topk(torch.abs(r), self.sparse_n, dim=1)
            _r = torch.zeros_like(r)
            _r = _r.scatter_(1, m, r.gather(1,m))
            return _r

    def fit(self, X):

        batch_size = 10000
        batch_num = -int(len(X)//-batch_size)
        self.packed_hashes = numpy.zeros((len(X), self.packed_hash_size), dtype=numpy.uint8)

        print("Hashing and packing...")
        t0 = time.time()
        i = 0
        for x in numpy.array_split(X, batch_num):
            h = numpy.array(self.hash_func(self.hash_func_constant, x, self.iterations))

            faiss.fvecs2bitvecs(
                faiss.swig_ptr(h),
                faiss.swig_ptr(self.packed_hashes[i]),
                self.hash_size,
                len(x)
            )
            i += len(x)
        print(f"Hashed and packed in {time.time() - t0}")

        self.faiss_index = faiss.IndexBinaryFlat(self.packed_hash_size * 8)
        self.faiss_index.add(self.packed_hashes)

    def query(self, q, k=1):
        
        q = jax.numpy.array(q)
        q_hashf = numpy.array(self.hash_func(self.hash_func_constant, q, self.iterations))

        faiss.fvecs2bitvecs(
            faiss.swig_ptr(q_hashf),
            faiss.swig_ptr(self.packed_q_holder),
            self.hash_size,
            1,
        )

        _, I = self.faiss_index.search(self.packed_q_holder, k)
        return I[0]

    def delete(self, id):
        self.faiss_index.remove_ids(numpy.asarray([id]))

class UsrANN(BaseANN):

    def __init__(self, metric, iterations, hash_bits_per_dim):
        self.index = None
        self._metric = metric

        self.iterations = iterations
        self.hash_bits_per_dim = hash_bits_per_dim
    
    def __str__(self):
        return f"Usr(m={self.hash_bits_per_dim},T={self.iterations})"

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        
        D = X.shape[1]
        average_norm = torch.tensor(X).norm(dim=1).mean().item()
        C = 1 / average_norm

        self.index = UsrFaiss(
            iterations=self.iterations,
            dimension=D,
            hash_bits_per_dim=self.hash_bits_per_dim,
            hash_func_constant=C,
        )
        self.index.fit(X)

    def query(self, q, k):
        if q.dtype != numpy.float32:
            q = q.astype(numpy.float32)
        
        if self._metric == 'angular':
            q /= numpy.linalg.norm(q)

        ids = self.index.query(numpy.expand_dims(q,0), k)

        return ids

    def delete(self, id, X):
        X = numpy.delete(X, id, axis=0)

        D = X.shape[1]
        average_norm = torch.tensor(X).norm(dim=1).mean().item()
        C = 1 / average_norm

        self.index.delete(id)
        self.index.hash_func_constant = C

    def batch_query(self, Q, n):
        if Q.dtype != numpy.float32:
            Q = Q.astype(numpy.float32)
        if self._metric == 'angular':
            Q /= numpy.linalg.norm(Q)

        self.batch_result = self.index.query(Q,n)

    def get_batch_results(self):
        return self.batch_result