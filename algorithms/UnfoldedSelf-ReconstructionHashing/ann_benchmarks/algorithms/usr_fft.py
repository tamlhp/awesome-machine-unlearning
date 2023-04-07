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

class UsrFftFaiss():

    def __init__(self, iterations, hash_bits_per_dim, dimension, hash_func_constant, hash_num):
        self.iterations = iterations
        self.dim=dimension
        self.hash_func_constant = hash_func_constant
        self.hash_bits_per_dim = hash_bits_per_dim
        self.hash_num = hash_num
        self.scaler = StandardScaler(with_std=False)
        self.query_iterations = iterations

        self.fft_index = numpy.zeros((self.hash_num, int(dimension/2)), dtype=numpy.int32)
        for h in range(self.hash_num):
            half_prime_hash_size, self.fft_index[h] = UsrFftFaiss.fft_index_generator(int(dimension/2), int(dimension/2 * hash_bits_per_dim) - 1)

        # self.fft_index = jax.numpy.array(self.fft_index).astype(numpy.int32)

        self.hash_size = int(half_prime_hash_size * 2)
        self.packed_hash_size = -int(self.hash_num*self.hash_size // -8)
        
        self.packed_q_holder = numpy.zeros((1,self.packed_hash_size), dtype=numpy.uint8)

        def fft_hash(x, t):

            b = numpy.zeros((self.hash_num, len(x), self.hash_size), dtype=numpy.float32)
            
            for h in range(self.hash_num):
                print(f"Hash-{h}")
                for _t in range(t):
                    print(f"Iteration-{_t}")
                    c = self.hash_func_constant
                    half_hash_size = int(self.hash_size/2)
                    half_dim = int(self.dim/2)

                    # X_hat = torch.complex(x.T[half_dim:,:], x.T[:half_dim,:]) 
                    X_hat = c * (x.T[half_dim:,:] + 1j * x.T[:half_dim,:]) / (2*math.sqrt(half_dim))
                    y = numpy.zeros((half_hash_size, len(x)), dtype=numpy.complex64)
                    y[self.fft_index[h]] = X_hat
                    y = numpy.fft.fft(y, axis=0)

                    # b_hat = torch.complex(b.T[half_hash_size:,:], b.T[:half_hash_size,:])
                    b_hat = ( b[h].T[half_hash_size:,:] + 1j * b[h].T[:half_hash_size,:] ) / math.sqrt(self.dim)
                    y_b = numpy.zeros((half_hash_size, len(x)), dtype=numpy.complex64)
                    y_b[self.fft_index[h]] = numpy.fft.ifft(b_hat, axis=0)[self.fft_index[h]]
                    y_b_H = numpy.fft.fft(y_b, axis=0)

                    b[h] = numpy.tanh( \
                        numpy.concatenate((y.real, y.imag), axis=0).T + \
                        b[h] - numpy.concatenate((y_b_H.real, y_b_H.imag), axis=0).T \
                    )
            b = numpy.moveaxis(b,0,-1).reshape(len(x),-1)
            
            return b

        # self.fft_hash = jax.jit(fft_hash)
        self.fft_hash = fft_hash

    @staticmethod
    def fft_index_generator(dim, minsize):

        size = minsize
        while True:
            size = nextprime(size)
            if (size - 1) % (dim) == 0: break
        g = primitive_root(size)
        m = pow(g, (size-1)//(dim), size)
        return numpy.int64(size), numpy.array([pow(m, i, size) for i in range(dim)], dtype=numpy.int64)

    
    def fit(self, X):

        batch_size = 10000000
        batch_num = -int(len(X)//-batch_size)
        self.packed_hashes = numpy.zeros((len(X), self.packed_hash_size), dtype=numpy.uint8)

        print("Hashing and packing...")
        t0 = time.time()
        i = 0

        # X = jax.numpy.asarray(X)

        for x in numpy.array_split(X, batch_num):
            print("Hashing...")
            t00 = time.time()
            h = self.fft_hash(x, self.iterations)
            print(f"Hashed in {time.time() - t00}")

            print("Contiguosing...")
            t00 = time.time()
            h = numpy.ascontiguousarray(h)
            print(f"Contiguosed in {time.time() - t00}")


            print("Packing...")
            t00 = time.time()
            faiss.fvecs2bitvecs(
                faiss.swig_ptr(h),
                faiss.swig_ptr(self.packed_hashes[i]),
                self.hash_num*self.hash_size,
                len(x)
            )
            print(f"Packed in {time.time() - t00}")

            i += len(x)
        print(f"Hashed and packed in {time.time() - t0}")

        self.faiss_index = faiss.IndexBinaryFlat(self.hash_num*self.hash_size)
        self.faiss_index.add(self.packed_hashes)

        # print("Creating hash table...")
        # t0 = time.time()
        # hash_ptr = hashes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # self.create_hash_table(hash_ptr, len(X),self.hash_size*self.hash_num,self.table_dim)
        # print(f"Hash table created in {time.time() - t0}")

        # del hashes

    def query(self, q, k=1):
        
        # q = self.scaler.transform(q)
        
        # q = jax.numpy.array(q)
        q_hashf = self.fft_hash(q, self.query_iterations)

        faiss.fvecs2bitvecs(
            faiss.swig_ptr(numpy.ascontiguousarray(q_hashf)),
            faiss.swig_ptr(self.packed_q_holder),
            self.hash_num*self.hash_size,
            1,
        )

        # probe q
        # t0 = time.time()
        # q_ptr = q_hashf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # candidates_ptr = self.candidates_holder.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        # probed_candidates = self.probe(q_ptr,candidates_ptr,probe_candidates, len(self.candidates_holder))
        # print(f"Probe in {time.time() - t0}")

        # t0 = time.time()
        # dist = numpy.zeros((k), dtype=numpy.int32)
        # ids = numpy.zeros((k), dtype=numpy.int64)
        # probe_hashes = numpy.ascontiguousarray(self.packed_hashes[self.candidates_holder[:probed_candidates]])
        # faiss.hammings_knn_mc(
        #     faiss.swig_ptr(self.packed_q_holder),    # q
        #     faiss.swig_ptr(probe_hashes), # X
        #     1,  # len(q)
        #     probed_candidates,    # len(X)
        #     10, # k
        #     self.packed_hash_size,
        #     faiss.swig_ptr(dist),
        #     faiss.swig_ptr(ids),
        # )
        # print(f"Sort in {time.time() - t0}")
    

        # return self.candidates_holder[ids]
        
        _, I = self.faiss_index.search(self.packed_q_holder, k)
        return I[0]

class UsrFftANN(BaseANN):

    def __init__(self, metric, iterations, hash_bits_per_dim, hash_num):
        self.index = None
        self._metric = metric

        self.iterations = iterations
        self.hash_bits_per_dim = hash_bits_per_dim
        self.hash_num = hash_num
        self.query_iterations = iterations

    def __str__(self):
        return f"UsrFft(m={self.hash_bits_per_dim},ehs={self.effective_bits},n={self.hash_num},i={self.iterations},qi={self.query_iterations})"

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        
        D = X.shape[1]
        average_norm = torch.tensor(X).norm(dim=1).mean().item()

        self.index = UsrFftFaiss(
            iterations=self.iterations,
            dimension=D,
            hash_bits_per_dim=self.hash_bits_per_dim,
            hash_func_constant=average_norm,
            hash_num = self.hash_num,
        )

        self.effective_bits = self.index.hash_size
        self.index.fit(X)
        

    def set_query_arguments(self, t):
        self.query_iterations = t
        self.index.query_iterations = t

    def query(self, q, k):
        if q.dtype != numpy.float32:
            q = q.astype(numpy.float32)
        if self._metric == 'angular':
            q /= numpy.linalg.norm(q)

        ids = self.index.query(numpy.expand_dims(q,0), k)

        return ids

    def batch_query(self, Q, n):
        if Q.dtype != numpy.float32:
            Q = Q.astype(numpy.float32)
        if self._metric == 'angular':
            Q /= numpy.linalg.norm(Q)

        self.batch_result = self.index.query(Q,n)

    def get_batch_results(self):
        return self.batch_result