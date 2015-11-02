#!python
#cython: boundscheck=False
import time
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    float log(float theta)
    float exp(float theta)

def func(list X, list Y, np.ndarray failures):
    '''
    computes the conditional likelihood of the data with the current setting of the failures parameters.
    i.e., P(X|Y; failures)

    X is a list of binary values. X[i] holds the value of x in document i
    Y is a list of lists. Y[i] holds a list of parents that are active in document i.
     e.g. Y[0] = [1,5,12] X[0] = 0
          Y[1] = [1,2,12] X[0] = 1
          ...
    '''

    cdef int L = failures.shape[0] - 1 #failures has one extra variable for "noise"
    cdef int numDocs=len(X)
    cdef int x
    cdef int i, j,d
    cdef int N
    cdef double temp, z=0.0
    cdef double total = 0.0

    for d from 0 <= d < numDocs:
        y = Y[d]
        N = len(y)

        z = failures[L]
        for i from 0 <= i < N:
            j = y[i]
            temp = failures[j]
            z += temp

        x = X[d]
        if x == 0:
            total += z
        else:
            total += log(1-exp(z))

    return total


def grad(list X, list Y, np.ndarray failures):
    '''
    computes gradients of the conditional likelihood.
    arguments are the same as f
    '''

    cdef int L = failures.shape[0]-1
    cdef int numDocs=len(X)
    cdef int x
    cdef int i, j,d
    cdef int N
    cdef double temp, z=0.0
    cdef double total = 0.0
    cdef g = np.zeros(L+1, dtype=np.float)

    for d from 0 <= d < numDocs:
        x = X[d]
        y = Y[d]

        N = len(y)
        if x == 0:
            g[L] += 1.0
            for i from 0 <= i < N:
                j = y[i]
                g[j] += 1.0

        else:
            z = failures[L]
            for i from 0 <= i < N:
                j = y[i]
                temp = failures[j]
                z += temp

            temp = (1/(exp(z)-1) + 1)
            g[L] += temp
            for i from 0 <= i < N:
                j = y[i]
                g[j] += temp
    return g
