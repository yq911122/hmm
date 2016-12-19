import numpy as np
cimport numpy as np
cimport cython

np.import_array()

from numpy.math cimport isinf, INFINITY
from libc.math cimport log, exp

ctypedef double dtype_t
ctypedef np.int_t int_t

@cython.boundscheck(False)
cdef inline dtype_t _max(dtype_t[:] X) nogil:
    cdef dtype_t X_max = -INFINITY
    cdef int pos = 0
    cdef int i
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]
    return X_max

@cython.boundscheck(False)
cdef inline dtype_t _logsumexp(dtype_t[:] X) nogil:
    cdef dtype_t X_max = _max(X)
    if isinf(X_max):
        return -INFINITY

    cdef dtype_t acc = 0
    for i in range(X.shape[0]):
        acc += exp(X[i] - X_max)

    return log(acc) + X_max

@cython.boundscheck(False)
cpdef void cal_forward_log_proba(int n_states, int n_periods,
                         dtype_t[:] log_init_prob,
                         dtype_t[:, :] log_transmat,
                         dtype_t[:, :] log_obsmat,
                         dtype_t[:, :] log_proba,
                         int_t[:] indice,
                         dtype_t[:] tmp) nogil:
    cdef int t, i, j, o
    
    o = indice[0]
    for i in range(n_states):
            log_proba[i, 0] = log_init_prob[i] + log_obsmat[i, o]

    for t in range(1, n_periods):
        for i in range(n_states):
            for j in range(n_states):
                tmp[j] = log_transmat[j, i] + log_proba[j, t-1]

            o = indice[t]
            log_proba[i, t] = _logsumexp(tmp) + log_obsmat[i, o]

@cython.boundscheck(False)
cpdef void cal_backward_log_proba(int n_states, int n_periods,
                         dtype_t[:] log_init_prob,
                         dtype_t[:, :] log_transmat,
                         dtype_t[:, :] log_obsmat,
                         dtype_t[:, :] log_proba,
                         int_t[:] indice,
                         dtype_t[:] tmp) nogil:
    cdef int t, i, j, o

    for t in range(n_periods-2, -1, -1):
        for i in range(n_states):
            o = indice[t+1]
            for j in range(n_states):
                tmp[j] = log_transmat[i, j] + log_proba[j, t+1] + log_obsmat[j, o]

            log_proba[i, t] = _logsumexp(tmp)