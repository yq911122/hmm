"""Hidden Markov Model

This module contains methods for:
- calculating some useful probabilities;
- fitting Hidden Markov Model (HMM) w/ or w/o states provided;
- generating observation sequence with model being fitted;
- predicing state sequence with model being fitted.

"""

# Authors: Quan Yuan <yq911122@gmail.com>

import warnings
from warnings import warn

import scipy.sparse as sp
import numpy as np

from _hmm import cal_forward_log_proba, cal_backward_log_proba
from scipy.misc import logsumexp

EPS = 1e-5

# def _logsumexp(X):
#     X_max = np.max(X)
#     if np.isinf(X_max):
#         return -np.inf

#     return np.log(np.sum(np.exp(X - X_max))) + X_max

def log_mask_zero(a):
    """Computes the log of input probabilities masking divide by zero in log.
    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.
    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        a_log = np.log(a)
        a_log[a <= 0] = 0.0
        return a_log

def log_normalize(a, axis=None):
    """Normalizes the input array so that the exponent of the sum is 1.
    Parameters
    ----------
    a : array
        Non-normalized input data.
    axis : int
        Dimension along which normalization is performed.
    Notes
    -----
    Modifies the input **inplace**.
    """
    a_lse = logsumexp(a, axis)
    if axis is None or axis == 0:
        a -= a_lse
    else:
        a -= a_lse[:, np.newaxis]

def check_1d_array(y):
    """
    Check the array is of shape == (N, ) or (N, 1)

    Parameters
    ----------
    y : array-like
        array to be checked

    Returns
    -------
    checked array: numpy array, shape = [N, ]. If the array
        isn't of shape (N,) or (N,1), a ValueError will be raised.

    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        warn("A column-vector y was passed when a 1d array was"
             " expected. Please change the shape of y to "
              "(n_samples, ), for example using ravel().")
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))

def check_array(X):
    """
    Check the array is of shape == (N, M)

    Parameters
    ----------
    y : array-like
        array to be checked

    Returns
    -------
    checked array: numpy array, shape = [N, M]. If the array
        isn't of shape (N, M), a ValueError will be raised.

    """
    shape = np.shape(X)
    if len(shape) == 2:
        return np.ravel(X)

    raise ValueError("bad input shape {0}".format(shape))

def div0( a, b ):
    """ divide two values, if denominator  is 0, return 0
    
    Parameters
    ----------
    a : array-like, shape = [N, ]
        numerator 

    b : array-like, shape = [N, ]
        denominator 

    Returns
    -------
    quotient: numpy array, shape = [N, ]. 
        If denominator is 0, return 0
    
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def check_arrays(X1, X2):
    """Check if two arrays are of the same shape
    
    Parameters
    ----------
    X1 : array-like

    X2 : array-like

    Returns
    -------
    A1, A2 : numpy.array, shape = X1.shape = X2.shape
        if X1 and X2 are of the same shape, return them;
        otherwise, a ValueError will be raised.
    """

    if np.shape(X1) == np.shape(X2):
        return np.ravel(X1), np.ravel(X2)

    raise ValueError("inconsistent input shape {0} and {1}".format(np.shape(X1), np.shape(X2)))

class TwoEndedIndex(object):
    """Helper class for storing index and reverse index
    
    Parameters
    ----------
    l : array-like, shape = [N,]
        index key; each value is unique

    Attributes
    ----------
    keydict: dict; index

    valuedict: dict; reverse index

    n: integer; size of index

    keys: array-like; index keys

    values: array-like; index values    

    """
    def __init__(self, l):
        super(TwoEndedIndex, self).__init__()
        self.keydict = {k:i for i, k in enumerate(l)}
        self.valuedict = {i:k for i, k in enumerate(l)}
        self.n = len(l)
        self.keys = l
        self.values = range(self.n)

    def get_key(self, value):
        """Return key by value; if no such key exists, return None

        Parameters
        ----------
        value : integer; the value of which key will be returned
        
        Returns
        -------
        key: string; key of the value
            if no such key exists, return None
        """
        try:
            return self.valuedict[value]
        except KeyError:
            return None

    def get_value(self, key):
        """Return value by key; if no such value exists, return -1

        Parameters
        ----------
        key : string; the key of which value will be returned
        
        Returns
        -------
        value: integer; value of the key
            if no such value exists, return -1
        """
        try:
            return self.keydict[key]
        except KeyError:
            return -1

    def all_keys(self):
        """
        Returns
        -------
        keys: array-like; index keys
        """
        return self.keys

    def all_values(self):
        """
        Returns
        -------
        values: array-like; index values
        """
        return self.values

    def size(self):
        """
        Returns
        -------
        size: integer; index size
        """
        return self.n
        

class HiddenMarkovModel(object):
    """A Hidden Markov Model
    
    Parameters
    ----------
    max_iters : integer, optional(default=1000)
        the maximum iterations when fitting the model if
        no states data provided. In that case, the model may 
        not reach the maximum iterations and stop early, which
        means the model converages. For more details, check fit()
        method.

    Attributes
    ----------
    transit_matrix : array-like, shape = [n_states, n_states]
        probability transit matrix, with the (i, j) element as:

            a(i,j) = P(i(t+1) = q(j) | i(t) = q(i))

        q is the state sequence

    observation_matrix : array-like, shape = [n_states, n_observations]
        probability matrix of obesrvations, with the (i, j) element as:

            b(i, j) = P(o(t) = v(j) | i(t) = q(i))

        q, v are the state sequence and observation sequence

    init_prob : array-like, shape = [n_states]    
        initaite probability of states

    n_states : integer; total number of unique states

    n_observations : integer; total number of unique observations

    states_space : object (TwoEndedIndex), storing states information

    observations_space : object (TwoEndedIndex), storing observations information

    max_iters : integer; the maximum iterations when fitting the model if
        no states data provided
    """
    def __init__(self, max_iters=50):
        super(HiddenMarkovModel, self).__init__()
        self.transit_matrix = None
        self.observation_matrix = None
        self.init_prob = None
        self.states_space = None
        self.observations_space = None
        self.max_iters = max_iters

    @property
    def n_states(self):
        """total number of unique states"""
        cands = [self.transit_matrix, self.init_prob, self.states_space]
        for i in xrange(len(cands)):
            if cands[i] is not None:
                if i <= 1:
                    return cands[i].shape[0]
                else:
                    return cands[i].size()
        return 0

    @property
    def n_observations(self):
        """total number of unique observations"""
        cands = [self.observation_matrix, self.observations_space]
        for i in xrange(len(cands)):
            if cands[i] is not None:
                if i == 0:
                    return cands[i].shape[1]
                else:
                    return cands[i].size()
        return 0

    def _check_probabilities(self):
        """check if transit_matrix, observation_matrix and init_prob
        are not None and consistent

        Returns
        -------
        is_consistent : boolean; True if transit_matrix, 
            observation_matrix and init_prob are not None and consistent 
        """
        A, B, pi = self.transit_matrix, self.observation_matrix, self.init_prob
        if A is not None and B is not None and pi is not None and \
           A.shape[0] == B.shape[0] == pi.shape[0]:
            return True

        raise ValueError("inconsistent or empty paramters among transit matrix,"
                        " observation matrix and initiate probability.")

    def _normalize_by_column(self, A):
        """normailize matrix by its column sum:

                a(i, j) = a(i, j) / sum(a(:, j))

        Parameters
        ----------
        A : array-like, shape = [M, N]
            matrix to be normalized

        Returns
        -------
        A_norm : array-like, shape = [M, N]
            normalized matrix 
        """
        if sp.issparse(A):
            print A.shape
            A = self._normalize_by_row(A.T)
            return A.T

        return div0(A, A.sum(axis=0))

    def _normalize_by_row(self, A):
        """normailize matrix by its row sum:

        a(i, j) = a(i, j) / sum(a(i, :))

        Parameters
        ----------
        A : array-like, shape = [M, N]
            matrix to be normalized

        Returns
        -------
        A_norm : array-like, shape = [M, N]
            normalized matrix 
        """
        if sp.issparse(A):
            ccd = sp.spdiags(1./A.sum(1).T, 1, *A.shape)
            print ccd.shape
            print A.shape
            return ccd * A

        return div0(A, A.sum(axis=1)[:, None])

    def set_params(self, **params):
        """Set attributes of the model

        Parameters
        ----------
        params : key-value pairs, with key as the name
            of the attribute, value as the attribute value
            to be set. If no such attribute exists, an
            AttributeError will be raised, but the former 
            valid attributes will be changed

        Returns
        -------
        self : object; self will be returned
        """
        if not params:
            return self

        n_states, n_observations = [], []
        if "init_prob" in params:
            n_states.append(params["init_prob"].shape[0])
        if "transit_matrix" in params:
            n_states.append(params["transit_matrix"].shape[0])
        if "states_space" in params:
            n_states.append(params["states_space"].size())

        if "observation_matrix" in params:
            n_observations.append(params["observation_matrix"].shape[1])
        if "observations_space" in params:
            n_observations.append(params["observations_space"].size())

        if len(set(n_states)) > 1 or len(set(n_observations)) > 1:
            raise ValueError("Inconsistent shape of input")

        for key, value in params.iteritems():
            try:
                setattr(self, key, value)
            except AttributeError:
                raise AttributeError("Invalid paramter {0} for estimator {1}.".format(key, self.__class__.__name__))

        return self

    def _get_seq_indice(self, seq, mode='state', check_input=True, accept_invalid=False):
        """map a sequence to related index values.

        Parameters
        ----------
        seq : array-like, shape = [n_seq]
            sequence to be mapped from

        mode : string, optional(default = 'state')
            if 'state', the sequence will be regarded
            as a state sequence and mapped to state index
            value;
            if 'observation', the sequence will be regarded
            as a observation sequence and mapped to observation 
            index value;
            otherwise, a ValueError will be raised.

        check_input : boolean, optional(defaul='True')
            determine whether to check if the sequence is of
            valid shape

        accept_invalid : boolean, optional(default="False")
            determine whether to accept state/observation that
            doesn't exist in the state/observation space. If set
            True, the non-exist state/observation will be mapped
            to -1

        Returns
        -------
        indice : array-like, shape = [n_seq]
            indice of the sequence
        """

        if check_input:
            seq = check_1d_array(seq)
        seq_length = seq.shape[0]

        if mode not in ['state', 'observation']:
            raise ValueError("invalid mode, only 'state' or 'observation' are valid.")
        if mode == 'state':
            space = self.states_space
        else:
            space = self.observations_space
        indice = np.empty((seq_length,), dtype=np.int32)
        for i, key in enumerate(seq):
            val = space.get_value(key)
            if val != -1 or accept_invalid:
                indice[i] = val
            else:
                raise KeyError("Invalid value {0} in the sequence. Valid input should be in [0, {1}]".format(val, space.size()))
        return indice

    def _validate_period(self, t, end):
        """check if the period is valid
        
        Parameters
        ----------
        t : integer; period to be checked

        end : integer: the last acceptable period

        Returns
        -------
        valid_t : integer; checked period. If the
            period is not valid, a ValueError will
            be raised
        """
        if t < 0 or t > end:
            raise ValueError("invalid period {0}. The period should be in [{1}, {2}].".format(t, 0, end))
        return t

    def fit(self, obs, states=None):
        """fit the model

        Parameters
        ----------
        obs : array-like, shape = [n_seq]
            observations sequence

        states : array-like or None, optional(default=None),
            states sequence;
            if not None, shape = [n_seq], then a max-likelihood 
            method will be applied to fitting;
            if None, then Baum-Welch Algorithm will be applied to
            fitting
        
        Returns
        -------
        self : object; self will be returned
        """
        obs = check_1d_array(obs)
        T = obs.shape[0]

        unique_obs, obs_ids = np.unique(obs, return_inverse=True)
        self.observations_space = TwoEndedIndex(unique_obs)
        n_observations = self.n_observations

        if states is None:
            n_states = self.n_states
            if n_states == 0:
                raise ValueError("States must be specified if no states data provided.")

            indice = self._get_seq_indice(obs, mode='observation', check_input=False)
            A = np.random.rand(n_states, n_states)
            # # B = sp.csc_matrix((n_states, n_observations))
            B = np.random.rand(n_states, n_observations)
            pi = np.random.rand(n_states, )

            A = self._normalize_by_row(A)
            B = self._normalize_by_row(B)
            pi /= np.sum(pi)

            self.transit_matrix = A
            self.observation_matrix = B
            self.init_prob = pi

            prob_seq1, prob_seq2 = np.random.rand(n_observations, ), np.random.rand(n_observations, )

            for i in xrange(self.max_iters):
                A, B = log_mask_zero(A), log_mask_zero(B)

                p_forward = self._cal_forward_proba(indice, None, check_input=False)
                p_backward = self._cal_backward_proba(indice, None, check_input=False)

                tmp = p_forward + p_backward
                log_normalize(tmp, axis=0)
                p_state = np.exp(tmp)

                p_transit = np.zeros((n_states, n_states))
                for i in xrange(T-1):
                    tmp = p_forward[:, i] + p_backward[:, i+1].reshape((n_states, 1)) + A + B[:, indice[i+1]].reshape((n_states, 1))
                    log_normalize(tmp)
                    p_transit += np.exp(tmp)

                A = div0(p_transit, np.sum(p_state[:, :-1], axis=1)[:, None])
                A = self._normalize_by_row(A)
                # B = sp.csc_matrix((n_states, n_observations))
                B = np.zeros((n_states, n_observations))
                for i in xrange(T):
                    j = obs_ids[i]
                    B[:, j] += p_state[:, j]

                B = self._normalize_by_row(B)

                self.transit_matrix = A
                self.observation_matrix = B
                self.init_prob = p_state[:, 0]

                prob_seq1 = np.sum(np.exp(p_forward))

                diff = np.sum((prob_seq2 - prob_seq1) ** 2)

                if diff < EPS:
                    break

                prob_seq2 = prob_seq1

        else:
            states = check_1d_array(states)
            obs, states = check_arrays(obs, states)

            unique_states, state_ids, states_count = np.unique(states, return_counts=True, return_inverse=True)

            self.init_prob = states_count / float(T)
            self.states_space = TwoEndedIndex(unique_states)
            n_states = self.n_states
            
            A = np.zeros((n_states, n_states), dtype=np.int32)
            for t in xrange(T-1):
                i, j = state_ids[t], state_ids[t+1]
                A[i, j] += 1

            self.transit_matrix = self._normalize_by_row(A)

            B = np.zeros((n_states, n_observations), dtype=np.int32)
            # B = sp.csc_matrix((n_states, n_observations), dtype=np.int32)
            for t in xrange(T):
                i, k = state_ids[t], obs_ids[t]
                B[i, k] += 1
            self.observation_matrix = self._normalize_by_row(B)

        return self


    def predict(self, obs):
        """predict states sequence of obs by Viterbi Algorithm

        Parameters
        ----------
        obs : array-like, shape = [n_seq]
            observations sequence

        Returns
        -------
        states : array-like, shape = [n_seq]
            predicted states sequence
        """
        obs = check_1d_array(obs)
        self._check_probabilities()

        T = obs.shape[0]

        A, B, pi = log_mask_zero(self.transit_matrix), log_mask_zero(self.observation_matrix), log_mask_zero(self.init_prob)
        indice = self._get_seq_indice(obs, mode='observation', accept_invalid=True)
        n_observations = self.n_observations
        n_states = self.n_states

        log_default_obs_prob =  -n_observations

        pred_indice = np.empty((T, ), dtype=np.int32)
        
        delta = pi + log_default_obs_prob if indice[0] == -1 else pi + B[:, indice[0]]
        phi = np.empty((T, n_states), dtype=np.int32)

        for i in xrange(1, T):
            o = indice[i]
            tmp = delta[:, None] + A
            phi[i-1] = np.argmax(tmp, axis=0)
            for j in xrange(n_states):
                k = phi[i-1,j]
                delta[j] = tmp[k, j] + B[j,o]

        pred_indice[T-1] = np.argmax(delta)

        for i in xrange(T-2, -1, -1):
            pred_indice[i] = phi[i, pred_indice[i+1]]

        return np.vectorize(self.states_space.get_key)(pred_indice)

    def generate_observation_sequence(self, T):
        """generate observations sequence
        
        Parameters
        ----------
        T : integer; length of observations sequence

        Returns
        -------
        obs : array-like, shape = [T]
            generated observations sequence
        """

        self._check_probabilities()
        if T < 0:
            raise ValueError("bad input sequence lenght {0}.".format(T))
        
        A, B, pi = self.transit_matrix, self.observation_matrix, self.init_prob
        obs = np.empty((T,), dtype=np.string_)
        i = np.random.choice(self.states_space.all_values(), 1, p=pi)[0]
        for t in xrange(T):
            obs[t] = np.random.choice(self.observations_space.all_keys(), 1, p=B[i])[0]
            i = np.random.choice(self.states_space.all_values(), 1, p=A[i,:])[0]
        return obs

    def _cal_forward_proba(self, indice, check_input=True):
        """calculate forward probability based on the following recursing method:

                a(0, i) = pi(i) * b(i, o(0))
                a(t+1, i) = b(i, o(t+1)) * sum_j(a(t, j) * p(j, i)), t = 1, 2,..., T
        
            where a(t, i) is the calculated forward probability, that is,
                
                a(t, i) = P(o(1), o(2),..., o(t), i(t) = q(i))
            
            pi(i) is the initiate probability, o(t) is the observation in t, 
            b(i, o) is the observation probability and p(j, i) the transit 
            probability from state j to i

        Parameters
        ----------
        indice : array-like, shape = [n_seq]
            indice of observation sequence

        check_input : boolean, optional(defaul='True')
            determine whether to check if the sequence is of
            valid shape

        Returns
        -------
        forward_proba : array-like, shape = [n_states, n_seq]; forward probability
    
        """
        if check_input:
            indice = check_1d_array(indice)
        end = indice.shape[0]
        self._check_probabilities()

        A, B, pi = log_mask_zero(self.transit_matrix), log_mask_zero(self.observation_matrix), log_mask_zero(self.init_prob)
        n_states = self.n_states
        log_proba = np.empty((n_states, end))
        tmp = np.empty((n_states, ))
        cal_forward_log_proba(n_states, end, pi, A, B, log_proba, indice, tmp)

        return log_proba

    def _cal_backward_proba(self, indice, check_input=True):
        """calculate backward probability based on the following recursing method:

                a(T, i) = 1
                a(t, i) = sum_j(b(j, o(t+1)) * a(t+1, j) * p(i, j)), t = 1, 2,..., T
        
            where a(t, i) is the calculated backward probability, that is,
                
                a(t, i) = P(o(t+1), o(t+2),..., o(T) | i(t) = q(i))
            
            pi(i) is the initiate probability, o(t) is the observation in t, 
            b(i, o) is the observation probability and p(j, i) the transit 
            probability from state j to i

        Parameters
        ----------
        indice : array-like, shape = [n_seq]
            indice of observation sequence

        check_input : boolean, optional(defaul='True')
            determine whether to check if the sequence is of
            valid shape

        Returns
        -------
        backward_proba : array-like, shape = [n_states, n_seq]; backward probability
    
        """
        if check_input:
            indice = check_1d_array(indice)
        end = indice.shape[0]

        self._check_probabilities()
        A = self.transit_matrix
        B = self.observation_matrix
        pi = self.init_prob

        A, B, pi = log_mask_zero(self.transit_matrix), log_mask_zero(self.observation_matrix), log_mask_zero(self.init_prob)
        n_states = self.n_states

        log_proba = np.empty((self.n_states, end))
        log_proba[:, end-1] = 0.
        tmp = np.empty((n_states, ))
        cal_backward_log_proba(n_states, end, pi, A, B, log_proba, indice, tmp)

        return log_proba

    def cal_state_proba(self, indice, t=None, check_input=True):
        """calculate state probability based on the following recursing method:

                r(t, i) = a(t, i) * b(t, i) / sum_j(a(t, j) * b(t, j))
        
            where r(t, i) is the calculated state probability, that is,
                
                r(t, i) = P(i(t) = q(i) | O)
            
            a(t, i) and b(t, i) are forward and backward probability

        Parameters
        ----------
        indice : array-like, shape = [n_seq]
            indice of observation sequence

        t : integer or None, optional(default=None)
            if None, state probabilities at all period will be calculated;
            otherwise, state probability at period t will be calculated

        check_input : boolean, optional(defaul='True')
            determine whether to check if the sequence is of
            valid shape

        Returns
        -------
        state_proba : array-like; state probability
            if t is None, shape = [n_states, n_seq]
            otherwise, shape = [n_states]
    
        """
        if check_input:
            indice = check_1d_array(indice)
        end = indice.shape[0]

        forward_proba = self._cal_forward_proba(indice, check_input=False)
        back_proba = self._cal_backward_proba(indice, check_input=False)

        if t is None:
            # calculate proba for all times
            tmp = forward_proba + back_proba
            log_normalize(tmp, axis=1)
            return np.exp(tmp)
        else:
            t = self._validate_period(end, end, t)
            tmp = forward_proba[:,t] + back_proba[:, t]

            log_normalize(tmp, axis=0)
            return np.exp(tmp)

    def cal_trasit_proba(self, indice, t=None, check_input=True):
        """calculate transit probability based on the following recursing method:

                r(t, i, j) = a(t, i) * b(t, j) * p(i, j) * p2(j, o(t+1)) / 
                            sum_i,j(a(t, i) * b(t, j) * p(i, j) * p2(j, o(t+1)))
        
            where r(t, i, j) is the calculated transit probability, that is,
                
                r(t, i, j) = P(i(t) = q(i), i(t+1) = q(j) | O)
            
            a(t, i) and b(t, i) are forward and backward probability, o(t) is the 
            observation in t, p2(i, o) is the observation probability and p(j, i) 
            the transit probability from state j to i

        Parameters
        ----------
        indice : array-like, shape = [n_seq]
            indice of observation sequence

        t : integer or None, optional(default=None)
            if None, transit probabilities at all period will be calculated;
            otherwise, transit probability at period t will be calculated

        check_input : boolean, optional(defaul='True')
            determine whether to check if the sequence is of
            valid shape

        Returns
        -------
        transit_proba : array-like; transit probability
            if t is None, shape = [n_seq, n_states, n_states]
            otherwise, shape = [n_states, n_states]
    
        """
        if check_input:
            indice = check_1d_array(indice)
        end = indice.shape[0] - 1
        if end == 0:
            raise ValueError("Only 1 observation in the sequence. Insufficient data to estimate transit probability")

        A, B = log_mask_zero(self.transit_matrix), log_mask_zero(self.observation_matrix)

        indice = self._get_seq_indice(indice, mode='observation', check_input=False)
        n_states = self.n_states

        forward_proba = self._cal_forward_proba(indice, None, check_input=False)
        back_proba = self._cal_backward_proba(indice, None, check_input=False)

        if t is None:

            r = np.empty((end, n_states, n_states))
            for i in xrange(end):
                bi = B[:, indice[i+1]]
                tmpi = forward_proba[:, i] + back_proba[:, i].reshape((n_states, 1)) + A + bi
                log_normalize(tmp)
                r[i] = np.exp(tmp)
            return r
        else:
            t = self._validate_period(end, end, t)

            tmp = forward_proba[:, t] + back_proba[:, t+1].reshape((n_states, 1)) + A + B[:, indice[t+1]]
            log_normalize(tmp, axis=0)
            return np.exp(tmp)

    def get_observation_sequence_proba(self, obs_seq):
        """calculate probability P(O) of the given sequnce by:

                P(O) = sum_i(a(T, i))

            where T is the end period, a(t, i) is the forward probability

        Parameters
        ----------
        obs_seq : array-like, shape = [n_seq]
            observation sequence

        Returns
        -------
        sequence_proba : float; probability of the given sequence
        """
        obs_seq = check_1d_array(obs_seq)
        indice = self._get_seq_indice(obs_seq, mode='observation', check_input=False)         

        self._check_probabilities()
        
        # foward algorithm
        forward_proba = self._cal_forward_proba(indice, check_input=False)
        return np.sum(np.exp(forward_proba))