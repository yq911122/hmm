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

import numpy as np

EPS = 1e-5

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
        self.values = range(n)

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

    def keys(self):
        """
        Returns
        -------
        keys: array-like; index keys
        """
        return self.keys

    def values(self):
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

            b(i,j) = P(o(t) = v(j) | i(t) = q(i))

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
    def __init__(self, max_iters=1000):
        super(HiddenMarkovModel, self).__init__()
        self.transit_matrix = None
        self.observation_matrix = None
        self.init_prob = None
        self.n_states = 0
        self.n_observations = 0
        self.states_space = None
        self.observations_space = None
        self.max_iters = max_iters

    def _normalize_by_column(A):
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
        return div0(A, np.sum(A, axis=1)[:, None])

    def _normalize_by_row(A):
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
        return div0(A, np.sum(A, axis=0))

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
        for key, value in params.iteritems():
            try:
                setattr(self, key, value)
            except AttributeError:
                raise AttributeError("Invalid paramter %s for estimator %s.".format(key, self.__class__.__name__))
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
        self.n_observations = unique_obs.shape[0]
        
        if states is None:
            if self.n_states == 0 or self.states is None:
                raise ValueError("States must be specified if no states data provided.")

            indice = self._get_seq_indice(obs, mode='observation', check_input=False)
            A = np.random.rand((self.n_states, self.n_states))
            B = np.random.rand((self.n_states, self.n_observations))
            pi = np.random.rand((self.n_states, ))

            A = self._normalize_by_column(A)
            B = self._normalize_by_column(B)
            pi /= np.sum(pi)

            self.transit_matrix = A
            self.observation_matrix = B
            self.init_prob = pi

            for _ in xrange(self.max_iters):
                p_forward = self._cal_forward_proba(indice, None, check_input=False)
                p_backward = self._cal_backward_proba(indice, None, check_input=False)

                tmp = p_forward * p_backward
                p_state = tmp / np.sum(tmp, axis=0)

                p_transit = np.zeros((self.n_states, self.n_states))
                for i in xrange(T-1):
                    bi = B[:, indice[i+1]]
                    tmp = p_forward[:, i].dot(p_backward[:, i]) * A * bi
                    p_transit += self._normalize_by_row(tmp)

                A = div0(p_transit, np.sum(p_state[:, :-1], axis=1)[:, None])
                B = np.zeros((self.n_states, self.n_observations))
                for i in xrange(T):
                    j = obs_ids[i]
                    B[:, j] += p_state[:, j]
                B = self._normalize_by_row(B)
                pi = p_state[:, 0]

                diff = np.sum((A - self.transit_matrix) ** 2) + \
                       np.sum((B - self.observation_matrix) ** 2) + \
                       np.sum((pi - self.init_prob) ** 2)
                if diff < EPS:
                    break

                self.transit_matrix = A
                self.observation_matrix = B
                self.init_prob = pi

        else:
            states = check_1d_array(states)
            obs, states = check_arrays(obs, states)

            unique_states, states_count, state_ids = np.unique(states, return_counts=True, return_inverse=True)
            
            self.init_prob = states_count / T
            self.states_space = TwoEndedIndex(unique_states)
            self.n_states = unique_states.shape[0]
            
            A = np.zeros((self.n_states, self.n_states), dtype=np.int32)
            for t in xrange(T-1):
                i, j = state_ids[t], state_ids[t+1]
                A[i, j] += 1
            self.transit_matrix = self._normalize_by_column(A)

            B = np.zeros((self.n_states, self.n_observations), dtype=np.int32)
            for t in xrange(T):
                i, k = state_ids[t], obs_ids[t]
                B[i, k] += 1
            self.observation_matrix = self._normalize_by_column(B)

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
        # check (A, B, pi)

        T = obs.shape[0]
        A, B, pi = self.transit_matrix, self.observation_matrix, self.init_prob
        indice = self._get_seq_indice(obs, mode='observation', accept_invalid=True)
        default_obs_prob = 1 / self.n_observations

        pred_indice = np.empty((T, ))
        
        delta = pi * default_obs_prob if indice[0] == -1 else pi * B[:, indice[0]]
        phi = np.empty((T, self.n_states))

        for i in xrange(1, T-1):
            o = indice[i]
            tmp = delta[:, None] * A
            phi[i-1] = np.argmax(tmp, axis=0)
            for j in xrange(self.n_states):
                k = phi[i-1,j]
                delta[j] = tmp[k,j] * B[k,o]

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

        # check (A, B, pi)
        if T < 0:
            raise ValueError("bad input sequence lenght {0}.".format(T))
        
        A, B, pi = self.transit_matrix, self.observation_matrix, self.init_prob
        obs = np.empty((T,))
        i = np.random.choice(self.states_space.values(), 1, p=pi)
        for t in xrange(T):
            obs[t] = np.random.choice(self.observations_space.keys(), 1, p=B[:,i])
            i = np.random.choice(self.states_space.values(), 1, p=A[i,:])
        return obs

    def _cal_forward_proba(self, indice, t=None, check_input=True):
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

        t : integer or None, optional(default=None)
            if None, forward probabilities at all period will be calculated;
            otherwise, forward probability at period t will be calculated

        check_input : boolean, optional(defaul='True')
            determine whether to check if the sequence is of
            valid shape

        Returns
        -------
        forward_proba : array-like; forward probability
            if t is None, shape = [n_states, n_seq]
            otherwise, shape = [n_states]
    
        """
        if check_input:
            indice = check_1d_array(indice)
        end = indice.shape[0]
        # check (A,B,pi)

        A = self.transit_matrix
        B = self.observation_matrix
        pi = self.init_prob

        if t is None:
            proba = np.empty((self.n_states, end))
            proba[:,0] = pi * B[:,indice[0]]
            for i in xrange(1, end):
                proba[:,i] = np.dot(proba[:,i-1], A) * B[:, indice[i]]
        else:
            t = self._validate_period(t, end)
            proba = pi * B[:,indice[0]]
            for i in xrange(1, t):
                proba = np.dot(proba, A) * B[:, indice[i]]
        return proba


    def _cal_backward_proba(self, indice, t=None, check_input=True):
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

        t : integer or None, optional(default=None)
            if None, backward probabilities at all period will be calculated;
            otherwise, backward probability at period t will be calculated

        check_input : boolean, optional(defaul='True')
            determine whether to check if the sequence is of
            valid shape

        Returns
        -------
        backward_proba : array-like; backward probability
            if t is None, shape = [n_states, n_seq]
            otherwise, shape = [n_states]
    
        """
        if check_input:
            indice = check_1d_array(indice)
        end = indice.shape[0]

        # check (A,B,pi)
        A = self.transit_matrix
        B = self.observation_matrix
        pi = self.init_prob
        
        if t is None:
            proba = np.empty((self.n_states, end))
            proba[:, end-1] = 1.
            for i in xrange(end-2, -1, -1):
                proba[:, i] = np.dot(A, proba[:, i+1] * B[:, indice[i+1]])
        else:
            t = self._validate_period(t, end)
            proba = np.ones((self.n_states, ))
            for i in xrange(end-2, t-1, -1):
                proba = np.dot(A, proba * B[:, indice[i+1]])
        return proba

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
        if t is None:
            # calculate proba for all times
            forward_proba = self._cal_forward_proba(indice, None, check_input=False)
            back_proba = self._cal_backward_proba(indice, None, check_input=False)
            tmp = forward_proba * back_proba
            return tmp / np.sum(tmp, axis=0)
        else:
            t = self._validate_period(end, end, t)

            forward_proba = self._cal_forward_proba(indice, t, check_input=False),
            back_proba = self._cal_backward_proba(indice, t, check_input=False)
            tmp = forward_proba * back_proba
            return tmp / np.sum(tmp)

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

        A, B = self.transit_matrix, self.observation_matrix
        indice = self._get_seq_indice(indice, mode='observation', check_input=False)

        if t is None:
            forward_proba = self._cal_forward_proba(indice, None, check_input=False)
            back_proba = self._cal_backward_proba(indice, None, check_input=False)

            r = np.empty((end, self.n_states, self.n_states))
            for i in xrange(end):
                bi = B[:, indice[i+1]]
                tmpi = forward_proba[:, i].dot(back_proba[:, i]) * A * bi
                r[i] = tmpi / np.sum(tmpi)
            return r
        else:
            t = self._validate_period(end, end, t)
            b = B[:, indice[t+1]]

            forward_proba = self._cal_forward_proba(obs_seq, t, check_input=False),
            back_proba = self._cal_backward_proba(obs_seq, t+1, check_input=False)
            tmp = forward_proba.dot(back_proba) * A * b
            return tmp / np.sum(tmp)

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

        # check (A,B,pi)
        
        # foward algorithm
        forward_proba = self._cal_forward_proba(indice, check_input=False)
        return np.sum(forward_proba)





