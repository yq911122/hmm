import warnings
from warnings import warn

import numpy as np

def check_1d_array(y):
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        warn("A column-vector y was passed when a 1d array was"
             " expected. Please change the shape of y to "
              "(n_samples, ), for example using ravel().")
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))

class HiddenMarkovModel(object):
    """docstring for HiddenMarkovModel"""
    def __init__(self):
        super(HiddenMarkovModel, self).__init__()
        self.transit_matrix = None
        self.observation_matrix = None
        self.init_prob = None
        self.n_states = 0
        self.n_observations = 0

    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.iteritems():
            try:
                setattr(self, key, value)
            except AttributeError:
                raise AttributeError("Invalid paramter %s for estimator %s.".format(key, self.__class__.__name__))
        return self

    def _get_seq_indice(self, seq, mode='state', check_input=True):
        if check_input:
            seq = check_1d_array(seq)
        seq_length = seq.shape[0]

        if mode not in ['state', 'observation']:
            raise ValueError("invalid mode, only 'state' or 'observation' are valid.")
        if mode == 'state':
            max_label = self.n_states - 1
        else:
            max_label = self.n_observations - 1
        indice = np.empty((seq_length,), dtype=np.int32)
        for i, val in enumerate(seq):
            if 0 <= val <= max_label:
                indice[i] = val
            else:
                raise KeyError("Invalid value {0} in the sequence. Valid input should be in [0, {1}]".format(val, max_label))
        return indice

    def _validate_period(self, t, end):
        if t < 0 or t > end:
            raise ValueError("invalid period {0}. The period should be in [{1}, {2}].".format(t, 0, end))
        return t

    def fit(self):
        pass

    def predict(self):
        pass

    def generate_observation_sequence(self):
        pass

    def _cal_forward_proba(self, obs_seq, t=None, check_input=True):
        if check_input:
            obs_seq = check_1d_array(obs_seq)
        end = obs_seq.shape[0]
        # check (A,B,pi)

        A = self.transit_matrix
        B = self.observation_matrix
        pi = self.init_prob
        indice = self._get_seq_indice(obs_seq, mode='observation', check_input=False)         

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


    def _cal_backward_proba(self, obs_seq, t=None, check_input=True):
        if check_input:
            obs_seq = check_1d_array(obs_seq)
        end = obs_seq.shape[0]

        # check (A,B,pi)
        A = self.transit_matrix
        B = self.observation_matrix
        pi = self.init_prob
        indice = self._get_seq_indice(obs_seq, mode='observation', check_input=False)
        
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

    def cal_state_proba(self, obs_seq, t=None, check_input=True):
        if check_input:
            obs_seq = check_1d_array(obs_seq)
        end = obs_seq.shape[0]
        if t is None:
            # calculate proba for all times
            alpha = self._cal_forward_proba(obs_seq, None, check_input=False)
            beta = self._cal_backward_proba(obs_seq, None, check_input=False)
            tmp = alpha * beta
            return tmp / np.sum(tmp, axis=0)
        else:
            t = self._validate_period(end, end, t)

            alpha = self._cal_forward_proba(obs_seq, t, check_input=False),
            beta = self._cal_backward_proba(obs_seq, t, check_input=False)
            tmp = alpha * beta
            return tmp / np.sum(tmp)

    def cal_trasit_proba(self, obs_seq, t=None, check_input=True):
        if check_input:
            obs_seq = check_1d_array(obs_seq)
        end = obs_seq.shape[0] - 1
        if end == 0:
            raise ValueError("Only 1 observation in the sequence. Insufficient data to estimate transit probability")

        A, B = self.transit_matrix, self.observation_matrix
        indice = self._get_seq_indice(obs_seq, mode='observation', check_input=False)

        if t is None:
            alpha = self._cal_forward_proba(obs_seq, None, check_input=False)
            beta = self._cal_backward_proba(obs_seq, None, check_input=False)

            r = np.empty((end, self.n_states, self.n_states))
            for i in xrange(end):
                bi = B[:, indice[i+1]]
                tmpi = alpha[:, i].dot(beta[:, i]) * A * bi
                r[i] = tmpi / np.sum(tmpi)
            return r
        else:
            t = self._validate_period(end, end, t)
            b = B[:, indice[t+1]]

            alpha = self._cal_forward_proba(obs_seq, t, check_input=False),
            beta = self._cal_backward_proba(obs_seq, t+1, check_input=False)
            tmp = alpha.dot(beta) * A * b
            return tmp / np.sum(tmp)

    def get_observation_sequence_proba(self, obs_seq):
        obs_seq = check_1d_array(obs_seq)
        # check (A,B,pi)
        
        # foward algorithm
        alpha = self._cal_forward_proba(obs_seq, check_input=False)
        return np.sum(alpha)





