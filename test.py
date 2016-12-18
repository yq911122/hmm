from hmm import HiddenMarkovModel, TwoEndedIndex
import numpy as np

clf = HiddenMarkovModel()

def test_initiation():
    pi = np.array([0.2, 0.4, 0.4])
    A = np.array([
                  [0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5],
                 ])
    B = np.array([
                  [0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3],
                 ])
    V = TwoEndedIndex(['b1','b2','b3'])
    O = TwoEndedIndex(['red','white'])
    clf.set_params(init_prob=pi, transit_matrix=A, observation_matrix=B, n_states=3, n_observations=2, states_space=V, observations_space=O)
    # print clf.init_prob
    # print clf.transit_matrix
    # print clf.observation_matrix
    # print clf.n_states
    # print clf.n_observations
    return clf

def test_proba():
    clf = test_initiation()
    seq = [0,1,0]
    pi = np.array([0.2, 0.4, 0.4])
    b0 = np.array([0.5,0.4,0.7])

    for i in xrange(3):
        print clf._cal_forward_proba(seq, t=i)
    forward_proba = clf._cal_forward_proba(seq)
    # should return [[ 0.1       0.077     0.04187 ]
                 # [ 0.16      0.1104    0.035512]
                 # [ 0.28      0.0606    0.052836]]
    for i in xrange(3):
        print clf._cal_backward_proba(seq, t=i)
    backward_proba = clf._cal_backward_proba(seq)
    # should return [[ 0.2451  0.54    1.    ]
                 # [ 0.2622  0.49    1.    ]
                 # [ 0.2277  0.57    1.    ]]
    p1 = np.sum(forward_proba[:,-1])
    p2 = np.sum(pi * b0 * backward_proba[:, 0])
    return p1 == p2

# to be tested
def test_state_proba():
    clf = test_initiation()
    seq = [0,1,0]
    return clf.cal_state_proba(seq)

# to be tested
def test_trasit_proba():
    clf = test_initiation()
    seq = [0,1,0]
    return clf.cal_trasit_proba(seq)


def test_observations_sequence_generation():
    clf = test_initiation()
    return clf.generate_observation_sequence(5)

def test_fit():
    obs_seq = ['red','white','red']
    state_seq = ['b1', 'b2',' b3']
    V = TwoEndedIndex(['b1','b2','b3'])
    clf.states_space = V
    clf.n_states = 3
    clf.fit(obs_seq)
    return 

def test_predict():
    clf = test_initiation()
    obs_seq = ['red','white','red']
    return clf.predict(obs_seq)

print test_predict()
