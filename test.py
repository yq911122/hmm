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
    clf.set_params(init_prob=pi, transit_matrix=A, observation_matrix=B, states_space=V, observations_space=O)
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
    clf.fit(obs_seq)
    return 

def test_predict():
    clf = test_initiation()
    obs_seq = ['red','white','red']
    return clf.predict(obs_seq)


def test_pos_tag():
    import nltk
    from nltk.data import load
    # tagdict = load('help/tagsets/brown_tagset.pickle')
    # tags = tagdict.keys()

    words = nltk.corpus.brown.tagged_words(categories='news')[:10000]
    obs = [v[0] for v in words]
    states = [v[1] for v in words]
    tags = set(states)
    clf.set_params(states_space=TwoEndedIndex(tags))
    clf.fit(obs)

    # print clf.transit_matrix.shape
    # print clf.init_prob
    # print clf.observation_matrix.shape

print test_pos_tag()


# from scipy.sparse import coo_matrix, csc_matrix, isspmatrix_csc, isspmatrix, spdiags, isspmatrix_csr
# data = B = np.array([
#                   [0.5, 0.5],
#                   [0.4, 0.6],
#                   [0.7, 0.3],
#                  ])
# mat = csc_matrix(data)
# print clf._normalize_by_row(mat)