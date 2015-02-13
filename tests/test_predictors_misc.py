from nose.tools import *
import networkx as nx

from linkpred.evaluation import Pair
from linkpred.predictors.misc import *


class TestCopy:
    def setup(self):
        self.G = nx.Graph()
        self.G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])

    def test_copy_unweighted(self):
        expected = {Pair(0, 1): 1, Pair(1, 2): 1}
        assert_dict_equal(Copy(self.G).predict(), expected)

    def test_copy_weighted(self):
        expected = {Pair(0, 1): 3.0, Pair(1, 2): 7.5}
        assert_dict_equal(Copy(self.G).predict(weight="weight"), expected)


def test_community():
    pass


def test_random():
    G = nx.Graph()
    G.add_nodes_from(range(10), eligible=True)
    prediction = Random(G).predict()
    assert_equal(len(prediction), 45)


def test_random_exclude_noneligible():
    G = nx.Graph()
    G.add_nodes_from(range(5), eligible=True)
    G.add_nodes_from(range(5, 10), eligible=False)
    prediction = Random(G, eligible='eligible').predict()
    assert_equal(len(prediction), 10)
    for i in range(5):
        for j in range(5):
            if i != j:
                assert Pair(i, j) in prediction
