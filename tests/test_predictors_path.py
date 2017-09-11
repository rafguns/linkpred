from __future__ import division
from nose.tools import assert_almost_equal
from utils import assert_dict_almost_equal
import networkx as nx
import numpy as np

from linkpred.predictors.path import *
from linkpred.evaluation import Scoresheet


def test_katz():
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 1), (0, 2, 5), (2, 3, 1), (0, 4, 2),
                               (1, 4, 1), (3, 5, 1), (4, 5, 3)])

    beta = 0.01
    I = np.identity(6)
    for weight in ('weight', None):
        katz = Katz(G).predict(beta=beta, weight=weight)

        nodes = list(G.nodes())
        M = nx.to_numpy_matrix(G, nodelist=nodes, weight=weight)
        K = (I - beta * M) ** -1 - I

        x, y = np.asarray(K).nonzero()
        for i, j in zip(x, y):
            if i == j:
                continue
            u, v = nodes[i], nodes[j]
            assert_almost_equal(K[i, j], katz[(u, v)])


class TestGraphDistance:
    def setup(self):
        self.G = nx.Graph()
        self.G.add_weighted_edges_from([(0, 1, 1), (0, 2, 3), (1, 2, 1),
                                        (1, 3, 2), (2, 4, 1)])

    def test_unweighted(self):
        known = {(0, 1): 1, (0, 2): 1, (1, 2): 1, (1, 3): 1, (2, 4): 1,
                 (0, 3): 0.5, (0, 4): 0.5, (1, 4): 0.5, (2, 3): 0.5,
                 (3, 4): 1 / 3}
        known = Scoresheet(known)
        graph_distance = GraphDistance(self.G).predict(weight=None)
        assert_dict_almost_equal(graph_distance, known)

        graph_distance = GraphDistance(self.G).predict(alpha=0)
        assert_dict_almost_equal(graph_distance, known)

    def test_weighted(self):
        known = {(0, 1): 1, (0, 2): 3, (1, 2): 1, (1, 3): 2, (2, 4): 1,
                 (0, 3): 2 / 3, (0, 4): 0.75, (1, 4): 0.5, (2, 3): 2 / 3,
                 (3, 4): 0.4}
        known = Scoresheet(known)
        graph_distance = GraphDistance(self.G).predict()
        assert_dict_almost_equal(graph_distance, known)

    def test_weighted_alpha(self):
        from math import sqrt

        known = {(0, 1): 1, (0, 2): sqrt(3), (1, 2): 1, (1, 3): sqrt(2),
                 (2, 4): 1, (0, 3): 1 / (1 + 1 / sqrt(2)),
                 (0, 4): 1 / (1 + 1 / sqrt(3)), (1, 4): 0.5,
                 (2, 3): 1 / (1 + 1 / sqrt(2)), (3, 4):  1 / (2 + 1 / sqrt(2))}
        known = Scoresheet(known)
        graph_distance = GraphDistance(self.G).predict(alpha=0.5)
        assert_dict_almost_equal(graph_distance, known)
