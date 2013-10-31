from __future__ import division
from math import log, sqrt
from nose.tools import assert_almost_equal
import networkx as nx

from linkpred.evaluation import Scoresheet
from linkpred.predictors.neighbour import *


def assert_dict_almost_equal(d1, d2):
    for k in d1:
        assert_almost_equal(d1[k], d2[k])


class TestUnweighted:
    def setup(self):
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5)])

    def test_adamic_adar(self):
        known = {(1, 5): 1 / log(3), (2, 3): 2 / log(2),
                 (1, 4): 1 / log(2) + 1 / log(3), (4, 5): 1 / log(3)}
        found = AdamicAdar(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_association_strength(self):
        known = {(1, 5): 0.5, (2, 3): 1 / 3, (1, 4): 0.5, (4, 5): 0.5}
        found = AssociationStrength(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_common_neighbours(self):
        known = {(1, 5): 1, (2, 3): 2, (1, 4): 2, (4, 5): 1}
        found = CommonNeighbours(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_cosine(self):
        known = {(1, 5): 1 / sqrt(2), (2, 3): 2 / sqrt(6), (1, 4): 1,
                 (4, 5): 1 / sqrt(2)}
        found = Cosine(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_degree_product(self):
        known = {(1, 2): 4, (1, 3): 6, (1, 4): 4, (1, 5): 2, (2, 3): 6,
                 (2, 4): 4, (2, 5): 2, (3, 4): 6, (3, 5): 3, (4, 5): 2}
        found = DegreeProduct(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_jaccard(self):
        known = {(1, 5): 0.5, (2, 3): 2 / 3, (1, 4): 1, (4, 5): 0.5}
        found = Jaccard(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_nmeasure(self):
        known = {(1, 5): sqrt(2 / 5), (2, 3): sqrt(8 / 13), (1, 4): 1,
                 (4, 5): sqrt(2 / 5)}
        found = NMeasure(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_maxoverlap(self):
        known = {(1, 5): 0.5, (2, 3): 2 / 3, (1, 4): 1, (4, 5): 0.5}
        found = MaxOverlap(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_minoverlap(self):
        known = {(1, 5): 1, (2, 3): 1, (1, 4): 1, (4, 5): 1}
        found = MinOverlap(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_pearson(self):
        known = {(1, 5): 0.61237243, (2, 3): 2 / 3, (1, 4): 1,
                 (4, 5): 0.61237243}
        found = Pearson(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_resource_allocation(self):
        known = {(1, 5): 1 / 3, (2, 3): 1, (1, 4): 5 / 6, (4, 5): 1 / 3}
        found = ResourceAllocation(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))
