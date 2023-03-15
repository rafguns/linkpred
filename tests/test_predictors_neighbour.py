from math import log, sqrt

import networkx as nx
import pytest

import linkpred.predictors.neighbour as nbr
from linkpred.evaluation import Scoresheet


class TestUnweighted:
    def setup_method(self):
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5)])

    def test_adamic_adar(self):
        known = {
            (1, 5): 1 / log(3),
            (2, 3): 2 / log(2),
            (1, 4): 1 / log(2) + 1 / log(3),
            (4, 5): 1 / log(3),
        }
        found = nbr.AdamicAdar(self.G).predict()
        assert found == pytest.approx(Scoresheet(known))

    def test_association_strength(self):
        known = {(1, 5): 0.5, (2, 3): 1 / 3, (1, 4): 0.5, (4, 5): 0.5}
        found = nbr.AssociationStrength(self.G).predict()
        assert found == pytest.approx(Scoresheet(known))

    def test_common_neighbours(self):
        known = {(1, 5): 1, (2, 3): 2, (1, 4): 2, (4, 5): 1}
        found = nbr.CommonNeighbours(self.G).predict()
        assert found == pytest.approx(Scoresheet(known))

    def test_cosine(self):
        known = {
            (1, 5): 1 / sqrt(2),
            (2, 3): 2 / sqrt(6),
            (1, 4): 1,
            (4, 5): 1 / sqrt(2),
        }
        found = nbr.Cosine(self.G).predict()
        assert found == pytest.approx(Scoresheet(known))

    def test_degree_product(self):
        known = {
            (1, 2): 4,
            (1, 3): 6,
            (1, 4): 4,
            (1, 5): 2,
            (2, 3): 6,
            (2, 4): 4,
            (2, 5): 2,
            (3, 4): 6,
            (3, 5): 3,
            (4, 5): 2,
        }
        found = nbr.DegreeProduct(self.G).predict()
        assert found == pytest.approx(Scoresheet(known))

    def test_jaccard(self):
        known = {(1, 5): 0.5, (2, 3): 2 / 3, (1, 4): 1, (4, 5): 0.5}
        found = nbr.Jaccard(self.G).predict()
        assert found == pytest.approx(Scoresheet(known))

    def test_nmeasure(self):
        known = {
            (1, 5): sqrt(2 / 5),
            (2, 3): sqrt(8 / 13),
            (1, 4): 1,
            (4, 5): sqrt(2 / 5),
        }
        found = nbr.NMeasure(self.G).predict()
        assert found == pytest.approx(Scoresheet(known))

    def test_maxoverlap(self):
        known = {(1, 5): 0.5, (2, 3): 2 / 3, (1, 4): 1, (4, 5): 0.5}
        found = nbr.MaxOverlap(self.G).predict()
        assert found == pytest.approx(Scoresheet(known))

    def test_minoverlap(self):
        known = {(1, 5): 1, (2, 3): 1, (1, 4): 1, (4, 5): 1}
        found = nbr.MinOverlap(self.G).predict()
        assert found == pytest.approx(Scoresheet(known))

    def test_pearson(self):
        known = {(1, 5): 0.61237243, (2, 3): 2 / 3, (1, 4): 1, (4, 5): 0.61237243}
        found = nbr.Pearson(self.G).predict()
        assert found == pytest.approx(Scoresheet(known))

    def test_resource_allocation(self):
        known = {(1, 5): 1 / 3, (2, 3): 1, (1, 4): 5 / 6, (4, 5): 1 / 3}
        found = nbr.ResourceAllocation(self.G).predict()
        assert found == pytest.approx(Scoresheet(known))


class TestWeighted:
    def setup_method(self):
        self.G = nx.Graph()
        self.G.add_weighted_edges_from(
            [(1, 2, 1), (1, 3, 5), (2, 4, 2), (3, 4, 1), (3, 5, 2)]
        )

    def test_adamic_adar(self):
        known = {
            (1, 5): 10 / log(30),
            (1, 4): 2 / log(5) + 5 / log(30),
            (2, 3): 2 / log(5) + 5 / log(26),
            (4, 5): 2 / log(30),
        }
        found = nbr.AdamicAdar(self.G).predict(weight="weight")
        assert found == pytest.approx(Scoresheet(known))

    def test_association_strength(self):
        known = {(1, 5): 5 / 52, (2, 3): 7 / 150, (1, 4): 7 / 130, (4, 5): 0.1}
        found = nbr.AssociationStrength(self.G).predict(weight="weight")
        assert found == pytest.approx(Scoresheet(known))

    def test_common_neighbours(self):
        known = {(1, 5): 10, (2, 3): 7, (1, 4): 7, (4, 5): 2}
        found = nbr.CommonNeighbours(self.G).predict(weight="weight")
        assert found == pytest.approx(Scoresheet(known))

    def test_cosine(self):
        known = {
            (1, 5): 10 / sqrt(104),
            (2, 3): 7 / sqrt(150),
            (1, 4): 7 / sqrt(130),
            (4, 5): 2 / sqrt(20),
        }
        found = nbr.Cosine(self.G).predict(weight="weight")
        assert found == pytest.approx(Scoresheet(known))

    def test_degree_product(self):
        known = {
            (1, 2): 130,
            (1, 3): 780,
            (1, 4): 130,
            (1, 5): 104,
            (2, 3): 150,
            (2, 4): 25,
            (2, 5): 20,
            (3, 4): 150,
            (3, 5): 120,
            (4, 5): 20,
        }
        found = nbr.DegreeProduct(self.G).predict(weight="weight")
        assert found == pytest.approx(Scoresheet(known))

    def test_jaccard(self):
        known = {(1, 5): 0.5, (2, 3): 7 / 28, (1, 4): 7 / 24, (4, 5): 2 / 7}
        found = nbr.Jaccard(self.G).predict(weight="weight")
        assert found == pytest.approx(Scoresheet(known))

    def test_nmeasure(self):
        known = {
            (1, 5): sqrt(50 / 173),
            (2, 3): sqrt(98 / 925),
            (1, 4): sqrt(98 / 701),
            (4, 5): sqrt(8 / 41),
        }
        found = nbr.NMeasure(self.G).predict(weight="weight")
        assert found == pytest.approx(Scoresheet(known))

    def test_maxoverlap(self):
        known = {(1, 5): 5 / 13, (2, 3): 7 / 30, (1, 4): 7 / 26, (4, 5): 0.4}
        found = nbr.MaxOverlap(self.G).predict(weight="weight")
        assert found == pytest.approx(Scoresheet(known))

    def test_minoverlap(self):
        known = {(1, 5): 2.5, (2, 3): 1.4, (1, 4): 1.4, (4, 5): 0.5}
        found = nbr.MinOverlap(self.G).predict(weight="weight")
        assert found == pytest.approx(Scoresheet(known))

    def test_pearson(self):
        known = {(1, 5): 0.9798502, (2, 3): 0.2965401, (1, 4): 0.4383540, (4, 5): 0.25}
        found = nbr.Pearson(self.G).predict(weight="weight")
        assert found == pytest.approx(Scoresheet(known))

    def test_resource_allocation(self):
        known = {(1, 5): 1 / 3, (2, 3): 77 / 130, (1, 4): 17 / 30, (4, 5): 1 / 15}
        found = nbr.ResourceAllocation(self.G).predict(weight="weight")
        assert found == pytest.approx(Scoresheet(known))
