import networkx as nx
from linkpred.network.misc import from_biadjacency_matrix
from nose.tools import *

class TestMisc:

    def setup(self):
        self.G = nx.bipartite_gnmk_random_graph(40, 60, 50)
        self.M = nx.bipartite.biadjacency_matrix(self.G, range(40))

    def test_biadjacency_matrix1(self):
        H = from_biadjacency_matrix(self.M, range(40), range(40, 100))
        assert_equal(sorted(self.G.edges()), sorted(H.edges()))
        assert_equal(sorted(self.G.nodes()), sorted(H.nodes()))

    def test_biadjacency_matrix2(self):
        H = from_biadjacency_matrix(self.M)
        assert_equal(sorted(self.G.edges()), sorted(H.edges()))
        assert_equal(sorted(self.G.nodes()), sorted(H.nodes()))

    @raises(ValueError)
    def test_biadjacency_matrix_wrong_row_items(self):
        from_biadjacency_matrix(self.M, range(41), range(41, 101))

    @raises(ValueError)
    def test_biadjacency_matrix_wrong_col_items(self):
        from_biadjacency_matrix(self.M, range(40), range(40, 101))
