from nose.tools import assert_equal, assert_less_equal
import networkx as nx

from linkpred.predictors.eigenvector import RootedPageRank, SimRank


class TestEigenvectorRuns:
    """Test if eigenvector methods run

    This is a bit of a temporary hack to avoid regressions.We're not quite
    sure of the correct output of, especially, SimRank, and so having tests
    that at least run the code are better than nothing.

    """
    def setup(self):
        self.n = 20
        self.G = nx.gnm_random_graph(self.n, self.n * 3)

    def test_rooted_pagerank_runs(self):
        pred = RootedPageRank(self.G).predict()
        assert_less_equal(len(pred), self.n * (self.n - 1) // 2)

    def test_simrank_runs(self):
        pred = SimRank(self.G).predict()
        assert_equal(len(pred), self.n * (self.n - 1) // 2)


class TestEigenVector:

    def test_rooted_pagerank(self):
        pass

    def test_rooted_pagerank_weighted(self):
        pass

    def test_rooted_pagerank_alpha(self):
        pass

    def test_rooted_pagerank_beta(self):
        pass

    def test_rooted_pagerank_k(self):
        pass

    def test_simrank(self):
        pass

    def test_simrank_c(self):
        pass

    def test_simrank_weighted(self):
        pass
