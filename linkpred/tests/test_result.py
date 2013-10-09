from nose.tools import *

import networkx as nx
from linkpred.result import *

class TestResult:
    def setup(self):
        self.B = nx.bipartite_random_graph(50, 60, 0.2)
        nodes = [n for n in self.B if self.B.node[n]['bipartite']]
        self.G = nx.bipartite.weighted_projected_graph(self.B, nodes)

    def test_result_init(self):
        res = Result(self.B, eligible='bipartite')
        assert_equal(len(res), len(self.G))
        assert_equal(len(res.network), len(self.G))
        assert_equal(len(res.pathspec), len(self.B))

        res = Result(self.G)
        assert_equal(len(res), len(self.G))
        assert_equal(len(res.network), len(self.G))
        with assert_raises(AttributeError):
            res.pathspec

    def test_result_remove_items(self):
        res = Result(self.B, eligible='bipartite')
        # the bottom nodes (bipartite=True) start from 50.
        res.remove_items_from(range(50, 60))
        assert_equal(len(res), 50)

def test_filter_low_degree_nodes():
    B1 = nx.bipartite_random_graph(50, 60, 0.2)
    B2 = nx.bipartite_random_graph(50, 60, 0.2)
    res1 = Result(B1, eligible='bipartite')
    res2 = Result(B2, eligible='bipartite')

    filter_low_degree_nodes([res1, res2])
    assert_less_equal(len(res1), 60)
    assert_less_equal(len(res2), 60)
    assert_equal(len(res1), len([n for n in res1.pathspec\
                                 if res1.pathspec.node[n]['bipartite']]))
    assert_equal(len(res2), len([n for n in res2.pathspec\
                                 if res1.pathspec.node[n]['bipartite']]))
