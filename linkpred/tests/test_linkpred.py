from nose.tools import assert_less_equal

import networkx as nx
from linkpred.linkpred import filter_low_degree_nodes


def test_filter_low_degree_nodes():
    G1 = nx.erdos_renyi_graph(50, 0.1)
    G2 = nx.erdos_renyi_graph(50, 0.1)

    filter_low_degree_nodes([G1, G2])

    assert_less_equal(len(G1), 50)
    assert_less_equal(len(G2), 50)
