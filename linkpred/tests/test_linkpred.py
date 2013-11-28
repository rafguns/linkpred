from nose.tools import assert_equal, assert_less_equal, raises

import linkpred
import networkx as nx


def test_imports():
    linkpred.LinkPred
    linkpred.LinkPredError
    linkpred.filter_low_degree_nodes
    linkpred.read_network
    linkpred.network
    linkpred.evaluation
    linkpred.predictors


def test_filter_low_degree_nodes():
    G1 = nx.erdos_renyi_graph(50, 0.1)
    G2 = nx.erdos_renyi_graph(50, 0.1)
    linkpred.filter_low_degree_nodes([G1, G2])
    assert_less_equal(len(G1), 50)
    assert_equal(len(G2), len(G1))

    G = nx.star_graph(4)
    G.add_edge(1, 2)
    linkpred.filter_low_degree_nodes([G], minimum=2)
    assert_equal(sorted(G), [0, 1, 2])

    edge_sets = [[(0, 1), (0, 5), (2, 3), (2, 5), (4, 3)],
                 [(0, 1), (0, 5), (2, 3), (2, 5), (4, 3), (4, 1)]]
    expected = [0, 2]
    graphs = []
    for edges in edge_sets:
        G = nx.Graph()
        G.add_edges_from(edges)
        for n in G:
            G.node[n]['eligible'] = n % 2 == 0
        graphs.append(G)
    linkpred.filter_low_degree_nodes(graphs, minimum=2)
    for G in graphs:
        assert_equal(sorted(n for n in G if G.node[n]['eligible']), expected)


def test_for_comparison():
    from linkpred.linkpred import for_comparison
    from linkpred.evaluation import Pair

    G = nx.path_graph(10)
    expected = set(Pair(x) for x in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                                     (5, 6), (6, 7), (7, 8), (8, 9)])
    assert_equal(for_comparison(G), expected)

    to_delete = [Pair(2, 3), Pair(8, 9)]
    expected = expected.difference(to_delete)
    assert_equal(for_comparison(G, exclude=to_delete), expected)


def test_pretty_print():
    from linkpred.linkpred import pretty_print

    name = "foo"
    assert_equal(pretty_print(name), "foo")
    params = {"bar": 0.1, "baz": 5}
    assert_equal(pretty_print(name, params), "foo (baz = 5, bar = 0.1)")


@raises(linkpred.LinkPredError)
def test_LinkPred_without_predictors():
    linkpred.LinkPred()
