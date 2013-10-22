from nose.tools import assert_equal, assert_less_equal

import networkx as nx


def test_imports():
    import linkpred

    linkpred.LinkPred
    linkpred.LinkPredError
    linkpred.filter_low_degree_nodes
    linkpred.read_network
    linkpred.network
    linkpred.evaluation
    linkpred.predictors


def test_filter_low_degree_nodes():
    from linkpred import filter_low_degree_nodes

    G1 = nx.erdos_renyi_graph(50, 0.1)
    G2 = nx.erdos_renyi_graph(50, 0.1)
    filter_low_degree_nodes([G1, G2])
    assert_less_equal(len(G1), 50)
    assert_equal(len(G2), len(G1))

    G = nx.star_graph(4)
    G.add_edge(1, 2)
    filter_low_degree_nodes([G], minimum=2)
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
    filter_low_degree_nodes(graphs, minimum=2)
    for G in graphs:
        assert_equal(sorted(n for n in G if G.node[n]['eligible']), expected)
