import networkx as nx

from linkpred.preprocess import (
    without_low_degree_nodes,
    without_selfloops,
    without_uncommon_nodes,
)


def test_without_uncommon_nodes():
    G1 = nx.erdos_renyi_graph(50, 0.1)
    G2 = nx.erdos_renyi_graph(50, 0.1)
    G1, G2 = without_uncommon_nodes([G1, G2])
    assert len(G1) <= 50
    assert len(G2) == len(G1)

    node_sets = [range(5), range(1, 7)]
    graphs = []
    for nodes in node_sets:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for n in G:
            G.nodes[n]["eligible"] = n % 2 == 0
        graphs.append(G)

    for G in without_uncommon_nodes(graphs):
        assert sorted(n for n in G if G.nodes[n]["eligible"]) == [2, 4]


def test_without_low_degree_nodes():
    G = nx.star_graph(4)
    G.add_edge(1, 2)
    G = without_low_degree_nodes(G, minimum=2)
    assert sorted(G) == [0, 1, 2]

    edges = [(0, 1), (0, 5), (2, 3), (2, 5), (4, 3)]
    G = nx.Graph()
    G.add_edges_from(edges)
    for n in G:
        G.nodes[n]["eligible"] = n % 2 == 0
    G = without_low_degree_nodes(G, minimum=2)
    assert sorted(n for n in G if G.nodes[n]["eligible"]) == [0, 2]


def test_without_selfloops():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 2)])
    G = without_selfloops(G)
    assert sorted(G.edges()) == [(0, 1), (1, 2), (1, 3)]
