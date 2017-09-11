import networkx as nx
from linkpred.network.addremove import *
from nose.tools import *


def test_add_random_edges():
    G = nx.star_graph(10)
    edges = list(G.edges())

    add_random_edges(G, 0)
    assert_equal(edges, list(G.edges()))

    add_random_edges(G, 0.5)
    assert_equal(G.size(), 15)
    assert set(edges) < set(G.edges())

    assert_raises(ValueError, add_random_edges, G, 1.2)


def test_remove_random_edges():
    G = nx.star_graph(10)
    edges = list(G.edges())

    remove_random_edges(G, 0)
    assert_equal(edges, list(G.edges()))

    remove_random_edges(G, 0.5)
    assert_equal(G.size(), 5)
    assert set(G.edges()) < set(edges)

    assert_raises(ValueError, remove_random_edges, G, 10)


def test_add_remove_random_edges():
    G = nx.star_graph(10)
    edges = list(G.edges())

    add_remove_random_edges(G, 0, 0)
    assert_equal(edges, list(G.edges()))

    add_remove_random_edges(G, 0.3, 0.4)
    assert_equal(G.size(), 9)
    assert_equal(len(set(edges) & set(G.edges())), 6)

    assert_raises(ValueError, add_remove_random_edges, G, 0, 1.2)
    assert_raises(ValueError, add_remove_random_edges, G, 1.2, 0)

