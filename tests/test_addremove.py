import networkx as nx
import pytest
from linkpred.network.addremove import (
    add_random_edges,
    remove_random_edges,
    add_remove_random_edges,
)


def test_add_random_edges():
    G = nx.star_graph(10)
    edges = list(G.edges())

    add_random_edges(G, 0)
    assert edges == list(G.edges())

    add_random_edges(G, 0.5)
    assert G.size() == 15
    assert set(edges) < set(G.edges())

    with pytest.raises(ValueError):
        add_random_edges(G, 1.2)


def test_remove_random_edges():
    G = nx.star_graph(10)
    edges = list(G.edges())

    remove_random_edges(G, 0)
    assert edges == list(G.edges())

    remove_random_edges(G, 0.5)
    assert G.size() == 5
    assert set(G.edges()) < set(edges)

    with pytest.raises(ValueError):
        remove_random_edges(G, 10)


def test_add_remove_random_edges():
    G = nx.star_graph(10)
    edges = list(G.edges())

    add_remove_random_edges(G, 0, 0)
    assert edges == list(G.edges())

    add_remove_random_edges(G, 0.3, 0.4)
    assert G.size() == 9
    assert len(set(edges) & set(G.edges())) == 6

    with pytest.raises(ValueError):
        add_remove_random_edges(G, 0, 1.2)
    with pytest.raises(ValueError):
        add_remove_random_edges(G, 1.2, 0)
