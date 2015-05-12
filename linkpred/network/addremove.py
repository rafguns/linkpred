import logging
import networkx as nx
import random

log = logging.getLogger(__name__)

__all__ = ['add_random_edges', 'remove_random_edges',
           'add_remove_random_edges']


def assert_is_percentage(pct):
    if not 0 <= pct <= 1:
        raise ValueError("Percentage should be float between 0 and 1")


def add_random_edges(G, pct):
    """Add `n` random edges to G (`n` = fraction of current edge count)

    Parameters
    ----------
    G : a networkx.Graph
        the network

    pct : float
        A percentage (between 0 and 1)
    """
    assert_is_percentage(pct)
    m = G.size()
    to_add = int(m * pct)
    log.debug("Will add %d edges to %d (%f)", to_add, m, pct)

    new_edges = set(nx.non_edges(G))
    G.add_edges_from(random.sample(new_edges, to_add), weight=1)


def remove_random_edges(G, pct):
    """Randomly remove `n` edges from G (`n` = fraction of current edge count)

    Parameters
    ----------
    G : a networkx.Graph
        the network

    pct : float
        A percentage (between 0 and 1)
    """
    assert_is_percentage(pct)
    edges = G.edges()
    m = len(edges)
    to_remove = int(m * pct)

    log.debug("Will remove %d edges of %d (%f)", to_remove, m, pct)
    G.remove_edges_from(random.sample(edges, to_remove))


def add_remove_random_edges(G, pct_add, pct_remove):
    """Randomly add edges to and remove edges from G

    Parameters
    ----------
    G : a networkx.Graph
        the network

    pct_add : float
        A percentage (between 0 and 1)

    pct_remove : float
        A percentage (between 0 and 1)
    """
    assert_is_percentage(pct_add)
    assert_is_percentage(pct_remove)
    edges = G.edges()
    m = len(edges)
    to_add = int(m * pct_add)
    to_remove = int(m * pct_remove)
    log.debug("Will add %d (%f) edges to and remove %d (%f) edges of %d",
              to_add, pct_add, to_remove, pct_remove, m)

    new_edges = set(nx.non_edges(G))
    G.remove_edges_from(random.sample(edges, to_remove))
    G.add_edges_from(random.sample(new_edges, to_add))
