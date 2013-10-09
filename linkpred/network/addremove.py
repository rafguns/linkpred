from random import sample
from ..util import all_pairs, log

__all__ = ['add_random_edges', 'remove_random_edges',
           'add_remove_random_edges']


def add_random_edges(G, pct):
    edges = G.edges()
    m = len(edges)
    to_add = int(m * pct)
    log.logger.debug("Will add %d edges to %d (%f)" % (to_add, m, pct))

    new_edges = set(all_pairs(G.nodes())) - set(edges)
    G.add_edges_from(sample(new_edges, to_add), weight=1)


def remove_random_edges(G, pct):
    edges = G.edges()
    m = len(edges)
    to_remove = int(m * pct)

    log.logger.debug("Will remove %d edges of %d (%f)" % (to_remove, m, pct))
    G.remove_edges_from(sample(edges, to_remove))


def add_remove_random_edges(G, pct_add, pct_remove):
    edges = G.edges()
    m = len(edges)
    to_add = int(m * pct_add)
    to_remove = int(m * pct_remove)
    log.logger.debug("Will add %d (%f) edges to and remove"
                     "%d (%f) edges from %d" %
                     (to_add, pct_add, to_remove, pct_remove, m))

    new_edges = set(all_pairs(G.nodes())) - set(edges)
    G.remove_edges_from(sample(edges, to_remove))
    G.add_edges_from(sample(new_edges, to_add))
