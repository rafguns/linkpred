import logging
log = logging.getLogger(__name__)

import networkx as nx


def without_low_degree_nodes(G, minimum=1, eligible=None):
    """Return a copy of the graph without nodes with degree below minimum

    arguments
    ---------
    g : a networkx.graph

    minimum : int
        minimum node degree

    eligible : none or string
        only eligible nodes are considered for removal

    """
    def low_degree(G, threshold):
        """Get eligible nodes whose degree is below the threshold"""
        if eligible is None:
            return [n for n, d in G.degree() if d < threshold]
        else:
            return [n for n, d in G.degree()
                    if d < threshold and G.node[n][eligible]]

    to_remove = low_degree(G, minimum)
    H = G.copy()
    H.remove_nodes_from(to_remove)
    log.info("Removed %d nodes (degree < %d)", len(to_remove), minimum)

    return H


def without_uncommon_nodes(networks, eligible=None):
    """Return list of networks without nodes not common to all

    Arguments
    ---------
    networks : an iterable of `networkx.Graph`s

    eligible : None or string
        only eligible nodes are considered for removal

    Example
    -------
    >>> import networkx as nx
    >>> A, B = nx.Graph(), nx.Graph()
    >>> A.add_nodes_from('abcd')
    >>> B.add_nodes_from('cdef')
    >>> A2, B2 = without_uncommon_nodes((A, B))
    >>> sorted(A2.nodes())
    ['c', 'd']
    >>> sorted(B2.nodes())
    ['c', 'd']

    """
    def items_outside(G, nbunch):
        """Get eligible nodes outside nbunch"""
        if eligible is None:
            return [n for n in G.nodes() if n not in nbunch]
        else:
            return [n for n in G.nodes()
                    if G.node[n][eligible] and n not in nbunch]

    common = set.intersection(*[set(G) for G in networks])
    new_networks = []
    for G in networks:
        to_remove = items_outside(G, common)
        H = G.copy()
        H.remove_nodes_from(to_remove)
        new_networks.append(H)
        log.info("Removed %d nodes (not common)", len(to_remove))

    return new_networks


def without_selfloops(G):
    """return copy of G without selfloop edges"""
    H = G.copy()
    num_loops = nx.number_of_selfloops(G)

    if num_loops:
        log.warning("Network contains {} self-loops. "
                    "Removing...".format(num_loops))
        H.remove_edges_from(nx.selfloop_edges(G))

    return H
