import networkx as nx

#TODO Examine if we can use nx.single-source_shortest_path_length here


def edge_weights(G, weight='weight'):
    """Iterator over edge weights in G"""
    for _, nbrdict in G.adjacency_iter():
        for edgedata in nbrdict.itervalues():
            yield edgedata[weight]


def from_biadjacency_matrix(A, row_items=None, col_items=None,
                            weight='weight'):
    """Convert biadjacency matrix to bipartite graph

    This function is a counterpart to networkx.bipartite.biadjacency_matrix .

    Parameters
    ----------
    A : numpy matrix
        biadjacency matrix

    row_items : list of nodes, optional
        list of nodes corresponding to each row
        (if not supplied, numbers are used)

    col_items : list of nodes, optional
        list of nodes corresponding to each column
        (if not supplied, numbers are used)

    weight : None or string, optional
        If None, all edge weights are considered equal.
        Otherwise holds the name of the edge attribute used as weight.

    """
    import numpy

    kind_to_python_type = {'f': float,
                           'i': int,
                           'u': int,
                           'b': bool,
                           'c': complex,
                           'S': str}

    dt = A.dtype
    nrows, ncols = A.shape
    try:
        python_type = kind_to_python_type[dt.kind]
    except:
        raise TypeError("Unknown numpy data type: %s" % dt)

    if row_items is None:
        row_items = range(nrows)
    elif len(row_items) != nrows:
        raise ValueError("Expected %d row items, but got %d instead" %
                         (nrows, len(row_items)))
    if col_items is None:
        col_items = range(nrows, nrows + ncols)
    elif len(col_items) != ncols:
        raise ValueError("Expected %d col items, but got %d instead" %
                         (ncols, len(col_items)))

    G = nx.Graph()
    G.add_nodes_from(row_items)
    G.add_nodes_from(col_items)
    # get a list of edges
    x, y = numpy.asarray(A).nonzero()

    # handle numpy constructed data type
    G.add_edges_from((row_items[u], col_items[v],
                      {weight: python_type(A[u, v])})
                     for u, v in zip(x, y))

    return G
