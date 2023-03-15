import networkx as nx


def neighbourhood(G, n, k=1):
    """Get k-neighbourhood of node n"""
    if k == 1:
        return G[n]
    dist = nx.single_source_shortest_path_length(G, n, k)
    del dist[n]
    return dist.keys()


def neighbourhood_intersection_size(G, a, b, weight=None, k=1):
    """Get the summed weight of the common neighbours of a and b

    If weighted, we use the sum of the weight products. This is equivalent
    to the vector-based interpretation (dot product of the two vectors).

    """
    common_neighbours = set(neighbourhood(G, a, k)) & set(neighbourhood(G, b, k))
    if weight:
        return sum(G[a][n][weight] * G[b][n][weight] for n in common_neighbours)

    return len(common_neighbours)


def neighbourhood_size(G, u, weight=None, k=1, power=2):
    """Get the weight of the neighbours of u

    If weighted, we use the sum of the squared edge weight for compatibility
    with the vector-based measures.

    """
    # The fast route for default options
    if weight is None and k == 1:
        return len(G[u])
    # The slow route for everything else
    neighbours = neighbourhood(G, u, k)
    return (
        sum(G[u][v][weight] ** power for v in neighbours) if weight else len(neighbours)
    )


def neighbourhood_union_size(G, a, b, weight=None, k=1, power=2):
    """Get the weight of the neighbours union of a and b"""
    a_neighbours = set(neighbourhood(G, a, k))
    b_neighbours = set(neighbourhood(G, b, k))
    if weight:
        return (
            sum(G[a][n][weight] ** power for n in a_neighbours)
            + sum(G[b][n][weight] ** power for n in b_neighbours)
            - sum(
                G[a][n][weight] * G[b][n][weight] for n in a_neighbours & b_neighbours
            )
        )

    return len(a_neighbours | b_neighbours)
