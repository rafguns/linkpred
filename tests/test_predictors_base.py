import networkx as nx

from linkpred.evaluation import Pair
from linkpred.predictors import CommonNeighbours, Copy, Predictor, all_predictors


def test_bipartite_common_neighbour():
    B = nx.Graph()
    B.add_nodes_from(range(1, 5), eligible=0)
    B.add_nodes_from("abc", eligible=1)
    B.add_edges_from(
        [(1, "a"), (1, "b"), (2, "a"), (2, "b"), (2, "c"), (3, "c"), (4, "a")]
    )

    expected = {Pair("a", "b"): 2, Pair("b", "c"): 1, Pair("a", "c"): 1}
    assert CommonNeighbours(B, eligible="eligible").predict() == expected


def test_bipartite_common_neighbours_equivalent_projection():
    B = nx.bipartite.random_graph(30, 50, 0.1)
    nodes = [v for v in B if B.nodes[v]["bipartite"]]
    G = nx.bipartite.weighted_projected_graph(B, nodes)

    expected = CommonNeighbours(B, eligible="bipartite")()
    assert Copy(G).predict(weight="weight") == expected


def test_postprocessing():
    G = nx.karate_club_graph()
    prediction_all_links = CommonNeighbours(G)()
    prediction_only_new_links = CommonNeighbours(G, excluded=G.edges())()

    for link, score in prediction_all_links.items():
        if G.has_edge(*link):
            assert link not in prediction_only_new_links
        else:
            assert score == prediction_only_new_links[link]


def test_all_predictors():
    predlist = all_predictors()
    assert len(predlist) > 0
    for p in predlist:
        assert p.__base__ == Predictor
