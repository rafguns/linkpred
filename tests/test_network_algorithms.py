import networkx as nx
import numpy as np
import pytest
from linkpred.network.algorithms import raw_google_matrix

from .utils import assert_array_equal


@pytest.fixture
def langville_meyer_graph():
    """Network on p. 32 of Langville & Meyer's book

    See https://books.google.be/books?id=KsHTl_2Pfl8C&lpg=PA38&ots=rNlWZp97RF&dq=raw%20google%20matrix%20langville&hl=nl&pg=PA32#v=onepage&q&f=false
    """
    G = nx.DiGraph()
    G.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (3, 1),
            (3, 2),
            (3, 5),
            (4, 5),
            (4, 6),
            (5, 4),
            (5, 6),
            (6, 4),
        ]
    )
    return G


def test_raw_google_matrix_langville_meyer_graph(langville_meyer_graph):
    expected = np.array(
        [
            [0, 1 / 2, 1 / 2, 0, 0, 0],
            [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
            [1 / 3, 1 / 3, 0, 0, 1 / 3, 0],
            [0, 0, 0, 0, 1 / 2, 1 / 2],
            [0, 0, 0, 1 / 2, 0, 1 / 2],
            [0, 0, 0, 1, 0, 0],
        ]
    )
    M = raw_google_matrix(langville_meyer_graph, list(range(1, 7)))

    assert (M - expected <= 1e-6).all()
