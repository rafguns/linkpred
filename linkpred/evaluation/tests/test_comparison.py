import networkx as nx
from nose.tools import assert_equal

from linkpred.evaluation.comparison import DataSet


def test_dataset_init():
    name = "test"
    predictions = {("a", "b"): 1, ("b", "c"): 2}
    test_network = nx.Graph()
    test_network.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("c", "e")])
    test = test_network
    steps = 5

    d = DataSet(name, predictions, test, steps=steps)
    assert_equal(d.name, name)
    assert_equal(d.predictions, predictions)
    assert_equal(d.steps, steps)
    assert_equal(d.num_universe, 10)

    d = DataSet(name, predictions, test, exclude=set([("c", "d"), ("d", "e")]),
                steps=steps)
    assert_equal(d.num_universe, 8)
