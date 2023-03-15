# Should be at start of file
import io

import matplotlib
import networkx as nx
import pytest
import smokesignal

import linkpred
from linkpred.evaluation.listeners import (
    CacheEvaluationListener,
    FMaxListener,
    FScorePlotter,
    RecallPrecisionPlotter,
    ROCPlotter,
)

from .utils import temp_file

matplotlib.use("Agg")


def test_imports():
    linkpred.LinkPred
    linkpred.read_network
    linkpred.network
    linkpred.exceptions
    linkpred.evaluation
    linkpred.predictors


def test_for_comparison():
    from linkpred.evaluation import Pair
    from linkpred.linkpred import for_comparison

    G = nx.path_graph(10)
    expected = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)}
    assert for_comparison(G) == expected

    to_delete = [Pair(2, 3), Pair(8, 9)]
    expected = {Pair(t) for t in expected}
    expected = expected.difference(to_delete)
    assert for_comparison(G, exclude=to_delete) == expected


def test_pretty_print():
    from linkpred.linkpred import pretty_print

    name = "foo"
    assert pretty_print(name) == "foo"
    params = {"bar": 0.1, "baz": 5}

    # 2 possibilities because of hash randomization
    assert pretty_print(name, params) in [
        "foo (baz = 5, bar = 0.1)",
        "foo (bar = 0.1, baz = 5)",
    ]


def test_read_unknown_network_type():
    with temp_file(suffix=".foo") as fname:
        with pytest.raises(linkpred.exceptions.LinkPredError):
            linkpred.read_network(fname)


def test_read_network():
    with temp_file(suffix=".net") as fname:
        with open(fname, "w") as fh:
            fh.write(
                """*vertices 2
1 "A"
2 "B"
*arcs 2
1 2
2 1"""
            )
        expected = nx.DiGraph()
        expected.add_edges_from([("A", "B"), ("B", "A")])

        G = linkpred.read_network(fname)
        assert set(G.edges()) == set(expected.edges())

        with open(fname) as fh:
            G = linkpred.read_network(fname)
            assert set(G.edges()) == set(expected.edges())


def test_read_pajek():
    from linkpred.linkpred import _read_pajek

    with temp_file(suffix=".net") as fname:
        with open(fname, "w") as fh:
            fh.write(
                """*vertices 2
1 "A"
2 "B"
*arcs 2
1 2
1 2"""
            )
        expected = nx.DiGraph()
        expected.add_edges_from([("A", "B")])

        G = _read_pajek(fname)
        assert isinstance(G, nx.DiGraph)
        assert sorted(G.edges()) == sorted(expected.edges())

        with open(fname, "w") as fh:
            fh.write(
                """*vertices 2
1 "A"
2 "B"
*edges 2
1 2
1 2"""
            )
        expected = nx.Graph()
        expected.add_edges_from([("A", "B")])

        G = _read_pajek(fname)
        assert isinstance(G, nx.Graph)
        assert sorted(G.edges()) == sorted(expected.edges())


def test_LinkPred_without_predictors():
    with pytest.raises(linkpred.exceptions.LinkPredError):
        linkpred.LinkPred()


class TestLinkpred:
    def teardown_method(self):
        smokesignal.clear_all()

    def config_file(self, training=False, test=False, **kwargs):
        config = {"predictors": ["Random"], "label": "testing"}

        # add training and test files, if needed
        for var, name, fname, data in (
            (
                training,
                "training",
                "foo.net",
                b"*Vertices 3\n1 A\n2 B\n3 C\n" b"*Edges 1\n1 2 1\n",
            ),
            (
                test,
                "test",
                "bar.net",
                b"*Vertices 3\n1 A\n2 B\n3 C\n" b"*Edges 1\n3 2 1\n",
            ),
        ):
            if var:
                fh = io.BytesIO()
                fh.name = fname
                fh.write(data)
                fh.seek(0)
                config[f"{name}-file"] = fh

        config.update(kwargs)
        return config

    def test_init(self):
        lp = linkpred.LinkPred(self.config_file())
        assert lp.config["label"] == "testing"
        assert lp.training is None

        lp = linkpred.LinkPred(self.config_file(training=True))
        assert isinstance(lp.training, nx.Graph)
        assert len(lp.training.nodes()) == 3
        assert len(lp.training.edges()) == 1
        assert lp.test is None

    def test_excluded(self):
        for value, expected in zip(
            ("", "old", "new"), (set(), {("A", "B")}, {("B", "C"), ("A", "C")})
        ):
            lp = linkpred.LinkPred(self.config_file(training=True, exclude=value))
            assert {tuple(sorted(p)) for p in lp.excluded} == expected
        with pytest.raises(linkpred.exceptions.LinkPredError):
            lp = linkpred.LinkPred(self.config_file(exclude="bla"))
            lp.excluded

    def test_preprocess_only_training(self):
        lp = linkpred.LinkPred(self.config_file(training=True))
        lp.preprocess()
        assert set(lp.training.nodes()) == set("AB")

    def test_preprocess_training_and_test(self):
        lp = linkpred.LinkPred(self.config_file(training=True, test=True))
        lp.preprocess()
        assert set(lp.training.nodes()) == {"B"}
        assert set(lp.test.nodes()) == {"B"}

    def test_setup_output_evaluating_without_test(self):
        lp = linkpred.LinkPred(self.config_file(training=True))
        with pytest.raises(linkpred.exceptions.LinkPredError):
            lp.setup_output()

    def test_setup_output(self):
        # Make sure this also works is $DISPLAY is not set.
        # Should probably mock this out...

        for name, klass in (
            ("recall-precision", RecallPrecisionPlotter),
            ("f-score", FScorePlotter),
            # Should be able to handle uppercase
            ("ROC", ROCPlotter),
            ("fmax", FMaxListener),
            ("cache-evaluations", CacheEvaluationListener),
        ):
            config = self.config_file(training=True, test=True, output=[name])
            lp = linkpred.LinkPred(config)
            lp.setup_output()
            assert isinstance(lp.listeners[0], klass)
            smokesignal.clear_all()
        # Has an evaluator been set up?
        assert len(lp.evaluator.params["relevant"]) == 1
        assert lp.evaluator.params["universe"] == 2
        assert isinstance(lp.evaluator.params["universe"], int)

    def test_predict_all(self):
        # Mock out linkpred.predictors
        class Stub:
            def __init__(self, training, eligible, excluded):
                self.training = training
                self.eligible = eligible
                self.excluded = excluded

            def predict(self, **params):
                self.params = params
                return "scoresheet"

        linkpred.predictors.A = Stub
        linkpred.predictors.B = Stub

        config = self.config_file(training=True)
        config["predictors"] = [
            {"name": "A", "parameters": {"X": "x"}, "displayname": "prettyA"},
            {"name": "B", "displayname": "prettyB"},
        ]
        lp = linkpred.LinkPred(config)
        results = list(lp.predict_all())
        assert results == [("A", "scoresheet"), ("B", "scoresheet")]

    def test_process_predictions(self):
        @smokesignal.on("prediction_finished")
        def a(scoresheet, dataset, predictor):
            assert scoresheet.startswith("scoresheet")
            assert predictor.startswith("pred")
            assert dataset == "testing"
            a.called = True

        @smokesignal.on("dataset_finished")
        def b(dataset):
            assert dataset == "testing"
            b.called = True

        @smokesignal.on("run_finished")
        def c():
            c.called = True

        a.called = b.called = c.called = False
        lp = linkpred.LinkPred(self.config_file())
        lp.predictions = [("pred1", "scoresheet1"), ("pred2", "scoresheet2")]
        lp.process_predictions()
        assert a.called
        assert b.called
        assert c.called
        smokesignal.clear_all()
