import os
import re

import smokesignal
from linkpred.evaluation import BaseScoresheet, EvaluationSheet
from linkpred.evaluation.listeners import (
    CacheEvaluationListener,
    CachePredictionListener,
    EvaluatingListener,
    _timestamped_filename,
)

from .utils import assert_array_equal


def test_timestamped_filename():
    fname = _timestamped_filename("test")
    assert re.match(r"test_\d{4}-\d{2}-\d{2}_\d{2}.\d{2}.txt", fname)
    fname = _timestamped_filename("test", "foo")
    assert re.match(r"test_\d{4}-\d{2}-\d{2}_\d{2}.\d{2}.foo", fname)


def test_EvaluatingListener():
    @smokesignal.on("evaluation_finished")
    def t(evaluation, dataset, predictor):
        assert dataset == "dataset"
        assert isinstance(evaluation, EvaluationSheet)
        assert_array_equal(evaluation.tp, [1, 1, 2, 2])
        assert_array_equal(evaluation.fp, [0, 1, 1, 2])
        assert_array_equal(evaluation.fn, [1, 1, 0, 0])
        assert_array_equal(evaluation.tn, [2, 1, 1, 0])
        assert predictor == "predictor"
        t.called = True

    t.called = False
    relevant = {1, 2}
    universe = {1, 2, 3, 4}
    scoresheet = BaseScoresheet({1: 10, 3: 5, 2: 2, 4: 1})
    EvaluatingListener(relevant=relevant, universe=universe)
    smokesignal.emit(
        "prediction_finished",
        scoresheet=scoresheet,
        dataset="dataset",
        predictor="predictor",
    )
    assert t.called
    smokesignal.clear_all()


def test_CachePredictionListener():
    l = CachePredictionListener()
    scoresheet = BaseScoresheet({1: 10, 2: 5, 3: 2, 4: 1})
    smokesignal.emit("prediction_finished", scoresheet, "d", "p")

    with open(l.fname) as fh:
        # Line endings may be different across platforms
        assert fh.read().replace("\r\n", "\n") == "1\t10\n2\t5\n3\t2\n4\t1\n"
    smokesignal.clear_all()
    os.unlink(l.fname)


def test_CacheEvaluationListener():
    l = CacheEvaluationListener()
    scores = BaseScoresheet({1: 10, 2: 5})
    ev = EvaluationSheet(scores, {1})
    smokesignal.emit("evaluation_finished", ev, "d", "p")

    ev2 = EvaluationSheet.from_file(l.fname)
    assert_array_equal(ev.data, ev2.data)
    smokesignal.clear_all()
    os.unlink(l.fname)
