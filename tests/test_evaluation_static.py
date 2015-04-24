from __future__ import division

from nose.tools import *
from utils import assert_array_equal, temp_file

import numpy as np
from linkpred.evaluation import (StaticEvaluation, EvaluationSheet,
                                 Scoresheet, BaseScoresheet, UndefinedError)


class TestStaticEvaluation:

    def setup(self):
        self.ret = range(5)
        self.rel = [3, 4, 5, 6]
        self.num_universe = 20
        self.universe = range(self.num_universe)

    def test_init(self):
        e = StaticEvaluation(self.ret, self.rel, self.universe)
        assert_equal(len(e.tp), 2)
        assert_equal(len(e.fp), 3)
        assert_equal(len(e.tn), 13)
        assert_equal(len(e.fn), 2)

        e_no_universe = StaticEvaluation(self.ret, self.rel)
        assert_equal(len(e.tp), len(e_no_universe.tp))
        assert_equal(len(e.fp), len(e_no_universe.fp))
        assert_equal(len(e.fn), len(e_no_universe.fn))
        assert_equal(e_no_universe.tn, None)

        e_num_universe = StaticEvaluation(
            self.ret, self.rel, self.num_universe)
        assert_equal(len(e_num_universe.tp), 2)
        assert_equal(len(e_num_universe.fp), 3)
        assert_equal(len(e_num_universe.fn), 2)
        assert_equal(len(e_num_universe.tp), e_num_universe.num_tp)
        assert_equal(len(e_num_universe.fp), e_num_universe.num_fp)
        assert_equal(len(e_num_universe.fn), e_num_universe.num_fn)
        assert_equal(e_num_universe.num_tn, 13)

    def test_update_retrieved(self):
        e = StaticEvaluation(self.ret, self.rel, self.universe)
        e.update_retrieved([6, 7])
        assert_equal(len(e.tp), 3)
        assert_equal(len(e.fp), 4)
        assert_equal(len(e.tn), 12)
        assert_equal(len(e.fn), 1)

        assert_raises(ValueError, e.update_retrieved, [1])  # fp
        assert_raises(ValueError, e.update_retrieved, [3])  # tp
        assert_raises(ValueError, e.update_retrieved, ['a'])

    def test_update_retrieved_num_universe(self):
        e = StaticEvaluation(self.ret, self.rel, self.num_universe)
        e.update_retrieved([6, 7])
        assert_equal(len(e.tp), 3)
        assert_equal(len(e.fp), 4)
        assert_equal(len(e.fn), 1)
        assert_equal(e.num_tp, 3)
        assert_equal(e.num_fp, 4)
        assert_equal(e.num_tn, 12)
        assert_equal(e.num_fn, 1)

        assert_raises(ValueError, e.update_retrieved, [1])  # fp
        assert_raises(ValueError, e.update_retrieved, [3])  # tp

    def test_update_retrieved_full(self):
        e = StaticEvaluation(relevant=range(5), universe=20)
        e.update_retrieved(range(10))
        e.update_retrieved(range(10, 20))
        assert_equal(e.num_tp, 5)
        assert_equal(e.num_fp, 15)
        assert_equal(e.num_fn, 0)
        assert_equal(e.num_tn, 0)

    @raises(ValueError)
    def test_ret_no_universe_subset(self):
        StaticEvaluation([1, 2, 'a'], [2, 3], range(10))

    @raises(ValueError)
    def test_rel_no_universe_subset(self):
        StaticEvaluation([1, 2], [2, 3, 'a'], range(10))

    @raises(ValueError)
    def test_ret_larger_than_universe(self):
        StaticEvaluation(range(11), [2, 3], 10)

    @raises(ValueError)
    def test_rel_larger_than_universe(self):
        StaticEvaluation([1, 2], range(11), 10)


class TestEvaluationSheet:

    def setup(self):
        self.rel = [3, 4, 5, 6]
        self.scores = BaseScoresheet({7: 0.9, 4: 0.8, 6: 0.7, 1: 0.6, 3: 0.5,
                                      5: 0.2, 2: 0.1})
        self.num_universe = 20
        self.universe = range(self.num_universe)

    def test_init(self):
        sheet = EvaluationSheet(self.scores, relevant=self.rel)
        expected = np.array([[0, 1, 2, 2, 3, 4, 4],
                             [1, 1, 1, 2, 2, 2, 3],
                             [4, 3, 2, 2, 1, 0, 0],
                             [-1, -1, -1, -1, -1, -1, -1]]).T
        assert_array_equal(sheet.data, expected)

        sheet = EvaluationSheet(self.scores, relevant=self.rel,
                                universe=self.universe)
        expected = np.array([[0, 1, 2, 2, 3, 4, 4],
                             [1, 1, 1, 2, 2, 2, 3],
                             [4, 3, 2, 2, 1, 0, 0],
                             [15, 15, 15, 14, 14, 14, 13]]).T
        assert_array_equal(sheet.data, expected)

        sheet = EvaluationSheet(self.scores, relevant=self.rel,
                                universe=self.num_universe)
        # Same expected applies as above
        assert_array_equal(sheet.data, expected)

        data = np.array([[1, 0, 0, 1], [1, 1, 0, 0]])
        sheet = EvaluationSheet(data)
        assert_array_equal(sheet.data, data)

    def test_to_file_from_file(self):
        data = np.array([[1, 0, 0, 1], [1, 1, 0, 0]])
        sheet = EvaluationSheet(data)

        with temp_file() as fname:
            sheet.to_file(fname)
            newsheet = EvaluationSheet.from_file(fname)
            assert_array_equal(sheet.data, newsheet.data)

    def test_measures(self):
        sheet_num_universe = EvaluationSheet(self.scores, relevant=self.rel,
                                             universe=self.num_universe)
        sheet_universe = EvaluationSheet(self.scores, relevant=self.rel,
                                         universe=self.universe)
        sheet_no_universe = EvaluationSheet(self.scores, relevant=self.rel)

        # Measures that don't require universe

        for sheet in (sheet_num_universe, sheet_universe, sheet_no_universe):
            assert_array_equal(sheet.precision(),
                               np.array([0, 0.5, 2 / 3, 0.5, 3 / 5,
                                         2 / 3, 4 / 7]))
            assert_array_equal(sheet.recall(),
                               np.array([0, 0.25, 0.5, 0.5, 0.75, 1, 1]))

        # Measures that do require universe

        for sheet in (sheet_num_universe, sheet_universe):
            # XXX The following ones look wrong?!
            expected = np.array([1 / 16, 1 / 16, 1 / 16, 1 / 8, 1 / 8, 1 / 8,
                                 3 / 16])
            assert_array_equal(sheet.fallout(), expected)
            expected = np.array([4 / 19, 3 / 18, 2 / 17, 2 / 16, 1 / 15, 0, 0])
            assert_array_equal(sheet.miss(), expected)
            expected = np.array([0.75, 0.8, 17 / 20, 0.8, 17 / 20, 0.9,
                                 17 / 20])
            assert_array_equal(sheet.accuracy(), expected)
            assert_array_equal(sheet.generality(), 0.2)

        assert_raises(UndefinedError, sheet_no_universe.fallout)
        assert_raises(UndefinedError, sheet_no_universe.miss)
        assert_raises(UndefinedError, sheet_no_universe.accuracy)
        assert_raises(UndefinedError, sheet_no_universe.generality)

    def test_measures_with_empty_rel_and_ret(self):
        sheet1 = EvaluationSheet(Scoresheet(), [], [])
        sheet2 = EvaluationSheet(Scoresheet(), [], 10)
        sheet3 = EvaluationSheet(Scoresheet(), [])

        for sheet in (sheet1, sheet2, sheet3):
            assert_raises(UndefinedError, sheet.precision)
            assert_raises(UndefinedError, sheet.recall)
            assert_raises(UndefinedError, sheet.f_score)
            assert_raises(UndefinedError, sheet.fallout)
            assert_raises(UndefinedError, sheet.miss)
            assert_raises(UndefinedError, sheet.accuracy)
            assert_raises(UndefinedError, sheet.generality)

    def test_f_score(self):
        sheet = EvaluationSheet(self.scores, relevant=self.rel)
        expected = np.array([0, 2 / 6, 4 / 7, 4 / 8, 6 / 9, 8 / 10, 8 / 11])
        assert_array_equal(sheet.f_score(), expected)
        # $F_\beta = \frac{\beta^2 + 1 |rel \cap ret|}{\beta^2 |rel| + |ret|}$
        expected = np.array([
            0,
            1.25 * 1 / (0.25 * 4 + 2),
            1.25 * 2 / (0.25 * 4 + 3),
            1.25 * 2 / (0.25 * 4 + 4),
            1.25 * 3 / (0.25 * 4 + 5),
            1.25 * 4 / (0.25 * 4 + 6),
            1.25 * 4 / (0.25 * 4 + 7)
        ])
        assert_array_equal(sheet.f_score(0.5), expected)
        expected = np.array([
            0,
            5 * 1 / (4 * 4 + 2),
            5 * 2 / (4 * 4 + 3),
            5 * 2 / (4 * 4 + 4),
            5 * 3 / (4 * 4 + 5),
            5 * 4 / (4 * 4 + 6),
            5 * 4 / (4 * 4 + 7)
        ])
        assert_array_equal(sheet.f_score(2), expected)
