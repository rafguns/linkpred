from nose.tools import *

from linkpred.evaluation.static import StaticEvaluation

class TestStaticEvaluation:
    def setup(self):
        self.ret          = range(5)
        self.rel          = [3, 4, 5, 6]
        self.num_universe = 20
        self.universe     = range(self.num_universe)

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

        e_num_universe = StaticEvaluation(self.ret, self.rel, self.num_universe)
        assert_equal(len(e.tp), 2)
        assert_equal(len(e.fp), 3)
        assert_equal(len(e.fn), 2)
        assert_equal(len(e.tp), e.num_tp)
        assert_equal(len(e.fp), e.num_fp)
        assert_equal(len(e.fn), e.num_fn)
        assert_equal(e.num_tn, 13)

    def test_update_retrieved(self):
        e = StaticEvaluation(self.ret, self.rel, self.universe)
        e.update_retrieved([6, 7])
        assert_equal(len(e.tp), 3)
        assert_equal(len(e.fp), 4)
        assert_equal(len(e.tn), 12)
        assert_equal(len(e.fn), 1)

        assert_raises(ValueError, e.update_retrieved, [1]) # fp
        assert_raises(ValueError, e.update_retrieved, [3]) # tp
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

        assert_raises(ValueError, e.update_retrieved, [1]) # fp
        assert_raises(ValueError, e.update_retrieved, [3]) # tp

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
        e = StaticEvaluation([1, 2, 'a'], [2, 3], range(10))

    @raises(ValueError)
    def test_rel_no_universe_subset(self):
        e = StaticEvaluation([1, 2], [2, 3, 'a'], range(10))

    @raises(ValueError)
    def test_ret_larger_than_universe(self):
        e = StaticEvaluation(range(11), [2, 3], 10)

    @raises(ValueError)
    def test_rel_larger_than_universe(self):
        e = StaticEvaluation([1, 2], range(11), 10)

    def test_measures(self):
        e = StaticEvaluation(self.ret, self.rel, self.universe)
        assert_equal(e.precision(), float(2) / 5)
        assert_equal(e.recall(), float(2) / 4)
        assert_equal(e.fallout(), float(3) / 16)
        assert_equal(e.miss(), float(2) / 15)
        assert_equal(e.accuracy(), float(15) / 20)
        assert_equal(e.generality(), float(4) / 20)

        e = StaticEvaluation(self.ret, self.rel)
        assert_equal(e.precision(), float(2) / 5)
        assert_equal(e.recall(), float(2) / 4)
        assert_raises(ValueError, e.fallout)
        assert_raises(ValueError, e.miss)
        assert_raises(ValueError, e.accuracy)
        assert_raises(ValueError, e.generality)

        e = StaticEvaluation(self.ret, self.rel, self.num_universe)
        assert_equal(e.precision(), float(2) / 5)
        assert_equal(e.recall(), float(2) / 4)
        assert_equal(e.fallout(), float(3) / 16)
        assert_equal(e.miss(), float(2) / 15)
        assert_equal(e.accuracy(), float(15) / 20)
        assert_equal(e.generality(), float(4) / 20)

    def test_measures_with_zero_universe(self):
        e = StaticEvaluation([], [], [])
        assert_equal(e.precision(), 0.)
        assert_equal(e.recall(), 0.)
        assert_equal(e.f_score(), 0.)
        assert_equal(e.fallout(), 0.)
        assert_equal(e.miss(), 0.)
        assert_equal(e.accuracy(), 0.)
        assert_equal(e.generality(), 0.)

    def test_measures_with_zero_num_universe(self):
        e = StaticEvaluation([], [], 0)
        assert_equal(e.precision(), 0.)
        assert_equal(e.recall(), 0.)
        assert_equal(e.f_score(), 0.)
        assert_equal(e.fallout(), 0.)
        assert_equal(e.miss(), 0.)
        assert_equal(e.accuracy(), 0.)
        assert_equal(e.generality(), 0.)

    def test_measures_with_zero_no_universe(self):
        e = StaticEvaluation([], [])
        assert_equal(e.precision(), 0.)
        assert_equal(e.recall(), 0.)
        assert_equal(e.f_score(), 0.)
        assert_raises(ValueError, e.fallout)
        assert_raises(ValueError, e.miss)
        assert_raises(ValueError, e.accuracy)
        assert_raises(ValueError, e.generality)

    def test_f_score(self):
        e = StaticEvaluation(self.ret, self.rel)
        assert_almost_equal(e.f_score(), 4. / 9.)
        # $F_\beta = \frac{\beta^2 + 1 |rel \cap ret|}{\beta^2 |rel| + |ret|}$
        assert_almost_equal(e.f_score(0.5), 1.25 * 2. / 6.)
        assert_almost_equal(e.f_score(2), 10. / 21.)
