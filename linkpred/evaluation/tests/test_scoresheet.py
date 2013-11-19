import networkx as nx
from nose.tools import assert_dict_equal, assert_equal, assert_less, raises

from linkpred.evaluation.scoresheet import BaseScoresheet, Pair, Scoresheet


class TestBaseScoresheet:
    def setup(self):
        self.n = 3
        self.scoresheet = BaseScoresheet(
            zip("abcdefghijklmnopqrstuvwx", range(24)), n=self.n)

    def test_sets(self):
        for i, s in enumerate(self.scoresheet.sets(), start=1):
            assert_equal(len(s), i * self.n)
        for i, s in enumerate(self.scoresheet.successive_sets()):
            assert_equal(len(s), self.n)

    def test_sets_with_n(self):
        n = 8
        for i, s in enumerate(self.scoresheet.sets(n=n), start=1):
            assert_equal(len(s), i * n)
        for s in self.scoresheet.successive_sets(n=n):
            assert_equal(len(s), n)

    def test_sets_with_even_threshold(self):
        threshold = 12
        for i, s in enumerate(self.scoresheet.sets(threshold=threshold), start=1):
            assert_equal(len(s), i * self.n)
        for s in self.scoresheet.successive_sets(threshold=threshold):
            assert_equal(len(s), self.n)

    def test_with_too_large_threshold(self):
        threshold = 25
        for s in self.scoresheet.sets(threshold=threshold):
            assert_less(len(s), threshold)
        for s in self.scoresheet.successive_sets(threshold=threshold):
            assert_equal(len(s), self.n)

    def test_sets_with_uneven_threshold(self):
        """
        If the threshold does not nicely fit a 'boundary', only the last set
        should be affected.
        """
        threshold = 10

        result = list(enumerate(self.scoresheet.sets(threshold=threshold), start=1))
        for i, s in result:
            assert_equal(len(s), i * self.n)

        result = list(self.scoresheet.successive_sets(threshold=threshold))
        for s in result:
            assert_equal(len(s), self.n)

    def test_top(self):
        top = self.scoresheet.top()
        assert_dict_equal(top, dict(zip("opqrstuvwx", range(14, 24))))

        top = self.scoresheet.top(2)
        assert_dict_equal(top, dict(zip("wx", range(22, 24))))

        top = self.scoresheet.top(100)
        assert_equal(len(top), 24)


def test_pair():
    t = ('a', 'b')
    pair = Pair(t)
    assert_equal(pair, Pair(*t))
    assert_equal(pair, Pair('b', 'a'))
    assert_equal(pair, eval(repr(pair)))
    assert_equal(str(pair), "b - a")
    assert_equal(unicode(pair), u"b - a")

    # Non-ASCII characters should not cause exception if they're proper UTF-8
    pair = Pair("a", "\xc4\x87")
    str(pair)
    unicode(pair)


@raises(AssertionError)
def test_pair_identical_elements():
    Pair('a', 'a')


def test_scoresheet():
    sheet = Scoresheet()
    t = ('a', 'b')
    sheet[t] = 5
    assert_equal(len(sheet), 1)
    assert_equal(sheet.items(), [(Pair('a', 'b'), 5.0)])
    assert_equal(sheet[t], 5.0)
    del sheet[t]
    assert_equal(len(sheet), 0)


def test_scoresheet_process_data():
    t = ('a', 'b')
    d = {t: 5}
    G = nx.Graph()
    G.add_edge(*t, weight=5)
    s = [(t, 5)]

    for x in (d, G, s):
        sheet = Scoresheet(x)
        assert_equal(sheet[t], 5.0)
