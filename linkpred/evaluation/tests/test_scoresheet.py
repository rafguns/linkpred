import networkx as nx
from nose.tools import assert_dict_equal, assert_equal, assert_less, raises

from linkpred.evaluation.scoresheet import BaseScoresheet, Pair, Scoresheet


class TestBaseScoresheet:
    def setup(self):
        self.scoresheet = BaseScoresheet(
            zip("abcdefghijklmnopqrstuvwx", range(24)))

    def test_ranked_items(self):
        d = dict(self.scoresheet.ranked_items())
        assert_dict_equal(d, dict(self.scoresheet))

        s = self.scoresheet.ranked_items()
        assert_equal(next(s), ('x', 23))
        assert_equal(next(s), ('w', 22))
        assert_equal(next(s), ('v', 21))

    def test_sets_with_threshold(self):
        threshold = 12
        d = dict(self.scoresheet.ranked_items(threshold=threshold))
        assert_dict_equal(d, dict(zip("xwvutsrqponm",
                                      reversed(range(12, 24)))))

    def test_with_too_large_threshold(self):
        threshold = 25
        for s in self.scoresheet.ranked_items(threshold=threshold):
            assert_less(len(s), threshold)

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
    assert_equal(list(sheet.items()), [(Pair('a', 'b'), 5.0)])
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
