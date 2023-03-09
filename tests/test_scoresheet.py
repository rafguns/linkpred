import networkx as nx
import pytest

from linkpred.evaluation.scoresheet import BaseScoresheet, Pair, Scoresheet
from .utils import temp_file


class TestBaseScoresheet:
    def setup_method(self):
        self.scoresheet = BaseScoresheet(zip("abcdefghijklmnopqrstuvwx", range(24)))

    def test_ranked_items(self):
        d = dict(self.scoresheet.ranked_items())
        assert d == dict(self.scoresheet)

        s = self.scoresheet.ranked_items()
        assert next(s) == ("x", 23)
        assert next(s) == ("w", 22)
        assert next(s) == ("v", 21)

    def test_sets_with_threshold(self):
        threshold = 12
        d = dict(self.scoresheet.ranked_items(threshold=threshold))
        assert d == dict(zip("xwvutsrqponm", reversed(range(12, 24))))

    def test_with_too_large_threshold(self):
        threshold = 25
        for s in self.scoresheet.ranked_items(threshold=threshold):
            assert len(s) < threshold

    def test_top(self):
        top = self.scoresheet.top()
        assert top == dict(zip("opqrstuvwx", range(14, 24)))

        top = self.scoresheet.top(2)
        assert top == dict(zip("wx", range(22, 24)))

        top = self.scoresheet.top(100)
        assert len(top) == 24

    def test_to_file_from_file(self):
        with temp_file() as fname:
            self.scoresheet.to_file(fname)

            newsheet = BaseScoresheet.from_file(fname)
            assert self.scoresheet == newsheet


class TestScoresheetFile:
    def setup_method(self):
        self.sheet = Scoresheet()
        self.sheet[(u"a", u"b")] = 2.0
        self.sheet[(u"b", u"\xe9")] = 1.0
        self.expected = b"b\ta\t2.0\n\xc3\xa9\tb\t1.0\n"

    def test_to_file(self):
        with temp_file() as fname:
            self.sheet.to_file(fname)

            with open(fname, "rb") as fh:
                assert fh.read() == self.expected

    def test_from_file(self):
        with temp_file() as fname:
            with open(fname, "wb") as fh:
                fh.write(self.expected)

            sheet = Scoresheet.from_file(fname)
            assert sheet == self.sheet


def test_pair():
    t = ("a", "b")
    pair = Pair(t)
    assert pair == Pair(*t)
    assert pair == t
    assert pair == Pair("b", "a")
    assert pair == eval(repr(pair))
    assert str(pair) == "b - a"

    # Test unicode (C4 87 -> latin small letter C with acute)
    pair = Pair("a", "\xc4\x87")
    assert str(pair) == "\xc4\x87 - a"


def test_pair_identical_elements():
    with pytest.raises(AssertionError):
        Pair("a", "a")


def test_pair_different_types():
    # Should not raise an error
    assert Pair("a", 1) == Pair(1, "a")


def test_scoresheet():
    sheet = Scoresheet()
    t = ("a", "b")
    sheet[t] = 5
    assert len(sheet) == 1
    assert list(sheet.items()) == [(Pair("a", "b"), 5.0)]
    assert sheet[t] == 5.0
    del sheet[t]
    assert len(sheet) == 0


def test_scoresheet_process_data():
    t = ("a", "b")
    d = {t: 5}
    G = nx.Graph()
    G.add_edge(*t, weight=5)
    s = [(t, 5)]

    for x in (d, G, s):
        sheet = Scoresheet(x)
        assert sheet[t] == 5.0
