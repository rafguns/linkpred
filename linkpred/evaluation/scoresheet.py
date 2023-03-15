import logging
from collections import defaultdict

import networkx as nx
from networkx.readwrite.pajek import make_qstr

log = logging.getLogger(__name__)
__all__ = ["Pair", "BaseScoresheet", "Scoresheet"]


class BaseScoresheet(defaultdict):
    """Score sheet for evaluation of IR and similar

    This is a simple dict-like object, whose values are numeric
    (floats). It adds the methods `ranked_items` and `top`.

    Example
    -------
    >>> data = {('a', 'b'): 0.8, ('b', 'c'): 0.5, ('c', 'a'): 0.2}
    >>> sheet = Scoresheet(data)
    >>> for (x, y), score in sheet.ranked_items():
    ...     print("{}-{}: {}".format(x, y, score))
    b-a: 0.8
    c-b: 0.5
    c-a: 0.2

    """

    def __init__(self, data=None):
        defaultdict.__init__(self, float)
        if data:
            self.update(self.process_data(data))

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, float(val))

    def process_data(self, data):
        """Can be overridden by child classes"""
        return data

    def ranked_items(self, threshold=None):
        """Return items in decreasing order of their score

        Arguments
        ---------
        threshold : int
            Maximum number of items to return (in total)

        Returns
        -------
        (item, score) : tuple of item and score

        """
        threshold = threshold or len(self)
        log.debug("Called Scoresheet.ranked_items(): threshold=%d", threshold)

        # Sort first by score, then by key. This way, we always get the same
        # ranking, even in case of ties.
        # We use the tmp structure because it is much faster than
        # itemgetter(1, 0).
        tmp = ((score, key) for key, score in self.items())
        ranked_data = sorted(tmp, reverse=True)

        for score, key in ranked_data[:threshold]:
            yield key, score

    def top(self, n=10):
        return dict(self.ranked_items(threshold=n))

    @staticmethod
    def from_record(line, delimiter="\t"):
        line = line.rstrip("\n")
        return line.rstrip("\n").split(delimiter)

    @staticmethod
    def to_record(key, value, delimiter="\t"):
        key, value = map(make_qstr, (key, value))
        return f"{key}{delimiter}{value}\n"

    @classmethod
    def from_file(cls, fname, delimiter="\t", encoding="utf-8"):
        """Create new instance from CSV file *fname*"""
        d = cls()
        with open(fname, "rb") as fh:
            for line in fh:
                key, score = cls.from_record(line.decode(encoding), delimiter)
                d[key] = score
        return d

    def to_file(self, fname, delimiter="\t", encoding="utf-8"):
        """Save to CSV file *fname*"""
        with open(fname, "wb") as fh:
            for key, score in self.ranked_items():
                fh.write(self.to_record(key, score, delimiter).encode(encoding))


class Pair:
    """An unsorted pair of things.

    We could probably also use frozenset for this, but a Pair class opens
    possibilities for the future, such as extensions to 'directed' pairs
    (where the order is important) or to self-loops (where the two elements
    are the same).

    Example
    -------
    >>> t = ('a', 'b')
    >>> Pair(t) == Pair(*t) == Pair('b', 'a')
    True

    """

    def __init__(self, *args):
        if len(args) == 1:
            key = args[0]
            if isinstance(key, Pair):
                a, b = key.elements
            elif isinstance(key, tuple) and len(key) == 2:
                a, b = key
            else:
                raise TypeError("Key '%s' is not a Pair or tuple." % (key))
        elif len(args) == 2:
            a, b = args
        else:
            msg = "__init__() takes 1 or 2 arguments in addition to self"
            raise TypeError(msg)
        # For link prediction, a and b are two different nodes
        assert a != b, f"Predicted link ({a}, {b}) is a self-loop!"
        self.elements = self._sorted_tuple((a, b))

    @staticmethod
    def _sorted_tuple(t):
        a, b = t
        try:
            return (a, b) if a > b else (b, a)
        except TypeError:
            # Different node types. This does not hande all possible edge
            # cases but should be enough for most real-world scenarios.
            return (a, b) if str(a) > str(b) else (b, a)

    def __eq__(self, other):
        try:
            return self.elements == other.elements
        except AttributeError:
            return self.elements == self._sorted_tuple(other)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        try:
            return self.elements < other.elements
        except AttributeError:
            return self.elements < self._sorted_tuple(other)

    def __gt__(self, other):
        try:
            return self.elements > other.elements
        except AttributeError:
            return self.elements > self._sorted_tuple(other)

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

    def __getitem__(self, idx):
        return self.elements[idx]

    def __hash__(self):
        return hash(self.elements)

    def __str__(self):
        return "{} - {}".format(*self.elements)

    def __repr__(self):
        return "Pair%s" % repr(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)


class Scoresheet(BaseScoresheet):
    """Scoresheet for link prediction

    Scoresheet's keys are always Pairs.

    """

    def __getitem__(self, key):
        return BaseScoresheet.__getitem__(self, Pair(key))

    def __setitem__(self, key, val):
        BaseScoresheet.__setitem__(self, Pair(key), float(val))

    def __delitem__(self, key):
        return dict.__delitem__(self, Pair(key))

    def process_data(self, data, weight="weight"):
        if isinstance(data, dict):
            return {Pair(k): float(v) for k, v in data.items()}
        if isinstance(data, nx.Graph):
            return {Pair(u, v): float(d[weight]) for u, v, d in data.edges(data=True)}
        # We assume that data is some sort of iterable, like a list or tuple
        return {Pair(k): float(v) for k, v in data}

    @staticmethod
    def from_record(line, delimiter="\t"):
        u, v, score = line.rstrip("\n").split(delimiter)
        return (u, v), score

    @staticmethod
    def to_record(key, value, delimiter="\t"):
        u, v = key
        u, v, score = map(make_qstr, (u, v, value))
        return f"{u}{delimiter}{v}{delimiter}{score}\n"
