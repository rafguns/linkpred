from __future__ import print_function, unicode_literals

import networkx as nx

from collections import defaultdict
from ..util import log

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

    def process_data(self, data, *args, **kwargs):
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
        log.logger.debug("Called Scoresheet.ranked_items(): "
                         "threshold=%d" % threshold)

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


class Pair(object):
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
            raise TypeError(
                "__init__() takes 1 or 2 arguments in addition to self")
        # For link prediction, a and b are two different nodes
        assert a != b, "Predicted link (%s, %s) is a self-loop!" % (a, b)
        self.elements = (a, b) if a > b else (b, a)

    def __eq__(self, other):
        return self.elements == other.elements

    def __ne__(self, other):
        return self.elements != other.elements

    def __lt__(self, other):
        return self.elements < other.elements

    def __gt__(self, other):
        return self.elements > other.elements

    def __getitem__(self, idx):
        return self.elements[idx]

    def __hash__(self):
        return hash(self.elements)

    def __unicode__(self):
        a, b = map(str, self.elements)
        return u"%s - %s" % (a.decode('utf-8'), b.decode('utf-8'))

    def __str__(self):
        return unicode(self).encode('utf-8')

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

    def process_data(self, data, weight='weight'):
        if isinstance(data, dict):
            return {Pair(k): float(v) for k, v in data.items()}
        if isinstance(data, nx.Graph):
            return {Pair(u, v): float(d[weight]) for u, v, d
                    in data.edges(data=True)}
        # We assume that data is some sort of iterable, like a list or tuple
        return {Pair(k): float(v) for k, v in data}
