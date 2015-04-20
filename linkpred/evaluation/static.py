import logging
import numpy as np

from .scoresheet import BaseScoresheet

log = logging.getLogger(__name__)

__all__ = ["EvaluationSheet", "StaticEvaluation", "UndefinedError"]


class UndefinedError(Exception):
    """Raised when the method's result is undefined"""


class StaticEvaluation(object):
    """Static evaluation of IR"""

    def __init__(self, retrieved=None, relevant=None, universe=None):
        """
        Initialize IR evaluation.

        We determine the following table:

        +--------------+---------------+
        | tp           | fp            |
        | ret & rel    | ret & ~rel    |
        +--------------+---------------+
        | fn           | tn            |
        | ~ret & rel   | ~ret & ~rel   |
        +--------------+---------------+

        Arguments
        ---------
        retrieved : a list or set
            iterable of the retrieved items

        relevant : a list or set
            iterable of the relevant items

        universe : a list or set, an int or None
            If universe is an iterable, it is interpreted as the set of all
            items in the system. If universe is an int, it is interpreted as
            the *number* of items in the system. This allows for fewer checks
            but is more memory-efficient. If universe is None, it is supposed
            to be unknown. This still allows for some measures, including
            precision and recall, to be calculated.

        """
        retrieved = set(retrieved) if retrieved else set()
        relevant = set(relevant) if relevant else set()

        self.fp = retrieved - relevant
        self.fn = relevant - retrieved
        self.tp = retrieved & relevant
        if universe is None:
            self.tn = None
            self.num_universe = -1
        elif isinstance(universe, int):
            self.tn = None
            self.num_universe = universe
            if len(retrieved) > self.num_universe:
                raise ValueError("Retrieved cannot be larger than universe.")
            if len(relevant) > self.num_universe:
                raise ValueError("Retrieved cannot be larger than universe.")
        else:
            universe = set(universe)
            if not (retrieved <= universe and relevant <= universe):
                raise ValueError("Retrieved and relevant should be "
                                 "subsets of universe.")
            self.num_universe = len(universe)
            self.tn = universe - retrieved - relevant
            del universe
        self.update_counts()

    def update_counts(self):
        self.num_fp = len(self.fp)
        self.num_fn = len(self.fn)
        self.num_tp = len(self.tp)
        if self.tn is not None:
            self.num_tn = len(self.tn)
        elif self.num_universe == -1:
            self.num_tn = -1
        else:
            self.num_tn = self.num_universe - self.num_fp \
                                            - self.num_fn - self.num_tp
            assert self.num_tn >= 0

    def add_retrieved_item(self, item):
        self.update_retrieved({item})

    def update_retrieved(self, new):
        new = set(new)

        if not (new.isdisjoint(self.tp) and new.isdisjoint(self.fp)):
            raise ValueError("One or more elements in `new` have "
                             "already been retrieved.")

        relevant_new = new & self.fn
        nonrelevant_new = new - relevant_new

        self.tp |= relevant_new
        self.fp |= nonrelevant_new
        if self.tn:
            if not new <= self.fn | self.tn:
                raise ValueError("Newly retrieved items should be a subset "
                                 "of currently unretrieved items.")
            self.tn -= nonrelevant_new
        self.fn -= relevant_new
        self.update_counts()


def ensure_defined(func):
    def _wrapper(self, *args, **kwargs):
        if self.data.shape[0] == 0:
            raise UndefinedError("Measure is undefined if there are no "
                                 "relevant or retrieved items")
        return func(self, *args, **kwargs)
    return _wrapper


def ensure_universe_known(func):
    def _wrapper(self, *args, **kwargs):
        # If tn is -1 somewhere, we know that universe is not defined.
        if np.where(self.tn == -1, True, False).any():
            raise UndefinedError("Measure is undefined if universe is unknown")
        return func(self, *args, **kwargs)
    return _wrapper


class EvaluationSheet(object):

    def __init__(self, data=None, relevant=None, universe=None):
        if isinstance(data, BaseScoresheet):
            if relevant is None:
                raise TypeError("Cannot create evaluation sheet from "
                                "scoresheet without set of relevant items")
            log.debug("Counting for evaluation sheet...")
            static = StaticEvaluation(relevant=relevant, universe=universe)
            # Initialize empty array of right dimensions
            # 4 columns for tp, fp, fn, tn
            self.data = np.empty((len(data), 4))
            for i, (prediction, _) in enumerate(data.ranked_items()):
                static.add_retrieved_item(prediction)
                self.data[i] = (static.num_tp, static.num_fp, static.num_fn,
                                static.num_tn)
            log.debug("Finished counting evaluation sheet...")
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError("Cannot create evaluation sheet from "
                            "unknown data type {}".format(type(data)))

    def __len__(self):
        return len(self.data)

    @property
    def tp(self):
        return self.data[:, 0]

    @property
    def fp(self):
        return self.data[:, 1]

    @property
    def fn(self):
        return self.data[:, 2]

    @property
    def tn(self):
        return self.data[:, 3]

    def to_file(self, fname, *args, **kwargs):
        np.savetxt(fname, self.data, *args, **kwargs)

    @classmethod
    def from_file(cls, fname, *args, **kwargs):
        data = np.loadtxt(fname, *args, **kwargs)
        return cls(data)

    @ensure_defined
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @ensure_defined
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @ensure_defined
    @ensure_universe_known
    def fallout(self):
        return self.fp / (self.fp + self.tn)

    @ensure_defined
    @ensure_universe_known
    def miss(self):
        return self.fn / (self.fn + self.tn)

    @ensure_defined
    @ensure_universe_known
    def accuracy(self):
        return (self.tp + self.tn) / self.data.sum(axis=1)

    @ensure_defined
    def f_score(self, beta=1):
        r"""Compute F-score

        F is the harmonic mean of precision and recall:

            F = 2PR / (P + R)

        We use the generalized form:

            F = (beta^2 + 1)PR / (beta^2 P + R)
              = (beta^2 + 1)tp / ((beta^2 + 1)tp + beta^2fn + fp)

        The parameter beta allows assigning more weight to precision or recall.
        If beta > 1, recall is emphasized over precision. If beta < 1,
        precision is emphasized over recall.

        """
        beta2 = beta ** 2
        beta2_tp = (beta2 + 1) * self.tp
        return beta2_tp / (beta2_tp + beta2 * self.fn + self.fp)

    @ensure_defined
    @ensure_universe_known
    def generality(self):
        """Compute generality of the query

        Generality G is defined as:

            G = (tp + fn) / (tp + fn + fp + tp)

        Returns
        -------

        G : float

        """
        # Return single number: this is constant wrt what is retrieved
        return ((self.tp + self.fn) / self.data.sum(axis=1))[0]
