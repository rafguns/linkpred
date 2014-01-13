import numpy as np


class StaticEvaluation(object):

    """
    Static evaluation of IR
    """
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


class EvaluationSheet(object):

    def __init__(self, scoresheet, relevant=None, universe=None, n=1):
        static = StaticEvaluation(relevant=relevant, universe=universe)
        # Initialize empty array of right dimensions
        self.data = np.empty(self._data_size(len(scoresheet), n))
        for i, prediction in enumerate(scoresheet.successive_sets(n=n)):
            static.update_retrieved(prediction)
            self.data[i] = (static.num_tp, static.num_fp, static.num_fn,
                            static.num_tn)
        self.data = self.data.transpose()

    # XXX Static/class method?
    def _data_size(self, length, n):
        """Determine dimensions needed to store data"""
        # 4 columns for tp, fp, fn, tn
        return (int(np.ceil(float(length) / n)), 4)

    @property
    def tp(self):
        return self.data[0]

    @property
    def fp(self):
        return self.data[1]

    @property
    def fn(self):
        return self.data[2]

    @property
    def tn(self):
        return self.data[3]

    def tofile(self, fid, sep="\t", format="%s"):
        return self.data.tofile(fid, sep, format)

    def _universe_unknown(self):
        return np.where(self.tn == -1, True, False).any()

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def fallout(self):
        if self._universe_unknown():
            raise ValueError(
                "Cannot determine fallout if universe is undefined")

        return self.fp / (self.fp + self.tn)

    def miss(self):
        if self._universe_unknown():
            raise ValueError("Cannot determine miss if universe is undefined")

        return self.fn / (self.fn + self.tn)

    def accuracy(self):
        if self._universe_unknown():
            raise ValueError(
                "Cannot determine accuracy if universe is undefined")

        return (self.tp + self.tn) / self.data.sum(axis=1)

    def f_score(self, beta=1):
        r"""Compute F-score

        F is the harmonic mean of precision and recall

        .. math ::
            F = \frac{2PR}{P + R}

        We use the generalized form

        .. math ::
            F &= \frac{(\beta^2 + 1)PR}{\beta^2 P + R} \\
              &= \frac{(\beta^2 + 1)tp}{(\beta^2 + 1)tp + \beta^2fn + fp}

        """
        beta2 = beta ** 2
        beta2_tp = (beta2 + 1) * self.tp
        return beta2_tp / (beta2_tp + beta2 * self.fn + self.fp)

    def generality(self):
        if self._universe_unknown():
            raise ValueError(
                "Cannot determine generality if universe is undefined")

        return (self.tp + self.fn) / self.data.sum(axis=1)
