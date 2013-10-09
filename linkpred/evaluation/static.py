from ..util import log


class StaticEvaluation(object):
    """
    Static evaluation of IR
    """
    def __init__(self, retrieved=[], relevant=[], universe=None):
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
            If universe is an iterable, it is interpreted as the set of all items
            in the system.
            If universe is an int, it is interpreted as the *number* of items in
            the system. This allows for fewer checks but is more memory-efficient.
            If universe is None, it is supposed to be unknown. This still allows for
            some measures, including precision and recall, to be calculated.

        """
        retrieved = set(retrieved)
        relevant = set(relevant)

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

    def precision(self):
        try:
            return float(self.num_tp) / (self.num_tp + self.num_fp)
        except ZeroDivisionError:
            log.logger.warning("Division by 0 in calculating precision: "
                               "tp = %d, fp = %d, fn = %d, tn = %d" %
                               (self.num_tp, self.num_fp, self.num_tn, self.num_tn))
            return 0.0

    def recall(self):
        try:
            return float(self.num_tp) / (self.num_tp + self.num_fn)
        except ZeroDivisionError:
            log.logger.warning("Division by 0 in calculating recall: "
                               "tp = %d, fp = %d, fn = %d, tn = %d" %
                               (self.num_tp, self.num_fp, self.num_tn, self.num_tn))
            return 0.0

    def fallout(self):
        if self.num_tn == -1:
            raise ValueError(
                "Cannot determine fallout if universe is undefined")
        try:
            return float(self.num_fp) / (self.num_fp + self.num_tn)
        except ZeroDivisionError:
            log.logger.warning("Division by 0 in calculating fallout: "
                               "tp = %d, fp = %d, fn = %d, tn = %d" %
                               (self.num_tp, self.num_fp, self.num_tn, self.num_tn))
            return 0.0

    def miss(self):
        if self.num_tn == -1:
            raise ValueError("Cannot determine miss if universe is undefined")
        try:
            return float(self.num_fn) / (self.num_fn + self.num_tn)
        except ZeroDivisionError:
            log.logger.warning("Division by 0 in calculating miss: "
                               "tp = %d, fp = %d, fn = %d, tn = %d" %
                               (self.num_tp, self.num_fp, self.num_tn, self.num_tn))
            return 0.0

    def accuracy(self):
        """Compute accuracy = |correct| / |universe|

        Not appropriate for IR, since over 99.9% is nonrelevant. A system that
        labels everything as nonrelevant, would still have high accuracy.
        """
        if self.num_tn == -1:
            raise ValueError(
                "Cannot determine accuracy if universe is undefined")
        try:
            return float(self.num_tp + self.num_tn) / \
                (self.num_tp + self.num_fp + self.num_tn + self.num_fn)
        except ZeroDivisionError:
            log.logger.warning("Division by 0 in calculating accuracy: "
                               "tp = %d, fp = %d, fn = %d, tn = %d" %
                               (self.num_tp, self.num_fp, self.num_tn, self.num_tn))
            return 0.0

    def f_score(self, beta=1):
        """Compute F-measure or F-score.

        F is the weighted harmonic mean of recall R and precision P:
            F = 2PR / (P + R)
        In this case, R and P are evenly weighted. More generally:
            F = (1 + b^2)PR / (b^2 * P + R)
        If beta = 2, R is weighted twice as much as P.
        If beta = 0.5, R is weighted half as much as P.

        """
        p = self.precision()
        r = self.recall()
        beta_squared = beta ** 2
        try:
            return float((1 + beta_squared) * p * r) / (beta_squared * p + r)
        except ZeroDivisionError:
            return 0.0

    def generality(self):
        """Compute generality = |relevant| / |universe|"""
        if self.num_tn == -1:
            raise ValueError(
                "Cannot determine generality if universe is undefined")
        try:
            return float(self.num_tp + self.num_fn) / \
                (self.num_tp + self.num_fp + self.num_tn + self.num_fn)
        except ZeroDivisionError:
            log.logger.warning("Division by 0 in calculating generality: "
                               "tp = %d, fp = %d, fn = %d, tn = %d" %
                               (self.num_tp, self.num_fp, self.num_tn, self.num_tn))
            return 0.0
