from ..util import log
from .scoresheet import Pair
from .signals import new_prediction, datagroup_finished, dataset_finished

__all__ = ["DataSet"]


def for_comparison(G, exclude=[]):
    """Return the result in a format, suitable for comparison.

    In practice this means we return it as a set of Pairs.

    """
    exclude = set(Pair(u, v) for u, v in exclude)
    return set(Pair(u, v) for u, v in G.edges_iter()) - exclude


class DataSet(object):
    def __init__(self, name, predictions, test, exclude=set(), steps=1):
        self.name = name
        self.predictions = predictions
        self.steps = steps
        if test:
            self.test = for_comparison(test, exclude=exclude)
            nnodes = len(test)
            # Universe = all possible edges, except for the ones that we no
            # longer consider (because they're already in the training network)
            self.num_universe = nnodes * (nnodes - 1) / 2 - len(exclude)
            log.logger.debug("Constructed dataset '%s': num_universe = %d" %
                             (self.name, self.num_universe))
        else:
            self.test = None
            log.logger.debug("Constructed dataset '%s': " % self.name)

    def run(self):
        # The following loop actually executes the predictors
        for predictorname, scoresheet in self.predictions:
            for prediction in scoresheet.successive_sets(n=self.steps):
                new_prediction.send(sender=self, prediction=prediction,
                                    dataset=self.name, predictor=predictorname)
            datagroup_finished.send(sender=self, dataset=self.name,
                                    predictor=predictorname)
        dataset_finished.send(sender=self, dataset=self.name)
