from ..util import log
from .scoresheet import Pair
from .signals import new_prediction, new_evaluation, datagroup_finished,\
    dataset_finished, run_finished
from .static import StaticEvaluation

__all__ = ["DataSet", "Comparison"]


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


def prediction_evaluation(retrieved, relevant, n=None, **kwargs):
    """Generator for ranked evaluation of IR

    Arguments
    ---------
    retrieved : a Scoresheet
        score sheet of ranked retrieved results

    relevant : a set
        set of relevant results

    n : an integer
        At each step, the next n items on the retrieved score sheet are
        added to the set of retrieved items that are compared to the relevant
        ones.

    """
    evaluation = StaticEvaluation(relevant=relevant, **kwargs)
    for ret in retrieved.successive_sets(n=n):
        evaluation.update_retrieved(ret)
        yield ret, evaluation


class Comparison(object):

    def __init__(self):
        self.datasets = []

    def __iter__(self):
        return iter(self.datasets)

    def register_dataset(self, dataset):
        self.datasets.append(dataset)

    def run(self):
        # XXX Do we need to support multiple datasets?
        # If not we could remove Comparison and move this function to DataSet
        for d in self.datasets:
            # The following loop actually executes the predictors
            for predictorname, scoresheet in d.predictions:
                for prediction, evaluation in prediction_evaluation(
                        scoresheet, d.test, n=d.steps, universe=d.num_universe):
                    new_prediction.send(sender=self, prediction=prediction,
                                        dataset=d, predictor=predictorname)
                    new_evaluation.send(sender=self, evaluation=evaluation,
                                        dataset=d.name, predictor=predictorname)
                datagroup_finished.send(sender=self, dataset=d.name,
                                        predictor=predictorname)
            dataset_finished.send(sender=self, dataset=d.name)
        run_finished.send(sender=self)
