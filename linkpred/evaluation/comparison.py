from .ranked import ranked_evaluation
from .signals import new_evaluation, datagroup_finished,\
    dataset_finished, run_finished
from ..util import log

__all__ = ["DataSet", "Comparison"]


class DataSet(object):
    def __init__(self, name, predictions, test, exclude=set(), steps=1):
        self.name = name
        self.predictions = predictions
        self.test = test.for_comparison(exclude=exclude)
        self.steps = steps
        nnodes = len(test)
        # Universe = all possible edges, except for the ones that we no longer
        # consider (because they're already in the training network)
        self.num_universe = nnodes * (nnodes - 1) / 2 - len(exclude)
        log.logger.debug("Constructed dataset '%s': "
                         "num_universe = %d" % (self.name, self.num_universe))


class Comparison(object):

    def __init__(self):
        self.datasets = []

    def __iter__(self):
        return iter(self.datasets)

    def register_dataset(self, dataset):
        self.datasets.append(dataset)

    def register_datasets(self, datasets):
        for d in datasets:
            self.register_dataset(d)

    def run(self):
        for d in self.datasets:
            for predictorname, scoresheet in d.predictions:
                for evaluation in ranked_evaluation(scoresheet, d.test,
                                                    n=d.steps,
                                                    universe=d.num_universe):
                    new_evaluation.send(sender=self, evaluation=evaluation,
                                        dataset=d.name, predictor=predictorname)
                datagroup_finished.send(sender=self, dataset=d.name,
                                        predictor=predictorname)
            dataset_finished.send(sender=self, dataset=d.name)
        run_finished.send(sender=self)
