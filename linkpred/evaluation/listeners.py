from __future__ import print_function

import copy
import logging
import smokesignal

from time import localtime, strftime

from .static import EvaluationSheet
from ..util import interpolate

log = logging.getLogger(__name__)

__all__ = ["EvaluatingListener",
           "CachePredictionListener",
           "Listener",
           "Plotter",
           "CacheEvaluationListener",
           "FMaxListener",
           "RecallPrecisionPlotter",
           "FScorePlotter",
           "ROCPlotter",
           "PrecisionAtKListener",
           "MarkednessPlotter"]


def _timestamped_filename(basename, ext="txt"):
    return basename + strftime("_%Y-%m-%d_%H.%M.", localtime()) + ext


class Listener(object):

    def __init__(self):
        smokesignal.on('dataset_finished', self.on_dataset_finished)
        smokesignal.on('run_finished', self.on_run_finished)

    def on_dataset_finished(self, dataset):
        pass

    def on_run_finished(self):
        pass


class EvaluatingListener(Listener):

    def __init__(self, **kwargs):
        smokesignal.on('prediction_finished', self.on_prediction_finished)
        self.params = kwargs

        super(EvaluatingListener, self).__init__()

    def on_prediction_finished(self, scoresheet, dataset, predictor):
        evaluation = EvaluationSheet(scoresheet, **self.params)
        smokesignal.emit('evaluation_finished', evaluation=evaluation,
                         dataset=dataset, predictor=predictor)


class CachePredictionListener(Listener):

    def __init__(self):
        smokesignal.on('prediction_finished', self.on_prediction_finished)
        super(CachePredictionListener, self).__init__()
        self.encoding = 'utf-8'

    def on_prediction_finished(self, scoresheet, dataset, predictor):
        self.fname = _timestamped_filename("%s-%s-predictions" % (dataset,
                                                                  predictor))
        scoresheet.to_file(self.fname)


class CacheEvaluationListener(Listener):

    def __init__(self):
        smokesignal.on('evaluation_finished', self.on_evaluation_finished)
        super(CacheEvaluationListener, self).__init__()

    def on_evaluation_finished(self, evaluation, dataset, predictor):
        self.fname = _timestamped_filename("%s-%s-predictions" % (dataset,
                                                                  predictor))
        evaluation.to_file(self.fname)


class FMaxListener(Listener):

    def __init__(self, name, beta=1):
        self.beta = beta
        self.fname = _timestamped_filename("%s-Fmax" % name)

        smokesignal.on('evaluation_finished', self.on_evaluation_finished)
        super(FMaxListener, self).__init__()

    def on_evaluation_finished(self, evaluation, dataset, predictor):
        fmax = evaluation.f_score(self.beta).max()

        status = "%s\t%s\t%.4f\n" % (dataset, predictor, fmax)

        with open(self.fname, 'a') as f:
            f.write(status)
        print(status)


class PrecisionAtKListener(Listener):

    def __init__(self, name, k=10):
        self.k = k
        self.fname = _timestamped_filename(
            "%s-precision-at-%d" % (name, self.k))

        smokesignal.on('evaluation_finished', self.on_evaluation_finished)
        super(PrecisionAtKListener, self).__init__()

    def on_evaluation_finished(self, evaluation, dataset, predictor):
        precision = evaluation.precision()[self.k]

        status = "%s\t%s\t%.4f\n" % (dataset, predictor, precision)
        with open(self.fname, 'a') as f:
            f.write(status)
        print(status)


GENERIC_CHART_LOOKS = ['k-', 'k--', 'k.-', 'k:',
                       'r-', 'r--', 'r.-', 'r:',
                       'b-', 'b--', 'b.-', 'b:',
                       'g-', 'g--', 'g.-', 'g:',
                       'c-', 'c--', 'c.-', 'c:',
                       'y-', 'y--', 'y.-', 'y:']


class Plotter(Listener):

    def __init__(self, name, xlabel="", ylabel="", filetype="pdf",
                 chart_looks=None):
        import matplotlib.pyplot as plt

        self.name = name
        self.filetype = filetype
        self.chart_looks = chart_looks
        self._charttype = ""
        self._legend_props = {'prop': {'size': 'x-small'}}
        self.fig = plt.figure()
        ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.8],
                               xlabel=xlabel, ylabel=ylabel)
        ax.set_ylim((0, 1))
        self._x = []
        self._y = []

        smokesignal.on('evaluation_finished', self.on_evaluation_finished)
        super(Plotter, self).__init__()

    def add_line(self, predictor=""):
        ax = self.fig.axes[0]
        ax.plot(self._x, self._y, self.chart_look(), label=predictor)

        log.debug("Added line with %d points: "
                  "start = (%.2f, %.2f), end = (%.2f, %.2f)",
                  len(self._x), self._x[0], self._y[0],
                  self._x[-1], self._y[-1])

    def chart_look(self, default=None):
        if not self.chart_looks:
            if not default:
                default = GENERIC_CHART_LOOKS
            self.chart_looks = copy.copy(default)
        return self.chart_looks.pop(0)

    def on_evaluation_finished(self, evaluation, dataset, predictor):
        self.setup_coords(evaluation)
        self.add_line(predictor)

    def on_run_finished(self):
        # Fix looks
        for ax in self.fig.axes:
            ax.legend(**self._legend_props)

        # Save to file
        self.fname = _timestamped_filename("%s-%s" % (self.name,
                                                      self._charttype),
                                           self.filetype)
        self.fig.savefig(self.fname)


class RecallPrecisionPlotter(Plotter):

    def __init__(self, name, xlabel="Recall", ylabel="Precision",
                 interpolation=True, **kwargs):
        super(RecallPrecisionPlotter, self).__init__(name, xlabel, ylabel,
                                                     **kwargs)
        self._charttype = "recall-precision"
        self.interpolation = interpolation

    def add_line(self, predictor=""):
        if self.interpolation:
            self._y = interpolate(self._y)
        Plotter.add_line(self, predictor)

    def setup_coords(self, evaluation):
        self._x = evaluation.recall()
        self._y = evaluation.precision()


class FScorePlotter(Plotter):

    def __init__(self, name, xlabel="#", ylabel="F-score",
                 beta=1, **kwargs):
        super(FScorePlotter, self).__init__(name, xlabel, ylabel, **kwargs)
        self._charttype = "F-Score"
        self.beta = beta

    def setup_coords(self, evaluation):
        self._x = range(len(evaluation))
        self._y = evaluation.f_score(self.beta)


class ROCPlotter(Plotter):

    def __init__(self, name, xlabel="False pos. rate",
                 ylabel="True pos. rate", **kwargs):
        super(ROCPlotter, self).__init__(name, xlabel, ylabel, **kwargs)
        self._charttype = "ROC"

    def setup_coords(self, evaluation):
        self._x = evaluation.fallout()
        self._y = evaluation.recall()


class MarkednessPlotter(Plotter):

    def __init__(self, name, xlabel="Miss", ylabel="Precision", **kwargs):
        super(MarkednessPlotter, self).__init__(name, xlabel, ylabel, **kwargs)
        self._charttype = "Markedness"
        self._legend_props["loc"] = "upper left"

    def setup_coords(self, evaluation):
        self._x = evaluation.miss()
        self._y = evaluation.precision()
