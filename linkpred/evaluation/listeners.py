import copy

from time import localtime, strftime

from .signals import evaluation_finished
from .static import EvaluationSheet
from ..util import interpolate, log

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

    def on_dataset_finished(self, sender, **kwargs):
        pass

    def on_run_finished(self, sender, **kwargs):
        pass


class EvaluatingListener(Listener):

    def __init__(self, **kwargs):
        self.params = kwargs

    def on_prediction_finished(self, sender, **kwargs):
        scoresheet, dataset, predictor = \
            kwargs['scoresheet'], kwargs['dataset'], kwargs['predictor']
        evaluation = EvaluationSheet(scoresheet, **self.params)
        evaluation_finished.send(sender=self, evaluation=evaluation,
                                 dataset=dataset, predictor=predictor)


class CacheMixin(object):

    def line(self, *args):
        return "\t".join(map(str, args)) + "\n"


class CachePredictionListener(Listener, CacheMixin):

    def __init__(self, n):
        self.n = n

    def on_prediction_finished(self, sender, **kwargs):
        scoresheet, dataset, predictor = kwargs['scoresheet'], \
            kwargs['dataset'], kwargs['predictor']

        with open(_timestamped_filename("%s-%s-predictions" %
                                        (dataset, predictor))) as fh:
            for prediction in scoresheet.successive_sets(n=self.n):
                for (u, v), W in prediction.iteritems():
                    fh.writeline(self.line(u, v, W))


class CacheEvaluationListener(Listener, CacheMixin):

    def on_evaluation_finished(self, sender, **kwargs):
        evaluation, dataset, predictor = kwargs['evaluation'], \
            kwargs['dataset'], kwargs['predictor']

        evaluation.tofile(_timestamped_filename(
            "%s-%s-predictions" % (dataset, predictor)))


class FMaxListener(Listener):

    def __init__(self, name, beta=1):
        self.beta = beta
        self.fname = _timestamped_filename("%s-Fmax" % name)

    def on_evaluation_finished(self, sender, **kwargs):
        evaluation = kwargs['evaluation']
        fmax = evaluation.f_score(self.beta).max()

        status = "%s\t%s\t%.4f\n" % (
            kwargs['dataset'], kwargs['predictor'], fmax)

        with open(self.fname, 'a') as f:
            f.write(status)
        print status


class PrecisionAtKListener(Listener):

    def __init__(self, name, k=10, steps=1):
        self.k = k
        self.steps = steps
        self.fname = _timestamped_filename(
            "%s-precision-at-%d" % (name, self.k))

    def on_evaluation_finished(self, sender, **kwargs):
        evaluation, dataset, predictor = kwargs['evaluation'], \
            kwargs['dataset'], kwargs['predictor']

        precision = evaluation.precision()[self.k / self.steps]

        status = "%s\t%s\t%.4f\n" % (dataset, predictor, precision)
        with open(self.fname, 'a') as f:
            f.write(status)
        print status


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
        self.fig.add_axes([0.1, 0.1, 0.8, 0.8], xlabel=xlabel, ylabel=ylabel)
        self._x = []
        self._y = []

    def add_line(self, predictor=""):
        ax = self.fig.axes[0]
        ax.plot(self._x, self._y, self.chart_look(), label=predictor)

        log.logger.debug("Added line with %d points: "
                         "start = (%.2f, %.2f), end = (%.2f, %.2f)" %
                         (len(self._x), self._x[0], self._y[0],
                          self._x[-1], self._y[-1]))

    def chart_look(self, default=None):
        if not self.chart_looks:
            if not default:
                default = GENERIC_CHART_LOOKS
            self.chart_looks = copy.copy(default)
        return self.chart_looks.pop(0)

    def on_evaluation_finished(self, sender, **kwargs):
        self.setup_coords(kwargs['evaluation'])
        self.add_line(kwargs['predictor'])

    def on_run_finished(self, sender, **kwargs):
        # Fix looks
        for ax in self.fig.axes:
            ax.legend(**self._legend_props)

        # Save to file
        fname = _timestamped_filename("%s-%s" % (self.name, self._charttype),
                                      self.filetype)
        self.fig.savefig(fname)


class RecallPrecisionPlotter(Plotter):

    def __init__(self, name, xlabel="Recall", ylabel="Precision",
                 interpolation=True, **kwargs):
        Plotter.__init__(self, name, xlabel, ylabel, **kwargs)
        self._charttype = "recall-precision"
        self.interpolation = interpolation

    def add_line(self, predictor=""):
        if self.interpolation:
            self._y = interpolate(self._y)
        Plotter.add_line(self, predictor)

    def setup_coords(self, evaluation):
        # XXX Start at (0, 1)
        self._x = evaluation.recall()
        self._y = evaluation.precision()


class FScorePlotter(Plotter):

    def __init__(self, name, xlabel="#", ylabel="F-score",
                 beta=1, steps=1, **kwargs):
        Plotter.__init__(self, name, xlabel, ylabel, **kwargs)
        self._charttype = "F-Score"
        self.beta = beta
        self.steps = steps

    def setup_coords(self, evaluation):
        self._x = [self.steps * i for i in range(len(evaluation))]
        self._y = evaluation.f_score(self.beta)


class ROCPlotter(Plotter):

    def __init__(self, name, xlabel="False pos. rate",
                 ylabel="True pos. rate", **kwargs):
        Plotter.__init__(self, name, xlabel, ylabel, **kwargs)
        self._charttype = "ROC"

    def setup_coords(self, evaluation):
        self._x = evaluation.fallout()
        self._y = evaluation.recall()


class MarkednessPlotter(Plotter):

    def __init__(self, name, xlabel="Miss", ylabel="Precision", **kwargs):
        Plotter.__init__(self, name, xlabel, ylabel, **kwargs)
        self._charttype = "Markedness"
        self._legend_props["loc"] = "upper left"

    def setup_coords(self, evaluation):
        self._x = evaluation.miss()
        self._y = evaluation.precision()
