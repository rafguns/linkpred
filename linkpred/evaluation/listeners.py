import copy
import matplotlib.pyplot as plt

from time import localtime, strftime

from .signals import new_evaluation
from .static import StaticEvaluation
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
    def on_datagroup_finished(self, sender, **kwargs):
        pass

    def on_dataset_finished(self, sender, **kwargs):
        pass

    def on_run_finished(self, sender, **kwargs):
        pass


class EvaluatingListener(Listener):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.evaluation = None

    def on_new_prediction(self, sender, **kwargs):
        if not self.evaluation:
            self.evaluation = StaticEvaluation(**self.params)

        prediction, dataset, predictor = \
            kwargs['prediction'], kwargs['dataset'], kwargs['predictor']
        self.evaluation.update_retrieved(prediction)

        new_evaluation.send(sender=self, evaluation=self.evaluation,
                            dataset=dataset, predictor=predictor)

    def on_datagroup_finished(self, sender, **kwargs):
        self.evaluation = None


class CachePredictionListener(Listener):
    def __init__(self):
        self.cachefile = None

    def writeline(self, *args):
        line = "\t".join(map(str, args))
        self.cachefile.write("%s\n" % line)

    def on_new_prediction(self, sender, **kwargs):
        prediction, dataset, predictor = kwargs['prediction'], \
            kwargs['dataset'], kwargs['predictor']

        if not self.cachefile:
            fname = _timestamped_filename("%s-%s-cache" % (dataset, predictor))
            self.cachefile = open(fname, 'w')
            # Header row
            self.writeline('u', 'v', 'W(u,v)')
        # If we take steps larger than 1, this may write lines in 'wrong'
        # (non-descending) order.
        for (u, v), W in prediction.iteritems():
            self.writeline(u, v, W)

    def on_datagroup_finished(self, sender, **kwargs):
        if not self.cachefile:
            return
        log.logger.debug("Closing file '%s'" % self.cachefile.name)
        self.cachefile.close()
        self.cachefile = None


class CacheEvaluationListener(Listener):

    def __init__(self):
        self.cachefile = None

    def writeline(self, *args):
        line = "\t".join(map(str, args))
        self.cachefile.write("%s\n" % line)

    def on_new_evaluation(self, sender, **kwargs):
        evaluation, dataset, predictor = kwargs['evaluation'], \
            kwargs['dataset'], kwargs['predictor']
        tp, fp, fn, tn = evaluation.num_tp, evaluation.num_fp, \
            evaluation.num_fn, evaluation.num_tn

        if not self.cachefile:
            fname = "%s-%s-cache.txt" % (dataset, predictor)
            self.cachefile = open(fname, 'w')
            # Header row
            self.writeline('tp', 'fp', 'fn', 'tn')
        self.writeline(tp, fp, fn, tn)

    def on_datagroup_finished(self, sender, **kwargs):
        if not self.cachefile:
            return
        self.cachefile.close()
        self.cachefile = None


class FMaxListener(Listener):
    def __init__(self, name, beta=1):
        self.beta = beta
        self.reset_data()
        self.fname = _timestamped_filename("%s-Fmax" % name)

    def reset_data(self):
        self._f = []

    def on_new_evaluation(self, sender, **kwargs):
        evaluation = kwargs['evaluation']
        self._f.append(evaluation.f_score(self.beta))

    def on_datagroup_finished(self, sender, **kwargs):
        fmax = max(self._f) if self._f else 0
        self.reset_data()

        status = "%s\t%s\t%.4f\n" % (
            kwargs['dataset'], kwargs['predictor'], fmax)

        with open(self.fname, 'a') as f:
            f.write(status)
        print status


class PrecisionAtKListener(Listener):
    def __init__(self, name, k=10, steps=1):
        self.k = k
        self.steps = steps
        self.reset_data()

        self.fname = _timestamped_filename(
            "%s-precision-at-%d" % (name, self.k))

    def reset_data(self):
        self.precision = 0.0
        self.count = 0

    def on_new_evaluation(self, sender, **kwargs):
        self.count += 1
        if self.count / self.steps == self.k:
            self.precision = kwargs['evaluation'].precision()

    def on_datagroup_finished(self, sender, **kwargs):
        status = "%s\t%s\t%.4f\n" % (kwargs['dataset'],
                                     kwargs['predictor'],
                                     self.precision)

        with open(self.fname, 'a') as f:
            f.write(status)
        print status

        self.reset_data()


GENERIC_CHART_LOOKS = ['k-', 'k--', 'k.-', 'k:',
                       'r-', 'r--', 'r.-', 'r:',
                       'b-', 'b--', 'b.-', 'b:',
                       'g-', 'g--', 'g.-', 'g:',
                       'c-', 'c--', 'c.-', 'c:',
                       'y-', 'y--', 'y.-', 'y:']


class Plotter(Listener):
    def __init__(self, name, xlabel="", ylabel="", filetype="pdf",
                 chart_looks=[]):
        self.name = name
        self.filetype = filetype
        self.chart_looks = chart_looks
        self._charttype = ""
        self._legend_props = {'prop': {'size': 'x-small'}}
        self.fig = plt.figure()
        self.fig.add_axes([0.1, 0.1, 0.8, 0.8], xlabel=xlabel, ylabel=ylabel)
        self.reset_data()

    def reset_data(self):
        self._x = []
        self._y = []

    def add_line(self, dataset="", predictor=""):
        label = self.build_label(dataset, predictor)
        ax = self.fig.axes[0]
        ax.plot(self._x, self._y, self.chart_look(), label=label)

        log.logger.debug("Added line with %d points: "
                         "start = (%.2f, %.2f), end = (%.2f, %.2f)" %
                         (len(self._x), self._x[0], self._y[0],
                          self._x[-1], self._y[-1]))

    def build_label(self, dataset="", predictor=""):
        return predictor

    def chart_look(self, default=GENERIC_CHART_LOOKS):
        if not self.chart_looks:
            self.chart_looks = copy.copy(default)
        return self.chart_looks.pop(0)

    def on_datagroup_finished(self, sender, **kwargs):
        self.add_line(kwargs['dataset'], kwargs['predictor'])
        self.reset_data()

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
                 interpolate=True, **kwargs):
        Plotter.__init__(self, name, xlabel, ylabel, **kwargs)
        self._charttype = "recall-precision"
        self.interpolate = interpolate

    def reset_data(self):
        # Make sure that we always start in the top-left corner
        self._x = [0.]
        self._y = [1.]

    def add_line(self, dataset="", predictor=""):
        if self.interpolate:
            self._y = interpolate(self._y)
        Plotter.add_line(self, dataset, predictor)

    def on_new_evaluation(self, sender, **kwargs):
        evaluation = kwargs['evaluation']

        self._x.append(evaluation.recall())
        self._y.append(evaluation.precision())


class FScorePlotter(Plotter):
    def __init__(self, name, xlabel="#", ylabel="F-score",
                 beta=1, steps=1, **kwargs):
        Plotter.__init__(self, name, xlabel, ylabel, **kwargs)
        self._charttype = "F-Score"
        self.beta = beta
        self.steps = steps

    def on_new_evaluation(self, sender, **kwargs):
        evaluation = kwargs['evaluation']

        self._x.append(self.steps * len(self._x))
        self._y.append(evaluation.f_score(self.beta))


class ROCPlotter(Plotter):
    def __init__(self, name, xlabel="False pos. rate",
                 ylabel="True pos. rate", **kwargs):
        Plotter.__init__(self, name, xlabel, ylabel, **kwargs)
        self._charttype = "ROC"

    def on_new_evaluation(self, sender, **kwargs):
        evaluation = kwargs['evaluation']

        self._x.append(evaluation.fallout())
        self._y.append(evaluation.recall())


class MarkednessPlotter(Plotter):
    def __init__(self, name, xlabel="Miss", ylabel="Precision", **kwargs):
        Plotter.__init__(self, name, xlabel, ylabel, **kwargs)
        self._charttype = "Markedness"
        self._legend_props["loc"] = "upper left"

    def on_new_evaluation(self, sender, **kwargs):
        evaluation = kwargs['evaluation']

        self._x.append(evaluation.miss())
        self._y.append(evaluation.precision())
