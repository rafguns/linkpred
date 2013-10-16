import copy
import matplotlib.pyplot as plt

from time import localtime, strftime

from ..util import interpolate

__all__ = ["Listener", "Plotter", "CachingListener", "FMaxListener",
           "RecallPrecisionPlotter", "FScorePlotter", "ROCPlotter",
           "PrecisionAtKListener", "MarkednessPlotter"]


class Listener(object):

    def on_new_evaluation(self, sender, **kwargs):
        pass

    def on_new_prediction(self, sender, **kwargs):
        pass

    def on_datagroup_finished(self, sender, **kwargs):
        pass

    def on_dataset_finished(self, sender, **kwargs):
        pass

    def on_run_finished(self, sender, **kwargs):
        pass


class CachingListener(Listener):

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


class PredictionCache(CachingListener):
    def on_new_evaluation(self, sender, **kwargs):
        pass

    def on_new_prediction(self, sender, **kwargs):
        predictions, dataset, predictor = kwargs['prediction'], \
            kwargs['dataset'], kwargs['predictor']

        if not self.cachefile:
            fname = "%s-%s-cache.txt" % (dataset.name, predictor)
            self.cachefile = open(fname, 'w')
            # Header row
            self.writeline('u', 'v', 'W(u, v)')
        # TODO XXX `predictions` is actually a set of predictions
        # Problems:
        # - we can only handle one prediction at a time
        # - we need a dict rather than a set, becaause we need the score as
        #   well
        self.writeline()


class FMaxListener(Listener):
    def __init__(self, name, beta=1):
        self.beta = beta
        self.reset_data()
        self.fname = "%s-Fmax" % name + \
            strftime("_%Y-%m-%d_%H.%M.txt", localtime())

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

        self.fname = "%s-precision-at-%d" % (name, self.k) + \
            strftime("_%Y-%m-%d_%H.%M.txt", localtime())

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


generic_chart_looks = ['k-', 'k--', 'k.-', 'k:',
                       'r-', 'r--', 'r.-', 'r:',
                       'b-', 'b--', 'b.-', 'b:',
                       'g-', 'g--', 'g.-', 'g:',
                       'c-', 'c--', 'c.-', 'c:',
                       'y-', 'y--', 'y.-', 'y:']


class Plotter(Listener):
    def __init__(self, name, xlabel="", ylabel="", filetype="pdf", chart_looks=[]):
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

    def add_line(self, dataset="", predictor="", default_look=generic_chart_looks):
        label = self.build_label(dataset, predictor)
        ax = self.fig.axes[0]
        ax.plot(self._x, self._y, self.chart_look(default_look), label=label)

    def build_label(self, dataset="", predictor=""):
        return predictor

    def chart_look(self, default):
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
        fname = "%s-%s" % (self.name, self._charttype) + \
            strftime("_%Y-%m-%d_%H.%M.", localtime()) + self.filetype
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
    def __init__(self, name, xlabel="#", ylabel="F-score", beta=1, steps=1, **kwargs):
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
