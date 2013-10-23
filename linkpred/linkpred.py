import networkx as nx
import os

from . import predictors
from .evaluation import Pair, signals, listeners
from .network import read_pajek
from .util import log

__all__ = ["LinkPredError", "LinkPred", "filter_low_degree_nodes",
           "read_network"]


class LinkPredError(Exception):
    pass


def filter_low_degree_nodes(networks, minimum=1, eligible=None):
    """Only retain nodes with minimum degree in all networks

    This also removes nodes that are not present in all networks.
    Changes are made in place.

    Arguments
    ---------
    networks : a list or iterable of networkx.Graph instances

    minimum : int
        minimum node degree

    """
    def low_degree(G, threshold):
        """Get eligible nodes whose degree is below the threshold"""
        if eligible is None:
            return [n for n, d in G.degree_iter() if d < threshold]
        else:
            return [n for n, d in G.degree_iter()
                    if d < threshold and G.node[n][eligible]]

    def items_outside(G, nbunch):
        """Get eligible nodes outside nbunch"""
        if eligible is None:
            return [n for n in G.nodes_iter() if n not in nbunch]
        else:
            return [n for n in G.nodes_iter()
                    if G.node[n][eligible] and n not in nbunch]

    log.logger.info("Filtering low degree nodes...")
    for G in networks:
        to_remove = low_degree(G, minimum)
        G.remove_nodes_from(to_remove)
        log.logger.info("Removed %d nodes "
                        "(degree < %d)" % (len(to_remove), minimum))
    common = set.intersection(*[set(G) for G in networks])
    for G in networks:
        to_remove = items_outside(G, common)
        G.remove_nodes_from(to_remove)
        log.logger.info("Removed %d nodes (not common)" % len(to_remove))
    log.logger.info("Finished filtering low degree nodes.")


def for_comparison(G, exclude=[]):
    """Return the result in a format, suitable for comparison.

    In practice this means we return it as a set of Pairs.

    """
    exclude = set(Pair(u, v) for u, v in exclude)
    return set(Pair(u, v) for u, v in G.edges_iter()) - exclude


def pretty_print(name, params={}):
    """Pretty print a predictor name"""
    if not params:
        return name

    pretty_params = ", ".join("%s = %s" % (k, str(v))
                              for k, v in params.iteritems())
    return "%s (%s)" % (name, pretty_params)


def read_network(fname):
    filetype_readers = {'net': read_pajek,
                        'gml': nx.read_gml,
                        'graphml': nx.read_graphml,
                        'gexf': nx.read_gexf,
                        'edgelist': nx.read_edgelist,
                        'adjlist': nx.read_adjlist}

    for ext, read in filetype_readers.iteritems():
        if fname.lower().endswith(ext):
            log.logger.info("Reading file '%s'..." % fname)
            network = read(fname)
            log.logger.info("Successfully read file.")
            return network


class LinkPred(object):

    def __init__(self, config={}):
        # default config
        self.config = {
            'chart_filetype': 'pdf',
            'eligible':       None,
            'interpolation':  False,
            'label':          '',
            'min_degree':     1,
            'only_new':       False,
            'output':         ['recall-precision'],
            'predictors':     [],
            'steps':          1,
            'test':           None,
            'training':       None
        }
        self.config.update(config)
        log.logger.debug(u"Config: %s" % unicode(config))

        if not self.config['predictors']:
            raise LinkPredError("No predictor specified. Aborting...")

        self.training = self.network('training')
        self.test = self.network('test')
        self.label = os.path.splitext(self.config['training'])[0]
        self.evaluator = None

    @property
    def excluded(self):
        return set(self.training.edges_iter()) \
            if self.config['only_new'] else set()

    def network(self, key):
        try:
            return read_network(self.config[key])
        except (KeyError, AttributeError):
            pass

    def preprocess(self):
        networks = [self.training]
        if self.test:
            networks.append(self.test)

        filter_low_degree_nodes(networks, minimum=self.config['min_degree'])

    def do_predict_all(self):
        """Generator that yields predictions on the basis of training network G

        Returns
        -------
        (label, scoresheet) : a 2-tuple
            2-tuple consisting of a string (label of the prediction) and
            a Scoresheet (actual predictions)

        """
        for predictor_profile in self.config['predictors']:
            params = predictor_profile.get('parameters', {})
            name = predictor_profile['name']
            predictor_class = getattr(predictors, name)
            label = predictor_profile.get('displayname',
                                          pretty_print(name, params))

            log.logger.info("Executing %s..." % label)
            predictor = predictor_class(self.training,
                                        eligible=self.config['eligible'],
                                        only_new=self.config['only_new'])
            scoresheet = predictor.predict(**params)
            log.logger.info("Finished executing %s." % label)

            # XXX TODO Do we need label?
            yield label, scoresheet

    def predict_all(self):
        self.predictions = self.do_predict_all()
        return self.predictions

    def process_predictions(self):
        filetype = self.config['chart_filetype']
        interpolation = self.config['interpolation']
        steps = self.config['steps']

        prediction_listeners = {
            'cache-predictions': listeners.CachePredictionListener()
        }
        evaluation_listeners = {
            'recall-precision': listeners.RecallPrecisionPlotter(
                self.label, filetype=filetype, interpolation=interpolation),
            'f-score': listeners.FScorePlotter(self.label, filetype=filetype,
                                               xlabel="# predictions",
                                               steps=steps),
            'roc': listeners.ROCPlotter(self.label, filetype=filetype),
            'fmax': listeners.FMaxListener(self.label),
            'cache-evaluations': listeners.CacheEvaluationListener()
        }

        for output in self.config['output']:
            name = output.lower()
            if name in evaluation_listeners:
                # We're evaluating!
                listener = evaluation_listeners[name]
                signals.new_evaluation.connect(listener.on_new_evaluation)

                if not self.test:
                    raise LinkPredError("Cannot evaluate (%s) without "
                                        "test network" % output)

                test_set = for_comparison(self.test, exclude=self.excluded)
                nnodes = len(self.test)
                # Universe = all possible edges, except for the ones that we no
                # longer consider (because they're already in the training network)
                num_universe = nnodes * (nnodes - 1) / 2 - len(self.excluded)

                if not self.evaluator:
                    self.evaluator = listeners.EvaluatingListener(
                        relevant=test_set, universe=num_universe)
                signals.new_prediction.connect(self.evaluator.on_new_prediction)
            else:
                # We assume that if it's not an evaluation listener, it must
                # be a prediction listener
                listener = prediction_listeners[name]
                signals.new_prediction.connect(listener.on_new_prediction)

            signals.datagroup_finished.connect(listener.on_datagroup_finished)
            signals.dataset_finished.connect(listener.on_dataset_finished)
            signals.run_finished.connect(listener.on_run_finished)

            log.logger.debug("Added listener for '%s'" % output)

        # The following loop actually executes the predictors
        for predictorname, scoresheet in self.predictions:

            log.logger.debug("Predictor '%s' yields %d predictions" % (
                predictorname, len(scoresheet)))

            for prediction in scoresheet.successive_sets(n=steps):
                signals.new_prediction.send(sender=self, prediction=prediction,
                                            dataset=self.label,
                                            predictor=predictorname)
            signals.datagroup_finished.send(sender=self, dataset=self.label,
                                            predictor=predictorname)

        signals.dataset_finished.send(sender=self, dataset=self.label)
        signals.run_finished.send(sender=self)
