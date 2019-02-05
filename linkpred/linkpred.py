"""linkpred main module"""
import logging
import networkx as nx
import os
import smokesignal

from . import predictors
from .evaluation import Pair, listeners as l
from .exceptions import LinkPredError
from .preprocess import (without_low_degree_nodes, without_uncommon_nodes,
                         without_selfloops)

log = logging.getLogger(__name__)

__all__ = ["LinkPred", "read_network"]


def for_comparison(G, exclude=None):
    """Return the result in a format, suitable for comparison.

    In practice this means we return it as a set of Pairs.

    """
    if not exclude:
        return set(G.edges())

    exclude = set(Pair(u, v) for u, v in exclude)
    return set(Pair(u, v) for u, v in G.edges()) - exclude


def pretty_print(name, params=None):
    """Pretty print a predictor name

    Arguments
    ---------
    name : string
        predictor name

    params : dict or None
        dictionary of parameter name -> value

    """
    if not params:
        return name

    pretty_params = ", ".join("%s = %s" % (k, str(v))
                              for k, v in params.items())
    return "%s (%s)" % (name, pretty_params)


def _read_pajek(*args, **kwargs):
    """Read Pajek file and make sure that we get an nx.Graph or nx.DiGraph"""
    G = nx.read_pajek(*args, **kwargs)
    edges = G.edges()
    if len(set(edges)) < len(edges):  # multiple edges
        log.warning("Network contains multiple edges. These will be ignored.")
    if G.is_directed():
        return nx.DiGraph(G)
    else:
        return nx.Graph(G)


FILETYPE_READERS = {'.net': _read_pajek,
                    '.gml': nx.read_gml,
                    '.graphml': nx.read_graphml,
                    '.gexf': nx.read_gexf,
                    '.edgelist': nx.read_edgelist,
                    '.adjlist': nx.read_adjlist}


def read_network(fh):
    """Read the network file and return as nx.Graph or nx.DiGraph

    Arguments
    ---------
    fh : string
        file handle or file name

    """
    if nx.utils.is_string_like(fh):
        fname = fh
    else:
        # We assume that fh is a file handle
        fname = fh.name

    ext = os.path.splitext(fname.lower())[1]
    try:
        read = FILETYPE_READERS[ext]
        log.info("Reading file '%s'...", fname)
        network = read(fh)
        log.info("Successfully read file.")
    except KeyError:
        raise LinkPredError("File '%s' is of an unknown type. Known types "
                            "are: %s.", fname, ", ".join(FILETYPE_READERS))

    return network


class LinkPred(object):

    """linkpred main object

    LinkPred stores all configuration and provides a high-level interface to
    most functionality.

    """

    def __init__(self, config=None):
        # default config
        self.config = {
            'chart_filetype': 'pdf',
            'eligible':       None,
            'interpolation':  False,
            'label':          '',
            'min_degree':     1,
            'exclude':        'old',
            'output':         ['recall-precision'],
            'predictors':     [],
            'test-file':      None,
            'training-file':  None
        }
        if config:
            self.config.update(config)
        log.debug("Config: %s", self.config)

        if not self.config['predictors']:
            raise LinkPredError("No predictor specified. Aborting...")

        self.label = self.config['label'] or \
            os.path.splitext(self.config['training-file'])[0]
        self.training = self.network('training-file')
        self.test = self.network('test-file')
        self.evaluator = None
        self.listeners = []

    @property
    def excluded(self):
        """Get set of links that should not be predicted"""
        exclude = self.config['exclude']
        if not exclude:
            return set()  # No nodes are excluded
        elif exclude == 'old':
            return set(self.training.edges())
        elif exclude == 'new':
            return set(nx.non_edges(self.training))
        raise LinkPredError("Value '{}' for exclude is unexpected. Use either "
                            "'old', 'new' or empty string '' (for no "
                            "exclusions)".format(exclude))

    def network(self, key):
        """Get network for given key"""
        try:
            network_file = self.config[key]
        except KeyError:
            pass
        if network_file:
            return read_network(network_file)

    def preprocess(self):
        """Preprocess all networks according to configuration"""

        log.info("Starting preprocessing...")

        preprocessed = lambda G: without_low_degree_nodes(
            without_selfloops(G), minimum=self.config['min_degree'])

        if self.test:
            networks = [preprocessed(G) for G in (self.training, self.test)]
            self.training, self.test = without_uncommon_nodes(networks)
        else:  # Only a training network
            self.training = preprocessed(self.training)

        log.info("Finished preprocessing.")

    def setup_output(self):
        """Configure listeners"""
        filetype = self.config['chart_filetype']
        interpolation = self.config['interpolation']

        listeners = {
            'cache-predictions': (
                l.CachePredictionListener, False, {}),
            'recall-precision': (
                l.RecallPrecisionPlotter,
                True,
                dict(name=self.label, filetype=filetype,
                     interpolation=interpolation)),
            'f-score': (
                l.FScorePlotter,
                True,
                dict(name=self.label, filetype=filetype)),
            'roc': (
                l.ROCPlotter,
                True,
                dict(name=self.label, filetype=filetype)),
            'fmax': (
                l.FMaxListener, True, {'name': self.label}),
            'cache-evaluations': (
                l.CacheEvaluationListener, True, {})
        }

        for output in self.config['output']:
            name = output.lower()
            listener, evaluating, kwargs = listeners[name]

            if evaluating:
                if not self.test:
                    raise LinkPredError("Cannot evaluate (%s) without "
                                        "test network" % output)

                # Set up an 'evaluator': a listener that routes predictions
                # and turns them into evaluations
                if not self.evaluator:
                    test_set = for_comparison(self.test, exclude=self.excluded)
                    n = len(self.test)
                    # Universe = all possible edges, except for the ones that
                    # we no longer consider because they're excluded
                    # Make sure we get an int here.
                    num_universe = n * (n - 1) // 2 - len(self.excluded)
                    self.evaluator = l.EvaluatingListener(
                        relevant=test_set, universe=num_universe)

            self.listeners.append(listener(**kwargs))
            log.debug("Added listener for '%s'", output)

    def do_predict_all(self):
        """Generator that yields predictions based on training network

        Yields
        ------
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

            log.info("Executing %s...", label)
            predictor = predictor_class(self.training,
                                        eligible=self.config['eligible'],
                                        excluded=self.excluded)
            scoresheet = predictor.predict(**params)
            log.info("Finished executing %s.", label)

            # XXX TODO Do we need name?
            yield name, scoresheet

    def predict_all(self):
        """Perform all predictions according to configuration

        The predictions are only executed when `process_predictions` is called
        or when `LinkPred.predictions` is accessed in some other way.

        """
        self.predictions = self.do_predict_all()
        return self.predictions

    def process_predictions(self):
        """Process (evaluate, log...) all predictions according to config"""

        # The following loop actually executes the predictors
        for predictorname, scoresheet in self.predictions:
            log.debug("Predictor '%s' yields %d predictions",
                      predictorname, len(scoresheet))
            smokesignal.emit('prediction_finished',
                             scoresheet=scoresheet,
                             dataset=self.label,
                             predictor=predictorname)

        smokesignal.emit('dataset_finished', dataset=self.label)
        smokesignal.emit('run_finished')
        log.info("Prediction run finished")
