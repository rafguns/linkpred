import networkx as nx
from . import predictors
from .evaluation import Comparison, signals, listeners
from .network import read_pajek
from .util import log


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
        """
        Find low-degree nodes

        Parameters
        ----------
        threshold : int
            Only nodes whose degree is below the threshold are retained

        """
        if eligible is None:
            return [n for n, d in G.degree_iter() if d < threshold]
        else:
            return [n for n, d in G.degree_iter()
                    if d < threshold and G.node[n][eligible]]

    def items_outside(G, container):
        if eligible is None:
            return [n for n in G.nodes_iter() if n not in container]
        else:
            return [n for n in G.nodes_iter()
                    if G.node[n][eligible] and n not in container]

    log.logger.info("Filtering low degree nodes...")
    for G in networks:
        to_remove = low_degree(G, minimum)
        G.remove_nodes_from(to_remove)
        log.logger.info("Removed %d items" % len(to_remove))
    common = set.intersection(*[set(G) for G in networks])
    for G in networks:
        to_remove = items_outside(G, common)
        G.remove_nodes_from(to_remove)
        log.logger.info("Removed %d items" % len(to_remove))
    log.logger.info("Finished filtering low degree nodes.")


def pretty_print(name, params={}):
    """Pretty print a predictor name"""
    if not params:
        return name

    pretty_params = ", ".join("%s = %s" % (k, str(v))
                              for k, v in params.iteritems())
    return "%s (%s)" % (name, pretty_params)


def predict(G, predictor, eligible=None, only_new=False, **kwargs):
    scoresheet = predictor(G, eligible=eligible,
                           only_new=only_new).predict(**kwargs)
    return scoresheet


def connect_signals(listeners):
    for listener in listeners:
        signals.new_evaluation.connect(listener.on_new_evaluation)
        signals.datagroup_finished.connect(listener.on_datagroup_finished)
        signals.dataset_finished.connect(listener.on_dataset_finished)
        signals.run_finished.connect(listener.on_run_finished)


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
        self.config = {}  # default config
        self.config.update(config)

        self.training = self.network('training')
        self.test = self.network('test')
        if self.test:
            filter_low_degree_nodes([self.training, self.test],
                                    minimum=self.config['min_degree'])

        self.excluded = set(self.training.edges_iter()) \
            if self.config['only_new'] else set()

    def network(self, key):
        return read_network(self.config[key])

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
            predictor = getattr(predictors, name)
            label = predictor_profile.get('displayname',
                                          pretty_print(name, params))

            log.logger.info("Executing %s..." % label)
            scoresheet = predict(self.training, predictor,
                                 self.config['eligible'],
                                 self.config['only_new'], **params)
            log.logger.info("Finished executing %s." % label)

            # XXX TODO Do we need label?
            yield label, scoresheet

    def predict_all(self):
        self.predictions = self.do_predict_all()

    def process_predictions(self):
        # Evaluations etc.
        pass
