import networkx as nx
from . import predictors
from .evaluation import Comparison, signals, listeners
from .network import read_pajek
from .util import log


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


def predict(G, predictortype, eligible=None, only_new=False, **kwargs):
    predictor = predictortype(G, eligible=eligible, only_new=only_new)
    scoresheet = predictor(**kwargs)
    return scoresheet


def predict_all(G, profile, only_new=False, eligible=None):
    """Generator that yields predictions on the basis of training network G

    Arguments
    ---------
    G : a networkx.Graph
        Training network

    profile : a dict
        Profile detailing which predictors should be used

    only_new : True|False
        Whether or not we restrict ourselves to predicting only new links

    eligible : a string or None
        If a string, the attribute according to which 'eligible' nodes are
        found. If None, this is ignored.

    Returns
    -------
    (label, scoresheet) : a 2-tuple
        2-tuple consisting of a string (label of the prediction) and
        a Scoresheet (actual predictions)

    """
    for predictor_profile in profile['predictors']:
        params = predictor_profile.get('parameters', {})
        name = predictor_profile['name']
        predictortype = getattr(predictors, name)
        label = predictor_profile.get('displayname',
                                      pretty_print(name, params))

        log.logger.info("Executing %s..." % label)
        scoresheet = predict(G, predictortype, label, eligible,
                             only_new, **params)
        log.logger.info("Finished executing %s." % label)

        yield label, scoresheet


def connect_signals(listeners):
    for listener in listeners:
        signals.new_evaluation.connect(listener.on_new_evaluation)
        signals.datagroup_finished.connect(listener.on_datagroup_finished)
        signals.dataset_finished.connect(listener.on_dataset_finished)
        signals.run_finished.connect(listener.on_run_finished)


def evaluate(datasets, name, filetype="pdf", interpolate=True, steps=1):
    # TODO figure out easy way to specify which listeners we want
    cache = listeners.CachingListener()
    rp = listeners.RecallPrecisionPlotter(name, filetype=filetype,
                                          interpolate=interpolate)
    f = listeners.FScorePlotter(name, filetype=filetype,
                                xlabel="# predictions", steps=steps)
    roc = listeners.ROCPlotter(name, filetype=filetype)
    fmax = listeners.FMaxListener(name)
    connect_signals((cache, rp, f, roc, fmax))

    comp = Comparison()
    try:
        comp.register_datasets(datasets)
    except TypeError:  # Oops, not iterable!
        comp.register_dataset(datasets)
    comp.run()
