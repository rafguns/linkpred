from . import predictors
from .evaluation import Comparison
from .result import ResultDict  # XXX
from .util import log


def training_test_data(profile, minimum_degree=1, **kwargs):
    endpoint = profile["sparql_endpoint"]
    query = profile.get("query")
    training = profile["training"]
    test = profile["test"]

    results = ResultDict()
    for dataprofile in (training, test):
        name = dataprofile["name"]
        parameters = dataprofile["parameters"]
        if query is None:
            query = dataprofile["query"]

        log.logger.info("Collecting data (%s)..." % name)
        # XXX TODO XXX
        results[name] = "TODO"
        log.logger.info("Finished collecting data.")

    if minimum_degree:
        results.filter_all_low_degree_nodes(minimum_degree)
    return results[training['name']], results[test['name']]


def pretty_print(name, bipartite=False, tfidf=False, params={}):
    """Pretty print a predictor name"""
    retval = name
    if bipartite:
        retval += " bipartite"
    if tfidf:
        retval += " TF-IDF"
    if not params:
        return retval

    pretty_params = ", ".join("%s = %s" % (k, str(v))
                              for k, v in params.iteritems())
    return "%s (%s)" % (retval, pretty_params)


def to_tfidf(G):
    """TF-IDF transform the edges of G

    This is done by transforming its adjacency matrix and then converting back
    to a network.
    """
    import networkx as nx
    from linkpred.matrix import tfidf_matrix
    from linkpred.network import from_biadjacency_matrix

    assert nx.is_bipartite(G)
    row_items = [n for n in G.nodes_iter() if G.node[n]['eligible']]
    col_items = [n for n in G.nodes_iter() if not G.node[n]['eligible']]
    matrix = nx.bipartite.biadjacency_matrix(G, row_items, col_items)
    matrix = tfidf_matrix(matrix)
    G2 = from_biadjacency_matrix(matrix, row_items, col_items)
    G2.node = G.node
    return G2


def do_predict(G, predictortype, label, eligible=None, only_new=False, **kwargs):
    log.logger.info("Executing %s..." % label)
    predictor = predictortype(G, eligible=eligible, only_new=only_new)
    scoresheet = predictor(**kwargs)
    log.logger.info("Finished executing %s." % label)
    return scoresheet


def predict(training, profile, only_new=False, eligible=None):
    """Generator that yields predictions on the basis of training

    Arguments
    ---------
    training : a Result
        Training data

    profile : a dict
        Profile detailing which predictors should be used

    only_new : True|False
        Whether or not we should restrict ourselves to predicting only new links

    eligible : a string or None
        If a string, the attribute according to which 'eligible' nodes are found.
        If None, this is ignored.

    Returns
    -------
    (label, scoresheet) : a 2-tuple
        2-tuple consisting of a string (label of the prediction) and
        a Scoresheet (actual predictions)

    """
    for predictor_profile in profile['predictors']:
        bipartite = predictor_profile.get('bipartite', False)
        tfidf = predictor_profile.get('tfidf', False)
        parameters = predictor_profile.get('parameters', {})
        name = predictor_profile['name']
        predictortype = getattr(predictors, name)
        label = predictor_profile.get('displayname',
                                      pretty_print(name, bipartite, tfidf, parameters))

        if bipartite and tfidf:
            # Create a reusable TF-IDF network, so we don't have to do this
            # transformation for each predictor.
            if not hasattr(predict, 'tfidf_network'):
                predict.tfidf_network = to_tfidf(training.pathspec)
            G = predict.tfidf_network
        elif bipartite:
            G = training.pathspec
        else:
            G = training.network

        scoresheet = do_predict(
            G, predictortype, label, eligible, only_new, **parameters)

        yield label, scoresheet


def connect_signals(listeners):
    from linkpred.evaluation import signals
    for listener in listeners:
        signals.new_evaluation.connect(listener.on_new_evaluation)
        signals.datagroup_finished.connect(listener.on_datagroup_finished)
        signals.dataset_finished.connect(listener.on_dataset_finished)
        signals.run_finished.connect(listener.on_run_finished)


def evaluate(datasets, name, filetype="pdf", interpolate=True, steps=1):
    import linkpred.evaluation.listeners as l
    # TODO figure out easy way to specify which listeners we want
    cache = l.CachingListener()
    rp = l.RecallPrecisionPlotter(name, filetype=filetype,
                                  interpolate=interpolate)
    f = l.FScorePlotter(name, filetype=filetype, xlabel="# predictions",
                        steps=steps)
    roc = l.ROCPlotter(name, filetype=filetype)
    fmax = l.FMaxListener(name)
    connect_signals((cache, rp, f, roc, fmax))

    comp = Comparison()
    try:
        comp.register_datasets(datasets)
    except TypeError:  # Oops, not iterable!
        comp.register_dataset(datasets)
    comp.run()
