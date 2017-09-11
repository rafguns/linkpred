from .util import neighbourhood

__all__ = ["Predictor",
           "all_predictors"]


class Predictor(object):
    """
    Predictor based on graph structure

    This can also be used for bipartite networks or other networks
    involving nodes that should not be included in the predictions.
    To distinguish between 'eligible' and 'non-eligible' nodes, the
    graph can set a node attribute that returns true for eligible
    nodes and false for non-eligible ones.

    For instance:

    >>> import networkx as nx
    >>> B = nx.Graph()
    >>> # Add the node attribute "bipartite"
    >>> B.add_nodes_from([1,2,3,4], bipartite=0)
    >>> B.add_nodes_from(['a','b','c'], bipartite=1)
    >>> B.add_edges_from([(1,'a'), (1,'b'), (2,'b'), (2,'c'),
    ...                   (3,'c'), (4,'a')])
    >>> p = Predictor(B, eligible='bipartite')
    >>> p.eligible_node(1)
    0
    >>> sorted(p.eligible_nodes())
    ['a', 'b', 'c']

    """

    def __init__(self, G, eligible=None, excluded=None):
        """
        Initialize predictor

        Arguments
        ---------
        G : nx.Graph
            a graph

        eligible : a string or None
            If this is a string, it is used to distinguish between eligible
            and non-eligible nodes. We only try to predict links between
            two eligible nodes.

        excluded : iterable or None
            A list or iterable of node pairs that should be excluded (i.e., not
            predicted). This is useful to, for instance, make sure that we only
            predict new links that are not currently in G.

        """
        self.G = G
        self.eligible_attr = eligible
        self.name = self.__class__.__name__
        self.excluded = [] if excluded is None else excluded

        # Add a decorator to predict(), to do the necessary postprocessing for
        # filtering out links if `excluded` is not empty. We do this in
        # __init__() such that child classes need not be changed.
        def add_postprocessing(func):
            def predict_and_postprocess(*args, **kwargs):
                scoresheet = func(*args, **kwargs)
                for u, v in self.excluded:
                    try:
                        del scoresheet[(u, v)]
                    except KeyError:
                        pass
                return scoresheet
            predict_and_postprocess.__name__ = func.__name__
            predict_and_postprocess.__doc__ = func.__doc__
            predict_and_postprocess.__dict__.update(func.__dict__)
            return predict_and_postprocess

        self.predict = add_postprocessing(self.predict)

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def eligible(self, u, v):
        """Check if link between nodes u and v is eligible

        Eligibility allows us to ignore some nodes/links for link prediction.

        """
        return self.eligible_node(u) and self.eligible_node(v) and u != v

    def eligible_node(self, v):
        """Check if node v is eligible

        Eligibility allows us to ignore some nodes/links for link prediction.

        """
        if self.eligible_attr is None:
            return True
        return self.G.node[v][self.eligible_attr]

    def eligible_nodes(self):
        """Get list of eligible nodes

        Eligibility allows us to ignore some nodes/links for link prediction.

        """
        return [v for v in self.G if self.eligible_node(v)]

    def likely_pairs(self, k=2):
        """
        Yield node pairs from the same neighbourhood

        Arguments
        ---------
        k : int
            size of the neighbourhood (e.g., if k = 2, the neighbourhood
            consists of all nodes that are two links away)

        """
        for a in self.G.nodes():
            if not self.eligible_node(a):
                continue
            for b in neighbourhood(self.G, a, k):
                if not self.eligible_node(b):
                    continue
                yield (a, b)


def all_predictors():
    """Returns a list of all predictors"""
    from ..util import itersubclasses
    from operator import itemgetter

    predictors = sorted(((s, s.__name__) for s in itersubclasses(Predictor)),
                        key=itemgetter(1))
    return list(zip(*predictors))[0]
