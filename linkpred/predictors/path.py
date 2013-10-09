import networkx as nx

from ..evaluation import Scoresheet
from .base import Predictor

__all__ = ["GraphDistance",
           "WeightedGraphDistance",
           "Katz"]


class GraphDistance(Predictor):
    def predict(self):
        res = Scoresheet()
        shortest_paths = nx.shortest_path_length(self.G)
        for a, reachables in shortest_paths.iteritems():
            if not self.eligible_node(a):
                continue
            for b, length in reachables.iteritems():
                if a == b or not self.eligible_node(b):
                    continue
                if length > 0:  # same node
                    w = 1.0 / length
                    res[(a, b)] = w
        return res


class WeightedGraphDistance(Predictor):
    def predict(self, weight='weight', alpha=1):
        r"""Predict by weighted graph distance

        This is based on the dissimilarity measures of Egghe & Rousseau (2003):

        $d(i, j) = \min(\sum 1/w_k)$

        The parameter alpha was introduced by Opsahl et al. (2010):

        $d_\alpha(i, j) = \min(\sum 1 / w_k^\alpha)$

        If alpha = 0, this reduces to unweighted graph distance, i.e. only keep
        track of number of intermediate nodes and not of edge weights. If alpha = 1,
        we only keep track of edge weights and not of the number of intermediate
        nodes. (In practice, setting alpha equal to around 0.1 seems to yield the
        best results.)

        """
        res = Scoresheet()
        inverted = nx.Graph()
        inverted.add_weighted_edges_from((u, v, 1.0 / d[weight] ** alpha)
                                         for u, v, d in self.G.edges_iter(data=True))
        dist = nx.shortest_path_length(inverted, weight=weight)
        for a, others in dist.iteritems():
            if not self.eligible_node(a):
                continue
            for b, length in others.iteritems():
                if a == b or not self.eligible_node(b):
                    continue
                if a != b:
                    w = 1.0 / length
                    res[(a, b)] = w
        return res


class Katz(Predictor):
    def predict(self, beta=0.001, max_power=5, weight='weight', all_walks=True,
                dtype=None):
        """Predict by Katz (1953) measure

        Let $A$ be an adjacency matrix for the directed network $self.G$.
        We assume that $self.G$ is unweighted, hence $A$ only contains values 1 and 0.
        Then, each element $a_{ij}^{(k)}$ of $A^k$ (the $k$-th power of $A$) has a
        value equal to the number of walks with length $k$ from $i$ to $j$.

        The probability of a link rapidly decreases as the walks grow longer.
        Katz therefore introduces an extra parameter (here beta) to weigh
        longer walks less.

        Parameters
        ----------
        beta : a float
            the value of beta in the formula of the Katz equation

        max_power : an int
            the maximum number of powers to take into account

        weight : string or None
            The edge attribute that holds the numerical value used for
            the edge weight.  If None then treat as unweighted.

        all_walks : True|False
            can walks contain the same node/link more than once?

        dtype : a data type
            data type of edge weights (default numpy.int32)

        """
        from linkpred.util import progressbar
        from itertools import izip

        if dtype is None:
            import numpy
            dtype = numpy.int32

        nodelist = self.G.nodes()
        adj = nx.to_scipy_sparse_matrix(
            self.G, dtype=dtype, weight=weight)
        res = Scoresheet()

        if not all_walks:
            from scipy.sparse import triu
            # Make triangular upper matrix
            adj = triu(adj)

        for k in progressbar(range(1, max_power + 1), "Computing matrix powers: "):
            # The below method is found to be fastest for iterating through a
            # sparse matrix, see
            # http://stackoverflow.com/questions/4319014/iterating-through-a-scipy-sparse-vector-or-matrix
            matrix = (adj ** k).tocoo()
            for i, j, d in izip(matrix.row, matrix.col, matrix.data):
                if i == j:
                    continue
                u, v = nodelist[i], nodelist[j]
                if self.eligible(u, v):
                    w = d * (beta ** k)
                    res[(u, v)] += w
        return res
