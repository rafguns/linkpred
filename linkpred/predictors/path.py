from __future__ import division
import networkx as nx

from ..evaluation import Scoresheet
from ..util import progressbar
from .base import Predictor

__all__ = ["GraphDistance", "Katz"]


class GraphDistance(Predictor):
    def predict(self, weight='weight', alpha=1):
        r"""Predict by graph distance

        This is based on the dissimilarity measures of Egghe & Rousseau (2003):

        $d(i, j) = \min(\sum 1/w_k)$

        The parameter alpha was introduced by Opsahl et al. (2010):

        $d_\alpha(i, j) = \min(\sum 1 / w_k^\alpha)$

        If alpha = 0 or weight is None, we determine unweighted graph distance,
        i.e. only keep track of number of intermediate nodes and not of edge
        weights. If alpha = 1, we only keep track of edge weights and not of
        the number of intermediate nodes. (In practice, setting alpha equal to
        around 0.1 seems to yield the best results.)

        Parameters
        ----------
        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        alpha : float
            Parameter to determine relative importance of intermediate
            link strength

        """
        res = Scoresheet()

        if weight is None:
            G = self.G
        else:
            # We assume that edge weights denote proximities
            G = nx.Graph()
            G.add_weighted_edges_from((u, v, 1 / d[weight] ** alpha) for
                                      u, v, d in self.G.edges(data=True))

        dist = nx.shortest_path_length(G, weight=weight)
        for a, others in dist:
            if not self.eligible_node(a):
                continue
            for b, length in others.items():
                if a == b or not self.eligible_node(b):
                    continue
                w = 1 / length
                res[(a, b)] = w
        return res


class Katz(Predictor):
    def predict(self, beta=0.001, max_power=5, weight='weight', dtype=None):
        """Predict by Katz (1953) measure

        Let `A` be an adjacency matrix for the directed network `G`.
        Then, each element `a_{ij}` of `A^k` (the `k`-th power of `A`) has a
        value equal to the number of walks with length `k` from `i` to `j`.

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

        dtype : a data type
            data type of edge weights (default numpy.int32)

        """
        if dtype is None:
            import numpy
            dtype = numpy.int32

        nodelist = list(self.G.nodes)
        adj = nx.to_scipy_sparse_matrix(
            self.G, dtype=dtype, weight=weight)
        res = Scoresheet()

        for k in progressbar(range(1, max_power + 1),
                             "Computing matrix powers: "):
            # The below method is found to be fastest for iterating through a
            # sparse matrix, see
            # http://stackoverflow.com/questions/4319014/
            matrix = (adj ** k).tocoo()
            for i, j, d in zip(matrix.row, matrix.col, matrix.data):
                if i == j:
                    continue
                u, v = nodelist[i], nodelist[j]
                if self.eligible(u, v):
                    w = d * (beta ** k)
                    res[(u, v)] += w

        # We count double in case of undirected networks ((i, j) and (j, i))
        if not self.G.is_directed():
            for pair in res:
                res[pair] /= 2

        return res
