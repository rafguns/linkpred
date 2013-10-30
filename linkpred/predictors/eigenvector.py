from ..evaluation import Scoresheet
from ..network import neighbourhood_graph, rooted_pagerank, simrank
from ..util import progressbar
from .base import Predictor


class RootedPageRank(Predictor):
    def predict(self, nbunch=None, alpha=0.85, beta=0, weight='weight', k=None):
        """Predict using rooted PageRank.

        Parameters
        ----------

        G : a networkx.Graph

        nbunch : iterable collection of nodes
            node(s) to calculate PR for (default: all)

        alpha : float
            PageRank probability that we will advance to a neighbour of the
            current node in a random walk

        beta : float or int
            Normally, we return to the root node with probability 1 - alpha.
            With this parameter, we can also advance to a random other node in the
            network with probability beta. Thus, we get back to the root node with
            probability 1 - alpha - beta. This is off (0) by default.

        weight : string or None
            The edge attribute that holds the numerical value used for
            the edge weight.  If None then treat as unweighted.

        k : int or None
            If `k` is `None`, this predictor is applied to the entire network.
            If `k` is an int, the predictor is applied to a subgraph consisting
            of the k-neighbourhood of the current node.
            Results are often very similar but much faster.

        See documentation for linkpred.network.rooted_pagerank() for these
        parameters.

        """
        res = Scoresheet()
        if nbunch is None:
            nbunch = self.G.nodes()
        for u in progressbar(nbunch):
            if not self.eligible_node(u):
                continue
            if k is None:
                G = self.G
            else:
                # Restrict to the k-neighbourhood subgraph
                G = neighbourhood_graph(self.G, u, k)
            pagerank_scores = rooted_pagerank(G, u, alpha, beta, weight)
            for v, w in pagerank_scores.iteritems():
                if w > 0 and u != v and self.eligible_node(v):
                    res[(u, v)] += w
        return res


class SimRank(Predictor):
    def predict(self, c=0.8, num_iterations=10, weight='weight'):
        res = Scoresheet()
        nodelist = self.G.nodes()
        sim = simrank(self.G, nodelist, c, num_iterations, weight)
        (m, n) = sim.shape
        assert m == n

        for i in range(m):
            # sim(a, b) = sim(b, a), leading to a 'mirrored' matrix.
            # We start the column range at i + 1, such that we only look at the
            # upper triangle in the matrix, excluding the diagonal: sim(a, a) = 1.
            u = nodelist[i]
            for j in range(i + 1, n):
                if sim[i, j] > 0:
                    v = nodelist[j]
                    if self.eligible(u, v):
                        res[(u, v)] = sim[i, j]
        return res
