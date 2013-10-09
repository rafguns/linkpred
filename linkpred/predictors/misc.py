from ..evaluation import Scoresheet
from ..util import all_pairs
from .base import Predictor

__all__ = ["Community",
           "Copy",
           "Random"]


class Community(Predictor):
    def predict(self):
        from collections import defaultdict
        from linkpred.network import generate_dendogram, partition_at_level

        res = Scoresheet()
        dendogram = generate_dendogram(self.G)

        for i in range(len(dendogram)):
            partition = partition_at_level(dendogram, i)
            communities = defaultdict(list)
            weight = len(dendogram) - i  # Lower i, smaller communities

            for n, com in partition.iteritems():
                communities[com].append(n)
            for nodes in communities.itervalues():
                for u, v in all_pairs(nodes):
                    if not self.eligible(u, v):
                        continue
                    res[(u, v)] += weight
        return res


class Copy(Predictor):
    def predict(self, weight=None):
        if weight is None:
            return Scoresheet.fromkeys(self.G.edges_iter(), 1)
        return Scoresheet(((u, v), d[weight]) for u, v, d in self.G.edges(data=True))


class Random(Predictor):
    def predict(self):
        import random

        res = Scoresheet()
        for a, b in all_pairs(self.eligible_nodes()):
            res[(a, b)] = random.random()
        return res
