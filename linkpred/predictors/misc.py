import random
from collections import defaultdict

from ..evaluation import Scoresheet
from ..util import all_pairs
from .base import Predictor

__all__ = ["Community", "Copy", "Random"]


class Community(Predictor):
    def predict(self):  # pylint:disable=E0202
        """Predict using community structure

        If two nodes belong to the same community, they are predicted to form
        a link. This uses the Louvain algorithm, which determines communities
        at different granularity levels: the finer grained the community, the
        higher the resulting score.

        This needs the python-louvain package. Install linkpred as follows:

        $ pip install linkpred[community]

        """
        try:
            import community
        except ImportError as err:
            msg = (
                "Module 'community' could not be found. Please install linkpred with: "
                "$ pip install linkpred[community]"
            )
            raise ImportError(msg) from err

        res = Scoresheet()
        dendogram = community.generate_dendrogram(self.G)

        for i in range(len(dendogram)):
            partition = community.partition_at_level(dendogram, i)
            communities = defaultdict(list)
            weight = len(dendogram) - i  # Lower i, smaller communities

            for n, com in partition.items():
                communities[com].append(n)
            for nodes in communities.values():
                for u, v in all_pairs(nodes):
                    if not self.eligible(u, v):
                        continue
                    res[(u, v)] += weight
        return res


class Copy(Predictor):
    def predict(self, weight=None):  # pylint:disable=E0202
        """Predict by copying the training network

        If weights are used, the likelihood score is equal to the link weight.

        This predictor is mostly intended as a sort of baseline. By definition,
        it only yields predictions if we do not exclude links from the training
        network (with `excluded`).

        Parameters
        ----------
        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        """
        if weight is None:
            return Scoresheet.fromkeys(self.G.edges(), 1)
        return Scoresheet(((u, v), d[weight]) for u, v, d in self.G.edges(data=True))


class Random(Predictor):
    def predict(self):  # pylint:disable=E0202
        """Predict randomly

        This predictor can be used as a baseline.

        """
        res = Scoresheet()
        for a, b in all_pairs(self.eligible_nodes()):
            res[(a, b)] = random.random()
        return res
