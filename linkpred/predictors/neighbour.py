import math

from ..evaluation import Scoresheet
from ..util import all_pairs
from .base import Predictor
from .util import neighbourhood, neighbourhood_size,\
    neighbourhood_intersection_size, neighbourhood_union_size

__all__ = ["AdamicAdar",
           "AssociationStrength",
           "CommonNeighbours",
           "Cosine",
           "DegreeProduct",
           "Jaccard",
           "MaxOverlap",
           "MinOverlap",
           "NMeasure",
           "Pearson",
           "ResourceAllocation"]


class AdamicAdar(Predictor):
    def predict(self, weight=None):
        """Predict by Adamic/Adar measure of neighbours

        Parameters
        ----------
        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        """
        res = Scoresheet()
        for a, b in self.likely_pairs():
            intersection = set(neighbourhood(self.G, a)) & \
                set(neighbourhood(self.G, b))
            w = 0
            for c in intersection:
                if weight is not None:
                    numerator = self.G[a][c][weight] * self.G[b][c][weight]
                else:
                    numerator = 1
                w += numerator / \
                    math.log(neighbourhood_size(self.G, c, weight))
            if w > 0:
                res[(a, b)] = w
        return res


class AssociationStrength(Predictor):
    def predict(self, weight=None):
        """Predict by association strength of neighbours

        Parameters
        ----------
        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        """
        res = Scoresheet()
        for a, b in self.likely_pairs():
            w = neighbourhood_intersection_size(self.G, a, b, weight) / \
                (neighbourhood_size(self.G, a, weight) *
                 neighbourhood_size(self.G, b, weight))
            if w > 0:
                res[(a, b)] = w
        return res


class CommonNeighbours(Predictor):
    def predict(self, alpha=1.0, weight=None):
        r"""Predict using common neighbours

        This is loosely based on Opsahl et al. (2010):

        .. math ::

            k(u, v) = |N(u) \cap N(v)|
            s(u, v) = \sum_{i=1}^n x_i \cdot y_i
            W(u, v) = k(u, v)^{1 - \alpha} \cdot s(u, v)^{\alpha}

        Parameters
        ----------
        alpha : float, optional
            If alpha = 0, weights are ignored. If alpha = 1, only weights are
            used (ignoring the number of intermediary nodes).

        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        """
        res = Scoresheet()
        for a, b in self.likely_pairs():
            if weight is None or alpha == 0.0:
                w = neighbourhood_intersection_size(self.G, a, b, weight=None)
            elif alpha == 1.0:
                w = neighbourhood_intersection_size(
                    self.G, a, b, weight=weight)
            else:
                k = neighbourhood_intersection_size(self.G, a, b, weight=None)
                s = neighbourhood_intersection_size(
                    self.G, a, b, weight=weight)
                w = (k ** (1.0 - alpha)) * (s ** alpha)
            if w > 0:
                res[(a, b)] = w
        return res


class Cosine(Predictor):
    def predict(self, weight=None):
        """Predict by cosine measure of neighbours

        Parameters
        ----------
        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        """
        res = Scoresheet()
        for a, b in self.likely_pairs():
            w = neighbourhood_intersection_size(self.G, a, b, weight) / \
                math.sqrt(neighbourhood_size(self.G, a, weight) *
                          neighbourhood_size(self.G, b, weight))
            if w > 0:
                res[(a, b)] = w
        return res


class DegreeProduct(Predictor):
    def predict(self, weight=None, minimum=1):
        """Predict by degree product (preferential attachment)

        Parameters
        ----------
        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        minimum : int, optional (default = 1)
            If the degree product is below this minimum, the corresponding
            prediction is ignored.

        """
        res = Scoresheet()
        for a, b in all_pairs(self.eligible_nodes()):
            w = neighbourhood_size(self.G, a, weight) *\
                neighbourhood_size(self.G, b, weight)
            if w >= minimum:
                res[(a, b)] = w
        return res


class Jaccard(Predictor):
    def predict(self, weight=None):
        """Predict by Jaccard index of neighbours

        Parameters
        ----------
        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        """
        res = Scoresheet()
        for a, b in self.likely_pairs():
            # Best performance: weighted numerator, unweighted denominator.
            numerator = neighbourhood_intersection_size(self.G, a, b, weight)
            denominator = neighbourhood_union_size(self.G, a, b, weight)
            w = numerator / denominator
            if w > 0:
                res[(a, b)] = w
        return res


class NMeasure(Predictor):
    def predict(self, weight=None):
        """Predict by N measure of neighbours

        The N measure was defined by Egghe (2009).

        Parameters
        ----------
        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        """
        res = Scoresheet()
        for a, b in self.likely_pairs():
            w = math.sqrt(2) *\
                neighbourhood_intersection_size(self.G, a, b, weight) / \
                math.sqrt(neighbourhood_size(self.G, a, weight) ** 2 +
                          neighbourhood_size(self.G, b, weight) ** 2)
            if w > 0:
                res[(a, b)] = w
        return res


def _predict_overlap(predictor, function, weight=None):
    res = Scoresheet()
    for a, b in predictor.likely_pairs():
        # Best performance: weighted numerator, unweighted denominator.
        numerator = neighbourhood_intersection_size(predictor.G, a, b, weight)
        denominator = function(neighbourhood_size(predictor.G, a, weight),
                               neighbourhood_size(predictor.G, b, weight))
        w = numerator / denominator
        if w > 0:
            res[(a, b)] = w
    return res


class MaxOverlap(Predictor):
    def predict(self, weight=None):
        """Predict by maximum overlap between neighbours

        Parameters
        ----------
        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        """
        return _predict_overlap(self, max, weight)


class MinOverlap(Predictor):
    def predict(self, weight=None):
        """Predict by minimum overlap between neighbours

        Parameters
        ----------
        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        """
        return _predict_overlap(self, min, weight)


class Pearson(Predictor):
    def predict(self, weight=None):
        """Predict by Pearson correlation between neighbours

        Parameters
        ----------
        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        """
        res = Scoresheet()
        # 'Full' Pearson looks at all possible pairs. Since those are likely
        # of little value for link prediction, we restrict ourselves to pairs
        # with at least one common neighbour.
        for a, b in self.likely_pairs():
            n = len(self.G)
            a_l2norm = neighbourhood_size(self.G, a, weight)
            b_l2norm = neighbourhood_size(self.G, b, weight)
            a_l1norm = neighbourhood_size(self.G, a, weight, pow=1)
            b_l1norm = neighbourhood_size(self.G, b, weight, pow=1)
            intersect = neighbourhood_intersection_size(self.G, a, b, weight)

            numerator = (n * intersect) - (a_l1norm * b_l1norm)
            denominator = math.sqrt(n * a_l2norm - a_l1norm ** 2) * \
                math.sqrt(n * b_l2norm - b_l1norm ** 2)

            w = numerator / denominator
            if w > 0:
                res[(a, b)] = w
        return res


class ResourceAllocation(Predictor):
    def predict(self, weight=None):
        """Predict with resource allocation index of neighbours

        Resource allocation was defined by Zhou, Lu & Zhang (2009, Eur. Phys.
        J. B, 71, 623).

        Parameters
        ----------
        weight : None or string, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        """
        res = Scoresheet()
        for a, b in self.likely_pairs():
            intersection = set(neighbourhood(self.G, a)) & \
                set(neighbourhood(self.G, b))
            w = 0
            for c in intersection:
                if weight is not None:
                    numerator = float(self.G[a][c][weight] *
                                      self.G[b][c][weight])
                else:
                    numerator = 1
                w += numerator / neighbourhood_size(self.G, c, weight)
            if w > 0:
                res[(a, b)] = w
        return res
