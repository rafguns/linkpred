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
        res = Scoresheet()
        for a, b in self.likely_pairs():
            intersection = set(neighbourhood(self.G, a)) & \
                set(neighbourhood(self.G, b))
            w = 0
            for c in intersection:
                if weight is not None:
                    numerator = self.G[a][c][weight] * self.G[b][c][weight]
                else:
                    numerator = 1.0
                w += numerator / \
                    math.log(neighbourhood_size(self.G, c, weight))
            if w > 0:
                res[(a, b)] = w
        return res


class AssociationStrength(Predictor):
    def predict(self, weight=None):
        res = Scoresheet()
        for a, b in self.likely_pairs():
            w = neighbourhood_intersection_size(self.G, a, b, weight) / \
                float(neighbourhood_size(self.G, a, weight) *
                      neighbourhood_size(self.G, b, weight))
            if w > 0:
                res[(a, b)] = w
        return res


class CommonNeighbours(Predictor):
    def predict(self, alpha=1.0, weight=None):
        r"""Predict using common neighbours

        This is loosely based on Opsahl et al. (2010):

            k(u, v) = |N(u) \cap N(v)|
            s(u, v) = \sum_{i=1}^n x_i \cdot y_i
            w(u, v) = k(u, v)^{1 - \alpha} \cdot s(u, v)^{\alpha}

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
        res = Scoresheet()
        for a, b in all_pairs(self.eligible_nodes()):
            w = neighbourhood_size(self.G, a, weight) *\
                neighbourhood_size(self.G, b, weight)
            if w >= minimum:
                res[(a, b)] = w
        return res


class Jaccard(Predictor):
    def predict(self, weight=None):
        """Predict by Jaccard index, based on neighbours of a and b

        Jaccard index J = |A \cap B| / |A \cup B|

        """
        res = Scoresheet()
        for a, b in self.likely_pairs():
            # Best performance: weighted numerator, unweighted denominator.
            numerator = neighbourhood_intersection_size(self.G, a, b, weight)
            denominator = neighbourhood_union_size(self.G, a, b, weight)
            w = numerator / float(denominator)
            if w > 0:
                res[(a, b)] = w
        return res


class NMeasure(Predictor):
    def predict(self, weight=None):
        r"""Predict by N measure (Egghe, 2009)

        $N(A, B) = \srqt{2} \frac{|A \cap B|}{\sqrt{|A|^2 + |B|^2}}$

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


class Overlap(Predictor):
    def predict(self, function, weight=None):
        res = Scoresheet()
        for a, b in self.likely_pairs():
            # Best performance: weighted numerator, unweighted denominator.
            numerator = neighbourhood_intersection_size(self.G, a, b, weight)
            denominator = function(neighbourhood_size(self.G, a, weight),
                                   neighbourhood_size(self.G, b, weight))
            w = numerator / float(denominator)
            if w > 0:
                res[(a, b)] = w
        return res


class MaxOverlap(Overlap):
    def predict(self, weight=None):
        return Overlap.predict(self, max, weight)


class MinOverlap(Overlap):
    def predict(self, weight=None):
        return Overlap.predict(self, min, weight)


class Pearson(Predictor):
    def predict(self, weight=None):
        res = Scoresheet()
        # 'Full' Pearson looks at all possible pairs. Since those are likely
        # of little value for link prediction, we restrict ourselves to pairs
        # with at least one common neighbour.
        for a, b in self.likely_pairs():
            n = len(self.G) - 1
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
        """Predict with Resource Allocation index

        See T. Zhou, L. Lu, YC. Zhang (2009). Eur. Phys. J. B, 71, 623

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
                    numerator = 1.0
                w += numerator / neighbourhood_size(self.G, c, weight)
            if w > 0:
                res[(a, b)] = w
        return res
