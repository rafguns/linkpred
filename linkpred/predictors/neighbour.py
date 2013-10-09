import math

from ..evaluation import Scoresheet
from ..util import all_pairs
from .base import Predictor
from .util import neighbourhood, neighbourhood_size,\
    neighbourhood_intersection_size, neighbourhood_union_size

__all__ = ["AdamicAdar",
           "AssociationStrength",
           "CommonNeighbours",
           "CommonKNeighbours",
           "Cosine",
           "DegreeProduct",
           "Euclidean",
           "HirschCore",
           "Jaccard",
           "K50",
           "Manhattan",
           "Minkowski",
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


class CommonKNeighbours(Predictor):
    def predict(self, beta=0.01, max_k=3, weight=None):
        r"""A generalized version of common neighbours, somewhat inspired by Katz

        $w(u, v) = \sum_{k=1}^\infty \beta^k |\self.Gamma_k(u) \cap \self.Gamma_k(v)|$

        """
        res = Scoresheet()
        #for a, b in all_pairs(self.G.nodes()):
        for a, b in self.likely_pairs():
            w = 0
            for k in range(1, max_k + 1):
                w += (beta ** k) *\
                    neighbourhood_intersection_size(self.G, a, b, weight, k)
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


class Minkowski(Predictor):
    r"""
    Predictor based on Minkowski distance

    The distance `d` is defined as:

    .. math::

       d = ( \sum |x_i - x_j|^r )^{1/r}

    and hence the likelihood score `w` is:

    .. math::

       w = \frac{1}{d}

    """
    def predict(self, r=1, weight='weight'):

        def size(G, u, v, weight=None):
            if weight is None and G.has_edge(u, v):
                return 1
            try:
                return G[u][v][weight]
            except KeyError:
                return 0

        res = Scoresheet()
        for a, b in self.likely_pairs():
            nbr_a = set(neighbourhood(self.G, a))
            nbr_b = set(neighbourhood(self.G, b))
            d = sum(abs(size(self.G, a, v, weight) - size(self.G, b, v, weight)) ** r
                    for v in nbr_a & nbr_b)
            d += sum(size(self.G, a, v, weight) ** r for v in nbr_a - nbr_b)
            d += sum(size(self.G, b, v, weight) ** r for v in nbr_b - nbr_a)
            d = d ** 1.0 / r
            if d > 0:
                # d is a distance measure, so we take the inverse
                res[(a, b)] = 1.0 / d
        return res


class Euclidean(Minkowski):
    def predict(self, weight='weight'):
        return Minkowski.predict(self, r=2, weight=weight)


class HirschCore(Predictor):
    """
    Predictor based on overlap between the h-cores of nodes

    The h-index of a node n is the largest number h, such that each node has at
    least h neighbours.
    The h-core of a node n is then defined as the set of neighbours of n with h
    or more neighbours.

    References
    ----------
    Schubert, A. (2010). A reference-based Hirschian similarity measure for
    journals. Scientometrics 84(1), 133-147.

    Schubert, A. & Soos, S. (2010). Mapping of science journals based on
    h-similarity. Scientometrics 83(2), 589-600.

    """
    def predict(self):

        def h_core_set(G, nodes):
            from hirsch import h_index

            degree_dict = {n: len(G[n]) for n in nodes}
            h_degree = h_index(degree_dict.values())

            return set(k for k, v in degree_dict.iteritems() if v >= h_degree)

        res = Scoresheet()
        for a, b in self.likely_pairs():
            a_neighbours = set(neighbourhood(self.G, a))
            b_neighbours = set(neighbourhood(self.G, b))
            if a_neighbours & b_neighbours:
                a_core = h_core_set(self.G, a_neighbours)
                b_core = h_core_set(self.G, b_neighbours)
                if a_core & b_core:
                    # Jaccard index of Hirsch cores or peripheries
                    res[(a, b)] = len(
                        a_core & b_core) / float(len(a_core | b_core))
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


class K50(Predictor):
    def predict(self, weight=None):
        """K50, proposed by Boyack & Klavans (2006)"""
        res = Scoresheet()
        nbr_all = sum(neighbourhood_size(self.G, x, weight)
                      for x in self.G.nodes_iter())
        for a, b in self.likely_pairs():
            intersection = neighbourhood_intersection_size(
                self.G, a, b, weight)
            nbr_a = neighbourhood_size(self.G, a, weight)
            nbr_b = neighbourhood_size(self.G, b, weight)
            den = nbr_a * nbr_b
            expected = min(
                den / float(nbr_all - nbr_a), den / float(nbr_all - nbr_b))
            w = (intersection - expected) / math.sqrt(den)
            if w > 0:
                res[(a, b)] = w
        return res


class Manhattan(Minkowski):
    def predict(self, weight='weight'):
        return Minkowski.predict(self, r=1, weight=weight)


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
