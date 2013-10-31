from __future__ import division
from math import log, sqrt
from nose.tools import *
import networkx as nx

from linkpred.evaluation import Pair, Scoresheet
from linkpred.predictors.neighbour import *


def assert_dict_almost_equal(d1, d2):
    for k in d1:
        assert_almost_equal(d1[k], d2[k])


class TestFlorentineFamily:

    def setup(self):
        self.G = nx.florentine_families_graph()
        nx.set_node_attributes(self.G, 'eligible', dict.fromkeys(self.G, True))

    def test_adamic_adar(self):
        answer = {
            Pair('Ridolfi', 'Barbadori'): 0.55811062655124721,
            Pair('Medici', 'Guadagni'): 1.8204784532536746,
            Pair('Peruzzi', 'Bischeri'): 0.72134752044448169,
            Pair('Lamberteschi', 'Bischeri'): 0.72134752044448169,
            Pair('Salviati', 'Albizzi'): 0.55811062655124721,
            Pair('Lamberteschi', 'Albizzi'): 0.72134752044448169,
            Pair('Peruzzi', 'Guadagni'): 0.91023922662683732,
            Pair('Strozzi', 'Medici'): 0.91023922662683732,
            Pair('Pazzi', 'Medici'): 1.4426950408889634,
            Pair('Ridolfi', 'Albizzi'): 0.55811062655124721,
            Pair('Tornabuoni', 'Lamberteschi'): 0.72134752044448169,
            Pair('Tornabuoni', 'Salviati'): 0.55811062655124721,
            Pair('Ridolfi', 'Acciaiuoli'): 0.55811062655124721,
            Pair('Strozzi', 'Guadagni'): 0.91023922662683732,
            Pair('Salviati', 'Acciaiuoli'): 0.55811062655124721,
            Pair('Guadagni', 'Ginori'): 0.91023922662683732,
            Pair('Strozzi', 'Barbadori'): 0.91023922662683732,
            Pair('Peruzzi', 'Barbadori'): 0.91023922662683732,
            Pair('Tornabuoni', 'Ridolfi'): 0.55811062655124721,
            Pair('Albizzi', 'Acciaiuoli'): 0.55811062655124721,
            Pair('Tornabuoni', 'Medici'): 0.91023922662683732,
            Pair('Ridolfi', 'Medici'): 0.91023922662683732,
            Pair('Peruzzi', 'Castellani'): 0.72134752044448169,
            Pair('Tornabuoni', 'Strozzi'): 0.91023922662683732,
            Pair('Tornabuoni', 'Bischeri'): 0.72134752044448169,
            Pair('Barbadori', 'Albizzi'): 0.55811062655124721,
            Pair('Castellani', 'Bischeri'): 1.631586747071319,
            Pair('Ridolfi', 'Guadagni'): 0.91023922662683732,
            Pair('Ridolfi', 'Bischeri'): 0.72134752044448169,
            Pair('Ridolfi', 'Peruzzi'): 0.72134752044448169,
            Pair('Medici', 'Castellani'): 1.4426950408889634,
            Pair('Bischeri', 'Albizzi'): 0.72134752044448169,
            Pair('Medici', 'Ginori'): 0.91023922662683732,
            Pair('Salviati', 'Ridolfi'): 0.55811062655124721,
            Pair('Tornabuoni', 'Barbadori'): 0.55811062655124721,
            Pair('Strozzi', 'Castellani'): 0.91023922662683732,
            Pair('Salviati', 'Barbadori'): 0.55811062655124721,
            Pair('Strozzi', 'Peruzzi'): 1.8204784532536746,
            Pair('Strozzi', 'Bischeri'): 0.91023922662683732,
            Pair('Tornabuoni', 'Albizzi'): 1.2794581469957289,
            Pair('Barbadori', 'Acciaiuoli'): 0.55811062655124721,
            Pair('Ridolfi', 'Castellani'): 0.72134752044448169,
            Pair('Tornabuoni', 'Acciaiuoli'): 0.55811062655124721}
        assert_dict_equal(AdamicAdar(self.G).predict(), answer)

    def test_adamic_adar_weighted(self):
        pass

    def test_association_strength(self):
        answer = {
            Pair('Ridolfi', 'Barbadori'): 0.16666666666666666,
            Pair('Medici', 'Guadagni'): 0.083333333333333329,
            Pair('Peruzzi', 'Bischeri'): 0.1111111111111111,
            Pair('Lamberteschi', 'Bischeri'): 0.33333333333333331,
            Pair('Salviati', 'Albizzi'): 0.16666666666666666,
            Pair('Lamberteschi', 'Albizzi'): 0.33333333333333331,
            Pair('Peruzzi', 'Guadagni'): 0.083333333333333329,
            Pair('Strozzi', 'Medici'): 0.041666666666666664,
            Pair('Pazzi', 'Medici'): 0.16666666666666666,
            Pair('Ridolfi', 'Albizzi'): 0.1111111111111111,
            Pair('Tornabuoni', 'Lamberteschi'): 0.33333333333333331,
            Pair('Tornabuoni', 'Salviati'): 0.16666666666666666,
            Pair('Ridolfi', 'Acciaiuoli'): 0.33333333333333331,
            Pair('Strozzi', 'Guadagni'): 0.0625,
            Pair('Salviati', 'Acciaiuoli'): 0.5,
            Pair('Guadagni', 'Ginori'): 0.25,
            Pair('Strozzi', 'Barbadori'): 0.125,
            Pair('Peruzzi', 'Barbadori'): 0.16666666666666666,
            Pair('Tornabuoni', 'Ridolfi'): 0.1111111111111111,
            Pair('Albizzi', 'Acciaiuoli'): 0.33333333333333331,
            Pair('Tornabuoni', 'Medici'): 0.055555555555555552,
            Pair('Ridolfi', 'Medici'): 0.055555555555555552,
            Pair('Peruzzi', 'Castellani'): 0.1111111111111111,
            Pair('Tornabuoni', 'Strozzi'): 0.083333333333333329,
            Pair('Tornabuoni', 'Bischeri'): 0.1111111111111111,
            Pair('Barbadori', 'Albizzi'): 0.16666666666666666,
            Pair('Castellani', 'Bischeri'): 0.22222222222222221,
            Pair('Ridolfi', 'Guadagni'): 0.083333333333333329,
            Pair('Ridolfi', 'Bischeri'): 0.1111111111111111,
            Pair('Ridolfi', 'Peruzzi'): 0.1111111111111111,
            Pair('Medici', 'Castellani'): 0.055555555555555552,
            Pair('Bischeri', 'Albizzi'): 0.1111111111111111,
            Pair('Medici', 'Ginori'): 0.16666666666666666,
            Pair('Salviati', 'Ridolfi'): 0.16666666666666666,
            Pair('Tornabuoni', 'Barbadori'): 0.16666666666666666,
            Pair('Strozzi', 'Castellani'): 0.083333333333333329,
            Pair('Salviati', 'Barbadori'): 0.25,
            Pair('Strozzi', 'Peruzzi'): 0.16666666666666666,
            Pair('Strozzi', 'Bischeri'): 0.083333333333333329,
            Pair('Tornabuoni', 'Albizzi'): 0.22222222222222221,
            Pair('Barbadori', 'Acciaiuoli'): 0.5,
            Pair('Ridolfi', 'Castellani'): 0.1111111111111111,
            Pair('Tornabuoni', 'Acciaiuoli'): 0.33333333333333331}
        assert_dict_equal(AssociationStrength(self.G).predict(), answer)

    def test_association_strength_weighted(self):
        pass

    def test_common_neighbours(self):
        answer = {
            Pair('Ridolfi', 'Barbadori'): 1.0,
            Pair('Medici', 'Guadagni'): 2.0,
            Pair('Peruzzi', 'Bischeri'): 1.0,
            Pair('Lamberteschi', 'Bischeri'): 1.0,
            Pair('Salviati', 'Albizzi'): 1.0,
            Pair('Lamberteschi', 'Albizzi'): 1.0,
            Pair('Peruzzi', 'Guadagni'): 1.0,
            Pair('Strozzi', 'Medici'): 1.0,
            Pair('Pazzi', 'Medici'): 1.0,
            Pair('Ridolfi', 'Albizzi'): 1.0,
            Pair('Tornabuoni', 'Lamberteschi'): 1.0,
            Pair('Tornabuoni', 'Salviati'): 1.0,
            Pair('Ridolfi', 'Acciaiuoli'): 1.0,
            Pair('Strozzi', 'Guadagni'): 1.0,
            Pair('Salviati', 'Acciaiuoli'): 1.0,
            Pair('Guadagni', 'Ginori'): 1.0,
            Pair('Strozzi', 'Barbadori'): 1.0,
            Pair('Peruzzi', 'Barbadori'): 1.0,
            Pair('Tornabuoni', 'Ridolfi'): 1.0,
            Pair('Albizzi', 'Acciaiuoli'): 1.0,
            Pair('Tornabuoni', 'Medici'): 1.0,
            Pair('Ridolfi', 'Medici'): 1.0,
            Pair('Peruzzi', 'Castellani'): 1.0,
            Pair('Tornabuoni', 'Strozzi'): 1.0,
            Pair('Tornabuoni', 'Bischeri'): 1.0,
            Pair('Barbadori', 'Albizzi'): 1.0,
            Pair('Castellani', 'Bischeri'): 2.0,
            Pair('Ridolfi', 'Guadagni'): 1.0,
            Pair('Ridolfi', 'Bischeri'): 1.0,
            Pair('Ridolfi', 'Peruzzi'): 1.0,
            Pair('Medici', 'Castellani'): 1.0,
            Pair('Bischeri', 'Albizzi'): 1.0,
            Pair('Medici', 'Ginori'): 1.0,
            Pair('Salviati', 'Ridolfi'): 1.0,
            Pair('Tornabuoni', 'Barbadori'): 1.0,
            Pair('Strozzi', 'Castellani'): 1.0,
            Pair('Salviati', 'Barbadori'): 1.0,
            Pair('Strozzi', 'Peruzzi'): 2.0,
            Pair('Strozzi', 'Bischeri'): 1.0,
            Pair('Tornabuoni', 'Albizzi'): 2.0,
            Pair('Barbadori', 'Acciaiuoli'): 1.0,
            Pair('Ridolfi', 'Castellani'): 1.0,
            Pair('Tornabuoni', 'Acciaiuoli'): 1.0}
        assert_dict_equal(CommonNeighbours(self.G).predict(), answer)

    def test_common_neighbours_alpha(self):
        pass

    def test_cosine(self):
        answer = {
            Pair('Ridolfi', 'Barbadori'): 0.40824829046386307,
            Pair('Medici', 'Guadagni'): 0.40824829046386307,
            Pair('Peruzzi', 'Bischeri'): 0.33333333333333331,
            Pair('Lamberteschi', 'Bischeri'): 0.57735026918962584,
            Pair('Salviati', 'Albizzi'): 0.40824829046386307,
            Pair('Lamberteschi', 'Albizzi'): 0.57735026918962584,
            Pair('Peruzzi', 'Guadagni'): 0.28867513459481292,
            Pair('Strozzi', 'Medici'): 0.20412414523193154,
            Pair('Pazzi', 'Medici'): 0.40824829046386307,
            Pair('Ridolfi', 'Albizzi'): 0.33333333333333331,
            Pair('Tornabuoni', 'Lamberteschi'): 0.57735026918962584,
            Pair('Tornabuoni', 'Salviati'): 0.40824829046386307,
            Pair('Ridolfi', 'Acciaiuoli'): 0.57735026918962584,
            Pair('Strozzi', 'Guadagni'): 0.25,
            Pair('Salviati', 'Acciaiuoli'): 0.70710678118654746,
            Pair('Guadagni', 'Ginori'): 0.5,
            Pair('Strozzi', 'Barbadori'): 0.35355339059327373,
            Pair('Peruzzi', 'Barbadori'): 0.40824829046386307,
            Pair('Tornabuoni', 'Ridolfi'): 0.33333333333333331,
            Pair('Albizzi', 'Acciaiuoli'): 0.57735026918962584,
            Pair('Tornabuoni', 'Medici'): 0.23570226039551587,
            Pair('Ridolfi', 'Medici'): 0.23570226039551587,
            Pair('Peruzzi', 'Castellani'): 0.33333333333333331,
            Pair('Tornabuoni', 'Strozzi'): 0.28867513459481292,
            Pair('Tornabuoni', 'Bischeri'): 0.33333333333333331,
            Pair('Barbadori', 'Albizzi'): 0.40824829046386307,
            Pair('Castellani', 'Bischeri'): 0.66666666666666663,
            Pair('Ridolfi', 'Guadagni'): 0.28867513459481292,
            Pair('Ridolfi', 'Bischeri'): 0.33333333333333331,
            Pair('Ridolfi', 'Peruzzi'): 0.33333333333333331,
            Pair('Medici', 'Castellani'): 0.23570226039551587,
            Pair('Bischeri', 'Albizzi'): 0.33333333333333331,
            Pair('Medici', 'Ginori'): 0.40824829046386307,
            Pair('Salviati', 'Ridolfi'): 0.40824829046386307,
            Pair('Tornabuoni', 'Barbadori'): 0.40824829046386307,
            Pair('Strozzi', 'Castellani'): 0.28867513459481292,
            Pair('Salviati', 'Barbadori'): 0.5,
            Pair('Strozzi', 'Peruzzi'): 0.57735026918962584,
            Pair('Strozzi', 'Bischeri'): 0.28867513459481292,
            Pair('Tornabuoni', 'Albizzi'): 0.66666666666666663,
            Pair('Barbadori', 'Acciaiuoli'): 0.70710678118654746,
            Pair('Ridolfi', 'Castellani'): 0.33333333333333331,
            Pair('Tornabuoni', 'Acciaiuoli'): 0.57735026918962584}
        assert_dict_equal(Cosine(self.G).predict(), answer)

    def test_cosine_weighted(self):
        pass

    def test_degree_product(self):
        answer = {
            Pair('Peruzzi', 'Bischeri'): 9.0,
            Pair('Lamberteschi', 'Albizzi'): 3.0,
            Pair('Tornabuoni', 'Ginori'): 3.0,
            Pair('Salviati', 'Pazzi'): 2.0,
            Pair('Guadagni', 'Castellani'): 12.0,
            Pair('Tornabuoni', 'Castellani'): 9.0,
            Pair('Castellani', 'Albizzi'): 9.0,
            Pair('Ginori', 'Barbadori'): 2.0,
            Pair('Pazzi', 'Guadagni'): 4.0,
            Pair('Castellani', 'Barbadori'): 6.0,
            Pair('Lamberteschi', 'Acciaiuoli'): 1.0,
            Pair('Ginori', 'Acciaiuoli'): 1.0,
            Pair('Lamberteschi', 'Ginori'): 1.0,
            Pair('Peruzzi', 'Barbadori'): 6.0,
            Pair('Medici', 'Castellani'): 18.0,
            Pair('Ginori', 'Castellani'): 3.0,
            Pair('Guadagni', 'Barbadori'): 8.0,
            Pair('Salviati', 'Medici'): 12.0,
            Pair('Ridolfi', 'Lamberteschi'): 3.0,
            Pair('Salviati', 'Ginori'): 2.0,
            Pair('Salviati', 'Barbadori'): 4.0,
            Pair('Strozzi', 'Pazzi'): 4.0,
            Pair('Pazzi', 'Acciaiuoli'): 1.0,
            Pair('Tornabuoni', 'Medici'): 18.0,
            Pair('Strozzi', 'Albizzi'): 12.0,
            Pair('Guadagni', 'Acciaiuoli'): 4.0,
            Pair('Lamberteschi', 'Bischeri'): 3.0,
            Pair('Ridolfi', 'Ginori'): 3.0,
            Pair('Castellani', 'Bischeri'): 9.0,
            Pair('Strozzi', 'Medici'): 24.0,
            Pair('Bischeri', 'Acciaiuoli'): 3.0,
            Pair('Strozzi', 'Guadagni'): 16.0,
            Pair('Medici', 'Acciaiuoli'): 6.0,
            Pair('Medici', 'Albizzi'): 18.0,
            Pair('Pazzi', 'Albizzi'): 3.0,
            Pair('Peruzzi', 'Medici'): 18.0,
            Pair('Guadagni', 'Albizzi'): 12.0,
            Pair('Strozzi', 'Acciaiuoli'): 4.0,
            Pair('Bischeri', 'Barbadori'): 6.0,
            Pair('Peruzzi', 'Castellani'): 9.0,
            Pair('Strozzi', 'Ridolfi'): 12.0,
            Pair('Barbadori', 'Albizzi'): 6.0,
            Pair('Ridolfi', 'Peruzzi'): 9.0,
            Pair('Bischeri', 'Albizzi'): 9.0,
            Pair('Ridolfi', 'Barbadori'): 6.0,
            Pair('Peruzzi', 'Pazzi'): 3.0,
            Pair('Strozzi', 'Peruzzi'): 12.0,
            Pair('Pazzi', 'Ginori'): 1.0,
            Pair('Medici', 'Lamberteschi'): 6.0,
            Pair('Strozzi', 'Bischeri'): 12.0,
            Pair('Salviati', 'Lamberteschi'): 2.0,
            Pair('Ridolfi', 'Castellani'): 9.0,
            Pair('Peruzzi', 'Lamberteschi'): 3.0,
            Pair('Ginori', 'Albizzi'): 3.0,
            Pair('Peruzzi', 'Guadagni'): 12.0,
            Pair('Strozzi', 'Lamberteschi'): 4.0,
            Pair('Medici', 'Guadagni'): 24.0,
            Pair('Salviati', 'Bischeri'): 6.0,
            Pair('Tornabuoni', 'Salviati'): 6.0,
            Pair('Medici', 'Barbadori'): 12.0,
            Pair('Guadagni', 'Bischeri'): 12.0,
            Pair('Salviati', 'Ridolfi'): 6.0,
            Pair('Salviati', 'Peruzzi'): 6.0,
            Pair('Pazzi', 'Barbadori'): 2.0,
            Pair('Ridolfi', 'Medici'): 18.0,
            Pair('Ridolfi', 'Guadagni'): 12.0,
            Pair('Ridolfi', 'Bischeri'): 9.0,
            Pair('Tornabuoni', 'Guadagni'): 12.0,
            Pair('Castellani', 'Acciaiuoli'): 3.0,
            Pair('Tornabuoni', 'Barbadori'): 6.0,
            Pair('Ginori', 'Bischeri'): 3.0,
            Pair('Lamberteschi', 'Castellani'): 3.0,
            Pair('Tornabuoni', 'Albizzi'): 9.0,
            Pair('Salviati', 'Guadagni'): 8.0,
            Pair('Tornabuoni', 'Pazzi'): 3.0,
            Pair('Salviati', 'Albizzi'): 6.0,
            Pair('Lamberteschi', 'Guadagni'): 4.0,
            Pair('Ridolfi', 'Pazzi'): 3.0,
            Pair('Peruzzi', 'Albizzi'): 9.0,
            Pair('Strozzi', 'Salviati'): 8.0,
            Pair('Strozzi', 'Barbadori'): 8.0,
            Pair('Tornabuoni', 'Lamberteschi'): 3.0,
            Pair('Pazzi', 'Medici'): 6.0,
            Pair('Ridolfi', 'Acciaiuoli'): 3.0,
            Pair('Guadagni', 'Ginori'): 4.0,
            Pair('Ridolfi', 'Albizzi'): 9.0,
            Pair('Albizzi', 'Acciaiuoli'): 3.0,
            Pair('Tornabuoni', 'Strozzi'): 12.0,
            Pair('Tornabuoni', 'Bischeri'): 9.0,
            Pair('Tornabuoni', 'Peruzzi'): 9.0,
            Pair('Salviati', 'Castellani'): 6.0,
            Pair('Peruzzi', 'Ginori'): 3.0,
            Pair('Medici', 'Ginori'): 6.0,
            Pair('Peruzzi', 'Acciaiuoli'): 3.0,
            Pair('Pazzi', 'Lamberteschi'): 1.0,
            Pair('Pazzi', 'Castellani'): 3.0,
            Pair('Strozzi', 'Castellani'): 12.0,
            Pair('Lamberteschi', 'Barbadori'): 2.0,
            Pair('Salviati', 'Acciaiuoli'): 2.0,
            Pair('Pazzi', 'Bischeri'): 3.0,
            Pair('Strozzi', 'Ginori'): 4.0,
            Pair('Tornabuoni', 'Ridolfi'): 9.0,
            Pair('Barbadori', 'Acciaiuoli'): 2.0,
            Pair('Tornabuoni', 'Acciaiuoli'): 3.0,
            Pair('Medici', 'Bischeri'): 18.0}
        assert_dict_equal(DegreeProduct(self.G).predict(), answer)

    def test_degree_product_weighted(self):
        pass

    def test_jaccard(self):
        answer = {
            Pair('Ridolfi', 'Barbadori'): 0.25,
            Pair('Medici', 'Guadagni'): 0.25,
            Pair('Peruzzi', 'Bischeri'): 0.20000000000000001,
            Pair('Lamberteschi', 'Bischeri'): 0.33333333333333331,
            Pair('Salviati', 'Albizzi'): 0.25,
            Pair('Lamberteschi', 'Albizzi'): 0.33333333333333331,
            Pair('Peruzzi', 'Guadagni'): 0.16666666666666666,
            Pair('Strozzi', 'Medici'): 0.1111111111111111,
            Pair('Pazzi', 'Medici'): 0.16666666666666666,
            Pair('Ridolfi', 'Albizzi'): 0.20000000000000001,
            Pair('Tornabuoni', 'Lamberteschi'): 0.33333333333333331,
            Pair('Tornabuoni', 'Salviati'): 0.25,
            Pair('Ridolfi', 'Acciaiuoli'): 0.33333333333333331,
            Pair('Strozzi', 'Guadagni'): 0.14285714285714285,
            Pair('Salviati', 'Acciaiuoli'): 0.5,
            Pair('Guadagni', 'Ginori'): 0.25,
            Pair('Strozzi', 'Barbadori'): 0.20000000000000001,
            Pair('Peruzzi', 'Barbadori'): 0.25,
            Pair('Tornabuoni', 'Ridolfi'): 0.20000000000000001,
            Pair('Albizzi', 'Acciaiuoli'): 0.33333333333333331,
            Pair('Tornabuoni', 'Medici'): 0.125,
            Pair('Ridolfi', 'Medici'): 0.125,
            Pair('Peruzzi', 'Castellani'): 0.20000000000000001,
            Pair('Tornabuoni', 'Strozzi'): 0.16666666666666666,
            Pair('Tornabuoni', 'Bischeri'): 0.20000000000000001,
            Pair('Barbadori', 'Albizzi'): 0.25,
            Pair('Castellani', 'Bischeri'): 0.5,
            Pair('Ridolfi', 'Guadagni'): 0.16666666666666666,
            Pair('Ridolfi', 'Bischeri'): 0.20000000000000001,
            Pair('Ridolfi', 'Peruzzi'): 0.20000000000000001,
            Pair('Medici', 'Castellani'): 0.125,
            Pair('Bischeri', 'Albizzi'): 0.20000000000000001,
            Pair('Medici', 'Ginori'): 0.16666666666666666,
            Pair('Salviati', 'Ridolfi'): 0.25,
            Pair('Tornabuoni', 'Barbadori'): 0.25,
            Pair('Strozzi', 'Castellani'): 0.16666666666666666,
            Pair('Salviati', 'Barbadori'): 0.33333333333333331,
            Pair('Strozzi', 'Peruzzi'): 0.40000000000000002,
            Pair('Strozzi', 'Bischeri'): 0.16666666666666666,
            Pair('Tornabuoni', 'Albizzi'): 0.5,
            Pair('Barbadori', 'Acciaiuoli'): 0.5,
            Pair('Ridolfi', 'Castellani'): 0.20000000000000001,
            Pair('Tornabuoni', 'Acciaiuoli'): 0.33333333333333331}
        assert_dict_equal(Jaccard(self.G).predict(), answer)

    def test_jaccard_weighted(self):
        pass

    def test_n_measure(self):
        answer = {
            Pair('Ridolfi', 'Barbadori'): 0.39223227027636809,
            Pair('Medici', 'Guadagni'): 0.39223227027636809,
            Pair('Peruzzi', 'Bischeri'): 0.33333333333333337,
            Pair('Lamberteschi', 'Bischeri'): 0.44721359549995793,
            Pair('Salviati', 'Albizzi'): 0.39223227027636809,
            Pair('Lamberteschi', 'Albizzi'): 0.44721359549995793,
            Pair('Peruzzi', 'Guadagni'): 0.28284271247461901,
            Pair('Strozzi', 'Medici'): 0.19611613513818404,
            Pair('Pazzi', 'Medici'): 0.2324952774876386,
            Pair('Ridolfi', 'Albizzi'): 0.33333333333333337,
            Pair('Tornabuoni', 'Lamberteschi'): 0.44721359549995793,
            Pair('Tornabuoni', 'Salviati'): 0.39223227027636809,
            Pair('Ridolfi', 'Acciaiuoli'): 0.44721359549995793,
            Pair('Strozzi', 'Guadagni'): 0.25,
            Pair('Salviati', 'Acciaiuoli'): 0.63245553203367588,
            Pair('Guadagni', 'Ginori'): 0.34299717028501769,
            Pair('Strozzi', 'Barbadori'): 0.31622776601683794,
            Pair('Peruzzi', 'Barbadori'): 0.39223227027636809,
            Pair('Tornabuoni', 'Ridolfi'): 0.33333333333333337,
            Pair('Albizzi', 'Acciaiuoli'): 0.44721359549995793,
            Pair('Tornabuoni', 'Medici'): 0.21081851067789195,
            Pair('Ridolfi', 'Medici'): 0.21081851067789195,
            Pair('Peruzzi', 'Castellani'): 0.33333333333333337,
            Pair('Tornabuoni', 'Strozzi'): 0.28284271247461901,
            Pair('Tornabuoni', 'Bischeri'): 0.33333333333333337,
            Pair('Barbadori', 'Albizzi'): 0.39223227027636809,
            Pair('Castellani', 'Bischeri'): 0.66666666666666674,
            Pair('Ridolfi', 'Guadagni'): 0.28284271247461901,
            Pair('Ridolfi', 'Bischeri'): 0.33333333333333337,
            Pair('Ridolfi', 'Peruzzi'): 0.33333333333333337,
            Pair('Medici', 'Castellani'): 0.21081851067789195,
            Pair('Bischeri', 'Albizzi'): 0.33333333333333337,
            Pair('Medici', 'Ginori'): 0.2324952774876386,
            Pair('Salviati', 'Ridolfi'): 0.39223227027636809,
            Pair('Tornabuoni', 'Barbadori'): 0.39223227027636809,
            Pair('Strozzi', 'Castellani'): 0.28284271247461901,
            Pair('Salviati', 'Barbadori'): 0.5,
            Pair('Strozzi', 'Peruzzi'): 0.56568542494923801,
            Pair('Strozzi', 'Bischeri'): 0.28284271247461901,
            Pair('Tornabuoni', 'Albizzi'): 0.66666666666666674,
            Pair('Barbadori', 'Acciaiuoli'): 0.63245553203367588,
            Pair('Ridolfi', 'Castellani'): 0.33333333333333337,
            Pair('Tornabuoni', 'Acciaiuoli'): 0.44721359549995793}
        assert_dict_equal(NMeasure(self.G).predict(), answer)

    def test_n_measure_weighted(self):
        pass

    def test_max_overlap(self):
        answer = {
            Pair('Ridolfi', 'Barbadori'): 0.33333333333333331,
            Pair('Medici', 'Guadagni'): 0.33333333333333331,
            Pair('Peruzzi', 'Bischeri'): 0.33333333333333331,
            Pair('Lamberteschi', 'Bischeri'): 0.33333333333333331,
            Pair('Salviati', 'Albizzi'): 0.33333333333333331,
            Pair('Lamberteschi', 'Albizzi'): 0.33333333333333331,
            Pair('Peruzzi', 'Guadagni'): 0.25,
            Pair('Strozzi', 'Medici'): 0.16666666666666666,
            Pair('Pazzi', 'Medici'): 0.16666666666666666,
            Pair('Ridolfi', 'Albizzi'): 0.33333333333333331,
            Pair('Tornabuoni', 'Lamberteschi'): 0.33333333333333331,
            Pair('Tornabuoni', 'Salviati'): 0.33333333333333331,
            Pair('Ridolfi', 'Acciaiuoli'): 0.33333333333333331,
            Pair('Strozzi', 'Guadagni'): 0.25,
            Pair('Salviati', 'Acciaiuoli'): 0.5,
            Pair('Guadagni', 'Ginori'): 0.25,
            Pair('Strozzi', 'Barbadori'): 0.25,
            Pair('Peruzzi', 'Barbadori'): 0.33333333333333331,
            Pair('Tornabuoni', 'Ridolfi'): 0.33333333333333331,
            Pair('Albizzi', 'Acciaiuoli'): 0.33333333333333331,
            Pair('Tornabuoni', 'Medici'): 0.16666666666666666,
            Pair('Ridolfi', 'Medici'): 0.16666666666666666,
            Pair('Peruzzi', 'Castellani'): 0.33333333333333331,
            Pair('Tornabuoni', 'Strozzi'): 0.25,
            Pair('Tornabuoni', 'Bischeri'): 0.33333333333333331,
            Pair('Barbadori', 'Albizzi'): 0.33333333333333331,
            Pair('Castellani', 'Bischeri'): 0.66666666666666663,
            Pair('Ridolfi', 'Guadagni'): 0.25,
            Pair('Ridolfi', 'Bischeri'): 0.33333333333333331,
            Pair('Ridolfi', 'Peruzzi'): 0.33333333333333331,
            Pair('Medici', 'Castellani'): 0.16666666666666666,
            Pair('Bischeri', 'Albizzi'): 0.33333333333333331,
            Pair('Medici', 'Ginori'): 0.16666666666666666,
            Pair('Salviati', 'Ridolfi'): 0.33333333333333331,
            Pair('Tornabuoni', 'Barbadori'): 0.33333333333333331,
            Pair('Strozzi', 'Castellani'): 0.25,
            Pair('Salviati', 'Barbadori'): 0.5,
            Pair('Strozzi', 'Peruzzi'): 0.5,
            Pair('Strozzi', 'Bischeri'): 0.25,
            Pair('Tornabuoni', 'Albizzi'): 0.66666666666666663,
            Pair('Barbadori', 'Acciaiuoli'): 0.5,
            Pair('Ridolfi', 'Castellani'): 0.33333333333333331,
            Pair('Tornabuoni', 'Acciaiuoli'): 0.33333333333333331}
        assert_dict_equal(MaxOverlap(self.G).predict(), answer)

    def test_max_overlap_weighted(self):
        pass

    def test_min_overlap(self):
        answer = {
            Pair('Ridolfi', 'Barbadori'): 0.5,
            Pair('Medici', 'Guadagni'): 0.5,
            Pair('Peruzzi', 'Bischeri'): 0.33333333333333331,
            Pair('Lamberteschi', 'Bischeri'): 1.0,
            Pair('Salviati', 'Albizzi'): 0.5,
            Pair('Lamberteschi', 'Albizzi'): 1.0,
            Pair('Peruzzi', 'Guadagni'): 0.33333333333333331,
            Pair('Strozzi', 'Medici'): 0.25,
            Pair('Pazzi', 'Medici'): 1.0,
            Pair('Ridolfi', 'Albizzi'): 0.33333333333333331,
            Pair('Tornabuoni', 'Lamberteschi'): 1.0,
            Pair('Tornabuoni', 'Salviati'): 0.5,
            Pair('Ridolfi', 'Acciaiuoli'): 1.0,
            Pair('Strozzi', 'Guadagni'): 0.25,
            Pair('Salviati', 'Acciaiuoli'): 1.0,
            Pair('Guadagni', 'Ginori'): 1.0,
            Pair('Strozzi', 'Barbadori'): 0.5,
            Pair('Peruzzi', 'Barbadori'): 0.5,
            Pair('Tornabuoni', 'Ridolfi'): 0.33333333333333331,
            Pair('Albizzi', 'Acciaiuoli'): 1.0,
            Pair('Tornabuoni', 'Medici'): 0.33333333333333331,
            Pair('Ridolfi', 'Medici'): 0.33333333333333331,
            Pair('Peruzzi', 'Castellani'): 0.33333333333333331,
            Pair('Tornabuoni', 'Strozzi'): 0.33333333333333331,
            Pair('Tornabuoni', 'Bischeri'): 0.33333333333333331,
            Pair('Barbadori', 'Albizzi'): 0.5,
            Pair('Castellani', 'Bischeri'): 0.66666666666666663,
            Pair('Ridolfi', 'Guadagni'): 0.33333333333333331,
            Pair('Ridolfi', 'Bischeri'): 0.33333333333333331,
            Pair('Ridolfi', 'Peruzzi'): 0.33333333333333331,
            Pair('Medici', 'Castellani'): 0.33333333333333331,
            Pair('Bischeri', 'Albizzi'): 0.33333333333333331,
            Pair('Medici', 'Ginori'): 1.0,
            Pair('Salviati', 'Ridolfi'): 0.5,
            Pair('Tornabuoni', 'Barbadori'): 0.5,
            Pair('Strozzi', 'Castellani'): 0.33333333333333331,
            Pair('Salviati', 'Barbadori'): 0.5,
            Pair('Strozzi', 'Peruzzi'): 0.66666666666666663,
            Pair('Strozzi', 'Bischeri'): 0.33333333333333331,
            Pair('Tornabuoni', 'Albizzi'): 0.66666666666666663,
            Pair('Barbadori', 'Acciaiuoli'): 1.0,
            Pair('Ridolfi', 'Castellani'): 0.33333333333333331,
            Pair('Tornabuoni', 'Acciaiuoli'): 1.0}
        assert_dict_equal(MinOverlap(self.G).predict(), answer)

    def test_min_overlap_weighted(self):
        pass

    def test_pearson(self):
        answer = {
            Pair('Medici', 'Guadagni'): 0.091287092917527679,
            Pair('Peruzzi', 'Bischeri'): 0.15151515151515152,
            Pair('Lamberteschi', 'Bischeri'): 0.53108500454379437,
            Pair('Salviati', 'Albizzi'): 0.28426762180748061,
            Pair('Lamberteschi', 'Albizzi'): 0.53108500454379437,
            Pair('Peruzzi', 'Guadagni'): 0.055048188256318034,
            Pair('Strozzi', 'Barbadori'): 0.19364916731037085,
            Pair('Pazzi', 'Medici'): 0.32025630761017432,
            Pair('Ridolfi', 'Albizzi'): 0.15151515151515152,
            Pair('Tornabuoni', 'Lamberteschi'): 0.53108500454379437,
            Pair('Tornabuoni', 'Salviati'): 0.28426762180748061,
            Pair('Ridolfi', 'Acciaiuoli'): 0.53108500454379437,
            Pair('Barbadori', 'Acciaiuoli'): 0.67936622048675754,
            Pair('Guadagni', 'Ginori'): 0.43852900965351466,
            Pair('Peruzzi', 'Barbadori'): 0.28426762180748061,
            Pair('Tornabuoni', 'Ridolfi'): 0.15151515151515152,
            Pair('Albizzi', 'Acciaiuoli'): 0.53108500454379437,
            Pair('Strozzi', 'Castellani'): 0.055048188256318034,
            Pair('Ridolfi', 'Barbadori'): 0.28426762180748061,
            Pair('Peruzzi', 'Castellani'): 0.15151515151515152,
            Pair('Tornabuoni', 'Strozzi'): 0.055048188256318034,
            Pair('Tornabuoni', 'Bischeri'): 0.15151515151515152,
            Pair('Barbadori', 'Albizzi'): 0.28426762180748061,
            Pair('Castellani', 'Bischeri'): 0.5757575757575758,
            Pair('Ridolfi', 'Guadagni'): 0.055048188256318034,
            Pair('Ridolfi', 'Bischeri'): 0.15151515151515152,
            Pair('Ridolfi', 'Peruzzi'): 0.15151515151515152,
            Pair('Bischeri', 'Albizzi'): 0.15151515151515152,
            Pair('Medici', 'Ginori'): 0.32025630761017432,
            Pair('Salviati', 'Ridolfi'): 0.28426762180748061,
            Pair('Tornabuoni', 'Barbadori'): 0.28426762180748061,
            Pair('Salviati', 'Acciaiuoli'): 0.67936622048675754,
            Pair('Salviati', 'Barbadori'): 0.41666666666666674,
            Pair('Strozzi', 'Peruzzi'): 0.44038550605054427,
            Pair('Strozzi', 'Bischeri'): 0.055048188256318034,
            Pair('Tornabuoni', 'Albizzi'): 0.5757575757575758,
            Pair('Ridolfi', 'Castellani'): 0.15151515151515152,
            Pair('Tornabuoni', 'Acciaiuoli'): 0.53108500454379437}
        assert_dict_equal(Pearson(self.G).predict(), answer)

    def test_pearson_weighted(self):
        pass

    def test_resource_allocation(self):
        answer = {
            Pair('Ridolfi', 'Barbadori'): 0.16666666666666666,
            Pair('Medici', 'Guadagni'): 0.66666666666666663,
            Pair('Peruzzi', 'Bischeri'): 0.25,
            Pair('Lamberteschi', 'Bischeri'): 0.25,
            Pair('Salviati', 'Albizzi'): 0.16666666666666666,
            Pair('Lamberteschi', 'Albizzi'): 0.25,
            Pair('Peruzzi', 'Guadagni'): 0.33333333333333331,
            Pair('Strozzi', 'Medici'): 0.33333333333333331,
            Pair('Pazzi', 'Medici'): 0.5,
            Pair('Ridolfi', 'Albizzi'): 0.16666666666666666,
            Pair('Tornabuoni', 'Lamberteschi'): 0.25,
            Pair('Tornabuoni', 'Salviati'): 0.16666666666666666,
            Pair('Ridolfi', 'Acciaiuoli'): 0.16666666666666666,
            Pair('Strozzi', 'Guadagni'): 0.33333333333333331,
            Pair('Salviati', 'Acciaiuoli'): 0.16666666666666666,
            Pair('Guadagni', 'Ginori'): 0.33333333333333331,
            Pair('Strozzi', 'Barbadori'): 0.33333333333333331,
            Pair('Peruzzi', 'Barbadori'): 0.33333333333333331,
            Pair('Tornabuoni', 'Ridolfi'): 0.16666666666666666,
            Pair('Albizzi', 'Acciaiuoli'): 0.16666666666666666,
            Pair('Tornabuoni', 'Medici'): 0.33333333333333331,
            Pair('Ridolfi', 'Medici'): 0.33333333333333331,
            Pair('Peruzzi', 'Castellani'): 0.25,
            Pair('Tornabuoni', 'Strozzi'): 0.33333333333333331,
            Pair('Tornabuoni', 'Bischeri'): 0.25,
            Pair('Barbadori', 'Albizzi'): 0.16666666666666666,
            Pair('Castellani', 'Bischeri'): 0.58333333333333326,
            Pair('Ridolfi', 'Guadagni'): 0.33333333333333331,
            Pair('Ridolfi', 'Bischeri'): 0.25,
            Pair('Ridolfi', 'Peruzzi'): 0.25,
            Pair('Medici', 'Castellani'): 0.5,
            Pair('Bischeri', 'Albizzi'): 0.25,
            Pair('Medici', 'Ginori'): 0.33333333333333331,
            Pair('Salviati', 'Ridolfi'): 0.16666666666666666,
            Pair('Tornabuoni', 'Barbadori'): 0.16666666666666666,
            Pair('Strozzi', 'Castellani'): 0.33333333333333331,
            Pair('Salviati', 'Barbadori'): 0.16666666666666666,
            Pair('Strozzi', 'Peruzzi'): 0.66666666666666663,
            Pair('Strozzi', 'Bischeri'): 0.33333333333333331,
            Pair('Tornabuoni', 'Albizzi'): 0.41666666666666663,
            Pair('Barbadori', 'Acciaiuoli'): 0.16666666666666666,
            Pair('Ridolfi', 'Castellani'): 0.25,
            Pair('Tornabuoni', 'Acciaiuoli'): 0.16666666666666666}
        assert_dict_equal(ResourceAllocation(self.G).predict(), answer)

    def test_resource_allocation_weighted(self):
        pass


class TestUnweighted:
    def setup(self):
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5)])

    def test_adamic_adar(self):
        known = {(1, 5): 1 / log(3), (2, 3): 2 / log(2),
                 (1, 4): 1 / log(2) + 1 / log(3), (4, 5): 1 / log(3)}
        found = AdamicAdar(self.G).predict()
        print found
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_association_strength(self):
        known = {(1, 5): 0.5, (2, 3): 1 / 3, (1, 4): 0.5, (4, 5): 0.5}
        found = AssociationStrength(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_common_neighbours(self):
        known = {(1, 5): 1, (2, 3): 2, (1, 4): 2, (4, 5): 1}
        found = CommonNeighbours(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_cosine(self):
        known = {(1, 5): 1 / sqrt(2), (2, 3): 2 / sqrt(6), (1, 4): 1,
                 (4, 5): 1 / sqrt(2)}
        found = Cosine(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_degree_product(self):
        known = {(1, 2): 4, (1, 3): 6, (1, 4): 4, (1, 5): 2, (2, 3): 6,
                 (2, 4): 4, (2, 5): 2, (3, 4): 6, (3, 5): 3, (4, 5): 2}
        found = DegreeProduct(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_jaccard(self):
        known = {(1, 5): 0.5, (2, 3): 2 / 3, (1, 4): 1, (4, 5): 0.5}
        found = Jaccard(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_nmeasure(self):
        known = {(1, 5): sqrt(2 / 5), (2, 3): sqrt(8 / 13), (1, 4): 1,
                 (4, 5): sqrt(2 / 5)}
        found = NMeasure(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_maxoverlap(self):
        known = {(1, 5): 0.5, (2, 3): 2 / 3, (1, 4): 1, (4, 5): 0.5}
        found = MaxOverlap(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_minoverlap(self):
        known = {(1, 5): 1, (2, 3): 1, (1, 4): 1, (4, 5): 1}
        found = MinOverlap(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_pearson(self):
        known = {(1, 5): 0.5, (2, 3): 2 / 3, (1, 4): 1, (4, 5): 0.5}
        found = Pearson(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))

    def test_resource_allocation(self):
        known = {(1, 5): 1 / 3, (2, 3): 1, (1, 4): 5 / 6, (4, 5): 1 / 3}
        found = ResourceAllocation(self.G).predict()
        assert_dict_almost_equal(found, Scoresheet(known))
