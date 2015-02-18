from nose.tools import (assert_almost_equal, assert_equal)


def assert_array_equal(a1, a2):
    try:
        if not (a1 == a2).all():
            raise AssertionError("{} != {}".format(a1, a2))
    except AttributeError:  # a1 and a2 are lists or empty ndarrays
        assert_equal(a1, a2)


def assert_dict_almost_equal(d1, d2):
    for k in d1:
        assert_almost_equal(d1[k], d2[k])
