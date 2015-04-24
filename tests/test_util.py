from nose.tools import assert_equal, assert_raises, raises

import linkpred.util as u
import six


def test_all_pairs():
    s = [1, 2, 3, 4]
    expected = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    assert_equal(sorted(u.all_pairs(s)), expected)


def test_load_function():
    import os
    assert_equal(u.load_function('os.path.join'), os.path.join)


@raises(ValueError)
def test_load_function_no_modulename():
    u.load_function('join')


def test_interpolate():
    a = [10, 8, 9, 6, 6, 7, 3, 5, 6, 2, 1, 2]
    assert_equal(u.interpolate(a), [10, 9, 9, 7, 7, 7, 6, 6, 6, 2, 2, 2])

    a = list(range(5))
    assert_equal(u.interpolate(a), [4] * 5)


def test_itersubclasses():
    class A(object):
        pass

    class Aa(A):
        pass

    class Ab(A):
        pass

    class Aaa(Aa):
        pass

    class OldStyleClass:
        pass

    name = lambda x: x.__name__
    assert_equal(list(map(name, u.itersubclasses(A))), ['Aa', 'Aaa', 'Ab'])

    if six.PY2:
        with assert_raises(TypeError):
            list(u.itersubclasses(OldStyleClass))


# This is silly but hey... 100% test coverage for this file :-)
def test_itersubclasses_from_type():
    list(u.itersubclasses(type))
