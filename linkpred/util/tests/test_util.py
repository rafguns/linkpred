from nose.tools import assert_equal, raises

import linkpred.util as u


def test_all_pairs():
    s = [1, 2, 3, 4]
    expected = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    assert_equal(sorted(u.all_pairs(s)), expected)


def test_slugify():
    # example from http://farmdev.com/talks/unicode/#id5
    s = u"Ivan Krsti\u0107"
    expected = "ivan-krstic"
    assert_equal(u.slugify(s), expected)


def test_load_function():
    import os
    assert_equal(u.load_function('os.path.join'), os.path.join)


@raises(ValueError)
def test_load_function_no_modulename():
    u.load_function('join')
