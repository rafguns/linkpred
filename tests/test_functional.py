import os
import subprocess

from nose.tools import assert_equal


def test_simple_run():
    import matplotlib
    matplotlib.use('Agg')

    num_files = len(os.listdir('examples'))

    subprocess.check_call(
        'python scripts/linkpred examples/inf1990-2004.net '
        'examples/inf2005-2009.net -p CommonNeighbours --quiet'.split())

    assert_equal(len(os.listdir('examples')), num_files + 1)
