import contextlib
import os
import tempfile


def assert_array_equal(a1, a2):
    try:
        if not (a1 == a2).all():
            raise AssertionError("{} != {}".format(a1, a2))
    except AttributeError:  # a1 and a2 are lists or empty ndarrays
        assert a1 == a2


@contextlib.contextmanager
def temp_file(suffix=".tmp"):
    fd, fname = tempfile.mkstemp(suffix)
    yield fname
    os.close(fd)
    os.unlink(fname)
