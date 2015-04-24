import itertools
import sys


def all_pairs(iterable):
    """Return iterator over all possible pairs in l"""
    return itertools.combinations(iterable, 2)


def progressbar(it, prefix="", size=60):
    """Show progress bar

    Taken from http://code.activestate.com/recipes/576986-progress-bar-for-console-programs-as-iterator/

    """
    count = len(it)

    def _show(_i):
        x = int(size * _i / count)
        sys.stdout.write(
            "%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x),
                                  _i, count))
        sys.stdout.flush()

    _show(0)
    for i, item in enumerate(it, start=1):
        yield item
        _show(i)
    sys.stdout.write("\n")
    sys.stdout.flush()


def load_function(full_functionname):
    """Return the function given by full_functionname

    This loads function names of the form 'module.submodule.function'

    """
    try:
        modulename, functionname = full_functionname.rsplit('.', 1)
    except ValueError:
        raise ValueError("No module name given in " + full_functionname)
    # Dynamically load module and function
    __import__(modulename)
    module = sys.modules[modulename]
    function = getattr(module, functionname)
    return function


def interpolate(curve):
    """Make curve decrease."""
    for i in range(-1, -len(curve), - 1):
        if curve[i] > curve[i - 1]:
            curve[i - 1] = curve[i]
    return curve


def itersubclasses(cls, _seen=None):
    """Generator over all subclasses of a given class, in depth first order.

    Source:
    http://code.activestate.com/recipes/576949-find-all-subclasses-of-a-given-class/

    """
    if not isinstance(cls, type):
        raise TypeError('itersubclasses must be called with '
                        'new-style classes, not %.100r' % cls)
    if _seen is None:
        _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError:  # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub
