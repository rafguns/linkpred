import re
import sys


def all_pairs(l):
    """Return list of all possible pairs in l"""
    try:
        from itertools import combinations
        return combinations(l, 2)
    except ImportError:
        return (tuple(sorted((x, y))) for i, x in enumerate(l, start=1)
                for y in l[:i] if x != y)


def slugify(value):
    """
    Normalize string to 'slug'

    Converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.

    Taken from http://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename-in-python/295466#295466

    """
    import unicodedata
    value = unicodedata.normalize(
        'NFKD', unicode(value)).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
    return unicode(re.sub('[-\s]+', '-', value))


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


def load_function(functionname):
    """Return the function given by functionname

    This loads function names of the form <module.submodule.function>

    """
    try:
        # Find rightmost point. Everything to the left is module name.
        index = functionname.rindex('.')
        modulename = functionname[:index]
    except ValueError:
        raise Exception("No module name given in " + functionname)
    # Dynamically load module and function
    __import__(modulename)
    module = sys.modules[modulename]
    function = getattr(module, functionname[index + 1:])
    return function


def ensure_dir(fname):
    """Make sure all the intermediate directories exist for given file name"""
    import os

    d = os.path.dirname(fname)
    if not os.path.isdir(d):
        os.makedirs(d)


def interpolate(l):
    """Make curve l decrease."""
    l.reverse()
    for i in xrange(len(l) - 1):
        if l[i] >= l[i + 1]:
            l[i + 1] = l[i]
    l.reverse()
    return l


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
