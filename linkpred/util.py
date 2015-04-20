import sys


def all_pairs(l):
    """Return list of all possible pairs in l"""
    try:
        from itertools import combinations
        return combinations(l, 2)
    except ImportError:
        return (tuple(sorted((x, y))) for i, x in enumerate(l, start=1)
                for y in l[:i] if x != y)


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


def ensure_dir(fname):
    """Make sure all the intermediate directories exist for given file name"""
    import os

    d = os.path.dirname(fname)
    if not os.path.isdir(d):
        os.makedirs(d)


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


def python_2_unicode_compatible(klass):
    """
    A decorator that defines __unicode__ and __str__ methods under Python 2.
    Under Python 3 it does nothing.

    To support Python 2 and 3 with a single code base, define a __str__ method
    returning text and apply this decorator to the class.

    Source: https://docs.djangoproject.com/en/1.7/ref/utils/

    """
    if sys.version_info[0] == 2:  # Python 2
        if '__str__' not in klass.__dict__:
            raise ValueError("@python_2_unicode_compatible cannot be applied "
                             "to %s because it doesn't define __str__()." %
                             klass.__name__)
        klass.__unicode__ = klass.__str__
        klass.__str__ = lambda self: self.__unicode__().encode('utf-8')
    return klass
