import logging
import sys

logger = logging.getLogger('linkpred')
streamhandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                              "%H:%M:%S")
streamhandler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(streamhandler)


def called_by(n=0):
    """Returns caller of current function, useful for debugging

    Example
    -------
    >>> def foo():
    ....    from linkpred.util import log
    ....    log.logger.debug("Called by %s, %s, l. %s" % log.called_by(2))

    """
    f = sys._getframe(n)
    c = f.f_code
    return c.co_filename, c.co_name, f.f_lineno
