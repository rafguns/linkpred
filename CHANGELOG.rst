Changelog
=========

**Note**: I only started keeping this changelog from version 0.5 onwards.

Version 0.5
-----------

- Python 3.8 officially supported!

- Behind-the-scenes work: testing is now done with pytest and tox, formatting is done by black, and we are based on the latest version of networkx (2.4).

- The Community predictor is now easier to use, also because its optional dependency (`python-louvain <https://github.com/taynaud/python-louvain>`_) is now on PyPI. If you want to use it, install as follows:

    $ pip install linkpred[all]

- Some bug fixes.
