linkpred
========

**linkpred** is a Python package for link prediction: given a network, **linkpred** provides a number of heuristics (known as *predictors*) that assess the likelihood of potential links in a future snapshot of the network.

While some predictors are fairly straightforward (e.g., if two people have a large number of mutual friends, it seems likely that eventually they will meet and become friends), others are more involved.


.. image:: https://travis-ci.org/rafguns/linkpred.svg?branch=master
    :target: https://travis-ci.org/rafguns/linkpred

.. image:: https://coveralls.io/repos/rafguns/linkpred/badge.svg?branch=master
    :target: https://coveralls.io/r/rafguns/linkpred?branch=master


Example
-------

::

    >>> import linkpred
    >>> G = linkpred.read_network("examples/training.net")
    11:49:00 - INFO - Reading file 'examples/training.net'...
    11:49:00 - INFO - Successfully read file.
    >>> len(G)   # number of nodes
    632
    >>> # We exclude edges already present, to predict only new links
    >>> simrank = linkpred.predictors.SimRank(G, excluded=G.edges())
    >>> simrank_results = simrank.predict(c=0.5)
    >>> top = simrank_results.top(5)
    >>> for authors, score in top.items():
    ...    print authors, score
    ...
    Tomizawa, H - Fujigaki, Y 0.188686630053
    Shirabe, M - Hayashi, T 0.143866427916
    Garfield, E - Fuseler, EA 0.148097050146
    Persson, O - Larsen, IM 0.138516589957
    Vanleeuwen, TN - Noyons, ECM 0.185040358711
