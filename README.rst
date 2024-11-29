⚠️ **Note: This package is in maintenance mode**.
Critical bugs will continue to be resolved,
but no new features will be implemented (`more information <https://github.com/rafguns/linkpred/issues/35>`_).

Linkpred
========

**Linkpred** is a Python package for link prediction: given a network, Linkpred provides a number of heuristics (known as *predictors*) that assess the likelihood of potential links in a future snapshot of the network.

While some predictors are fairly straightforward (e.g., if two people have a large number of mutual friends, it seems likely that eventually they will meet and become friends), others are more involved.

.. image:: https://codecov.io/gh/rafguns/linkpred/branch/master/graph/badge.svg?token=JVZIVHWJXY 
   :target: https://codecov.io/gh/rafguns/linkpred

**linkpred** can both be used as a command-line tool and as a Python library in your own code.


Installation
------------

**linkpred** (v0.6 and later) works under Python 3.8 to 3.12.
It depends on:

- matplotlib
- networkx
- numpy
- pyyaml
- scipy
- smokesignal

You should be able to install Linkpred and its dependencies using pip (``pip install linkpred`` or ``python -m pip install linkpred``).
If you do not yet have Python installed, I recommend starting with `Anaconda <https://www.continuum.io/downloads>`_,
which includes optimized versions of packages like numpy.
If you want to use the Community predictor, which relies on community structure of the network,
make sure you also have the `python-louvain <https://github.com/taynaud/python-louvain>`_ package by installing with ``pip install linkpred[community]``.


Example usage as command-line tool
----------------------------------

A good starting point is ``linkpred --help``, which lists all the available options. To save the predictions of the ``CommonNeighbours`` predictor, for instance, run::

    $ linkpred examples/inf1990-2004.net -p CommonNeighbours --output cache-predictions

where ``examples/inf1990-2004.net`` is a network file in Pajek format. Other supported formats include GML and GraphML. The full output looks like this:

.. code:: console

    $ linkpred examples/inf1990-2004.net -p CommonNeighbours --output cache-predictions
    16:43:13 - INFO - Reading file 'examples/inf1990-2004.net'...
    16:43:13 - INFO - Successfully read file.
    16:43:13 - INFO - Starting preprocessing...
    16:43:13 - INFO - Removed 35 nodes (degree < 1)
    16:43:13 - INFO - Finished preprocessing.
    16:43:13 - INFO - Executing CommonNeighbours...
    16:43:14 - INFO - Finished executing CommonNeighbours.
    16:43:14 - INFO - Prediction run finished

    $ head examples/inf1990-2004-CommonNeighbours-predictions_2016-04-22_16.43.txt
    "Ikogami, K"    "Ikegami, K"    5.0
    "Durand, T"     "Abd El Kader, M"       5.0
    "Sharma, L"     "Kumar, S"      4.0
    "Paul, A"       "Durand, T"     4.0
    "Paul, A"       "Dudognon, G"   4.0
    "Paul, A"       "Abd El Kader, M"       4.0
    "Karisiddippa, CR"      "Garg, KC"      4.0
    "Wu, YS"        "Kretschmer, H" 3.0
    "Veugelers, R"  "Deleus, F"     3.0
    "Veugelers, R"  "Andries, P"    3.0


Example usage within Python
---------------------------

.. code:: pycon

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
    ...    print(authors, score)
    ...
    Tomizawa, H - Fujigaki, Y 0.188686630053
    Shirabe, M - Hayashi, T 0.143866427916
    Garfield, E - Fuseler, EA 0.148097050146
    Persson, O - Larsen, IM 0.138516589957
    Vanleeuwen, TN - Noyons, ECM 0.185040358711
