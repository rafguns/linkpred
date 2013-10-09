import networkx as nx
from networkx.algorithms import bipartite

from .util import log

__all__ = ["ResultDict", "Result", "filter_low_degree_nodes"]


class Result(object):
    """Result represents a query result as a pathspec, a network, or both."""

    def __init__(self, data, eligible='eligible',
                 project=bipartite.weighted_projected_graph):
        self.eligible = eligible
        if isinstance(data, nx.Graph):
            if nx.is_bipartite(data):
                self.pathspec = data
                if self.eligible is None:
                    bottom = [n for n, d in data.nodes(data=True)]
                else:
                    bottom = [n for n, d in data.nodes(
                        data=True) if d[self.eligible]]
                self.network = project(data, bottom)
            else:
                self.network = data
        elif isinstance(data, Result):
            self.pathspec = data.pathspec
            self.network = data.network
        else:
            raise TypeError("Unexpected data type!")

    def __iter__(self):
        return iter(self.network)

    def __len__(self):
        return len(self.network)

    def for_comparison(self, exclude=set()):
        """Return the result in a format, suitable for comparison.

        In practice this means we return it as a set of Pairs.

        """
        from .evaluation import Pair

        exclude = set(Pair(u, v) for u, v in exclude)
        return set(Pair(u, v) for u, v in self.network.edges_iter()) - exclude

    def remove_items_from(self, l):
        self.network.remove_nodes_from(l)
        try:
            self.pathspec.remove_nodes_from(l)
        except AttributeError:
            pass

    def add_remove_random_edges(self, pct_to_remove=None, pct_to_add=None):
        from . import network

        if not pct_to_remove and not pct_to_add:
            return

        # For simplicity, we do not do this for pathspecs
        self.pathspec = None

        if pct_to_remove and pct_to_add:
            network.add_remove_random_edges(
                self.network, pct_to_add, pct_to_remove)
        elif pct_to_remove:
            network.remove_random_edges(self.network, pct_to_remove)
        elif pct_to_add:
            network.add_random_edges(self.network, pct_to_add)

    def low_degree(self, threshold):
        """
        Find low-degree nodes

        Parameters
        ----------
        threshold : int
            Only nodes whose degree is below the threshold are retained

        """
        if self.eligible is not None:
            return [n for n, d in self.network.degree_iter()
                    if d < threshold and self.network.node[n][self.eligible]]
        else:
            return [n for n, d in self.network.degree_iter() if d < threshold]

    def items_outside(self, container):
        if self.eligible is not None:
            return [n for n in self.network.nodes_iter()
                    if self.network.node[n][self.eligible] and n not in container]
        else:
            return [n for n in self.network.nodes_iter() if n not in container]


class ResultDict(dict):
    """A dict of Results, along with some methods for manipulating them."""

    def merge(self, mergespec, skipzero=True, weight='weight'):
        """Merge Results according to mergespec into new Result with given name

        Parameters
        ----------
        mergespec : a dict
            dictionary of result names and their weight
            (more weight = more importance)

        skipzero : True|False
            If an entry in the mergespec has zero weight, skip it
            (default: True)

        weight : string
            Edge attribute for edge weight

        Returns
        -------
        A networkx.Graph instance

        """
        log.logger.info("Merging...")
        g = nx.Graph()
        for resultname, resultweight in mergespec.iteritems():
            if resultweight == 0 and skipzero:
                continue
            result = self[resultname].network
            # We also copy node data, so that 'eligible' keywords are retained
            # in the merged network.
            g.add_nodes_from(result.nodes(data=True))
            for u, v, edgedata in result.edges_iter(data=True):
                w = edgedata[weight] * resultweight
                if g.has_edge(u, v):
                    g.edge[u][v][weight] += w
                else:
                    g.add_edge(u, v, attr_dict={weight: w})
        log.logger.info("Finished merging.")
        return g

    def filter_all_low_degree_nodes(self, minimum=1):
        networks = self.values()
        filter_low_degree_nodes(networks, minimum)


def filter_low_degree_nodes(results, minimum=1):
    """
    Only retain nodes that occur in all networks with at least a degree of k

    Changes are made in place.

    Arguments
    ---------
    networks : a list or iterable of networkx.Graph instances

    minimum : int
        minimum node degree

    """
    log.logger.info("Filtering low degree nodes...")
    for res in results:
        to_remove = res.low_degree(minimum)
        res.remove_items_from(to_remove)
        log.logger.info("Removed %d items" % len(to_remove))
    common = set.intersection(*[set(res) for res in results])
    for res in results:
        to_remove = res.items_outside(common)
        res.remove_items_from(to_remove)
        log.logger.info("Removed %d items" % len(to_remove))
    log.logger.info("Finished filtering low degree nodes.")
