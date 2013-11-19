# Fork of networkx.readwrite.pajek
import csv
import networkx
from networkx.utils import is_string_like

__all__ = ['read_pajek', 'parse_pajek', 'write_pajek']


def write_pajek(G, path, weight='weight', clusterpath=None, clusterlabel='cluster'):
    """Write in Pajek format to path.

    Parameters
    ----------
    G : graph
       A networkx graph
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.
    weight : string
        Edge attribute for edge weight
    clusterpath : file or string
        Optional path of partition file
    clusterlabel : string
        Label of the partition. Default: 'cluster'

    Examples
    --------
    >>> G=nx.path_graph(4)
    >>> nx.write_pajek(G, "test.net")
    """

    with open(path, mode="w") as fh:
        if G.name:
            fh.write("*network \"%s\"\n" % G.name)

        # write nodes with attributes
        fh.write("*vertices %s\n" % G.order())
        clu = "*vertices %s\n" % G.order()
        nodes = G.nodes()
        # make dictionary mapping nodes to integers
        nodenumber = dict(zip(nodes, range(1, len(nodes) + 1)))
        clusters = {}
        i = 0
        for n in nodes:
            na = G.node[n].copy()
            x = na.pop('x', None)
            y = na.pop('y', None)
            # It seems better if we just avoid the node_id in the dict altogether...
            node_id = nodenumber[n]
            shape = na.pop('shape', None)
            fh.write("%d \"%s\" %f %f %s " % (node_id, n,
                     float(x), float(y), shape))
            fh.write("%d \"%s\" " % (node_id, n))
            for attr in (x, y):
                if attr is not None:
                    fh.write("%f " % float(x))
            if shape:
                fh.write("%s " % shape)
            for k, v in na.iteritems():
                fh.write("%s \"%s\" " % (k, v))
            fh.write("\n")

            if clusterpath:
                if G.node[n][clusterlabel] not in clusters:
                    i += 1
                    clusters[G.node[n][clusterlabel]] = i
                clu += "%d\n" % clusters[G.node[n][clusterlabel]]

        # write edges with attributes
        if G.is_directed():
            fh.write("*arcs\n")
        else:
            fh.write("*edges\n")
        for u, v, edgedata in G.edges(data=True):
            d = edgedata.copy()
            value = d.pop(weight, 1.0)  # use 1 as default edge value
            fh.write("%d %d %f" % (nodenumber[u], nodenumber[v], float(value)))
            for k, v in d.iteritems():
                if is_string_like(v) and " " in v:
                    # add quotes to any values with a blank space
                    v = "\"%s\"" % v
                fh.write("%s %s " % (k, v))
            fh.write("\n")
    fh.close()

    if clusterpath:
        with open(clusterpath, mode="w") as fh:
            clusterpath.write(clu)


def read_pajek(path, weight='weight'):
    """Read graph in Pajek format from path.

    Returns a MultiGraph or MultiDiGraph.

    Parameters
    ----------
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.
    weight : string
        Edge attribute for edge weight

    Examples
    --------
    >>> G=nx.path_graph(4)
    >>> nx.write_pajek(G, "test.net")
    >>> G=nx.read_pajek("test.net")

    To create a Graph instead of a MultiGraph use

    >>> G1=nx.Graph(G)

    """
    with open(path) as fh:
        G = parse_pajek(fh, weight=weight)
    return G


def parse_line(l):
    # XXX This is not ideal: we instantiate a new object for each line...
    return csv.reader(l, delimiter=' ', skipinitialspace=True).next()


def parse_pajek(lines, weight='weight'):
    """Parse pajek format graph from string or iterable.

    Primarily used as a helper for read_pajek().

    See Also
    --------
    read_pajek()

    """
    G = networkx.DiGraph()
    nodelabels = {}
    nnodes = 0
    for l in lines:
        if not l.split():  # Ignore empty lines
            pass
        elif l.startswith("*"):
            if l.lower().startswith("*network"):
                try:
                    G.name = l.split()[1]
                except:
                    pass
            elif l.lower().startswith("*vertices"):
                state = "vertices"
                nnodes = int(l.split()[1])
            elif l.lower().startswith("*edges"):
                state = "edges"
            elif l.lower().startswith("*arcs"):
                state = "arcs"
            elif l.lower().startswith("*matrix"):
                raise NotImplementedError(
                    "Pajek matrix format is not yet supported.")
        elif state == "vertices":
            splitline = parse_line([l])
            node_id, label = splitline[0:2]
            if label in G.adj:
                raise networkx.NetworkXException(
                    "Node already added: " + label)
            G.add_node(label)
            nodelabels[node_id] = label
            G.node[label] = {'node_id': node_id}
            try:
                x = float(splitline[2])
                y = float(splitline[3])
                try:
                    z = float(splitline[4])
                    shape = splitline.pop(5)
                    G.node[label].update({'z': z})
                except ValueError:
                    shape = splitline[4]
                G.node[label].update({'x': x, 'y': y, 'shape': shape})
                extra_attr = zip(splitline[5::2], splitline[6::2])
            except (ValueError, IndexError):
                extra_attr = zip(splitline[2::2], splitline[3::2])
            G.node[label].update(extra_attr)
        elif state == "edges" or state == "arcs":
            if G.is_directed() and state == "edges":
                # The Pajek format supports networks with both directed and
                # edges. Since networkx does not, make this an undirected
                # network as soon as we encounter one undirected edge.
                G = networkx.Graph(G)
            splitline = l.split()
            ui, vi = splitline[0:2]
            u = nodelabels.get(ui, ui)
            v = nodelabels.get(vi, vi)
            edge_data = {}
            try:
                w = float(splitline[2])
                edge_data.update({weight: w})
                extra_attr = zip(splitline[3::2], splitline[4::2])
            except (ValueError, IndexError):
                extra_attr = zip(splitline[2::2], splitline[3::2])
            edge_data.update(extra_attr)
            G.add_edge(u, v, **edge_data)
    if nnodes != len(G):
        raise networkx.NetworkXException(
            "Wrong number of nodes in Pajek stream!")
    return G
