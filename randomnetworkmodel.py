import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from numpy.random import random
from numpy.linalg import norm
from matplotlib import pyplot as plt


def get_rwpoints(n, c, r):
    """
    Return the positions of n * c points, so that there are n points randomly placed around c centers in a circle
    with distance r. The centers are placed randomly on a circle around the origin with a radius of sqrt(c).
    :param n: number of nodes around a center
    :param c: number of centers
    :param r: radius of circle around center
    :return: array of shape (n * c, 2) containing the positions of the points
    """
    rc = np.sqrt(c)  # radius for centers
    phic = 2 * np.pi * random(c)
    xc, yc = rc * np.cos(phic), rc * np.sin(phic)
    phi = 2 * np.pi * random((c, n))  # distribution of nodes

    x, y = xc[..., None] + r * np.cos(phi), yc[..., None] + r * np.sin(phi)
    return np.array((x.flatten(), y.flatten())).T


def get_Gabriel_graph(points):
    """
    Create a Gabriel graph from a given number of points
    :param points: ndarray of shape (n * 2, ndim) containing the positions of points
    :return: networkx Graph instance and dictionary of positions
    """
    tri = Delaunay(points)
    indptr, indices = tri.vertex_neighbor_vertices
    pos = points
    posdict = {i: pos[i] for i in range(len(pos))}
    G = nx.Graph()
    for i in range(len(points)):
        nbs = indices[indptr[i]:indptr[i + 1]]
        bunch = [(i, nb, norm(pos[i] - pos[nb])) for nb in nbs]
        G.add_weighted_edges_from(bunch)

    nx.set_node_attributes(G, posdict, name='pos')
    lengths = nx.get_edge_attributes(G, 'weight')
    for u, v in G.edges():
        mp = 0.5 * (pos[u] + pos[v])
        l = lengths[(u, v)]
        for w in G:
            if w != u and w != v:
                if norm(mp - pos[w]) < l / 2:  # could be optimized by vectorization
                    G.remove_edge(u, v)
                    break

    return G


def get_randomnetwork(n, c, r):
    return get_Gabriel_graph(get_rwpoints(n, c, r))


def betweenness_impact(G, weight='weight'):
    """
    :param G:
    :return: dict of betweenness centrality impacts
    """
    bc = list(nx.edge_betweenness_centrality(G, weight=weight).values())
    meanbc = sum(bc) / len(bc)
    bci = {}
    Gtemp = G.copy(as_view=False)
    for u, v, d in G.edges(data=True):
        Gtemp.remove_edge(u, v)
        # edges = [(x, y, d['weight']) for x, y, d in G.edges(data=True) if (x, y) != (u, v)]
        # Gtemp.add_weighted_edges_from(edges)
        gGe = np.mean(list(nx.edge_betweenness_centrality(Gtemp, weight=weight).values()))
        delta1 = 1 - gGe / meanbc

        bci.update({(u, v): delta1})
        Gtemp.add_edge(u, v, data=d)

    return bc, bci

def edge_impact(G, weight='weight'):
    """
    :param G:
    :return: dict of betweenness centrality impacts
    """
    bc = list(nx.edge_betweenness_centrality(G, weight=weight).values())
    sp = nx.average_shortest_path_length(G, weight=weight)
    meanbc = sum(bc) / len(bc)
    bci = {}
    di = {}
    Gtemp = G.copy(as_view=False)
    for u, v, d in G.edges(data=True):
        Gtemp.remove_edge(u, v)
        # edges = [(x, y, d['weight']) for x, y, d in G.edges(data=True) if (x, y) != (u, v)]
        # Gtemp.add_weighted_edges_from(edges)
        gGe = np.mean(list(nx.edge_betweenness_centrality(Gtemp, weight=weight).values()))
        delta1 = 1 - gGe / meanbc
        try:
            spe = nx.average_shortest_path_length(Gtemp, weight=weight)
            delta2 = spe - sp
        except:
            delta2 = np.nan
        bci.update({(u, v): delta1})
        di.update({(u, v): delta2})
        Gtemp.add_edge(u, v, data=d)

    return bc, bci, di

def distance_impact(G, weight='weight'):
    """
    :param G:
    :return: dict of betweenness centrality impacts
    """

    sp = nx.average_shortest_path_length(G, weight='weight')
    bci = {}
    Gtemp = G.copy(as_view=False)
    for u, v, d in G.edges(data=True):
        Gtemp.remove_edge(u, v)
        # edges = [(x, y, d['weight']) for x, y, d in G.edges(data=True) if (x, y) != (u, v)]
        # Gtemp.add_weighted_edges_from(edges)
        try:
            spe = nx.average_shortest_path_length(Gtemp, weight=weight)
            delta = spe - sp
        except:
            delta = np.nan
        bci.update({(u, v): delta})
        Gtemp.add_edge(u, v, data=d)

    return bci

if __name__ == '__main__':
    G = get_randomnetwork(10, 2, .7)
    bc, bci, di = edge_impact(G)
    print(di.values())
    # print(di.values())
    fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, subplot_kw={'aspect': 'equal'})
    ed = nx.draw_networkx_edges(G, pos=nx.get_node_attributes(G, 'pos'), edge_color=bci.values(), ax=ax[0],
                     edge_cmap=plt.get_cmap('coolwarm'))
    plt.colorbar(ed, ax=ax[0], location='bottom')
    ed = nx.draw_networkx_edges(G, pos=nx.get_node_attributes(G, 'pos'), edge_color=di.values(), ax=ax[1],
                     edge_cmap=plt.get_cmap('cool'))
    plt.colorbar(ed, ax=ax[1], location='bottom')
    plt.show()
