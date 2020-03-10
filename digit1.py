import numpy as np
import pandas as pd
import graph_tool.all as gt
import scipy

import simplicial_vertices
from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx
from closure import compute_hull, dumb_compute_closed_interval

import pickle

from shortest_shortest_path_querying import local_global_strategy, label_propagation
from spcquerrying import spc_querying_naive
from twitch import are_convex


def is_convex(weighted):
    print("digit1")
    np.random.seed(0)
    X = np.genfromtxt('res/benchmark/SSL,set=' + str(1) + ',X.tab')
    # X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (np.genfromtxt('res/benchmark/SSL,set=' + str(1) + ',y.tab'))

    n = X.shape[0]
    dists = scipy.spatial.distance.cdist(X, X)
    y = y[:n]

    W = dists[:n, :n]  # np.exp(-(dists) ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    W[W > np.quantile(W, 0.004)] = np.inf
    # W2 = np.copy(W) less edges is slower strangely
    # W2[W2 <= 0.1] = 0

    weights = W[(W < np.inf) & (W > 0)].flatten()
    edges = np.array(np.where((W < np.inf) & (W > 0))).T

    np.random.seed(0)

    g = gt.Graph()

    # construct actual graph
    g.add_vertex(n)
    g.add_edge_list(edges)
    if weighted:
        weight_prop = g.new_edge_property("double", vals=weights)
    else:
        weight_prop = g.new_edge_property("double", val=1)

    comps, hist = gt.label_components(g)

    #print("simplicial=", len(simplicial_vertices(g)), "#coms=", hist.size)
    dist_map = gt.shortest_distance(g, weights=weight_prop)
    #paths = shortest_path_cover_logn_apx(g, weight_prop)
    if not weighted:
        spc = pickle.load(open("res/benchmark/spc_" + str(1) + "_q_" + str(0.004) + "_weighted_" + str(weighted) + ".p",
                     "rb"))
    else:
        spc = shortest_path_cover_logn_apx(g, weight_prop)
    labels = y

    a, b = spc_querying_naive(g, spc, labels)
    print(a)
    print(b, np.sum(b))
    print(np.sum(a == labels))
    print("len(spc)", len(spc))
    num_of_convex_paths = 0
    total_error = 0
    for p in spc:
        error = are_convex(labels[p])
        if error == 0:
            num_of_convex_paths += 1
        else:
            total_error += error

    print("#convex paths", num_of_convex_paths)
    print("total error on paths", total_error)


    return
    for c in np.unique(labels):
        print("class label", c)
        print("class size: ", np.sum(labels == c))
        cls = np.where(labels == c)[0]
        for sample_size in [5,10,20,len(cls)]:
            print("sample_size", sample_size)
            if sample_size <= 20:
                times = 5
            else:
                times = 1
            for _ in range(times):

                sample = np.random.choice(cls, sample_size, replace=False)

                hull_p = compute_hull(g, sample, dist_map=dist_map, comps=comps, hist=hist, compute_closure=False)
                print("size interval: ", np.sum(hull_p))
                print("number of correct in interval: ", np.sum(hull_p[cls]))

                hull_p = compute_hull(g, sample, dist_map=dist_map, comps=comps, hist=hist)
                print("size hull: ", np.sum(hull_p))
                print("number of correct in interval: ", np.sum(hull_p[cls]))



    print("==================================")

is_convex(False)
#is_convex(True)