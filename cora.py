import numpy as np
import pandas as pd
import graph_tool.all as gt

import simplicial_vertices
from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx
from closure import compute_hull, dumb_compute_closed_interval

import pickle

from shortest_shortest_path_querying import local_global_strategy, label_propagation
from spcquerrying import spc_querying_naive
from twitch import are_convex


def is_convex(directed):
    print("cora")
    np.random.seed(0)
    edges = np.genfromtxt('res/cora/cora.edges', dtype=np.int, delimiter=',')[:,:2] - 1

    labels = np.genfromtxt('res/cora/cora.node_labels', dtype=np.int, delimiter=',')[:,1]

    g = gt.Graph(directed=directed)

    g.add_edge_list(edges)

    weight = g.new_edge_property("double", val=1)

    comps, hist = gt.label_components(g)
    print(hist)
    dist_map = gt.shortest_distance(g, weights=weight)#, weights=weight)
    simple = simplicial_vertices.simplicial_vertices(g)

    print("n=",g.num_vertices(), "s=", len(simple))

    spc =pickle.load(open("res/cora/spc_"+str(directed)+".p", "rb")) #shortest_path_cover_logn_apx(g, weight)

    a,b = spc_querying_naive(g, spc, labels)
    print(a)
    print(b, np.sum(b))
    print(np.sum(a==labels))
    return

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
    pickle.dump(spc, open("res/cora/spc_"+str(directed)+".p", "wb"))

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

is_convex(True)
is_convex(False)