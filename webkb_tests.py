import json
import math

import numpy as np
import pandas as pd
import graph_tool.all as gt
import pickle

import simplicial_vertices
from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx
from closure import compute_hull, dumb_compute_closed_interval
from spcquerrying import spc_querying_with_shadow, spc_querying_naive, spc_querying_with_closure

if __name__== "__main__":
    edges_csv = np.genfromtxt("res/webwb/webkb-wisc.edges", dtype=np.int)[:,:2] - 1
    labels_csv = np.genfromtxt("res/webwb/webkb-wisc.node_labels", dtype=np.int)

    y = np.copy(labels_csv[:, 1])

    labels = np.unique(y)
    y[y != labels[4]] = labels[0]

    n = labels_csv.shape[0]
    print(n)
    np.random.seed(0)

    g = gt.Graph(directed=False)

    # construct actual graph
    g.add_vertex(n)
    g.add_edge_list(edges_csv)
    weight_prop = g.new_edge_property("double", vals=1)

    #spc = shortest_path_cover_logn_apx(g, weight_prop)

    spc = pickle.load( open("res/webwb/spc.p", "rb"))

    print(len(spc))

    sum = 0
    for p in spc:
        sum += max(2,math.ceil(math.log2(len(p))))

    print(sum)

    a,b = spc_querying_naive(g, spc, y)
    print(a)
    print(y)

    print(np.sum(a==y))

    print(b)
    print(np.sum(b))