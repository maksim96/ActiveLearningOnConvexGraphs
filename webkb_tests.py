import json
import math

import numpy as np
import pandas as pd
import graph_tool.all as gt
import pickle

import simplicial_vertices
from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx
from closure import compute_hull, dumb_compute_closed_interval


if __name__== "__main__":
    edges_csv = np.genfromtxt("res/webwb/webkb-wisc.edges", dtype=np.int)[:,:2] - 1
    labels_csv = np.genfromtxt("res/webwb/webkb-wisc.node_labels", dtype=np.int)

    y = np.copy(labels_csv[:, 1])

    n = labels_csv.shape[0]

    np.random.seed(0)

    g = gt.Graph()

    # construct actual graph
    g.add_vertex(n)
    g.add_edge_list(edges_csv)
    weight_prop = g.new_edge_property("double", vals=1)

    spc = shortest_path_cover_logn_apx(g, weight_prop)

    pickle.dump(spc, open("res/webwb/spc.p", "wb"))

    sum = 0
    for p in spc:
        sum += max(1,math.log2(len(p)))

    print(sum)