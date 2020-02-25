import numpy as np
import pandas as pd
import graph_tool.all as gt

import simplicial_vertices
from closure import compute_hull, dumb_compute_closed_interval


def is_convex():
    print("cora")
    np.random.seed(0)
    edges = np.genfromtxt('res/cora/cora.edges', dtype=np.int, delimiter=',')[:,:2] - 1

    labels = np.genfromtxt('res/cora/cora.node_labels', dtype=np.int, delimiter=',')[:,1]

    g = gt.Graph()

    g.add_edge_list(edges)

    #weight = g.new_edge_property("double", vals=weight)

    comps, hist = None,None# = gt.label_components(g)
    print(hist)
    dist_map = gt.shortest_distance(g)#, weights=weight)
    simple = simplicial_vertices.simplicial_vertices(g)

    print("n=",g.num_vertices(), "s=", len(simple))

    intersection_0 = []
    intersection_1 = []
    intersection_2 = []
    intersection_3 = []
    intersection_4 = []
    for c in np.unique(labels):
        print(c)
        print(np.sum(labels == c))
        cls = np.where(labels==c)[0][:5]
        hull_p = compute_hull(g, cls, dist_map=dist_map, comps=None, hist=hist, compute_closure=False)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        hull_p = compute_hull(g, cls, dist_map=dist_map, comps=comps, hist=hist)
        intersection_0.append(hull_p)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        cls = np.where(labels == c)[0][:10]
        hull_p = compute_hull(g, cls, dist_map=dist_map, comps=comps, hist=hist, compute_closure=False)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        hull_p = compute_hull(g, cls, dist_map=dist_map, comps=comps, hist=hist)
        intersection_1.append(hull_p)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        cls = np.where(labels == c)[0][:50]
        hull_p = compute_hull(g, cls, dist_map=dist_map,comps=comps, hist=hist,compute_closure=False)
        print(np.sum(hull_p),np.sum(hull_p)/g.num_vertices())
        hull_p = compute_hull(g, cls, dist_map=dist_map, comps=comps, hist=hist)
        intersection_2.append(hull_p)
        print(np.sum(hull_p),np.sum(hull_p) / g.num_vertices())
        cls = np.where(labels == c)[0][:100]
        hull_p = compute_hull(g, cls, dist_map=dist_map, comps=comps, hist=hist, compute_closure=False)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        hull_p = compute_hull(g, cls, dist_map=dist_map, comps=comps, hist=hist)
        intersection_3.append(hull_p)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        cls = np.where(labels == c)[0]
        hull_p = compute_hull(g, cls, dist_map=dist_map, comps=comps, hist=hist, compute_closure=False)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        hull_p = compute_hull(g, cls, dist_map=dist_map, comps=comps, hist=hist)
        intersection_4.append(hull_p)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        print("==========")

    print(np.sum(intersection_0[0] & intersection_0[1])/g.num_vertices())
    print(np.sum(intersection_1[0] & intersection_1[1]) / g.num_vertices())
    print(np.sum(intersection_2[0] & intersection_2[1]) / g.num_vertices())
    print(np.sum(intersection_3[0] & intersection_3[1]) / g.num_vertices())
    print(np.sum(intersection_4[0] & intersection_4[1]) / g.num_vertices())


    print("==================================")



is_convex()