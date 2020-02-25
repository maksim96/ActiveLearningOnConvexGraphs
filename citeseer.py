import numpy as np
import pandas as pd
import graph_tool.all as gt
import sagemath

import simplicial_vertices
from closure import compute_hull, dumb_compute_closed_interval

from scipy.spatial.distance import sqeuclidean, pdist
def is_convex():
    print("citeseer")
    print("unweighted")
    np.random.seed(0)

    attributes_df = pd.read_csv('res/citeseer/citeseer.content', sep="\t", header=None, dtype=np.str)
    features = attributes_df.iloc[:,1:-1].to_numpy(dtype=np.int)
    labels,_ = pd.factorize(attributes_df.iloc[:,-1])
    new_ids, old_ids = pd.factorize(attributes_df.iloc[:, 0])

    edges_df = pd.read_csv('res/citeseer/citeseer.cites', sep="\t", header=None, dtype=np.str)
    edges_df = edges_df[edges_df.iloc[:, 0].apply(lambda x: x in old_ids)]
    edges_df = edges_df[edges_df.iloc[:, 1].apply(lambda x: x in old_ids)]
    renamed = edges_df.replace(old_ids, new_ids)
    edges = renamed.to_numpy(dtype=np.int)
    edges = np.fliplr(edges)
    g = gt.Graph(directed=False)

    g.add_edge_list(edges)

    weight = None#np.sum(np.abs(features[edges[:,0]] - features[edges[:,1]]), axis=1)

    weight_prop = None#g.new_edge_property("int", val=1)

    #weight = g.new_edge_property("double", vals=weight)

    comps, hist = gt.label_components(g)
    print(hist)
    dist_map = gt.shortest_distance(g, weights=weight_prop)#, weights=weight)
    simple = simplicial_vertices.simplicial_vertices(g)

    print("n=",g.num_vertices(), "s=", len(simple))




    intersection_0 = []
    intersection_1 = []
    intersection_2 = []
    intersection_3 = []
    for c in np.unique(labels):
        print(c)
        cls = np.where(labels==c)[0][:5]
        hull_p = compute_hull(g, cls, weight_prop,dist_map=dist_map, comps=comps, hist=hist, compute_closure=False)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        hull_p = compute_hull(g, cls, weight_prop,dist_map=dist_map, comps=comps, hist=hist)
        intersection_0.append(hull_p)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        cls = np.where(labels == c)[0][:10]
        hull_p = compute_hull(g, cls, weight_prop,dist_map=dist_map, comps=comps, hist=hist, compute_closure=False)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        hull_p = compute_hull(g, cls, weight_prop,dist_map=dist_map, comps=comps, hist=hist)
        intersection_1.append(hull_p)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        cls = np.where(labels == c)[0][:50]
        hull_p = compute_hull(g, cls,weight_prop, dist_map=dist_map,comps=comps, hist=hist,compute_closure=False)
        print(np.sum(hull_p),np.sum(hull_p)/g.num_vertices())
        hull_p = compute_hull(g, cls,weight_prop, dist_map=dist_map, comps=comps, hist=hist)
        intersection_2.append(hull_p)
        print(np.sum(hull_p),np.sum(hull_p) / g.num_vertices())
        cls = np.where(labels == c)[0]
        hull_p = compute_hull(g, cls, weight_prop,dist_map=dist_map, comps=comps, hist=hist, compute_closure=False)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        hull_p = compute_hull(g, cls, weight_prop,dist_map=dist_map, comps=comps, hist=hist)
        intersection_3.append(hull_p)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        print("==========")

    print(np.sum(intersection_0[0] & intersection_0[1])/g.num_vertices())
    print(np.sum(intersection_1[0] & intersection_1[1]) / g.num_vertices())
    print(np.sum(intersection_2[0] & intersection_2[1]) / g.num_vertices())
    print(np.sum(intersection_3[0] & intersection_3[1]) / g.num_vertices())


    print("==================================")



is_convex()