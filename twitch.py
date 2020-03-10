import json

import numpy as np
import pandas as pd
import graph_tool.all as gt

import simplicial_vertices
from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx
from closure import compute_hull, dumb_compute_closed_interval

import pickle

from shortest_shortest_path_querying import local_global_strategy, label_propagation
from spcquerrying import spc_querying_naive


def are_convex(labelled_path):
    error = 0
    for c in np.unique(labelled_path):
        subpath = np.where(labelled_path==c)[0]
        if subpath.size != subpath[-1]-subpath[0]+1:
            error += subpath.size

    return error








def is_convex(dir, prefix, target_column, weighted=False):
    print(dir)
    np.random.seed(0)
    edges = np.genfromtxt(dir+prefix+'_edges.csv', skip_header=True, dtype=np.int, delimiter=',')

    df = pd.read_csv(dir+prefix+'_target.csv')#.sort_values('new_id')
    print(dir, "weighted", weighted)

    weight=1
    if weighted:
        if 'twitch' in dir:
            weight = np.zeros(edges.shape[0])
            max = df.iloc[:,1].max()
            min = df.iloc[:,1].min()
            df.iloc[:,1] =(df.iloc[:,1] - min)/(max - min)
            max = df.iloc[:, 3].max()
            min = df.iloc[:, 3].min()
            df.iloc[:, 3] = (df.iloc[:, 3] - min) / (max - min)


            for i, e in enumerate(edges):
                weight[i] = (df.iloc[e[0],1]-df.iloc[e[1],1])**2 + (df.iloc[e[0],3]-df.iloc[e[1],3])**2

        elif 'facebook' in dir:
            attributes = json.load(open('res/git/'+dir+'/facebook_features.json'))
            weight = np.zeros(edges.shape[0])
            for i, e in enumerate(edges):
                weight[i] = len(set(attributes[str(e[0])]).symmetric_difference(attributes[str(e[1])]))

    labels,_ = pd.factorize(df.iloc[:, target_column])

    g = gt.Graph(directed=False)

    g.add_edge_list(edges)

    if weighted:
        weight = g.new_edge_property("double", vals=weight)
    else:
        weight = g.new_edge_property("double", val=1)

    comps, hist = gt.label_components(g)
    #dist_map = gt.shortest_distance(g, weights=weight)
    #simple = simplicial_vertices.simplicial_vertices(g)
    #spc = shortest_path_cover_logn_apx(g, weight)
    if weighted:
        weighted_str = "_weigted_"
    else:
        weighted_str = ""
    spc = pickle.load(open(dir+'spc'+weighted_str+'.p', 'rb'))
    print(len(spc))
    a,b = spc_querying_naive(g, spc, labels)
    print(a)
    print(b, np.sum(b))
    print(np.sum(a==labels))



    print("len(spc)", len(spc))
    num_of_convex_paths=0
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

    dist_map = gt.shortest_distance(g, weights=weight)
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
if __name__ == "__main__":
    is_convex("res/git/twitch/PTBR/","PTBR",2)
    is_convex("res/git/twitch/PTBR/","PTBR",2,True)