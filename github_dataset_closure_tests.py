import numpy as np
import pandas as pd
import graph_tool.all as gt

import simplicial_vertices
from closure import compute_hull, dumb_compute_closed_interval


def is_convex(dir, prefix, target_column):
    print(dir)
    np.random.seed(0)
    edges = np.genfromtxt(dir+prefix+'_edges.csv', skip_header=True, dtype=np.int, delimiter=',')

    df = pd.read_csv(dir+prefix+'_target.csv')#.sort_values('new_id')


    weight=None
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

    labels,_ = pd.factorize(df.iloc[:, target_column])

    g = gt.Graph(directed=False)

    g.add_edge_list(edges)

    #weight = g.new_edge_property("double", vals=weight)

    comps, hist = gt.label_components(g)
    dist_map = gt.shortest_distance(g)#, weights=weight)
    simple = simplicial_vertices.simplicial_vertices(g)

    print("n=",g.num_vertices(), "s=", len(simple))

    intersection_0 = []
    intersection_1 = []
    intersection_2 = []
    intersection_3 = []
    for c in np.unique(labels):
        print(c)
        cls = np.where(labels==c)[0][:5]
        hull_p = compute_hull(g, cls, weight,dist_map=dist_map, comps=comps, hist=hist, compute_closure=False)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        hull_p = compute_hull(g, cls, weight,dist_map=dist_map, comps=comps, hist=hist)
        intersection_0.append(hull_p)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        cls = np.where(labels == c)[0][:10]
        hull_p = compute_hull(g, cls, weight,dist_map=dist_map, comps=comps, hist=hist, compute_closure=False)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        hull_p = compute_hull(g, cls, weight,dist_map=dist_map, comps=comps, hist=hist)
        intersection_1.append(hull_p)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        cls = np.where(labels == c)[0][:50]
        hull_p = compute_hull(g, cls, weight,dist_map=dist_map,comps=comps, hist=hist,compute_closure=False)
        print(np.sum(hull_p),np.sum(hull_p)/g.num_vertices())
        hull_p = compute_hull(g, cls, weight,dist_map=dist_map, comps=comps, hist=hist)
        intersection_2.append(hull_p)
        print(np.sum(hull_p),np.sum(hull_p) / g.num_vertices())
        cls = np.where(labels == c)[0]
        hull_p = compute_hull(g, cls, weight,dist_map=dist_map, comps=comps, hist=hist, compute_closure=False)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        hull_p = compute_hull(g, cls,weight, dist_map=dist_map, comps=comps, hist=hist)
        intersection_3.append(hull_p)
        print(np.sum(hull_p), np.sum(hull_p) / g.num_vertices())
        print("==========")

    print(np.sum(intersection_0[0] & intersection_0[1])/g.num_vertices())
    print(np.sum(intersection_1[0] & intersection_1[1]) / g.num_vertices())
    print(np.sum(intersection_2[0] & intersection_2[1]) / g.num_vertices())
    print(np.sum(intersection_3[0] & intersection_3[1]) / g.num_vertices())


    print("==================================")



'''is_convex("res/git/twitch/PTBR/","PTBR",2)
is_convex("res/git/twitch/PTBR/","PTBR",4)
is_convex("res/git/twitch/RU/","RU",2)
is_convex("res/git/twitch/RU/","RU",4)
is_convex("res/git/twitch/DE/","DE",2)
is_convex("res/git/twitch/DE/","DE",4)
is_convex("res/git/twitch/ENGB/","ENGB",2)
is_convex("res/git/twitch/ENGB/","ENGB",4)
is_convex("res/git/twitch/ES/","ES",2)
is_convex("res/git/twitch/ES/","ES",4)
is_convex("res/git/twitch/FR/","FR",2)
is_convex("res/git/twitch/FR/","FR",4)'''
#is_convex("res/git/facebook_large/","facebook",3)
is_convex("res/git/git_web_ml/","git",2)