import numpy as np
import graph_tool as gt
import scipy

import closure
from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx
from simplicial_vertices import simplicial_vertices
from spcquerrying import spc_querying_naive
import pickle

def is_convex(dataset,weighted,qs):
    X = np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',X.tab')
    #X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',y.tab'))
    print("========================================================================")
    print(dataset)
    n = X.shape[0]
    for  q in qs:
        print("q=",q)
        dists = scipy.spatial.distance.cdist(X, X)
        y = y[:n]

        W = dists[:n,:n]#np.exp(-(dists) ** 2 / (2 * sigma ** 2))
        np.fill_diagonal(W, 0)
        W[W > np.quantile(W,q)] = np.inf
        # W2 = np.copy(W) less edges is slower strangely
        # W2[W2 <= 0.1] = 0


        weights = W[(W<np.inf) & (W>0)].flatten()
        edges = np.array(np.where((W<np.inf) & (W>0))).T

        np.random.seed(0)

        g = gt.Graph()

        # construct actual graph
        g.add_vertex(n)
        g.add_edge_list(edges)
        if weighted:
            weight_prop = g.new_edge_property("double", vals=weights)
        else:
            weight_prop = g.new_edge_property("double", val=1)

        comps,hist = gt.topology.label_components(g)

        print("simplicial=",len(simplicial_vertices(g)), "#coms=",hist.size)

        paths = shortest_path_cover_logn_apx(g, weight_prop)

        pickle.dump(paths, open("res/benchmark/spc_"+str(dataset)+"_q_"+str(q)+"_weighted_"+str(weighted)+".p","wb"))

        sum = 0
        for i in paths:
            sum += max(1,np.ceil(np.log2(len(i))))

        print("|S|=", len(paths))
        print("#queries<=", sum, "%:", sum / n)

        continue
        pos = list(np.arange(n)[y > 0])[:n_prime]
        neg = list(np.arange(n)[y <= 0])[:n_prime]

        print(n,pos,neg)
        print("p",len(pos))
        print("n",len(neg))

        pos_hull = closure.compute_hull(g,pos, weight_prop,comps,hist)
        print(np.sum(pos_hull))
        neg_hull = closure.compute_hull(g, neg, weight_prop,comps,hist)
        print(np.sum(neg_hull))
        print(len(set(np.where(pos_hull)[0]).intersection(set(np.where(neg_hull)[0])))/n)


is_convex(1, False, [0.004, 0.02])
is_convex(2, False,[0.05, 0.4])
is_convex(3, False,[0.03, 0.2])
is_convex(4, False,[0.05, 0.4])
is_convex(5, False,[0.03, 0.2])
is_convex(7, False,[0.03, 0.2])

is_convex(1, True,[0.004, 0.02])
is_convex(2, True,[0.05, 0.4])
is_convex(3, True,[0.03, 0.2])
is_convex(4, True,[0.05, 0.4])
is_convex(5, True,[0.03, 0.2])
is_convex(7, True,[0.03, 0.2])


