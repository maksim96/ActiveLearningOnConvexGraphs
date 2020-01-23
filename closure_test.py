import numpy as np
import graph_tool as gt
import scipy

import closure
from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx


def is_convex(dataset):
    X = np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',X.tab')
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',y.tab'))

    n = 1500
    for n_prime in [60]:
        print("================================")
        print("n_prime=",n_prime)
        for q in [0.005,0.01,0.02,0.05]:
            print("q=",q)
            dists = scipy.spatial.distance.cdist(X, X)
            y = y

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
            weight_prop = g.new_edge_property("double", vals=weights)


            comps,hist = gt.topology.label_components(g)

            paths = shortest_path_cover_logn_apx(g, weight_prop)

            sum = 0
            for i in paths:
                sum += np.ceil(np.log2(len(i)))

            print("|S|=", len(paths))
            print("#queries<=", sum, "%:", sum / n)


            #pos = list(np.arange(n)[y > 0])[:n_prime]
            #neg = list(np.arange(n)[y <= 0])[:n_prime]

            #print(n,pos,neg)
            #print("p",len(pos))
            #print("n",len(neg))

            #pos_hull = closure.compute_hull(g,pos, weight_prop,comps,hist)
            #print(np.sum(pos_hull))
            #neg_hull = closure.compute_hull(g, neg, weight_prop,comps,hist)
            #print(np.sum(neg_hull))
            #print(len(set(np.where(pos_hull)[0]).intersection(set(np.where(neg_hull)[0])))/n)


is_convex(1)
#is_convex(2)
#is_convex(3)