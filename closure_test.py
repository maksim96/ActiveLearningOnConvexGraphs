import numpy as np
import graph_tool as gt
import scipy

import closure


def is_convex(dataset):
    X = np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',X.tab')
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',y.tab'))

    n = 200
    for sigma in [1,2,5,10,20]:
        print("================================")
        print("sigma=",sigma)
        for q in [0.1,0.2,0.3,0.5,0.9]:
            print("q=",q)
            dists = scipy.spatial.distance.cdist(X[:n,:n], X[:n,:n])
            y = y[:n]

            # average_knn_dist = np.average(np.sort(X)[:,:5])
            dists_without_diagonal = np.reshape(dists[~np.eye(dists.shape[0], dtype=bool)],
                                                (dists.shape[0], dists.shape[1] - 1))
            #sigma = np.average(np.sort(dists_without_diagonal)[5]) / 3

            W = dists#np.exp(-(dists) ** 2 / (2 * sigma ** 2))
            np.fill_diagonal(W, 0)
            W[W > np.quantile(W,q)] = np.inf
            # W2 = np.copy(W) less edges is slower strangely
            # W2[W2 <= 0.1] = 0


            weights = W.flatten()
            a, b = W.nonzero()
            edges = np.transpose(np.vstack((a, b, weights[weights != 0])))

            np.random.seed(0)

            g = gt.Graph()

            # construct actual graph
            g.add_vertex(W.shape[0])
            weight = g.new_edge_property("long double")
            eprops = [weight]
            g.add_edge_list(edges, eprops=eprops)

            pos = list(np.arange(n)[y > 0])
            neg = list(np.arange(n)[y <= 0])

            #print(n,pos,neg)
            #print("p",len(pos))
            #print("n",len(neg))

            pos_hull = closure.compute_hull(g,pos,weight)
            print(np.sum(pos_hull))
            neg_hull = closure.compute_hull(g, neg, weight)
            print(np.sum(neg_hull))
            print(len(set(np.where(pos_hull)[0]).intersection(set(np.where(neg_hull)[0])))/n)


is_convex(3)