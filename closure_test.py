import numpy as np
import graph_tool as gt
import scipy

import closure
from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx
from simplicial_vertices import simplicial_vertices
from spcquerrying import spc_querying_naive


def is_convex(dataset):
    X = np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',X.tab')
    #X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',y.tab'))

    n = 300
    for n_prime in [n]:
        print("================================")
        print("n_prime=",n_prime)
        for q in [0.001,0.002,0.005,0.01,0.02,0.05]:
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
            weight_prop = g.new_edge_property("double", vals=weights)


            comps,hist = gt.topology.label_components(g)

            print(len(simplicial_vertices(g)))
            continue
            paths = shortest_path_cover_logn_apx(g, weight_prop)

            sum = 0
            for i in paths:
                sum += np.ceil(np.log2(len(i)))

            print("|S|=", len(paths))
            print("#queries<=", sum, "%:", sum / n)


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


#is_convex(1)
#is_convex(2)
#is_convex(3)


edges_csv = np.genfromtxt("res/webwb/webkb-wisc.edges",dtype=np.int)
labels_csv = np.genfromtxt("res/webwb/webkb-wisc.node_labels",dtype=np.int)


y = np.copy(labels_csv[:,1])

y[y<5] = 0
y[y==5] = 1

n = labels_csv.shape[0]

W = np.ones((labels_csv.shape[0],labels_csv.shape[0]))*np.inf

for e in edges_csv:
    W[e[0]-1,e[1]-1] = e[2]


weights = W[(W<np.inf) & (W>0)].flatten()
edges = np.array(np.where((W<np.inf) & (W>0))).T

np.random.seed(0)

g = gt.Graph()

# construct actual graph
g.add_vertex(n)
g.add_edge_list(edges)
weight_prop = g.new_edge_property("double", vals=weights)


comps,hist = gt.topology.label_components(g)

print(len(simplicial_vertices(g)))

paths = shortest_path_cover_logn_apx(g, weight_prop)

sum = 0
for i in paths:
    sum += np.ceil(np.log2(len(i)))

print("|S|=", len(paths))
print("#queries<=", sum, "%:", sum / n)


pos = list(np.arange(n)[y > 0])
neg = list(np.arange(n)[y <= 0])

print(n,pos,neg)
print("p",len(pos))
print("n",len(neg))

#pos_hull = closure.compute_hull(g,pos, weight_prop,None,comps,hist)
#print(np.sum(pos_hull))
#neg_hull = closure.compute_hull(g, neg, weight_prop,None,comps,hist)
#print(np.sum(neg_hull))
#print(len(set(np.where(pos_hull)[0]).intersection(set(np.where(neg_hull)[0])))/n)

print("===============================================================")
known_labels, budget = spc_querying_naive(g, paths,y)#, labels_csv[:,1])
idx = (labels_csv[:,1] == 1) | (labels_csv[:,1] == 5)
print(np.sum(np.abs(known_labels[idx]-y[idx])/idx.size))
print(budget)