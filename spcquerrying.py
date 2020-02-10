import numpy as np
import graph_tool as gt
import scipy

import closure
from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx
from simplicial_vertices import simplicial_vertices


def binarySearch(arr, l, r, left_starting, known_labels):
    label_budget = 0
    while l <= r:

        mid = l + (r - l) // 2

        #if mid == l:
        #    return mid, label_budget

        # Check if x is present at mid
        if known_labels[mid] < 0:
            label_budget += 1
            known_labels[mid] = arr[mid]

        if arr[mid] == left_starting:
            l = mid + 1
        else:
            r = mid - 1

    # If we reach here, then the element was not present
    return l + (r - l) // 2,label_budget

def spc_querying_naive(g : gt.Graph, paths, y, weight, csv):
    known_labels = -np.ones(g.num_vertices())
    known_labels[(csv > 1) & (csv < 5)] = 0
    budget = 0
    for path in paths:
        if known_labels[path[0]] < 0:
            budget += 1
            known_labels[path[0]] = y[path[0]]
        if known_labels[path[-1]] < 0:
            budget += 1
            known_labels[path[-1]] = y[path[-1]]

        if known_labels[path[0]] ==  known_labels[path[-1]]:
            known_labels[path] = known_labels[path[0]]
        else:
            mid, label_budget = binarySearch(y[path], 0, len(path)-1, known_labels[path[0]], known_labels[path])
            budget += label_budget
            known_labels[path[0:mid+1]] = known_labels[path[0]]
            known_labels[path[mid+1:]] = known_labels[path[-1]]

        #p =closure.compute_hull(g, np.where(known_labels>0)[0], weight)
        #n = closure.compute_hull(g, np.where(known_labels==0)[0], weight)

        #known_labels[p] = 1
        #known_labels[n] = 0



    return known_labels, budget


def     is_convex(dataset,q,weighted=True):
    X = np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',X.tab')
    #X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',y.tab'))

    n = 1500
    dists = scipy.spatial.distance.cdist(X, X)
    y = y[:n]
    y = (y-np.min(y))//(np.max(y)-np.min(y))
    #q = 0.04
    W = dists[:n,:n]#np.exp(-(dists) ** 2 / (2 * sigma ** 2))
    q = np.quantile(W, q)
    W[W > q] = np.inf
    # W2 = np.copy(W) less edges is slower strangely
    if not weighted:
        W[W <= q] = 1
    np.fill_diagonal(W, 0)

    weights = W[(W<np.inf) & (W>0)].flatten()
    edges = np.array(np.where((W<np.inf) & (W>0))).T

    print("e",len(edges))
    #return

    np.random.seed(0)

    g = gt.Graph()

    # construct actual graph
    g.add_vertex(n)
    g.add_edge_list(edges)
    weight_prop = g.new_edge_property("double", val=1)


    comps,hist = gt.topology.label_components(g)

    simpl = simplicial_vertices(g)

    print(len(simpl), np.sum(closure.compute_hull(g, simpl, weight_prop)>0))
    return
    paths = shortest_path_cover_logn_apx(g, weight_prop)


    sum = 0
    for i in paths:
        sum += np.ceil(np.log2(len(i)))

    print("|S|=", len(paths))
    print("#queries<=", sum, "%:", sum / n)


    pos = list(np.arange(n)[y > 0])[:n]
    neg = list(np.arange(n)[y <= 0])[:n]

    print(n,pos,neg)
    print("p",len(pos))
    print("n",len(neg))

    #pos_hull = closure.compute_hull(g,pos, weight_prop,comps,hist)
    #print(np.sum(pos_hull))
    #neg_hull = closure.compute_hull(g, neg, weight_prop,comps,hist)
    #print(np.sum(neg_hull))
    #print(len(set(np.where(pos_hull)[0]).intersection(set(np.where(neg_hull)[0])))/n)

    print("===============================================================")
    known_labels, budget = spc_querying_naive(g, paths,y, weight_prop)
    print(np.sum(np.abs(known_labels-y)/n))
    print(budget)

print("========================================================")
print("is_convex(1,0.004)")
is_convex(1,0.004)
print("========================================================")
print("is_convex(1,0.005)")
is_convex(1,0.005)
print("========================================================")
print("is_convex(2,0.05)")
is_convex(2, 0.05)
print("========================================================")
print("is_convex(2,0.06)")
is_convex(2,0.06)
print("========================================================")
print("is_convex(3,0.03)")
is_convex(3, 0.03)
print("========================================================")
print("is_convex(3,0.04)")
is_convex(3,0.04)
print("========================================================")
print("========================================================")
print("is_convex(1,0.004,False)")
is_convex(1,0.004,False)
print("========================================================")
print("is_convex(1,0.005,False)")
is_convex(1,0.005,False)
print("========================================================")
print("is_convex(2,0.05,False)")
is_convex(2, 0.05,False)
print("========================================================")
print("is_convex(2,0.06,False)")
is_convex(2,0.06,False)
print("========================================================")
print("is_convex(3,0.03,False)")
is_convex(3, 0.03,False)
print("========================================================")
print("is_convex(3,0.04,False)")
is_convex(3,0.04,False)
print("========================================================")
'''
for dataset in [1,2,3]:
    for q in np.arange(1,101)/1000:
        X = np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',X.tab')
        # X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        y = (np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',y.tab'))

        n = 1500
        dists = scipy.spatial.distance.cdist(X, X)
        y = y[:n]
        y = (y - np.min(y)) // (np.max(y) - np.min(y))
        #q = 0.005
        W = dists[:n, :n]  # np.exp(-(dists) ** 2 / (2 * sigma ** 2))
        #q = np.quantile(W, q)
        W[W > np.quantile(W, q)] = np.inf
        # W2 = np.copy(W) less edges is slower strangely
        #W[W <= q] = 1
        np.fill_diagonal(W, 0)

        weights = W[(W < np.inf) & (W > 0)].flatten()
        edges = np.array(np.where((W < np.inf) & (W > 0))).T

        np.random.seed(0)

        g = gt.Graph()

        # construct actual graph
        g.add_vertex(n)
        g.add_edge_list(edges)
        weight_prop = g.new_edge_property("double", vals=weights)

        comps, hist = gt.topology.label_components(g)

        s = len(simplicial_vertices(g))
        print(dataset, q, s, np.sum(hist>0))

        if s == 0 and np.sum(hist>0) == 1:
            break
'''