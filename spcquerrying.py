import os
from queue import Queue

import numpy as np
import graph_tool.all as gt
import scipy

import closure
from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx
from closure import compute_hull
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

def spc_querying_naive(g : gt.Graph, paths, y, trust_own_predictions=True):
    '''

    :param g:
    :param paths: list of paths
    :param y: ground truth
    :param weight:
    :return:
    '''
    known_labels = -np.ones(g.num_vertices())
    budget = 0
    for path in paths:
        if not trust_own_predictions or known_labels[path[0]] < 0:
            budget += 1
            known_labels[path[0]] = y[path[0]]
        if not trust_own_predictions or known_labels[path[-1]] < 0:
            budget += 1
            known_labels[path[-1]] = y[path[-1]]

        if known_labels[path[0]] == known_labels[path[-1]]:
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


def spc_querying_with_closure(g: gt.Graph, paths, weights, y):
    '''

    :param g:
    :param paths: list of paths
    :param y: ground truth
    :param weight:
    :return:
    '''
    np.random.seed(55)
    #these two lines make repetitive closure computation a lot faster
    dist_map = gt.shortest_distance(g, weights=weights)
    comps, hist = gt.label_components(g)

    known_labels = -np.ones(g.num_vertices())
    num_of_known_labels = 0
    budget = 0

    pos_value, neg_value = np.unique(y)

    next_candidate_queues = [Queue() for _ in paths]
    left = np.zeros(len(paths), dtype=np.int)
    right = np.array([len(p)-1 for p in paths], dtype=np.int)
    queue_idxs = list(range(len(paths)))

    n = g.num_vertices()

    for i,path in enumerate(paths):
        next_candidate_queues[i].put(0)
        if len(path) > 1:
            next_candidate_queues[i].put(len(path)-1)

    starting_idx = np.random.choice(np.where(right>0)[0])
    starting_path = paths[starting_idx]

    budget += 2
    l = next_candidate_queues[starting_idx].get()
    r = next_candidate_queues[starting_idx].get()
    known_labels[starting_path[l]] = y[starting_path[l]]
    known_labels[starting_path[r]] = y[starting_path[r]]

    if known_labels[starting_path[0]] == known_labels[starting_path[-1]]:
        #color the hull of the path in the color of the endpoints
        path_closure = np.where(compute_hull(g, starting_path, weights, dist_map, comps, hist))[0]
        known_labels[path_closure] = known_labels[starting_path[0]]
        num_of_known_labels = len(path_closure)
        del queue_idxs[starting_idx]
    else:
        if (len(starting_path)>=3):
            next_candidate_queues[starting_idx].put(l + (r - l)//2)
        else:
            del queue_idxs[starting_idx]
        num_of_known_labels = 2

    pos = np.where(known_labels==pos_value)[0]
    neg = np.where(known_labels==neg_value)[0]

    candidates = np.zeros(len(paths), dtype=np.int)

    candidates[queue_idxs] = [next_candidate_queues[queue_idx].get() for queue_idx in queue_idxs] #this is always relative to the path

    candidate_pos_hulls = np.zeros((len(paths),n), dtype=np.bool)
    if len(pos) > 0:
        candidate_pos_hulls[queue_idxs] = [compute_hull(g, np.append(pos, paths[idx][candidates[idx]]), weights, dist_map, comps, hist) for idx in queue_idxs]
    else:
        for idx in queue_idxs:
            candidate_pos_hulls[idx][paths[idx][candidates[idx]]] = True
    candidate_neg_hulls = np.zeros((len(paths),n), dtype=np.bool)
    if len(neg) > 0:
        candidate_neg_hulls[queue_idxs] = [compute_hull(g, np.append(neg, paths[idx][candidates[idx]]), weights, dist_map, comps, hist) for idx in queue_idxs]
    else:
        for idx in queue_idxs:
            candidate_neg_hulls[idx][paths[idx][candidates[idx]]] = True
    pos_gains = np.zeros(len(paths))
    neg_gains = np.zeros(len(paths))

    while num_of_known_labels < n:
        to_remove = []
        changed = []
        for idx in queue_idxs:
            while known_labels[paths[idx][candidates[idx]]] >= 0:
                if not next_candidate_queues[idx].empty():
                    candidates[idx] = next_candidate_queues[idx].get()
                else:
                    maybe_remove = refill_queue_for_candidate(idx, candidates[idx], candidates, known_labels, left, next_candidate_queues, paths, queue_idxs, right)
                    if maybe_remove is not None:
                        to_remove.append(maybe_remove)
                        break
                    else:
                        candidates[idx] = next_candidate_queues[idx].get()
                changed.append(idx)

        for i in changed:
            candidate_pos_hulls[i] = compute_hull(g, np.append(pos, paths[i][candidates[i]]), weights, dist_map, comps, hist)
            candidate_neg_hulls[i] = compute_hull(g, np.append(neg, paths[i][candidates[i]]), weights, dist_map, comps, hist)

        for i in to_remove:
            queue_idxs.remove(i)
            if np.sum(known_labels[paths[i]] >= 0) != len(paths[i]):
                exit(555)

        pos_gains[queue_idxs] = np.sum(candidate_pos_hulls[queue_idxs], axis=1) - len(pos)
        neg_gains[queue_idxs] = np.sum(candidate_neg_hulls[queue_idxs], axis=1) - len(neg)

        heuristic = np.average(np.array([pos_gains[queue_idxs], neg_gains[queue_idxs]]), axis=0)

        candidate_idx = queue_idxs[np.argmax(heuristic)]
        candidate_vertex = candidates[candidate_idx]

        if known_labels[paths[candidate_idx][candidate_vertex]] == y[paths[candidate_idx][candidate_vertex]]:
            exit(9)
        known_labels[paths[candidate_idx][candidate_vertex]] = y[paths[candidate_idx][candidate_vertex]]

        budget += 1

        if known_labels[paths[candidate_idx][candidate_vertex]] == pos_value:
            pos =np.where(candidate_pos_hulls[candidate_idx])[0]
            known_labels[pos]  = pos_value
            #only recompute pos hulls, the negatives won't change
            candidate_pos_hulls[queue_idxs] = [compute_hull(g, np.append(pos, paths[idx][candidates[idx]]), weights, dist_map, comps, hist) for idx in queue_idxs]
        else:
            neg =np.where(candidate_neg_hulls[candidate_idx])[0]
            known_labels[neg] = neg_value
            # only recompute pos hulls, the negatives won't change
            candidate_neg_hulls[queue_idxs] = [compute_hull(g, np.append(neg, paths[idx][candidates[idx]]), weights, dist_map, comps, hist) for idx in queue_idxs]

        if next_candidate_queues[candidate_idx].empty():

            maybe_remove = refill_queue_for_candidate(candidate_idx, candidate_vertex, candidates, known_labels, left, next_candidate_queues, paths, queue_idxs, right)
            if maybe_remove is None:
                candidates[candidate_idx] = next_candidate_queues[candidate_idx].get()
            else:
                queue_idxs.remove(candidate_idx)
        else:
            candidates[candidate_idx] = next_candidate_queues[candidate_idx].get()

        candidate_pos_hulls[candidate_idx] = compute_hull(g, np.append(pos, paths[candidate_idx][candidates[candidate_idx]]), weights, dist_map, comps, hist)
        candidate_neg_hulls[candidate_idx] = compute_hull(g, np.append(neg, paths[candidate_idx][candidates[candidate_idx]]),weights, dist_map, comps, hist)

        pos = np.where(known_labels==pos_value)[0]
        neg = np.where(known_labels==neg_value)[0]

        num_of_known_labels = len(pos) + len(neg)

        print(num_of_known_labels, n)

    return known_labels, budget

def spc_querying_with_shadow(g: gt.Graph, paths, weights, y):
    '''

    :param g:
    :param paths: list of paths
    :param y: ground truth
    :param weight:
    :return:
    '''
    np.random.seed(55)
    #these two lines make repetitive closure computation a lot faster
    dist_map = gt.shortest_distance(g, weights=weights)
    comps, hist = gt.label_components(g)

    known_labels = -np.ones(g.num_vertices())
    num_of_known_labels = 0
    budget = 0

    pos_value, neg_value = np.unique(y)

    next_candidate_queues = [Queue() for _ in paths]
    left = np.zeros(len(paths), dtype=np.int)
    right = np.array([len(p)-1 for p in paths], dtype=np.int)
    queue_idxs = list(range(len(paths)))

    n = g.num_vertices()

    for i,path in enumerate(paths):
        next_candidate_queues[i].put(0)
        if len(path) > 1:
            next_candidate_queues[i].put(len(path)-1)

    starting_idx = np.random.choice(np.where(right>0)[0])
    starting_path = paths[starting_idx]

    budget += 2
    l = next_candidate_queues[starting_idx].get()
    r = next_candidate_queues[starting_idx].get()
    known_labels[starting_path[l]] = y[starting_path[l]]
    known_labels[starting_path[r]] = y[starting_path[r]]

    if known_labels[starting_path[0]] == known_labels[starting_path[-1]]:
        #color the hull of the path in the color of the endpoints
        path_closure = np.where(compute_hull(g, starting_path, weights, dist_map, comps, hist))[0]
        known_labels[path_closure] = known_labels[starting_path[0]]
        num_of_known_labels = len(path_closure)
        del queue_idxs[starting_idx]
    else:
        if (len(starting_path)>=3):
            next_candidate_queues[starting_idx].put(l + (r - l)//2)
        else:
            del queue_idxs[starting_idx]
        num_of_known_labels = 2

    pos = np.where(known_labels==pos_value)[0]
    neg = np.where(known_labels==neg_value)[0]

    candidates = np.zeros(len(paths), dtype=np.int)

    candidates[queue_idxs] = [next_candidate_queues[queue_idx].get() for queue_idx in queue_idxs] #this is always relative to the path

    candidate_pos_hulls = np.zeros((len(paths),n), dtype=np.bool)
    if len(pos) > 0:
        candidate_pos_hulls[queue_idxs] = [closure.compute_shadow(g, np.append(pos, paths[idx][candidates[idx]]), neg, weights, dist_map, comps, hist) for idx in queue_idxs]
    else:
        for idx in queue_idxs:
            candidate_pos_hulls[idx][paths[idx][candidates[idx]]] = True
    candidate_neg_hulls = np.zeros((len(paths),n), dtype=np.bool)
    if len(neg) > 0:
        candidate_neg_hulls[queue_idxs] = [closure.compute_shadow(g, np.append(neg, paths[idx][candidates[idx]]), pos, weights, dist_map, comps, hist) for idx in queue_idxs]
    else:
        for idx in queue_idxs:
            candidate_neg_hulls[idx][paths[idx][candidates[idx]]] = True
    pos_gains = np.zeros(len(paths))
    neg_gains = np.zeros(len(paths))

    while num_of_known_labels < n:
        to_remove = []
        changed = []
        for idx in queue_idxs:
            while known_labels[paths[idx][candidates[idx]]] >= 0:
                if not next_candidate_queues[idx].empty():
                    candidates[idx] = next_candidate_queues[idx].get()
                else:
                    maybe_remove = refill_queue_for_candidate(idx, candidates[idx], candidates, known_labels, left, next_candidate_queues, paths, queue_idxs, right)
                    if maybe_remove is not None:
                        to_remove.append(maybe_remove)
                        break
                    else:
                        candidates[idx] = next_candidate_queues[idx].get()
                changed.append(idx)

        for i in changed:
            candidate_pos_hulls[i] = closure.compute_shadow(g, np.append(pos, paths[i][candidates[i]]), neg, weights, dist_map, comps, hist)
            candidate_neg_hulls[i] = closure.compute_shadow(g, np.append(neg, paths[i][candidates[i]]), pos, weights, dist_map, comps, hist)

        for i in to_remove:
            queue_idxs.remove(i)
            if np.sum(known_labels[paths[i]] >= 0) != len(paths[i]):
                exit(555)

        pos_gains[queue_idxs] = np.sum(candidate_pos_hulls[queue_idxs], axis=1) - len(pos)
        neg_gains[queue_idxs] = np.sum(candidate_neg_hulls[queue_idxs], axis=1) - len(neg)

        heuristic = np.average(np.array([pos_gains[queue_idxs], neg_gains[queue_idxs]]), axis=0)

        candidate_idx = queue_idxs[np.argmax(heuristic)]
        candidate_vertex = candidates[candidate_idx]

        if known_labels[paths[candidate_idx][candidate_vertex]] == y[paths[candidate_idx][candidate_vertex]]:
            exit(9)
        known_labels[paths[candidate_idx][candidate_vertex]] = y[paths[candidate_idx][candidate_vertex]]

        budget += 1

        if known_labels[paths[candidate_idx][candidate_vertex]] == pos_value:
            pos =np.where(candidate_pos_hulls[candidate_idx])[0]
            known_labels[pos]  = pos_value
            #only recompute pos hulls, the negatives won't change
            candidate_pos_hulls[queue_idxs] = [closure.compute_shadow(g, np.append(pos, paths[idx][candidates[idx]]), neg, weights, dist_map, comps, hist) for idx in queue_idxs]
            candidate_neg_hulls[queue_idxs] = [closure.compute_shadow(g, np.append(neg, paths[idx][candidates[idx]]), pos, weights, dist_map, comps, hist) for idx in queue_idxs]

        else:
            neg =np.where(candidate_neg_hulls[candidate_idx])[0]
            known_labels[neg] = neg_value
            # only recompute pos hulls, the negatives won't change
            candidate_pos_hulls[queue_idxs] = [closure.compute_shadow(g, np.append(pos, paths[idx][candidates[idx]]), neg, weights, dist_map, comps, hist) for idx in queue_idxs]

            candidate_neg_hulls[queue_idxs] = [closure.compute_shadow(g, np.append(neg, paths[idx][candidates[idx]]), pos, weights, dist_map, comps, hist) for idx in queue_idxs]

        if next_candidate_queues[candidate_idx].empty():

            maybe_remove = refill_queue_for_candidate(candidate_idx, candidate_vertex, candidates, known_labels, left, next_candidate_queues, paths, queue_idxs, right)
            if maybe_remove is None:
                candidates[candidate_idx] = next_candidate_queues[candidate_idx].get()
            else:
                queue_idxs.remove(candidate_idx)
        else:
            candidates[candidate_idx] = next_candidate_queues[candidate_idx].get()

        candidate_pos_hulls[candidate_idx] = closure.compute_shadow(g, np.append(pos, paths[candidate_idx][candidates[candidate_idx]]), neg, weights, dist_map, comps, hist)
        candidate_neg_hulls[candidate_idx] = closure.compute_shadow(g, np.append(neg, paths[candidate_idx][candidates[candidate_idx]]), pos, weights, dist_map, comps, hist)

        pos = np.where(known_labels==pos_value)[0]
        neg = np.where(known_labels==neg_value)[0]

        num_of_known_labels = len(pos) + len(neg)

        print(num_of_known_labels, n)

    return known_labels, budget

def refill_queue_for_candidate(candidate_idx, candidate_vertex, candidates, known_labels, left, next_candidate_queues, paths, queue_idxs, right):
    l = left[candidate_idx]
    r = right[candidate_idx]
    if candidate_vertex != l and candidate_vertex != r:

        if known_labels[paths[candidate_idx][candidate_vertex]] == known_labels[paths[candidate_idx][l]]:
            left[candidate_idx] = candidate_vertex
        else:
            right[candidate_idx] = candidate_vertex
    mid = left[candidate_idx] + (right[candidate_idx] - left[candidate_idx]) // 2
    if mid != left[candidate_idx] and mid != right[candidate_idx]:
        next_candidate_queues[candidate_idx].put(mid)
        return None
    else:
        return candidate_idx


def is_convex(dataset,q,weighted=True):
    X = np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',X.tab')
    #X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',y.tab'))

    n = 100
    dists = scipy.spatial.distance.cdist(X, X)
    y = y[:n]
    y = (y-np.min(y))//(np.max(y)-np.min(y))
    #q = 0.04
    W = dists[:n,:n]#np.exp(-(dists) ** 2 / (2 * sigma ** 2))
    q = np.quantile(W, 0.1)
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


    comps,hist = gt.label_components(g)

    simpl = simplicial_vertices(g)

    print(len(simpl), np.sum(closure.compute_hull(g, simpl, weight_prop)>0))
    #return
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
    known_labels, budget = spc_querying_with_closure(g, paths,weight_prop,y)
    print(np.sum(np.abs(known_labels-y)/n))
    print(budget)
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    if "_" in text:
        a,b,c = text.split("_")
        a = int(a)
        b = int(b)
        c = int(c[0])
        return [a*b,a,c ]
    return [0,0]

if __name__ == "__main__":
    np.random.seed(43)
    files = os.listdir("res/synthetic/")
    files.sort(key=natural_keys)
    for filename in files:
        for label_idx in range(10):
            if ".csv" not in filename:
                continue
            if "4_5_0" in filename:
                continue
            instance = filename.split(".")[0]
            print("======================================================")
            print("file", instance, "label", label_idx)
            edges = np.genfromtxt("res/synthetic/" + instance + ".csv", delimiter=",", dtype=np.int)[:, :2]
            n = np.max(edges)+1
            g = gt.Graph(directed=False)
            g.add_vertex(n)
            g.add_edge_list(edges)
            weight_prop = g.new_edge_property("double", val=1)

            y = np.zeros(n)
            add_string = ""
            if label_idx >= 4:
                add_string = "_simplicial_start"
            y[np.genfromtxt("res/synthetic/labels/" + instance + "_"+str(label_idx)+add_string+"_positive.csv", dtype=np.int)] = True

            spc = shortest_path_cover_logn_apx(g, weight_prop)

            g.set_directed(False)

            a,b = spc_querying_with_shadow(g, spc, weight_prop, y)
            print(a)
            print(y)
            if not np.all(a==y):
                exit(22)
            print(b)

if __name__ == "__main2__":
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