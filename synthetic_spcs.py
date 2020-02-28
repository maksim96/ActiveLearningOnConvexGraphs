import os
import csv
import numpy as np
import graph_tool.all as gt
import pickle

from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx


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

if __name__=="__main__":
    np.random.seed(43)
    files = os.listdir("res/synthetic/")
    files.sort(key=natural_keys)
    for filename in files:
        if ".csv" not in filename:
            continue
        instance = filename.split(".")[0]
        print("======================================================")
        print("file: ", instance)
        edges = np.genfromtxt("res/synthetic/" + instance + ".csv", delimiter=",", dtype=np.int)[:, :2]
        n = np.max(edges - 1)
        g = gt.Graph(directed=False)
        g.add_vertex(n)
        g.add_edge_list(edges)
        weight_prop = g.new_edge_property("double", val=1)

        spc = shortest_path_cover_logn_apx(g, weight_prop)

        pickle.dump(spc, open("res/synthetic/spc/" + instance + ".p", "wb"))