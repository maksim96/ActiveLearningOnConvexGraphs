import os
import csv
import numpy as np
import graph_tool.all as gt

from closure import compute_hull
from simplicial_vertices import simplicial_vertices


def florians_procedure(g: gt.Graph, use_simplicial):
    n = g.num_vertices()

    if not use_simplicial:
        s = simplicial_vertices(g)
        a = s[0]
        while a in s:
            a = np.random.randint(0,n)

        b = a
        while a == b or b in s:
            b = np.random.randint(0, n)

    else:
        a = np.random.randint(0, n)

        b = a
        while a == b:
            b = np.random.randint(0, n)


    A = np.zeros(n, dtype=np.bool)
    A[a] = True
    B = np.zeros(n, dtype=np.bool)
    B[b] = True

    F = set(range(n)).difference(np.where(A|B==True)[0])

    i = 0
    while len(F) > 0:
        e = F.pop()

        if i %2 == 0:

            A[e] = True
            A_new =  (g,np.where(A==True)[0])
            if not np.any(B&A_new):
                A = A_new
                F = F.difference(set(np.where(A==True)[0]))
            else:
                A[e] = False
                B[e] = True
                B_new = compute_hull(g, np.where(B==True)[0])
                if not np.any(A & B_new):
                    B = B_new
                    F = F.difference(set(np.where(A==True)[0]))
                else:
                    B[e] = False
        else:
            B[e] = True
            B_new = compute_hull(g, np.where(B == True)[0])
            if not np.any(A & B_new):
                B = B_new
                F = F.difference(set(np.where(A == True)[0]))
            else:
                B[e] = False
                A[e] = True
                A_new = compute_hull(g, np.where(A == True)[0])
                if not np.any(B & A_new):
                    A = A_new
                    F = F.difference(set(np.where(A == True)[0]))

        i += 1
        print(len(F))
    return A,B

def label_graph(instance, i, use_simplicial):
    edges = np.genfromtxt("res/new_synthetic/"+instance+".csv", delimiter=",",dtype=np.int)[:,:2]
    n = np.max(edges-1)
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    g.add_edge_list(edges)

    #print(edges)

    #print(np.sort(g.degree_property_map("total").a))

    A,B = florians_procedure(g,use_simplicial)

    pos = np.where(A)[0]
    pos = g.new_vertex_property("bool", vals=A)

    simplicial_string = "_simplicial_start"
    if not use_simplicial:
        simplicial_string = ""
    file = open("res/new_synthetic/labels/"+instance+"_"+str(i)+simplicial_string+"_positive.csv", 'w')
    writer = csv.writer(file)
    writer.writerows(np.where(A)[0].reshape((-1,1)))
    #print(len(np.where(A)[0])/n)
    gt.graph_draw(g, pos=gt.arf_layout(g, max_iter=0),vertex_fill_color=pos,output="res/new_synthetic/images/"+instance+"_"+str(i)+"_"+simplicial_string+".svg")

    #print(A)
    #print(B)

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    if "_" in text:
        a,b = text.split("_")
        a = int(a)
        b = int(b[0])
        return [a,b]
    return [0,0]
if __name__=="__main__":
    np.random.seed(43)
    files = os.listdir("res/new_synthetic/")
    files.sort(key=natural_keys)
    for filename in files:
        if ".csv" not in filename or "500" not in filename:
            continue
        instance = filename.split(".")[0]
        print("============================")
        print(instance)
        for i in range(10):
            print(i)
            label_graph(instance,i,i>4)