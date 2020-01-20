import itertools
import queue

from graph_tool import _python_type, _prop
from graph_tool.all import *
import numpy as np


def dumb_compute_closed_interval(g, S, weight):

    visited_nodes = set(S)

    for i,j in itertools.combinations(S,2):
        for path in all_shortest_paths(g, i,j,weights=weight):
            visited_nodes = visited_nodes.union(path)

    return visited_nodes

def compute_hull(g, S, weight, compute_closure=True):
    class VisitorExample(DijkstraVisitor):

        def __init__(self, dists, dag, real_dists):
            self.dists = dists
            self.dag = dag
            self.real_dists = real_dists

        def examine_edge(self, e):
            if self.real_dists[e.source()] + self.dists[e] == self.real_dists[e.target()]:
                self.dag.add_edge(e.target(), e.source())
                #print(e)

    I_S = np.zeros(g.num_vertices(), dtype=np.bool)

    I_S[S] = True

    q = queue.Queue()

    for v in S:
        q.put(v)
    while not q.empty():
        v = q.get()
        dag = Graph()
        dag.add_vertex(g.num_vertices())

        #graph_draw(g, vertex_text=g.vertex_index, output="gtemp.png", output_size=(1000, 1000), vertex_font_size=20)

        dist_map, pred_map = dijkstra_search(g, weight, g.vertex(v))
        dist_map, pred_map = dijkstra_search(g, weight, g.vertex(v), VisitorExample(weight, dag, dist_map))

        #graph_draw(dag, vertex_text=dag.vertex_index, output="dag.png", output_size=(1000, 1000), vertex_font_size=20)

        # dag is the predeccesor dag marking all the nodes which are visitable by shartest paths from the source
        # now mark them all by bfs

        class VisitorExample2(BFSVisitor):

            def __init__(self, visited_nodes):
                self.visited_nodes = visited_nodes

            def examine_vertex(self, u):
                self.visited_nodes[u] = True

        visited_nodes = dag.new_vertex_property("bool", val=False)

        if compute_closure:
            for w in np.arange(g.num_vertices())[I_S == True]:
                bfs_search(dag, w, VisitorExample2(visited_nodes))
        else:
            for w in S:
                bfs_search(dag, w, VisitorExample2(visited_nodes))

        if compute_closure:
            for i in range(g.num_vertices()):
                if not I_S[i] and visited_nodes[i]:
                    q.put(i)

        I_S[visited_nodes.get_array() == 1] = True

        #print(np.sum(I_S))
    return I_S

'''
g = Graph(directed=False)

g.add_vertex(10)

S = [0, 5,2]

weight = g.new_edge_property("int")

e = g.add_edge(0, 1)
weight[e] = 4
e = g.add_edge(1, 2)
weight[e] = 4
e = g.add_edge(0, 3)
weight[e] = 1
e = g.add_edge(3, 4)
weight[e] = 1
e = g.add_edge(4, 5)
weight[e] = 1
e = g.add_edge(5, 6)
weight[e] = 1
e = g.add_edge(6, 7)
weight[e] = 1
e = g.add_edge(7, 8)
weight[e] = 1
e = g.add_edge(8, 9)
weight[e] = 1
e = g.add_edge(9, 2)
weight[e] = 1

#I_S = compute_hull(g, S,weight)
#print(I_S)
#print("==========================")
#I_S = compute_hull(g, S,weight, False)
#print(I_S)

g = Graph(directed=False)

g.add_vertex(5)

S = [0,4]
e = g.add_edge(0, 1)
e = g.add_edge(1, 2)
e = g.add_edge(1, 4)
e = g.add_edge(0, 3)
e = g.add_edge(2, 3)
e = g.add_edge(3, 4)

weight = g.new_edge_property("int", val=1)

I_S = compute_hull(g, S,weight)
print(I_S)
print("==========================")
I_S = compute_hull(g, S,weight,False)
print(I_S)

n = 50
for i in range(100):


    for j in range(1,20):
        print("===============================================")
        np.random.seed(i)
        seed_rng(i)
        S = np.random.choice(n, j)
        print(i)
        print(S)
        print("=====")

        deg_sampler = lambda: np.random.randint(2,10)
        g = random_graph(n,deg_sampler, directed=False)
        weight = g.new_edge_property("int", vals=np.random.randint(1,10,g.num_edges()))

        I_S = compute_hull(g, S,weight)
        hull = np.array(I_S)
        print(I_S)
        print("==========================")
        I_S = compute_hull(g, S,weight,False)
        I_S_dumb = dumb_compute_closed_interval(g, S, weight)
        if np.any(I_S_dumb != set(np.where(I_S)[0])):
            graph_draw(g, vertex_text=g.vertex_index, output="dag.png", output_size=(1000, 1000), edge_text=weight,
                       vertex_font_size=20, edge_font_size=20)
            I_S_dumb = dumb_compute_closed_interval(g, S, weight)
            exit(1)
        print(I_S)
        previous_I_S_dumb = set()
        for _ in range(n):
            I_S = compute_hull(g, np.arange(g.num_vertices())[I_S==True],weight,False)
            I_S_dumb = dumb_compute_closed_interval(g,I_S_dumb,weight)
            print(I_S)
            if np.any(I_S_dumb != set(np.where(I_S)[0])):
                exit(2)

            if previous_I_S_dumb == I_S_dumb:
                break

            previous_I_S_dumb = set(I_S_dumb)

        if np.any(hull != I_S):
            exit(3)


#graph_draw(g,vertex_text=g.vertex_index,output="two-nodes.png",  output_size=(1000,1000), vertex_font_size=20)'''