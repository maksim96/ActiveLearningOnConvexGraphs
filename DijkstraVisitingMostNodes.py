from graph_tool import _python_type, _prop
from graph_tool.all import *
import numpy as np
from graph_tool.search import libgraph_tool_search


class VisitorExample(DijkstraVisitor):

    def __init__(self, dists, visited_nodes):
        self.dists = dists
        self.visited_nodes = visited_nodes


    def edge_relaxed(self, e):
        self.visited_nodes[e.target()] = self.visited_nodes[e.source()]+1

def dijkstra_search2(g, weight, source=None, visitor=DijkstraVisitor(), dist_map=None,
                    pred_map=None, combine=lambda a, b: a + b,
                    compare=lambda a, b: a < b, zero=0, infinity=np.inf):

    if visitor is None:
        visitor = DijkstraVisitor()
    if dist_map is None:
        dist_map = g.new_vertex_property(weight.value_type())
    if pred_map is None:
        pred_map = g.new_vertex_property("int64_t")
    if pred_map.value_type() != "int64_t":
        raise ValueError("pred_map must be of value type 'int64_t', not '%s'." % \
                             pred_map.value_type())



    try:
        if source is None:
            source = 0
        else:
            source = int(source)
        libgraph_tool_search.dijkstra_search(g._Graph__graph,
                                             source,
                                             _prop("v", g, dist_map),
                                             _prop("v", g, pred_map),
                                             _prop("e", g, weight), visitor,
                                             compare, combine, zero, infinity)
    except StopSearch:
        pass

    return dist_map, pred_map

g = Graph(directed=False)

g.add_vertex(10)

weight = g.new_edge_property("vector<int>")

e = g.add_edge(0,1)
weight[e] = [4,1]
e = g.add_edge(1,2)
weight[e] = [4,1]
e = g.add_edge(0,3)
weight[e] = [1,1]
e = g.add_edge(3,4)
weight[e] = [1,1]
e = g.add_edge(4,5)
weight[e] = [1,1]
e = g.add_edge(5,6)
weight[e] = [1,1]
e = g.add_edge(6,7)
weight[e] = [1,1]
e = g.add_edge(7,8)
weight[e] = [1,1]
e = g.add_edge(8,9)
weight[e] = [1,1]
e = g.add_edge(9,2)
weight[e] = [1,1]
visited_nodes = g.new_vertex_property("int")
visited_nodes[0] = 1
combine = lambda a,b: [a[0] + b[0], a[1]+b[1]]
compare = lambda a,b: a[0] < b[0] or (a[0] == b[0] and a[1] > b[1])
dist_map,pred_map =dijkstra_search2(g, weight, g.vertex(0), VisitorExample(weight, visited_nodes), combine=combine, compare=compare,zero=[0,0],infinity=[100000,100000])
print(pred_map.get_array())
print([dist_map[i][0] for i in range(10)])
print(visited_nodes.get_array())

deg_sampler = lambda: (np.random.randint(1,20),np.random.randint(1,20))
n = 1000000
g = random_graph(n,deg_sampler)
weight = g.new_edge_property("vector<int>", val=[1,1])

dist_map,pred_map =dijkstra_search2(g, weight, g.vertex(0), VisitorExample(weight, visited_nodes), combine=combine, compare=compare,zero=[0,0],infinity=[100000000,100000000])
print("uw")
#print(np.sort(pred_map.get_array())[::-1])
x= [dist_map[i][0] for i in range(n)]
#print(x)

weight = g.new_edge_property("int", val=1)
dist_map,pred_map =dijkstra_search(g, weight, g.vertex(0))
#print(pred_map.get_array())
#print(dist_map.get_array())

print(np.all(x == dist_map.get_array()))