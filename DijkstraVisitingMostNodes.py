import itertools

from graph_tool import _python_type, _prop
from graph_tool.all import *
import numpy as np
from graph_tool.search import libgraph_tool_search



def combine(a,b):
    return [a[0] + b[0], a[1]+b[1]]
def compare(a,b):
    return a[0] < b[0] or (a[0] == b[0] and a[1] > b[1])


'''
bug in graph_tool when using a vector valued distance. zero and infinity are treated in the wrong way.
'''
def dijkstra_search_fix(g, weight, source=None, visitor=DijkstraVisitor(), dist_map=None,
                        pred_map=None, combine=combine,
                        compare=compare, zero=[0,0], infinity=[np.inf,np.inf]):

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

'''
g: Directed graph
weight: EdgeProperty map with vector<double> of length 2 values. First entry normal weight, second entry = 1 <=> not visited target
'''
def shortest_source_target_path_visiting_most_nodes(g, adjusted_weight, source):
    visited_nodes = g.new_vertex_property("int")
    visited_nodes[0] = 1
    dist_map, pred_map = dijkstra_search_fix(g, adjusted_weight, source)

    shortest_path = {source}

    target = np.argmax(dist_map.get_2d_array(range(2))[1])
    cursor = target
    while pred_map[cursor] != cursor:
        shortest_path.add(cursor)
        cursor = pred_map[cursor]

    num_visited_nodes = dist_map.get_2d_array(range(2))[1, target]
    if num_visited_nodes < np.inf:
        return shortest_path, num_visited_nodes
    else:
        return shortest_path, 0

'''
g: Directed graph
weight: EdgeProperty map with vector<double> of length 2 values. First entry normal weight, second entry = 1 <=> not visited target
'''
def shortest_path_visiting_most_nodes(g: Graph, adjusted_weight: EdgePropertyMap,covered_vertices):
    max_new_covered_nodes = 0
    for i in range(g.num_vertices()):
        i_path, new_covered_nodes = shortest_source_target_path_visiting_most_nodes(g, adjusted_weight, i)
        if i not in covered_vertices:
            new_covered_nodes += 1 #the adapted dijkstra is counting only the edges. not the starting vertex
        if new_covered_nodes > max_new_covered_nodes:
            max_new_covered_nodes =new_covered_nodes
            max_path = i_path

    return max_path

'''
g: Directed graph
weight: double valued EdgeProperty
'''
def shortest_path_cover_logn_apx(g: Graph, weight: EdgePropertyMap):
    if not g.is_directed():
        g.set_directed(True)


    adjusted_weight = g.new_edge_property("vector<double>")
    for e in g.edges():
        adjusted_weight[e] = [weight[e],1]#, vals=list(np.column_stack((weight.get_array(), np.repeat(1.0, g.num_edges())))))
    paths = []

    graph_draw(g, vertex_text=g.vertex_index, output="test.png", output_size=(1000, 1000), edge_text=adjusted_weight,
               vertex_font_size=20, edge_font_size=20)
    covered_vertices = set()

    while len(covered_vertices) != g.num_vertices():
        path = shortest_path_visiting_most_nodes(g, adjusted_weight,covered_vertices)
        paths.append(path)
        covered_vertices = covered_vertices.union(path)
        for v in path:
            for w in g.get_in_neighbors(v):
                adjusted_weight[g.edge(w,v)][1] = 0

    return paths

for k in range(1,11):
    print("n=", k*100)
    for i in range(10):

        deg_sampler = lambda: (np.random.randint(1,3),np.random.randint(1,3))
        n = k*100
        g= random_graph(n,deg_sampler)
        weight=g.new_edge_property("double", val=1)

        paths = shortest_path_cover_logn_apx(g, weight)

        sum = 0
        for i in paths:
            sum += np.ceil(np.log2(len(i)))

        print("|S|=",len(paths))
        print("#queries<=",sum, "%:", sum/n)


    print("==============================")