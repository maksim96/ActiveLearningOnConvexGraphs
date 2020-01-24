import itertools

from graph_tool import _python_type, _prop
from graph_tool.all import *
import numpy as np
from graph_tool.search import libgraph_tool_search



'''
g: Directed graph
weight: EdgeProperty map with vector<double> of length 2 values. First entry normal weight, second entry = 1 <=> not visited target
'''
def shortest_source_target_path_visiting_most_nodes(g, adjusted_weight, source, target):
    visited_nodes = g.new_vertex_property("int")
    visited_nodes[0] = 1
    dist_map, pred_map = dijkstra_search_fix(g, adjusted_weight, source, target)

    shortest_path = {source}

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
def shortest_path_visiting_most_nodes(g: Graph, adjusted_weight: EdgePropertyMap,covered_vertices,summed_edge_weight):

    dist_map= shortest_distance(g, weights=adjusted_weight)

    not_visited_source_vertex = np.ones(g.num_vertices(), dtype=np.bool)
    not_visited_source_vertex[list(covered_vertices)] = False
    not_visited_source_vertex = not_visited_source_vertex.reshape(g.num_vertices(), 1)

    all_dists = dist_map.get_2d_array(range(g.num_vertices())).T #shortest path does only count the edges. so we have add one if the starting vertex was not visited.

    all_dists[(all_dists > summed_edge_weight) | (all_dists < 0)] = 0

    all_dists = (g.num_vertices()+1 - all_dists) % (g.num_vertices()+1)

    source, target = np.unravel_index((all_dists+not_visited_source_vertex).argmax(), all_dists.shape)

    _,pred_map = dijkstra_search(g, adjusted_weight, source)

    shortest_path = {source}

    cursor = target
    while pred_map[cursor] != cursor:
        shortest_path.add(cursor)
        cursor = pred_map[cursor]

    if (all_dists+not_visited_source_vertex).max() != len(shortest_path.difference(covered_vertices)):
        exit(10)

    #trim covered vertices from start and end
    #...
    #better: build this step directly into the weight function s.t. |P| is minimized as a third priority?

    return shortest_path

'''
g: Directed graph
weight: double valued EdgeProperty
'''
def shortest_path_cover_logn_apx(g: Graph, weight: EdgePropertyMap):
    if not g.is_directed():
        g.set_directed(True)



    if weight.value_type() not in ["bool","int","int16_t", "int32_t", "int64_t"]:
        #min = np.min(weight.a)
        #min_second = np.min(weight.a[weight.a > min])

        eps = 1#min_second - min
        scaled_weight = (np.ceil(weight.a / eps) * (g.num_vertices()+1)).astype(np.int)  # ints >= 1
    else:
        scaled_weight = weight.a*(g.num_vertices()+1)

    summed_edge_weight = np.sum(scaled_weight)

    adjusted_weight = g.new_edge_property("long", vals=scaled_weight - 1)


    paths = []

    covered_vertices = set()

    while len(covered_vertices) != g.num_vertices():
        path = shortest_path_visiting_most_nodes(g, adjusted_weight,covered_vertices,summed_edge_weight)
        paths.append(path)

        #if len(path) <= 2 switch to fast mode and just add single edges/vertices until done.

        for v in path.difference(covered_vertices):
            for w in g.get_in_neighbors(v):
                adjusted_weight[g.edge(w,v)] += 1#.a[list()] -= 1
                if adjusted_weight[g.edge(w,v)] % (g.num_vertices()+1) != 0:
                    exit(5)
        new_covered = path.difference(covered_vertices)
        covered_vertices = covered_vertices.union(path)
        print(len(new_covered), len(path), len(covered_vertices), path)

    return paths

if __name__ == "__main__":
    g = Graph(directed=False)

    g.add_vertex(10)

    S = [0, 5, 2]

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

    shortest_path_cover_logn_apx(g, weight)