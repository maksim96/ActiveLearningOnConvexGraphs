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

    visited_source_vertex = np.zeros(g.num_vertices(), dtype=np.bool)
    visited_source_vertex[list(covered_vertices)] = True

    all_dists = dist_map.get_2d_array(range(g.num_vertices())) + (1-visited_source_vertex) #shortest path does only count the edges. so we have add one if the starting vertex was not visited.

    all_dists[(all_dists > summed_edge_weight) | (all_dists < 0)] = 0

    source, target = np.unravel_index((all_dists % g.num_vertices()).argmax(), all_dists.shape)

    _,pred_map = dijkstra_search(g, adjusted_weight, source)

    shortest_path = {source}

    cursor = target
    while pred_map[cursor] != cursor:
        shortest_path.add(cursor)
        cursor = pred_map[cursor]

    return shortest_path

'''
g: Directed graph
weight: double valued EdgeProperty
'''
def shortest_path_cover_logn_apx(g: Graph, weight: EdgePropertyMap):
    if not g.is_directed():
        g.set_directed(True)



    if weight.python_value_type() not in ["bool","int","int16_t", "int32_t", "int64_t"]:
        min = np.min(weight.a)
        min_second = np.min(weight.a[weight.a > min])

        eps = min_second - min
        scaled_weight = (np.floor(weight.a / eps) * g.num_vertices()).astype(np.int)  # ints >= 1
    else:
        scaled_weight = weight.a*g.num_vertices()

    summed_edge_weight = np.sum(scaled_weight)

    adjusted_weight = g.new_edge_property("long", vals=scaled_weight + 1)


    paths = []

    covered_vertices = set()

    while len(covered_vertices) != g.num_vertices():
        path = shortest_path_visiting_most_nodes(g, adjusted_weight,covered_vertices,summed_edge_weight)
        paths.append(path)

        for v in path.difference(covered_vertices):
            for w in g.get_in_neighbors(v):
                adjusted_weight[g.edge(v,w)] -= 1#.a[list()] -= 1
        covered_vertices = covered_vertices.union(path)
        print(len(path), len(covered_vertices))

    return paths

if __name__ == "__main__":
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