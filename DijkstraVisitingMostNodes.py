import itertools

from graph_tool import _python_type, _prop
from graph_tool.all import *
import numpy as np
from graph_tool.search import libgraph_tool_search

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

    shortest_paths = []
    all_currently_covered_vertices = set()
    current_length = -1
    z = 0

    '''if (all_dists + not_visited_source_vertex).max()>2:'''

    while True:

        source, target = np.unravel_index((all_dists+not_visited_source_vertex).argmax(), all_dists.shape)

        _,pred_map = dijkstra_search(g, adjusted_weight, source)

        shortest_path = []

        cursor = target
        while pred_map[cursor] != cursor:
            shortest_path.append(cursor)
            cursor = pred_map[cursor]
        shortest_path.append(source)

        shortest_path.reverse()

        if (all_dists+not_visited_source_vertex).max() != len(set(shortest_path).difference(covered_vertices)):
            exit(10)

        if len(all_currently_covered_vertices.intersection(shortest_path)) != 0:
            all_dists[source, target] = -1
            break
        if len(shortest_path) > 1 and len(shortest_path) < current_length:
            print(len(shortest_paths))
            return shortest_paths

        shortest_paths.append(shortest_path)
        all_currently_covered_vertices = all_currently_covered_vertices.union(shortest_path)
        if current_length < 0:
            current_length = len(shortest_path)
        #trim covered vertices from start and end
        #...
        #better: build this step directly into the weight function s.t. |P| is minimized as a third priority?

        if len(shortest_path) < 2:# and z >=10:
            break


        z += 1

    '''else:
        #edge and vertex covering fast mode
        for e in g.edges():
            if e.source() not in covered_vertices and e.target not in covered_vertices:
                shortest_paths.add([e.source, e.target])

        for v in g.vertices():
            if v not in covered_vertices:
                shortest_paths.add([v])'''

    return shortest_paths

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
        curr_paths = shortest_path_visiting_most_nodes(g, adjusted_weight,covered_vertices,summed_edge_weight)

        for path in curr_paths:
            paths.append(path)

            #if len(path) <= 2 switch to fast mode and just add single edges/vertices until done.
            path_vertices = set(path)
            for v in path_vertices.difference(covered_vertices):
                for w in g.get_in_neighbors(v):
                    adjusted_weight[g.edge(w,v)] += 1#.a[list()] -= 1
                    if adjusted_weight[g.edge(w,v)] % (g.num_vertices()+1) != 0:
                        exit(5)

            new_covered = path_vertices.difference(covered_vertices)
            covered_vertices = covered_vertices.union(path_vertices)
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