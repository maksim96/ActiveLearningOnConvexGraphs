

import itertools

import graph_tool as gt
import numpy as np
from graph_tool.generation import random_graph


def simplicial_vertices(g: gt.Graph):
    '''
    returns the (unweighted) simplicial vertices of g
    '''

    simplicial_vertices = []

    for v in g.vertices():

        #try to find a clique around v
        for x, y in itertools.combinations(g.get_all_neighbors(v), 2):
            if g.edge(x,y) is None:
                break
        else:
            simplicial_vertices.append(v)


    return simplicial_vertices


if __name__ == "__main__":
    for i in range(100):
        deg_sampler = lambda: np.random.randint(1, 50)
        g = random_graph(i*10, deg_sampler, directed=False)
        print(i*10,len(simplicial_vertices(g)))

