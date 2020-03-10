from sage.graphs.connectivity import is_cut_edge
from sage.all import *
import numpy as np
import csv

for n in [50,100,200,500,1000]:
    for i in range(5):
        g = Graph(n)
        g.add_cycle(range(n))
        while g.is_circular_planar() and i < 50:
            x = np.random.randint(0, 100 - 1)
            y = np.random.randint(x + 1, 100)
            g.add_edge(x, y)

        if not g.is_circular_planar():
            i += 1
            g.delete_edge(x, y)
        else:
            i = 0

        f = open(n + '_' + str(i) + '.csv', 'w')
        writer = csv.writer(f)
        writer.writerows(g.edges())
