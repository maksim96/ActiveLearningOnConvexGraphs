from sage.graphs.connectivity import is_cut_edge
from sage.all import *
from sage.graphs.connectivity import blocks_and_cut_vertices
from random import shuffle

for num_blocks,block_size in [(5,4),(4,5),(5,10),(10,5),(25,4),(4,25),(4,50),(50,4),(5,100),(100,5)]:
	expected_num_vertices = num_blocks*block_size
	max_block_size = 2*block_size-2
	for i in range(5):                               
		g = graphs.RandomBlockGraph(num_blocks,2,max_block_size)
		blocks,_ = blocks_and_cut_vertices(g)
		print("======================")
		print(i)

		for b in blocks:
			block = g.subgraph(b)

			edges = list(block.edges())
			shuffle(edges)
			while not block.is_circular_planar():

				#print(edges)
				for e in edges:
					block.delete_edge(e)
					g.delete_edge(e)
					print("remove ", e)
					if not block.is_biconnected():
						print("add ", e)
						block.add_edge(e)
						g.add_edge(e)

					if block.is_circular_planar():
						print("yes")
						break
				edges = list(block.edges())
				shuffle(edges)
				for e in edges:
					block.delete_edge(e)
					g.delete_edge(e)
					print("remove ", e)
					if not block.is_connected():
						print("add ", e)
						block.add_edge(e)
						g.add_edge(e)

					if block.is_circular_planar():
						break


		plot(g).save(''+str(num_blocks)+'_'+str(block_size)+'_'+str(i)+'.svg')




