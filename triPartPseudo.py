#See Networkx complete_multipartite_graph (*block sizes)
#	returns the complete multipartite graph with the specified block sizes


# Load GEM
# Load biochemical pathways
# Load SRA

# Run complete_multipartite_graph with these 3 graphs as params


#Tripartite Modularity
# calculate fraction of hyperedges E(lmn) for each community in each vertex type (3 types)

#{
# calculate two dimensional sums A(lx), A(my), A(nz) for the first community in Vx
# repeat for all communities
# calculate modularity for the first type of vertices Qx = sum(sum(sum(E(lmn) - A(xl)A(ym)A(zn))))

# repeat 2 more times for the other types of vertices


#calculate total modularity Q(t) = (1/3)(Q(x) + Q(y) + Q(z))

