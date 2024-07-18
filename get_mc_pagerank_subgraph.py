import numpy as np
import networkx as nx
import random
from scipy.sparse import isspmatrix_csr

def get_mc_pagerank_subgraph(S, G, walk_visited_count, R, nodelist=None, alpha = 0.85):

    '''
        Approximate Monte-Carlo pagerank on S subgraph of G
    
        INPUTS:
        - S is a directed networkx graph, subgraph of G
        - G is a directed networkx graph
        - walk_visited_count is a Compressed Sparse Row (CSR) matrix;
        element (i,j) is equal to the number of times v_j has been visited
        by a random walk started from v_i
        
        - R is the int number of random walks to be performed per node
        - nodelist is the list of nodes, used to order the nodes in a particular way
        - alpha is the dampening factor of Pagerank. default is 0.85

        OUTPUTS:
        - mc_pagerank is the dictionary {node: pg} of the pagerank value for each node is subgraph S
    '''
    
    # validate inputs and initialize variables
    n, N, S_nodes, G_nodes = _validate_inputs_and_initialize(S, G, walk_visited_count, R, nodelist, alpha)

    print(f'Computing positive and negative walks to do')

    if not nodelist:
        nodelist = list(G_nodes)

    # compute the inverse map of nodelist
    inverse_nodelist = {nodelist[j]: j for j in range(N)}

    # compute the indices of nodes in S
    S_indices = [inverse_nodelist[node] for node in S_nodes]

    # compute visits count from random walks starting from nodes in subgraph S
    visited_count_from_S = _get_visited_count_from_S(S_indices, walk_visited_count)
    
    # get the positive and negative walks to be performed for each node
    positive_walks_to_do, negative_walks_to_do = _get_walks_to_do(S_nodes, G_nodes, S, G, visited_count_from_S, inverse_nodelist, alpha)
    
    total_walks = sum(positive_walks_to_do.values()) + sum(negative_walks_to_do.values())
    print(f'Performing {total_walks} random walks instead of {n * R}')
    
    # perform the walks and get the visited counts
    positive_count = _perform_walks(N, S_indices, S_nodes, S, positive_walks_to_do, alpha)
    negative_count = _perform_walks(N, S_indices, S_nodes, S, negative_walks_to_do, alpha )

    # adding the effects of the random walk to the visited count of G
    visited_count_S = visited_count_from_S + positive_count - negative_count

    # computing the total visited count
    total_visited_count = np.sum(visited_count_S[S_indices])

    # computing the rank approx
    mc_pagerank = {nodelist[j]: visited_count_S[j] / total_visited_count 
                   for j in S_indices}
    
    return mc_pagerank

def _validate_inputs_and_initialize(S, G, walk_visited_count, R, nodelist, alpha):

    """
    Validates inputs and initializes necessary variables.
    """

    N = len(G)
    if N == 0:
        raise ValueError("graph G is empty")
        
    n = len(S)
    if n == 0:
        raise ValueError("subgraph S is empty")

    if not isinstance(R, int) or R <= 0:
        raise ValueError("R must be a positive integer")
    
    if not isinstance(alpha, float) or not (0 < alpha < 1):
        raise ValueError("alpha must be a float between 0 and 1")

    if not isspmatrix_csr(walk_visited_count) or walk_visited_count.shape != (N,N) :
        raise ValueError("walk_visited_count is not a " +str((N,N)) + "csr matrix")
    
    S_nodes = set(S.nodes())
    G_nodes = set(G.nodes())
    S_edges = set(S.edges())
    G_edges = set(G.edges())

    if not S_nodes <= G_nodes or not S_edges <= G_edges:
        raise ValueError("S is not a subgraph of G")
    
    if nodelist is not None and set(nodelist) != G_nodes:
        raise ValueError("nodelist does not match the nodes in G")

    return n, N, S_nodes, G_nodes

def _get_visited_count_from_S(S_indices, walk_visited_count):

    """
    Computes the visited count for random walks that started from nodes in subgraph S
    """

    # Extract the rows
    S_matrix = walk_visited_count[S_indices, :]

    # Sum the rows
    visited_count_from_S = np.array(S_matrix.sum(axis=0)).flatten()
    
    return visited_count_from_S

def _get_walks_to_do(S_nodes, G_nodes, S, G, visited_count_from_S, inverse_nodelist, alpha):

    """
    Determines the number of positive and negative random walks to perform for each node
    """

    N = len(G)
    
    # computing external nodes
    external_nodes = G_nodes - S_nodes
    
    # initialize positive and negative walks to do
    positive_walks_to_do = {node: 0 for node in S_nodes}
    negative_walks_to_do = {node: 0 for node in S_nodes}

    # compute nodes in S that point to G-S
    nodes_that_point_externally = {u for u,v in nx.edge_boundary(G, S_nodes, external_nodes)}

    # compute negative_walks_to_do
    for node in external_nodes:
        successors = set(G.successors(node)) & S_nodes

        if successors:

            # compute estimate_visits
            count = visited_count_from_S[inverse_nodelist[node]]
            degree = G.out_degree(node)
            estimate_visits = (alpha / degree) * count
            
            # add estimate_visits to all nodes in successors
            for v in successors:
                negative_walks_to_do[v] += estimate_visits

    # compute positive_walks_to_do
    for node in nodes_that_point_externally:
        successors = set(G.successors(node)) & S_nodes

        if successors:
            
            # compute estimate_visits
            count = visited_count_from_S[inverse_nodelist[node]]
            degree_G = G.out_degree(node)
            degree_S = S.out_degree(node)

            estimate_visits = alpha * count * (1/degree_S - 1/degree_G)
    
            for v in successors:
                positive_walks_to_do[v] += estimate_visits

    # compute the difference and store it
    for node in S_nodes:

        diff = round(positive_walks_to_do[node] - negative_walks_to_do[node])

        if diff >= 0:
            
            positive_walks_to_do[node] = diff
            negative_walks_to_do[node] = 0

        else:

            positive_walks_to_do[node] = 0
            negative_walks_to_do[node] = - diff
    
    return positive_walks_to_do, negative_walks_to_do


def _perform_walks(N, S_indices, S_nodes, S, walks_to_do, alpha = 0.85):

    """
    Performs the random walks based on the number of walks to do for each node.
    """

    # initializing the visited count
    visited_count_dict = {node: 0 for node in S_nodes}

    for node in walks_to_do.keys():

        num = walks_to_do[node]

        # performing num random walks
        for _ in range(num):

            current_node = node
            visited_count_dict[current_node] += 1

            # performing one random walk
            while np.random.uniform() < alpha:
    
                successors = list(S.successors(current_node))
        
                if not successors:
                    break
    
                current_node = random.choice(successors)

                # updating the visited count
                visited_count_dict[current_node] += 1

    # Create an array of zeros
    visited_count = np.zeros(N)
    
    # Convert dictionary keys and values to arrays
    visited_count[S_indices] = np.array(list(visited_count_dict.values()))

    return visited_count
