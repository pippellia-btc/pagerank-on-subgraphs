import networkx as nx
import numpy as np
from scipy.sparse import isspmatrix_csr
import random

def get_subrank(S, G, walk_visited_count, nodelist, alpha = 0.85):

    '''
        Subrank algorithm (stopping at dandling nodes);
        it aims to approximate the Pagerank over S subgraph of G
    
        INPUTS:
        - S is a directed networkx graph, subgraph of G
        - G is a directed networkx graph
        - walk_visited_count is a Compressed Sparse Row (CSR) matrix;
        element (i,j) is equal to the number of times v_j has been visited
        by a random walk in G started from v_i
        - nodelist is the ordered list of nodes in G; it's used to decode walk_visited_count
        - alpha is the dampening factor of Pagerank. default is 0.85

        OUTPUTS:
        - subrank is the dictionary {node: pg} of the pagerank value for each node in S
    '''
    
    # validate inputs and initialize variables
    N, S_nodes, G_nodes, inverse_nodelist = _validate_inputs_and_init(S, G, walk_visited_count, nodelist, alpha)

    # compute visited count from walks that started from S
    visited_count_from_S = _get_visited_count_from_S(N, S_nodes, walk_visited_count, nodelist, inverse_nodelist)

    # compute positive and negative walks to do
    positive_walks, negative_walks = _get_walks_to_do(S_nodes, G_nodes, S, G, visited_count_from_S, alpha)

    print(f'walks performed = {sum(positive_walks.values()) + sum(negative_walks.values())}')

    # perform the walks and get the visited counts
    positive_count = _perform_walks(S_nodes, S, positive_walks, alpha)
    negative_count = _perform_walks(S_nodes, S, negative_walks, alpha)

    # adding the effects of the random walk to the count of G
    new_visited_count = {node: visited_count_from_S[node] + positive_count[node] - negative_count[node]
                        for node in S_nodes}

    # computing the sum 
    total_visits = sum(new_visited_count.values())

    # computing the rank approx
    subrank = {node: visits / total_visits 
                   for node, visits in new_visited_count.items() }

    return subrank

def _validate_inputs_and_init(S, G, walk_visited_count, nodelist, alpha):

    '''
    This function validates all the inputs and initializes useful variables;
    Note: S being a subgraph of G is NOT checked because it's computationally expensive.
    '''

    if len(S) == 0:
        raise ValueError("graph S is empty")

    N = len(G)
    if N == 0:
        raise ValueError("graph G is empty")
    
    if not isinstance(alpha, float) or not (0 < alpha < 1):
        raise ValueError("alpha must be a float between 0 and 1")

    if not isspmatrix_csr(walk_visited_count) or walk_visited_count.shape != (N,N):
        raise ValueError(f"walk_visited_count must be a {(N,N)} CSR matrix")

    S_nodes = set(S.nodes())
    G_nodes = set(G.nodes())
    
    if not nodelist or set(nodelist) != set(G_nodes):
        raise ValueError("nodelist does not match the nodes in G")

    # compute the inverse map of nodelist
    inverse_nodelist = {nodelist[j]: j for j in range(N)}

    return N, S_nodes, G_nodes, inverse_nodelist


def _get_visited_count_from_S(N, S_nodes, walk_visited_count, nodelist, inverse_nodelist):

    '''
    This function extracts the number of visits that come from walks that started from S
    '''

    # getting the indices of nodes in S
    S_indices = [inverse_nodelist[node] for node in S_nodes]
    
    # Extract the rows
    S_matrix = walk_visited_count[S_indices, :]

    # Sum the rows
    visited_count_from_S = np.array(S_matrix.sum(axis=0)).flatten()

    # convert to a dictionary
    visited_count_from_S = {nodelist[j]: visited_count_from_S[j] for j in range(N)}
    
    return visited_count_from_S
    

def _get_walks_to_do(S_nodes, G_nodes, S, G, visited_count_from_S, alpha):

    '''
    This function calculates the positive and negative walks to be done for each node.
    It is a necessary step to take into account the different structure of S
    with respect to that of G.
    '''

    # compute nodes in G-S
    external_nodes = G_nodes - S_nodes

    # compute nodes in S that point to G-S
    nodes_that_point_externally = {u for u,v in nx.edge_boundary(G, S_nodes, external_nodes)}

    walks_to_do = {node: 0 for node in S_nodes}

    # add positive random walks to walks_to_do
    for node in nodes_that_point_externally:

        successors = set(G.successors(node)) & S_nodes

        if successors:

            # compute estimate visits
            visited_count = visited_count_from_S[node]
            degree_S = S.out_degree(node)
            degree_G = G.out_degree(node)
            estimate_visits = alpha * visited_count * (1/degree_S - 1/degree_G)

            for succ in successors:
                walks_to_do[succ] += estimate_visits

    # subtract number of negative random walks
    for node in external_nodes:

        successors = set(G.successors(node)) & S_nodes

        if successors:
            
            # compute estimate visits
            visited_count = visited_count_from_S[node]
            degree = G.out_degree(node)
            estimate_visits = alpha * visited_count / degree

            for succ in successors:
                walks_to_do[succ] -= estimate_visits

    # split the walks to do into positive and negative
    positive_walks_to_do = {node: round(value) for node,value in walks_to_do.items() if value > 0 }
    negative_walks_to_do = {node: round(-value) for node,value in walks_to_do.items() if value < 0 }

    return positive_walks_to_do, negative_walks_to_do
    

def _perform_walks(S_nodes, S, walks_to_do, alpha):

    '''
    This function performs a certain number of random walks on S for each node;
    It then returns the visited count for each node in S.
    '''

    # initializing the visited count
    visited_count = {node: 0 for node in S_nodes}

    for starting_node in walks_to_do.keys():

        num = walks_to_do[starting_node]

        # performing num random walks
        for _ in range(num):

            current_node = starting_node
            visited_count[current_node] += 1

            # performing one random walk
            while random.uniform(0,1) < alpha:
    
                successors = list(S.successors(current_node))
        
                if not successors:
                    break
    
                current_node = random.choice(successors)

                # updating the visited count
                visited_count[current_node] += 1

    return visited_count
