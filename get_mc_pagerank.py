import numpy as np
import random
from scipy.sparse import lil_matrix, csr_matrix

def get_mc_pagerank(G, R, nodelist = None, alpha=0.85):
    
    '''
        Monte-Carlo complete path stopping at dandling nodes
    
        INPUTS:
        - G is a directed networkx graph
        - R is the int number of random walks to be performed per node
        - nodelist is the list of nodes, used to order the nodes in a particular way
        - alpha is the dampening factor of Pagerank. default is 0.85

        OUTPUTS:
        - walk_visited_count is a Compressed Sparse Row (CSR) matrix;
        element (i,j) is equal to the number of times v_j has been visited
        by a random walk started from v_i
        
        - mc_pagerank is the dictionary {node: pg} of the pagerank value for each node
    '''

    # validate all the inputs
    _validate_inputs(G, R, nodelist, alpha)

    N = len(G)

    if not nodelist:
        nodelist = list(G.nodes())
            
    # compute the inverse map of nodelist
    inverse_nodelist = {nodelist[j]: j for j in range(N)}

    # initialize walk_visited_count as a sparse matrix
    walk_visited_count = lil_matrix((N, N), dtype='int')

    progress_count = 0

    # perform R random walks for each node
    for node in nodelist:

        # print progress every 100 nodes
        progress_count += 1
        if progress_count % 100 == 0:
            print('progress = {:.2f}%'.format(100 * progress_count / N), end='\r')
        
        for _ in range(R):

            node_pos = inverse_nodelist[node]
            walk_visited_count[node_pos, node_pos] += 1

            current_node = node

            while np.random.uniform() < alpha:
                
                successors = list(G.successors(current_node))
                if not successors:
                    break
                    
                current_node = random.choice(successors)
                current_node_pos = inverse_nodelist[current_node]
                
                walk_visited_count[node_pos, current_node_pos] += 1

    # Convert lil_matrix to csr_matrix for efficient storage and access
    walk_visited_count = walk_visited_count.tocsr()

    # sum all visits for each node into a numpy array
    total_visited_count = np.array(walk_visited_count.sum(axis=0)).flatten()

    # reciprocal of the number of total visits
    one_over_s = 1 / sum(total_visited_count)
    
    mc_pagerank = {nodelist[j]: total_visited_count[j] * one_over_s for j in range(N)}

    print('progress = 100%       ', end='\r')
    print('\nTotal walks performed: ', N * R )
    
    return walk_visited_count, mc_pagerank


def _validate_inputs(G, R, nodelist, alpha):
    if len(G) == 0:
        raise ValueError("Graph G is empty")
    
    if not isinstance(R, int) or R <= 0:
        raise ValueError("R must be a positive integer")
    
    if not isinstance(alpha, float) or not (0 < alpha < 1):
        raise ValueError("alpha must be a float between 0 and 1")
    
    if nodelist is not None and set(nodelist) != set(G.nodes()):
        raise ValueError("nodelist does not match the nodes in G")
