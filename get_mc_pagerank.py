import numpy as np
import random
from scipy.sparse import lil_matrix, csr_matrix

def get_mc_pagerank(G, R, nodelist = None, alpha=0.85):
    
    '''
        Monte-Carlo complete path stopping at dandling nodes
    
        INPUTS
        ------
        G: graph
            A directed Networkx graph. This function cannot work on directed graphs.
            
        R: int
            The number of random walks to be performed per node
            
        nodelist: list, optional
            the list of nodes in G networkx graph. 
            It is used to order the nodes in a specified way
            
        alpha: float, optional
            It is the dampening factor of Pagerank. default value is 0.85

        OUTPUTS
        -------
        walk_visited_count: CSR matrix
            a Compressed Sparse Row (CSR) matrix; element (i,j) is equal to 
            the number of times v_j has been visited by a random walk started from v_i
            
        mc_pagerank: dict
            The dictionary {node: pg} of the pagerank value for each node in G

        References
        ----------
        [1] K.Avrachenkov, N. Litvak, D. Nemirovsky, N. Osipova
        "Monte Carlo methods in PageRank computation: When one iteration is sufficient"
        https://www-sop.inria.fr/members/Konstantin.Avratchenkov/pubs/mc.pdf
    '''

    # validate all the inputs and initialize variables
    N, nodelist, inverse_nodelist = _validate_inputs_and_init_mc(G, R, nodelist, alpha)

    # initialize walk_visited_count as a sparse LIL matrix
    walk_visited_count = lil_matrix((N, N), dtype='int')

    progress_count = 0

    # perform R random walks for each node
    for node in nodelist:

        # print progress every 200 nodes
        progress_count += 1
        if progress_count % 200 == 0:
            print('progress = {:.2f}%'.format(100 * progress_count / N), end='\r')
        
        for _ in range(R):

            node_pos = inverse_nodelist[node]
            walk_visited_count[node_pos, node_pos] += 1

            current_node = node

            while random.uniform(0,1) < alpha:
                
                successors = list(G.successors(current_node))
                if not successors:
                    break
                    
                current_node = random.choice(successors)
                current_node_pos = inverse_nodelist[current_node]

                # add current node to the walk_visited_count
                walk_visited_count[node_pos, current_node_pos] += 1

    # convert lil_matrix to csr_matrix for efficient storage and access
    walk_visited_count = walk_visited_count.tocsr()

    # sum all visits for each node into a numpy array
    total_visited_count = np.array(walk_visited_count.sum(axis=0)).flatten()

    # reciprocal of the number of total visits
    one_over_s = 1 / sum(total_visited_count)
    
    mc_pagerank = {nodelist[j]: total_visited_count[j] * one_over_s for j in range(N)}

    print('progress = 100%       ', end='\r')
    print('\nTotal walks performed: ', N * R )
    
    return walk_visited_count, mc_pagerank


def _validate_inputs_and_init_mc(G, R, nodelist, alpha):

    '''
    This function validate the inputs and initialize the following variables:
    
    N: int
        the number of nodes in G Networkx graph

    nodelist : list
        the list of nodes in G Networkx graph

    inverse_nodelist : dict
       a dictionary that maps each node in G to its position in nodelist
    '''

    N = len(G)
    if N == 0:
        raise ValueError("Graph G is empty")
    
    if not isinstance(R, int) or R <= 0:
        raise ValueError("R must be a positive integer")
    
    if not isinstance(alpha, float) or not (0 < alpha < 1):
        raise ValueError("alpha must be a float between 0 and 1")
    
    if nodelist is not None and set(nodelist) != set(G.nodes()):
        raise ValueError("nodelist does not match the nodes in G")

    elif nodelist is None:
        nodelist = list(G.nodes())

    # compute the inverse map of nodelist
    inverse_nodelist = {nodelist[j]: j for j in range(N)}

    return N, nodelist, inverse_nodelist
