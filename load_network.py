import json

def load_network(name):

    '''
    This function loads the network graph and index map from storage.

    INPUTS
    ------
    name: str
        the name of the files, usually a timestamp like '1714823396'

    OUTPUTS:
    -------
    index_map: dict
        The dictionary {pk: node} that maps each public key to its corrispondent node id in the graph

    network_graph: graph
        A Networkx graph.
    '''

    if type(name) != str:
        name = str(name)

    # loading the index_map
    with open('index_map_' + name + '.json', 'r') as f:
        index_map = json.load(f)
    
    # loading the JSON for the graph
    with open('network_graph_' + name + '.json', 'r') as f:
        data = json.load(f)

    # convert JSON back to graph
    network_graph = nx.node_link_graph(data)

    return index_map, network_graph
