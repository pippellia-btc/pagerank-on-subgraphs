import json

def load_network(name):

    if type(name) != str:
        name = str(name)

    # loading the index_map
    with open('index_map_' + name + '.json', 'r') as f:
        index_map = json.load(f)
    
    # loading the JSON for the graph
    with open('network_graph_' + name + '.json', 'r') as f:
        data = json.load(f)

    # Convert JSON back to graph
    network_graph = nx.node_link_graph(data)

    return index_map, network_graph
