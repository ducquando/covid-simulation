import networkx as nx
import matplotlib.pyplot as plt

def visualize_situation(network, status, **kwargs):
    """
    Visualize the network with their member's Covid status. 

    Params:
        network (nx.Graph): The network of people
        statuses (np.ndarray): The status of each person in the network
    """
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize = (10,10))
    nx.draw_networkx_nodes(G, pos, node_size = 100, node_color = status, cmap = plt.cm.RdYlBu, ax = ax)
    # nx.draw_networkx_labels(G, pos, font_size = 8, ax = ax)
    plt.title("Covid situation")

    # location = kwargs.get('location', None)
    # filename = kwargs.get('filename', None)
    # plt.savefig(f"./{location}/{filename}.png")
    plt.show()