import networkx as nx
import matplotlib.pyplot as plt

def draw(data, path):
    # Create Graph
    G = nx.MultiDiGraph()
    for i in range(len(data)):
      G.add_node(i)
    for i in range(len(path) - 1):
      G.add_edge(path[i], path[i + 1], label = 0)
      G.add_edge(path[i], path[i + 1], label = 1)
    # Draw Graph
    nx.draw_networkx(G, pos = {i: data[i] for i in range(len(data))}, connectionstyle='arc3, rad = 0.1')
    plt.show()