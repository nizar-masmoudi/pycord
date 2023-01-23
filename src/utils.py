import networkx as nx
import matplotlib.pyplot as plt
from typing import Union
import numpy as np

# TODO - Add ax
def draw(data: Union[np.array, list], path: list):
  '''Plot nodes and directed path using `networkx` and `matplotlib`.

    # Parameters:
    data (Union[np.array, list]): Array of nodes coordinates
    path (list): List of sorted nodes according to a certain path

    # Returns:
    None
    
    # Example:
    >>> num_nodes = 20
    >>> data = np.random.randint(0, 100, size = num_nodes*2).reshape(-1, 2)
    >>> path = [0, 1, 3, 4, 2, 0]
    >>> draw(data, path)
    '''
  # Create Graph
  G = nx.MultiDiGraph()
  for i in range(len(data)):
    G.add_node(i)
  for i in range(len(path) - 1):
    G.add_edge(path[i], path[i + 1], label = 0)
    G.add_edge(path[i], path[i + 1], label = 1)
  # Draw Graph
  pos = {i: data[i] for i in range(len(data))}
  nx.draw_networkx(G, pos = pos, connectionstyle='arc3, rad = 0.1')
  plt.show()