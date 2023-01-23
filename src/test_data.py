# import numpy as np
# from scipy.spatial import distance_matrix
# from algorithms.exhaustive import BruteForce
# from algorithms.localsearch import OPT
# from algorithms.evolutionary import Genetic

# num_nodes = 5
# data = np.random.randint(0, 100, size = num_nodes*2).reshape(-1, 2)
# costmat = distance_matrix(data, data)

# bf = BruteForce()
# bf.fit(costmat)
# print(bf.path)
# print(bf.cost)

# opt = OPT()
# opt.fit(costmat)
# print(opt.path)

# ga = Genetic(10, .2, 1)
# ga.fit(None, max_generations = 20, costmat = costmat, verbose = True)
# print(ga.get_fittest(costmat))

import numpy as np
from algorithms.exhaustive import BruteForce
from scipy.spatial import distance_matrix
import networkx as nx
import matplotlib.pyplot as plt

def draw(data):
    # Create Graph
    G = nx.Graph()
    for i in range(len(data)):
      G.add_node(i)
    # Draw Graph
    nx.draw_networkx(G, pos = {i: data[i] for i in range(len(data))})
    plt.show()

num_nodes = 5
data = np.random.randint(0, 100, size = num_nodes*2).reshape(-1, 2)
costmat = distance_matrix(data, data)

opt_path, opt_cost = BruteForce.fit(costmat)
print(opt_path, opt_cost)
draw(data)