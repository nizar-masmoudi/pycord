import numpy as np
from algorithms.exhaustive import BruteForce
from algorithms.localsearch import Greedy
from scipy.spatial import distance_matrix, KDTree
import networkx as nx
import matplotlib.pyplot as plt
import numpy.ma as ma
from utils import draw

# np.random.seed(123)
num_nodes = 10
data = np.random.randint(0, 100, size = num_nodes*2).reshape(-1, 2)
costmat = distance_matrix(data, data)

# opt_path, opt_cost = BruteForce.fit(costmat)

# path, cost = Greedy.fit(costmat)



# draw(data, path)