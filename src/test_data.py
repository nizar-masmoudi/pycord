import numpy as np
from algorithms.exhaustive import BruteForce
from algorithms.localsearch import Greedy, OPT
from scipy.spatial import distance_matrix
from utils import draw

# np.random.seed(123)
num_nodes = 20
data = np.random.randint(0, 100, size = num_nodes*2).reshape(-1, 2)
costmat = distance_matrix(data, data)

# opt_path, opt_cost = BruteForce.fit(costmat)

# path, cost = Greedy.fit(costmat)

path, cost = OPT.fit(costmat)

draw(data, path)