import numpy as np
from algorithms.exhaustive import BruteForce
from algorithms.localsearch import Greedy, OPT
from algorithms.evolutionary import Genetic
from scipy.spatial import distance_matrix
from utils import draw

num_nodes = 10
# np.random.seed(111)
data = np.random.randint(0, 100, size = num_nodes*2).reshape(-1, 2)
costmat = distance_matrix(data, data)

# opt_path, opt_cost = BruteForce.fit(costmat)

# path, cost = Greedy.fit(costmat)

path, cost = OPT.fit(costmat)

gen = Genetic(population_size = 100, mutation_rate = .2, elitism = 2)
gen.simulate(max_generations = 30, costmat = costmat, verbose = True)
indv = gen.get_fittest(costmat)
path = indv.genomes

draw(data, path)