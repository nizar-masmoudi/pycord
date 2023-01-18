import numpy as np
from itertools import permutations

#####################################! Brute Force !#####################################
class BruteForce:
  def __init__(self) -> None:
    self.path = None
    self.cost = None
  def fit(self, data: list, cost_matrix: np.ndarray) -> np.ndarray:
    combinations = list(permutations(range(1, len(data) - 1)))
    combinations = [[0] + list(combination) + [0] for combination in combinations]
    costs = [np.sum(np.diagonal(cost_matrix[combination][:, combination], offset = 1)) for combination in combinations]
    self.path = combinations[np.argmin(costs)]
    self.cost = np.min(costs)