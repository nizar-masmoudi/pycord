import numpy as np
from itertools import permutations
from data import Data

#####################################! Brute Force !#####################################
# class BruteForce:
#   def __init__(self) -> None:
#     self.path = None
#     self.cost = None
#   def fit(self, data: list, costmat: np.ndarray) -> np.ndarray:
#     combinations = list(permutations(range(1, len(data) - 1)))
#     combinations = [[0] + list(combination) + [0] for combination in combinations]
#     costs = [np.sum(np.diagonal(costmat[combination][:, combination], offset = 1)) for combination in combinations]
#     self.path = combinations[np.argmin(costs)]
#     self.cost = np.min(costs)

class BruteForce:
  def __init__(self) -> None:
    self.path = None
    self.cost = None
  def fit(self, data: Data) -> np.ndarray:
    costmat = data.costmat
    combinations = list(permutations(range(1, len(data.coords) - 1)))
    combinations = [[0] + list(combination) + [0] for combination in combinations]
    costs = [np.sum(np.diagonal(costmat[combination][:, combination], offset = 1)) for combination in combinations]
    self.path = data[combinations[np.argmin(costs)]]
    self.cost = np.min(costs)