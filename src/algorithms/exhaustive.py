import numpy as np
from itertools import permutations
from typing import Union, Tuple

#####################################! Brute Force !#####################################  
class BruteForce:
  @staticmethod
  def fit(costmat: Union[np.array, list]) -> Tuple[list, float]:
    '''Run the brute force algorithm. This method is exhaustive as it tries all possible combinations and picks the best one.
    This method returns the absolute optimal solution.

    Parameters:
    costmat (Union[np.array, list]): Cost matrix

    Returns:
    Tuple[list, float]: Optimal path with its total cost (nodes are identified by their positions w.r. to the cost matrix)
    '''
    inter_stations = list(range(len(costmat)))[1:]
    combinations = [[0] + list(combination) + [0] for combination in permutations(inter_stations)]
    costs = [np.sum(np.diagonal(costmat[combination][:, combination], offset = 1)) for combination in combinations]
    opt_path = combinations[np.argmin(costs)]
    opt_cost = np.min(costs)
    return opt_path, opt_cost
    