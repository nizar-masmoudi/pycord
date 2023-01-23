import numpy as np
import numpy.ma as ma
from typing import Union, Tuple

#####################################! 2-Opt !#####################################
class OPT:
  def __init__(self) -> None:
    self.path = None

  def swap(self, i: int, j: int) -> None:
    path_ = self.path.copy()
    path_[i + 1], path_[j] = path_[j], path_[i + 1]
    path_[i + 2:j] = list(reversed(path_[i + 2:j]))
    self.path = path_

  def fit(self, costmat: np.ndarray) -> None:
    self.path = list(range(len(costmat)))
    self.path.append(self.path[0])
    improved = True
    while improved:
      improved = False
      for i in range(0, len(self.path) - 2):
        for j in range(i + 2, len(self.path) - 1):
          if costmat[self.path[i + 1], self.path[i]] + costmat[self.path[j + 1], self.path[j]] > costmat[self.path[j], self.path[i]] + costmat[self.path[j + 1], self.path[i + 1]]:
            self.swap(i, j)
            improved = True
      
#####################################! Greedy Search !#####################################
# TODO      
class Greedy:
  @staticmethod
  def fit(costmat: Union[np.array, list]) -> Tuple[list, float]:
    idxs = range(len(costmat))
    path = [0]
    cost = 0
    for _ in range(len(costmat)): # For each hop (number of hops is equal to number of stations)
      masked = ma.masked_where(np.logical_or(costmat[path[-1]] == 0, np.isin(idxs, path)), costmat[path[-1]]) # Mask current station and visited stations
      if masked.mask.sum() < len(idxs): 
        cost += np.min(masked)
        path.append(np.argmin(masked))
    # Return to initial station
    path.append(path[0])
    cost += costmat[path[-1], path[-2]]
    return path, cost