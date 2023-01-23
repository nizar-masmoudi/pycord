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
            
class OPT:
  @staticmethod
  def swap(path: list, i: int, j: int) -> list:
    path[i + 1], path[j] = path[j], path[i + 1]
    path[i + 2:j] = list(reversed(path[i + 2:j]))
    return path
  
  @staticmethod
  def fit(costmat: Union[np.array, list]) -> Tuple[list, float]:
    '''Runs the 2-Opt algorithm. This heuristic systematically swaps edges until finding a local minima.
    
    This method is a heuristic and returns only a local cost minima in exchange of execution speed.

    # Parameters:
    costmat (Union[np.array, list]): Cost matrix

    # Returns:
    Tuple[list, float]: Locally minimal path with its total cost (nodes are identified by their positions w.r. to the cost matrix)
    
    # Example:
    >>> num_nodes = 20
    >>> data = np.random.randint(0, 100, size = num_nodes*2).reshape(-1, 2)
    >>> costmat = scipy.spatial.distance_matrix(data, data)
    >>> path, cost = OPT.fit(costmat)
    '''
    path = list(range(len(costmat)))
    path.append(path[0])
    improved = True
    while improved:
      improved = False
      for i in range(0, len(path) - 2):
        for j in range(i + 2, len(path) - 1):
          if costmat[path[i + 1], path[i]] + costmat[path[j + 1], path[j]] > costmat[path[j], path[i]] + costmat[path[j + 1], path[i + 1]]:
            path = OPT.swap(path, i, j)
            improved = True
    min_cost = costmat[path, path[1:] + [0]]
    return path, min_cost
      
#####################################! Greedy Search !#####################################
class Greedy:
  @staticmethod
  def fit(costmat: Union[np.array, list]) -> Tuple[list, float]:
    '''Run the greedy search algorithm. This heuristic picks the closest station each hop.
    
    This method is a heuristic and returns only a local cost minima in exchange of execution speed.

    # Parameters:
    costmat (Union[np.array, list]): Cost matrix

    # Returns:
    Tuple[list, float]: Locally minimal path with its total cost (nodes are identified by their positions w.r. to the cost matrix)
    
    # Example:
    >>> num_nodes = 20
    >>> data = np.random.randint(0, 100, size = num_nodes*2).reshape(-1, 2)
    >>> costmat = scipy.spatial.distance_matrix(data, data)
    >>> path, cost = Greedy.fit(costmat)
    '''
    idxs = range(len(costmat))
    min_path = [0]
    min_cost = 0
    for _ in range(len(costmat)): # For each hop (number of hops is equal to number of stations)
      masked = ma.masked_where(np.logical_or(costmat[min_path[-1]] == 0, np.isin(idxs, min_path)), costmat[min_path[-1]]) # Mask current station and visited stations
      if masked.mask.sum() < len(idxs): 
        min_cost += np.min(masked)
        min_path.append(np.argmin(masked))
    # Return to initial station
    min_path.append(min_path[0])
    min_cost += costmat[min_path[-1], min_path[-2]]
    return min_path, min_cost