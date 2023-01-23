import numpy as np

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
  def __init__(self) -> None:
    pass