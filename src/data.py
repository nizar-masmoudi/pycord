import numpy as np
from tabulate import tabulate
from typing import Union
from scipy.spatial import distance_matrix
import networkx as nx # Draw
import matplotlib.pyplot as plt

class Data():
  def __init__(self, coords: Union[list, np.array], names: list = None) -> None:
    self.coords = np.array(coords)
    self.names = names if names else ['Node {}'.format(i) for i in range(1, len(self.coords) + 1)]

  @property
  def costmat(self, p: int = 2):
    return distance_matrix(self.coords, self.coords, p = p)

  def __repr__(self, decimals: int = 4) -> str:
    dict_ = {}
    dict_['Node'] = self.names
    dict_['X'] = self.coords[:, 0]
    dict_['Y'] = self.coords[:, 1]
    return tabulate(dict_, headers = 'keys', stralign = 'left', tablefmt='pretty')

  def __len__(self):
    return len(self.coords)

  def draw(self):
    raise NotImplementedError('This method is not implemented yet! I\'m just busy or lazy... most likely lazy :(')
  
def make_data(n_nodes: int, names: list = None) -> Data:
  coords = np.random.randint(0, 100, size = 2*n_nodes).reshape(-1, 2)
  return Data(coords, names)