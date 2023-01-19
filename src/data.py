import numpy as np
from tabulate import tabulate
from typing import Union
from scipy.spatial import distance_matrix
import networkx as nx # Draw
import matplotlib.pyplot as plt
from itertools import permutations
from collections.abc import Iterable

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
    dict_['X'] = np.around(self.coords[:, 0], decimals)
    dict_['Y'] = np.around(self.coords[:, 1], decimals)
    return tabulate(dict_, headers = 'keys', stralign = 'left', tablefmt='pretty')

  def __len__(self):
    return len(self.coords)
  
  def __getitem__(self, idx):
    if isinstance(idx, Iterable):
      return Data(coords = [self.coords[i] for i in idx], names = [self.names[i] for i in idx])
    return (self.names[idx], self.coords[idx])
  
  def draw(self, ax = None, **kwargs):
    # Create Graph
    G = nx.Graph()
    for i in range(len(self)):
      G.add_node(self.names[i])
    G.add_edges_from(list(permutations(self.names, 2)))
    # Draw Graph
    ax = ax if ax else plt.subplot()
    nx.draw_networkx(G, pos = {self.names[i]: self.coords[i] for i in range(len(self.names))}, ax = ax, **kwargs)

class Path():
  def __init__(self, coords: Union[list, np.array], names: list, cost: float) -> None:
    self.coords = np.array(coords)
    self.names = names
    self.cost = cost
    
  def __len__(self):
    return len(self.coords)
  
  def __getitem__(self, idx):
    if isinstance(idx, Iterable):
      return Data(coords = [self.coords[i] for i in idx], names = [self.names[i] for i in idx])
    return (self.names[idx], self.coords[idx])
  
  def draw(self, ax = None, **kwargs):
    raise NotImplemented

def make_data(n_nodes: int, names: list = None) -> Data:
  coords = np.random.randint(0, 100, size = 2*n_nodes).reshape(-1, 2)
  return Data(coords, names)