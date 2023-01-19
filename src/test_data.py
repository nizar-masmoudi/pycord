from data import Data, make_data
from algorithms.exhaustive import BruteForce
import numpy as np
import matplotlib.pyplot as plt

data = make_data(n_nodes = 5)
# print(data.coords)
# data.draw(font_size = 5, edge_color = (0, 0, 0, .3), width = .5)
# plt.show()

bf = BruteForce()
bf.fit(data)
print(data)
print(bf.path)
print(bf.cost)