from data import Data, make_data
import numpy as np
data = Data(np.array([[1, 1], [1, 2], [2, 1], [2, 2]]))
data = make_data(n_nodes = 5)
print(data)
data.draw()