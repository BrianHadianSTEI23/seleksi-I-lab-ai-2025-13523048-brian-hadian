
import numpy as np

def crossEntropy (z, y_onehot) : 
    return -np.sum(y_onehot * np.log(z + 1e-15))