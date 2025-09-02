
import numpy as np

def gradientDescent(alpha : float, difference : np.ndarray, x : np.ndarray, w : np.ndarray, b : np.ndarray):

    # calculate new w
    dLdW = np.outer(difference, x)
    w = w - alpha * dLdW

    # calculate new b
    dLdb = difference
    b = b - alpha * dLdb

    return w, b