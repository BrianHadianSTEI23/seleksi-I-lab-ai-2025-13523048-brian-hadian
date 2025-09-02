import pandas as pd
import numpy as np
import random
from src.utils.crossEntropy import crossEntropy
from src.utils.gradientDescent import gradientDescent

def softmax (training : pd.DataFrame, target : str, alpha : float, n : int = 100000) : # -> np.array

    # init
    parsedTraining = training.copy()
    parsedTraining = parsedTraining.drop([target])

    # get the dimension needed for w and b (target unique values x input length dimension)
    d = len(training.columns) - 1 # minus one is for the target column
    k = len(training[target].unique())

    # construct initial w 
    w = np.random.rand(d, k)
    w = w.transpose()

    # construct initial b
    b = np.random.random(k)

    # do loop of softmax given how many iterations / as many as the data can provide
    for iteration in range(n) : 
        # extract the value from the training
        y = training[target].iloc[iteration]
        x = parsedTraining.iloc(iteration).to_numpy()

        # get y value
        y_hat = w @ x + b

        # get z value (normalization)
        summation = 0
        for el in y_hat :
            summation += np.e ** el
        z = (np.e**y_hat) / summation

        # return the loss function
        loss = crossEntropy(z, y)

        # option to break if the loss is already low enough
        if loss < 0.0000001 :
            break

        # return the gradient descent
        y_onehot = np.zeros(k)
        y_onehot[y] = 1
        difference = z - y_onehot
        w, b = gradientDescent(alpha, difference, x, w, b)

    # return the w and b value
    return w, b