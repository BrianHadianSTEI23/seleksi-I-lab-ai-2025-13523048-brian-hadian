import pandas as pd
import numpy as np
from src.utils.crossEntropy import crossEntropy

class SoftmaxRegression:

    def __init__(self, training : pd.DataFrame, target : str, alpha : float, regularizationTerm : str, w : np.ndarray, b : np.ndarray, n : int = 100000) -> None:

        self.training = training
        self.target = target
        self.alpha = alpha
        self.n = n
        self.regularizationTerm = regularizationTerm
        if (regularizationTerm == "l1") :
            self.lambda_val = 0.01
        elif (regularizationTerm == "l2") :
            self.lambda_val = 0.1
        else :
            self.lambda_val = 0

    def train(self, regularizationTerm) : # -> np.array

        # init
        parsedTraining = self.training.copy()
        parsedTraining = parsedTraining.drop([self.target])

        # get the dimension needed for w and b (target unique values x input length dimension)
        d = len(self.training.columns) - 1 # minus one is for the target column
        k = len(self.training[self.target].unique())

        # construct initial w 
        w = np.random.rand(d, k)
        w = w.transpose()

        # construct initial b
        b = np.random.random(k)

        # do loop of softmax given how many iterations / as many as the data can provide
        for epoch in range(self.n) :

            for iteration in range(len(parsedTraining)) : 
                # extract the value from the training
                y = self.training[self.target].iloc[iteration]
                x = parsedTraining.iloc[iteration].to_numpy()

                # get y value
                y_hat = w @ x + b

                # get z value (normalization)
                exp_y = np.exp(y_hat - np.max(y_hat))
                z = exp_y / np.sum(exp_y)

                # return the loss function
                loss = crossEntropy(z, y)

                # option to break if the loss is already low enough
                if loss < 0.0000001 :
                    return w, b

                # return the gradient descent
                y_onehot = np.zeros(k)
                y_onehot[y] = 1
                difference = z - y_onehot
                w, b = self.gradientDescent(self.alpha, difference, x, w, b, self.regularizationTerm)

        # return the w and b value
        return w, b
    
    def gradientDescent(self, alpha : float, difference : np.ndarray, x : np.ndarray, w : np.ndarray, b : np.ndarray, regularizationTerm : str):

        dLdW = np.outer(difference, x)
        if (regularizationTerm == "l1") :
            # calculate new w
            dLdW += self.lambda_val * np.sign(w)
        elif (regularizationTerm == "l2") : 
            # calculate new w
            dLdW += self.lambda_val * 2 * w
        w = w - alpha * dLdW

        # calculate new b
        dLdb = difference
        b = b - alpha * dLdb

        return w, b