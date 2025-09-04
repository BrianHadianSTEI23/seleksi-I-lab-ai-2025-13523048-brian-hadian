import pandas as pd
import numpy as np

class SoftmaxRegression:

    def __init__(self, features : list[str], classes : list[str] , target : str, alpha : float, regularizationTerm : str, n : int = 100000) -> None:

        # init
        self.target = target
        self.alpha = alpha
        self.n = n
        self.features = features
        self.classes = classes
        self.regularizationTerm = regularizationTerm
        if (regularizationTerm == "l1") :
            self.lambda_val = 0.01
        elif (regularizationTerm == "l2") :
            self.lambda_val = 0.1
        else :
            self.lambda_val = 0

        # construct initial w 
        self.w = np.random.rand(len(features), len(classes)).transpose()

        # construct initial b
        self.b = np.random.random(len(classes))

    def train(self, x : np.ndarray, y : np.ndarray) : # -> np.array

        # do loop of softmax given how many iterations / as many as the data can provide

        # extract the value from the training
        # y = parsedTraining[self.target].iloc[iteration]
        # x = parsedTraining.iloc[iteration].to_numpy()

        # get y value
        y_pred = self.w @ x + self.b
        y_true = np.zeros(len(self.classes))
        y_true[y] = 1

        # get z value (normalization)
        exp_y = np.exp(y_pred - np.max(y_pred))
        z = exp_y / np.sum(exp_y)


        # return the loss function
        loss = self.crossEntropyLoss(z, y_true)
        # option to break if the loss is already low enough
        if loss < 0.0000001 :
            return self.w, self.b
        
        # return the gradient descent
        difference = z - y_true
        self.gradientDescent(self.alpha, difference, x, self.w, self.b, self.regularizationTerm)

        # return the w and b value
        return self.w, self.b
    
    def gradientDescent(self, alpha : float, difference : np.ndarray, x : np.ndarray, w : np.ndarray, b : np.ndarray, regularizationTerm : str):

        dLdW = np.outer(difference, x)
        if (regularizationTerm == "l1") :
            # calculate new w
            dLdW += self.lambda_val * np.sign(w)
        elif (regularizationTerm == "l2") : 
            # calculate new w
            dLdW += self.lambda_val * 2 * w
        self.w -= alpha * dLdW

        # calculate new b
        dLdb = difference
        self.b -= alpha * dLdb
    
    def crossEntropyLoss (self, z, y_onehot) : 
        return -np.sum(y_onehot * np.log(z + 1e-15))