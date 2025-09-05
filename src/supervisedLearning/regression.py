
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement

class PolynomialRegression : 

    def __init__(self, degree : int, learningRate : float, regularizationTrem, features : list[str], iteration = 100000) -> None:

        self.degree = degree
        self.features = features
        self.beta : np.ndarray = np.zeros(len(features))
        self.learningRate = learningRate
        self.regularizationTerm = regularizationTrem
        self.iteration = iteration

    def train(self, dataset: pd.DataFrame, y: np.ndarray):
        # Expand dataset into polynomial features
        xPoly = np.array([self.polynomialFeatures(x.to_numpy(), self.degree) 
                        for _, x in dataset[self.features].iterrows()])
        
        m, _ = xPoly.shape
        
        for epoch in range(self.iteration):
            yPred = xPoly @ self.beta
            error = yPred - y
            
            # gradient (MSE)
            grad = (1/m) * (xPoly.T @ error)
            
            # regularization
            if self.regularizationTerm == "l2":
                grad += (self.learningRate / m) * self.beta
            elif self.regularizationTerm == "l1":
                grad += (self.learningRate / m) * np.sign(self.beta)
            
            # update beta
            self.beta -= self.learningRate * grad
        
        return self.beta
        
    def polynomialFeatures(self, x: np.ndarray, degree: int):
        # this function will expand input features x into polynomial terms up to given degree.
        n_features = x.shape[0]
        features = [1.0]  # bias term
        
        for deg in range(1, degree + 1):
            for comb in combinations_with_replacement(range(n_features), deg):
                term = float(np.prod([x[i] for i in comb]))
                features.append(term)
        
        return np.array(features)
    
    def meanSquaredError(self, yPred : np.ndarray, yTrue : np.ndarray) : # this loss function will implement the mean squared error
        return (yTrue - yPred) ** 2
    
    def predict(self, x : np.ndarray):
        # this function will return the predicted value from x and beta that has been trained
        
        xPoly = self.polynomialFeatures(x, self.degree)
        return np.dot(self.beta, xPoly)