
import numpy as np
from itertools import combinations_with_replacement

class PolynomialRegression : 

    def __init__(self, degree : int, learningRate : float, datasetSize : int, regularizationTrem : str, classes : list[str], features : list[str], iteration = 100000) -> None:

        self.degree = degree
        self.classes = classes
        self.features = features
        self.beta : np.ndarray = np.ndarray([])
        self.learningRate = learningRate
        self.regularizationTerm = regularizationTrem
        self.iteration = iteration
        self.datasetSize = datasetSize

    def train(self, X : np.ndarray, y : np.ndarray) : # y is a 1x1 value (the label) 

        # train the model according to the iteration
        for epoch in range(self.iteration):
            for i in range(self.datasetSize):
                # given a dataset and its degree, compute the y value based on the polynomial function (with its corresponding degree)
                xPoly = np.array([self.polynomialFeatures(x, self.degree) for x in X])

                # # after getting the xPoly value, beta matrix can be found using the derivative of the mean squared error
                # newBeta = np.linalg.inv(xPoly.T @ xPoly) @ xPoly.transpose() @ y

                # # update the beta matrix
                # self.beta = newBeta

            # calculate the loss function 
            loss = 0
            for j in range(self.datasetSize):
                yPred = np.dot(self.beta, self.polynomialFeatures(, self.degree))
                loss += self.meanSquaredError()

        return self.beta

        
    def polynomialFeatures(self, x: np.ndarray, degree: int):
        # this function will expand input features x into polynomial terms up to given degree.
        n_features = x.shape[0]
        features = [1.0]  # bias term
        
        for deg in range(1, degree + 1):
            for comb in combinations_with_replacement(range(n_features), deg):
                term = np.prod([x[i] for i in comb])
                features.append(term)
        
        return np.array(features)
    
    def meanSquaredError(self, yPred : np.ndarray, yTrue : np.ndarray) : # this loss function will implement the mean squared error
        return (yTrue - yPred) ** 2
    
    def predict(self, x : np.ndarray):
        # this function will return the predicted value from x and beta that has been trained
        
        xPoly = self.polynomialFeatures(x, self.degree)
        return np.dot(self.beta, xPoly)