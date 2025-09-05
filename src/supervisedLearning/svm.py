
'''
TODO : 
1. list the dataset (already converted into all number) and target
2. keep a list and based on all unique value of target column, iterate each pair 
3. based on the function y = wx + b, create initial value of w (1 x K) and b (single value) (randomly and a small value)
4. '''

import numpy as np
import pandas as pd
from scipy.stats import mode


class SupportVectorMachine :
    def __init__(self, dataset : pd.DataFrame, target, alpha : float, learningRate, regularizationTerm, n : int = 100000) -> None:

        self.dataset = dataset
        self.target = target
        self.learningRate = learningRate
        self.alpha = alpha
        self.regularizationTerm = regularizationTerm
        self.n = n
        self.models = []
        if (regularizationTerm == "l1") :
            self.regularizationCoefficient = 0.01
        elif (regularizationTerm == "l2") :
            self.regularizationCoefficient = 0.1
        else :
            self.regularizationCoefficient = 0

    def train(self) :

        # init
        parsedDataset = self.dataset.copy()
        parsedDataset = parsedDataset.drop([self.target])

        # get the dimension needed for w and b (target unique values x input length dimension)
        d = len(self.dataset.columns) - 1 # minus one is for the target column
        k = len(self.dataset[self.target].unique())
        classes = self.dataset[self.target].unique()
        features = [col for col in self.dataset.columns if col != self.target]
    
        # construct initial w 
        w : np.ndarray = np.random.randn(d) * 0.01

        # construct initial b
        b : float = 0.0
        
        # iteration for epoch
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes):
                if i < j:
                    # generate a new dataset that has a new column "label" and assign -1 and 1 to rows with pairfeature and feature
                    transformedDataset = parsedDataset.copy()
                    transformedDataset["label"] = 0
                    transformedDataset.loc[self.dataset[self.target] == class1, "label"] = 1
                    transformedDataset.loc[self.dataset[self.target] == class2, "label"] = -1

                    # construct initial w 
                    w : np.ndarray = np.random.randn(d) * 0.01

                    # construct initial b
                    b : float = 0.0

                    #  here w and b has been initialized so that it can be inputted for hinge loss and gradient descent

                    for epoch in range(self.n) :
                        # objective function for loss value
                        loss = self.objectiveFunction(transformedDataset, classes, w, b)

                        # regularization by using gradient descent
                        if loss < 0.0000001 : 
                            return w, b

                        w, b = self.gradientDescent(transformedDataset, w, b)
                    # store the w and b according to class1 and class2
                    self.models.append((class1, class2, w, b))

        return w, b
    
    def objectiveFunction(self, trainedDataset, classes, w : np.ndarray, b: float) : 
        loss = self.hingeLoss(trainedDataset, classes, w, b)

        R = 0
        if self.regularizationTerm == "l2":
            R = 0.5 * np.sum(w ** 2)
        elif self.regularizationTerm == "l1":
            R = np.sum(np.abs(w))

        return loss + R * self.regularizationCoefficient
    

    def hingeLoss(self, trainedDataset : pd.DataFrame, classes : list, w, b) :

        summation = 0
        for iteration in range(len(trainedDataset)):
            # get the value of column "label"
            x = trainedDataset.drop(columns=[self.target, "label"]).iloc[iteration].to_numpy()
            y = trainedDataset["label"].iloc[iteration]
            summation += max(0, 1 - y * (np.dot(w, x) + b))

        return summation / len(trainedDataset)

    
    def gradientDescent(self, trainedDataset: pd.DataFrame, w: np.ndarray, b: float):
        for i in range(len(trainedDataset)):
            x = trainedDataset.drop(columns=[self.target, "label"]).iloc[i].to_numpy()
            y = trainedDataset["label"].iloc[i]

            condition = y * (np.dot(w, x) + b)

            # Gradient for w
            if condition >= 1:
                if self.regularizationTerm == "l2":
                    dw = self.regularizationCoefficient * w
                elif self.regularizationTerm == "l1":
                    dw = self.regularizationCoefficient * np.sign(w)
                else:
                    dw = 0
                db = 0
            else: # case for < 1 (based on the loss function)
                # Both hinge loss + regularization contribute
                if self.regularizationTerm == "l2":
                    dw = self.regularizationCoefficient * w - y * x
                elif self.regularizationTerm == "l1":
                    dw = self.regularizationCoefficient * np.sign(w) - y * x
                else:
                    dw = -y * x
                db = -y

            # Update
            w = w - self.learningRate * dw
            b = b - self.learningRate * db

        return w, b

    
    def predict(self, x: pd.Series) -> None:
        # this function will predict the borough name given a row of data
        preds = []
        for class1, class2, w, b in self.models:
            x_vec = x.drop(self.target).to_numpy()
            decision = np.dot(w, x_vec) + b
            pred = class1 if decision >= 0 else class2
            preds.append(pred)

        # mode with NumPy
        vals, counts = np.unique(preds, return_counts=True)
        return vals[np.argmax(counts)]
