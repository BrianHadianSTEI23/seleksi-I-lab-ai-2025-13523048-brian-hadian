
import numpy as np
import pandas as pd

class PCA :
    def __init__(self, componentNumber : int, features : list[str]) -> None:
        self.componentNumber = componentNumber
        self.eigenvectors : np.ndarray = np.ndarray(0)
        self.eigenvalues : np.ndarray = np.ndarray(0)
        self.features = features
        self.meanFeature = 0
        pass

    def train(self, dataset : pd.DataFrame):
        # init
        currDataset = dataset

        # calculate the mean for every feature
        self.meanFeature = currDataset.mean()

        # substract mean for every feature for each row
        currDataset -= self.meanFeature

        # calculate covariance matrix
        covariance = (1 / (len(currDataset) - 1)) * (currDataset.transpose() @ currDataset)

        # given covariance matrix (should be square matrix), find the eigen value and eigen vector
        self.eigenvalues, self.eigenvectors = np.linalg.eig(covariance)

        return
    
    def predict(self, x : np.ndarray):
        # this function will predict the classification of x based on the principal component that has been constructed
        # init
        pcaCoordinate = []

        # move the x into the center (by substracting it using the mean feature)
        for eigenvector in self.eigenvectors : 
            pcaCoordinate.append(eigenvector.transpose() @ x)
        
        # display the coordinate in the pca space and also its weight based on eigen values
        for i in range(len(pcaCoordinate)):
            print(f"PCA{i + 1} : {pcaCoordinate[i]}")

        print("===============================")

        for i in range(len(self.eigenvalues)):
            print(f"Weight of PC{i + 1} : {self.eigenvalues[i] / sum(self.eigenvalues)}")

        return