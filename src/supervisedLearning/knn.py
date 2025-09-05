
import numpy as np
import pandas as pd

class Point:
    def __init__(self, x : np.ndarray) -> None:
        self.value : np.ndarray = x
        self.corePoint : bool = False
        self.clusterId = -1 # current constant and this will be changed later when it is inputted into the dbscan

class Distance :
    @staticmethod
    def minkowskiDistance(df : pd.DataFrame, val2 : np.ndarray, p : float) -> pd.DataFrame :
        distances = df.apply(lambda row: (((row - val2) ** p).sum())**(1/p), axis=1)
        result = df.copy()
        result["distances"] = distances
        return result
    
    @staticmethod
    def manhattanDistance(df : pd.DataFrame, val2 : np.ndarray) -> pd.DataFrame :
        distances = df.apply(lambda row: (row - val2).abs().sum(), axis=1)
        result = df.copy()
        result["distances"] = distances
        return result 
    
    @staticmethod
    def euclideanDistance(df : pd.DataFrame, val2 : np.ndarray) -> pd.DataFrame :
        distances = df.apply(lambda row: ((row - val2) ** 2).sum(), axis=1)
        result = df.copy()
        result["distances"] = distances
        return result

class KNearestNeigbor : 
    
    def __init__(self, distanceFunction : str, neighborCount : int, target : str, minkowskiExp : float = 1) -> None: # default is manhattan
        self.distanceFunction = distanceFunction
        self.neighborCount = neighborCount
        self.minkowskiExp = minkowskiExp
        self.target = target

    def predict(self, dataset : pd.DataFrame, x : np.ndarray) :

        # algorithm check
        if self.distanceFunction == "minkowski" : # minkowski distance
            # get the dict
            dictionary = Distance.minkowskiDistance(dataset, x, self.minkowskiExp)
        elif self.distanceFunction == "euclidean" : # euclidean distance
            # get the dict
            dictionary = Distance.euclideanDistance(dataset, x)
        else :
            # get the dict
            dictionary = Distance.manhattanDistance(dataset, x)
            
        # attach label back
        dictionary[self.target] = dataset[self.target].values
        
        # get the n minimum value from dict based on distance
        neighbors = dictionary.sort_values(by="distances").head(self.neighborCount) # this should be per label

        prediction = neighbors[self.target].mode()[0]  
        return prediction
