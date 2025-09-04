
import numpy as np
import pandas as pd

class Point : 
    def __init__(self, x : np.ndarray) -> None:
        self.x = x
        pass
    
    def hammingDistance(self, x2: np.ndarray) -> float:
        difference = 0
        for i in range(len(x2)):
            if (self.x[i] != x2[i]):
                difference += 1
        return difference / len(x2)


class KMeans:

    def __init__(self, clusterNumber : int, maxIteration : int, features : list[str], classes : list[str]) -> None:
        self.clusterNumber = clusterNumber
        self.maxIteration = maxIteration
        self.features = features
        self.classes = classes
        self.clusterClasses : dict[Point, int] = {}
        self.currentVariance = 1
        self.THRESHOLD = 1 / (clusterNumber ** (2 ** 0.5))
        self.MAX_VARIANCE = 1 # constant

    def train(self, dataset: pd.DataFrame) : # dataset has been cleared of label column
        # get clusterNumber of random row
        for i in range(self.clusterNumber):
            randomRow : np.ndarray = dataset.iloc[np.random.randint(0, len(dataset))].to_numpy()
            self.clusterClasses[Point(randomRow)] = i + 1

        # calculate the distance of each point (except chosen one) to the random row above
        for iter, row in enumerate(dataset):
            # iterate for each cluster in cluster classes
            distanceDict = {}
            currentClass : Point = None
            for i, cluster in enumerate(self.clusterClasses.keys()):
                # rowPoint = Point(row)
                if Point(row) != cluster:
                    distanceDict[cluster] = cluster.hammingDistance(row)
            # get the smallest value out of distance dict
            for i, thing in enumerate(distanceDict.keys()) :
                if (distanceDict[thing] == min(distanceDict.values())):
                    # categorize the points (non chosen one) into each class (each class need to be )
                    dataset["label"] = i
                    currentClass = thing
                    break

            # get the mean value of each class and make it the new centroid
            unique_vals = dataset["label"].unique()
            for unique in unique_vals:
                currentData = dataset[dataset["label"] == unique]
                currentData = currentData[self.features]
                meanPoint = currentData.mean()
            
            # calculate the variance
            variance = (currentClass.hammingDistance(meanPoint) - self.currentVariance) / self.currentVariance

            # break if the variance is already low
            if variance < self.THRESHOLD or iter > self.maxIteration:
                return 
            # else continue training
                
        # do this until maxIteration or until it hit current variance of max : 1 / (clusterNumber ** (2 ** 0.5)) for each cluster
       

        return

    def predict(self, x : np.ndarray):
        # given an x, it will return the label
        p = np.inf
        currClass = 0
        for c in self.clusterClasses.keys():
            distance = c.hammingDistance(x)
            if (p > distance):
                p = distance
                currClass = self.clusterClasses[c]
        return currClass