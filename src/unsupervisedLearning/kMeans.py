
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
            randomRow = dataset.iloc[np.random.randint(0, len(dataset))][self.features].to_numpy()
            self.clusterClasses[Point(randomRow)] = i + 1

        dataset["label"] = -1 # add label column

        # calculate the distance of each point (except chosen one) to the random row above
        for iter in range(self.maxIteration):
            # iterate for each cluster in cluster classes
            for idx, row in dataset.iterrows():
                rowPoint = Point(row[self.features].to_numpy())
                distances :dict[Point, float] = {c: c.hammingDistance(rowPoint.x) for c in self.clusterClasses.keys()}
                bestCluster, _ = min(distances.items(), key=lambda item: item[1])
                dataset.at[idx, "label"] = self.clusterClasses[bestCluster]

            # get the mean value of each class and make it the new centroid
            newClusterClasses : dict[Point, int]= {}
            for clusterId in range(self.clusterNumber):
                clusterData = dataset[dataset["label"] == clusterId][self.features]
                if not clusterData.empty:
                    newCentroid = clusterData.mode().iloc[0].to_numpy()
                    newClusterClasses[Point(newCentroid)] = clusterId
            
            # calculate the variance
            totalVariance = 0
            for clusterPoint in newClusterClasses.keys():
                clusterData = dataset[dataset["label"] == newClusterClasses[clusterPoint]][self.features]
                for _, row in clusterData.iterrows():
                    totalVariance += clusterPoint.hammingDistance(row.to_numpy())

            variance = abs(totalVariance - self.currentVariance) / self.currentVariance

            # break if the variance is already low,             # else continue training
            if variance < self.THRESHOLD :
                break 

            self.clusterClasses = newClusterClasses
                
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