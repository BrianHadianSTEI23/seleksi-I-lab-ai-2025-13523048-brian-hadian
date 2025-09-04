
import numpy as np

class Point:
    def __init__(self, x : np.ndarray) -> None:
        self.value : np.ndarray = x
        self.corePoint : bool = False
        self.clusterId = -1 # current constant and this will be changed later when it is inputted into the dbscan
        pass

class Distance :
    @staticmethod
    def minkowski(x : Point, y : Point, p : float) : 
        return (np.sum((abs(x.value - y.value)) ** p)) ** (1/p)    
    
    @staticmethod
    def manhattan(x : Point, y : Point) : 
        p = 1
        return (np.sum((abs(x.value - y.value)) ** p)) ** (1/p)    
    
    @staticmethod
    def euclidean(x : Point, y : Point) : 
        p = 2
        return (np.sum((abs(x.value - y.value)) ** p)) ** (1/p)    
    
class Cluster : 
    def __init__(self, epsilon : float, points : list[Point], corePoints : list[Point], minkowskiExp : float, distanceFunction : str = "manhattan") -> None:
        self.maxDistance = epsilon
        self.corePoints = corePoints
        self.points = points
        self.minkowskiExp = minkowskiExp
        self.distanceFunction = distanceFunction
    
    def calculateMinDistance(self, x : np.ndarray):
        minVal = np.inf
        for core in self.corePoints:
            if (self.distanceFunction == "minkowski") :
                currVal = Distance.minkowski(core, Point(x), self.minkowskiExp)
                if (minVal > currVal):
                    minVal = currVal
            elif (self.distanceFunction == "euclidean") :
                currVal = Distance.euclidean(core, Point(x))
                if (minVal > currVal):
                    minVal = currVal
            else : # 
                currVal = Distance.manhattan(core, Point(x))
                if (minVal > currVal):
                    minVal = currVal
        return minVal

class DBScan :
    def __init__(self, epsilon : float, minSample : int, features : list[str], classes : list[str], minkowskiExp : float, distanceFunction : str = "manhattan") -> None:
        self.minSample = minSample
        self.maxDistance = epsilon
        self.features = features
        self.classes = classes
        self.distanceFunction = distanceFunction
        if distanceFunction == "minkowski":
            self.minkowskiExp = minkowskiExp
        else : 
            self.minkowskiExp = 1 # default is manhattan

        self.clusterClasses : dict[int, Cluster] = {}
        self.remainingPoints : list[Point] = []
        self.currCorePoint : list[Point] = []

    def train(self, dataset):

        # transform all row into point and put it into remaining points
        for k in range(len(dataset)):
            self.remainingPoints.append(Point(dataset.iloc[k][self.features].to_numpy()))

        # get random row from the dataset
        n : int = len(self.remainingPoints)
        i : int = 0
        while (len(self.remainingPoints) != 0):
            randIdx = np.random.randint(0, len(self.remainingPoints))
            currRandomPoint = self.remainingPoints[randIdx]

        # turn that row into Point and then based on that Point, count the distance from that point into any other point except itself
            for j in range(len(self.remainingPoints) - 1):
                if (j != randIdx):
                    iterPoint = Point(dataset.iloc[j][self.features].to_numpy())
                    if (self.distanceFunction == "minkowski"):
                        currDistance = Distance.minkowski(iterPoint, currRandomPoint, self.minkowskiExp)

                        # for every point that has distance < self.epsilon, its CorePoint bool value is turned on, else false (thus every row needs to be converted 
                        # into a Point data type and then this DBScan need to have array of those Points, either corepoint or not)
                        if (currDistance < self.maxDistance):
                            iterPoint.corePoint = True
                            self.currCorePoint.append(iterPoint)
                        else :
                            self.remainingPoints.append(iterPoint) # add all the row into array
                    elif (self.distanceFunction == "euclidean") : # euclidean
                        currDistance = Distance.euclidean(iterPoint, currRandomPoint)

                        # for every point that has distance < self.epsilon, its CorePoint bool value is turned on, else false (thus every row needs to be converted 
                        # into a Point data type and then this DBScan need to have array of those Points, either corepoint or not)
                        if (currDistance < self.maxDistance):
                            iterPoint.corePoint = True
                            self.currCorePoint.append(iterPoint)
                        else :
                            self.remainingPoints.append(iterPoint) # add all the row into array

                    else : # manhattan
                        currDistance = Distance.manhattan(iterPoint, currRandomPoint)

                        # for every point that has distance < self.epsilon, its CorePoint bool value is turned on, else false (thus every row needs to be converted 
                        # into a Point data type and then this DBScan need to have array of those Points, either corepoint or not)
                        if (currDistance < self.maxDistance):
                            iterPoint.corePoint = True
                            self.currCorePoint.append(iterPoint)
                        else :
                            self.remainingPoints.append(iterPoint) # add all the row into array

            
        # for every point that is CorePoint, it will be checked based on its distance with other core points
            for point in self.currCorePoint:
                potentialCore = []
                count = 0
                if point.corePoint :
                    for potentialPoint in self.remainingPoints : 
                        if self.distanceFunction == "minkowski":
                            currDistance = Distance.minkowski(point, potentialPoint, self.minkowskiExp)

                            if (currDistance < self.maxDistance) : 
                                potentialCore.append(potentialPoint)
                                count += 1
                        elif self.distanceFunction == "euclidean":
                            currDistance = Distance.euclidean(point, potentialPoint)

                            if (currDistance < self.maxDistance) : 
                                potentialCore.append(potentialPoint)
                                count += 1
                        else :
                            currDistance = Distance.manhattan(point, potentialPoint)

                            if (currDistance < self.maxDistance) : 
                                potentialCore.append(potentialPoint)
                                count += 1

        # for every corePoint that has distance < epsilon and there is a certain amount of it (>= minsample), then that corePoint is pushed into clusterClass along with the clusterId
                    if (count < self.minSample):
                        # create cluster and add the cluster into the cluster classes
                        newCluster = Cluster(self.maxDistance, potentialCore, self.currCorePoint, self.minkowskiExp, self.distanceFunction)
                
                        self.clusterClasses[i] = newCluster

                        self.remainingPoints = [] # reset
                        self.currCorePoint = [] # reset

        # then do for every possible point until the remaining is only the outlier, it will be pushed also into cluster with -1 class
        # adding the outlier class
        outlierCluster = Cluster(self.maxDistance, self.remainingPoints, self.remainingPoints, self.minkowskiExp, self.distanceFunction)
        self.clusterClasses[-1] = outlierCluster

        return
    
    def predict(self, x : np.ndarray) : 
        # this function will calculate the distance of x to each cluster


        return
    