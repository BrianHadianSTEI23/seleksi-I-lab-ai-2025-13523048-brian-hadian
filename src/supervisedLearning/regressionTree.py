

import numpy as np
import pandas as pd

class Point : 
    
    def __init__(self, x : dict) -> None:
        self.value = x

class Node : 
    
    def __init__(self, points : list[Point], feature = None, left = None, right = None) -> None: # left and right is of type Node (recursive datatype)
        self.points = points
        self.left = left
        self.right = right
        self.feature = feature

    def calculateCurrentFeatVariance(self): 
        # count the variance
        values = np.array([p.value[self.feature] for p in self.points])
        return  np.var(values)

    def calculateCurrentFeatMean(self) : 
        values = np.array([p.value[self.feature] for p in self.points])
        return  np.mean(values)

class RegressionTree : 

    # this regression tree assume left is true and right is false

    def __init__(self, features : list[str], maxDepth : int, minSample : int, maxVariance : float = 0.05) -> None:
        
        self.features = features
        self.maxDepth = maxDepth
        self.minSample = minSample
        self.MAX_VARIANCE = maxVariance
        self.FEAT_MAX_VARIANCE = maxVariance

    def train(self, dataset : pd.DataFrame) :

        # init : push all Point (rows) into root node
        points = []
        for _, row in dataset[self.features].iterrows():
            points.append(Point(row.to_dict()))
        root = Node(points)

        # based on the features, pick one feature and calculate the variance of that particular feture according to the current sample
        self.split(root)

        return root
    
    def split(self, node : Node) : 

        if len(node.points) < self.minSample:
            return

        bestGain = 0
        bestFeature = None
        bestLeft : list[Point ]= []
        bestRight : list[Point]= []

        parentVar = np.var([np.mean(list(p.value.values())) for p in node.points])  

        for feature in self.features:

            #  if the variance is high enough, get the mean of that particular feature (split point = mean of that feature)
            splitPoint = np.mean(np.array([p.value[feature] for p in node.points]))

            # branching : left tree contains list of Point that is true according to the branch, and the right is false

            # points for left (true)
            leftPoints = []
            rightPoints = []
            for p in node.points:
                if p.value[feature] < splitPoint :
                    leftPoints.append(p)
                else :
                    rightPoints.append(p)
            
            if len(leftPoints) == 0 or len(rightPoints) == 0:
                continue

            # do split again for left (recur)
            leftVar = np.var([p.value[feature] for p in leftPoints])
            rightVar = np.var([p.value[feature] for p in rightPoints])
            weightingVar = ((len(leftPoints) / len(node.points)) * leftVar) + ((len(rightPoints) / len(node.points)) * rightVar)

            currGain = weightingVar - parentVar

            # calculate the variance each tree and sum it. if it's >  currFeatVariance, split again, else prune
            if currGain > bestGain: 
                # do split again for right and left(recur)
                currGain = bestGain
                bestFeature = feature
                bestLeft = leftPoints
                bestRight = rightPoints

        # if no gain, stop
        if bestFeature is None or bestGain < self.MAX_VARIANCE:
            node.feature = None
            return    
        
        # now all featu has been tested, push the node and split if it could
        node.feature = bestFeature
        node.left = Node(bestLeft)
        node.right = Node(bestRight)

        # recursive split
        self.split(node.left)
        self.split(node.right)
    
    def predict(self, x : np.ndarray) :



        return
        