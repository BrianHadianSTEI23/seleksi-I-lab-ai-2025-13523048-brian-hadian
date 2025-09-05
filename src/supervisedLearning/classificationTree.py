
import numpy as np

class Node : 
    def __init__(self, featureIndex = None, threshold= None, left=None, right=None, infoGain = None, value = None):
        # for decision node
        self.featureIndex = featureIndex,
        self.threshold = threshold
        self.left = left
        self.right = right
        self.infoGain = infoGain

        # for leaf
        self.value = value


class ClassificationTree : 
    def __init__(self, minSampleSplit = 2, maxDepth = 10**5) -> None:
        self.root = None,
        self.minSampleSplit = minSampleSplit
        self.maxDepth = maxDepth

    def buildTree(self, dataset, currDepth = 0) :
        
        X, Y = dataset[:,:-1], dataset[:, -1]
        numSamples, numFeatures = np.shape(X)

        if numSamples >= self.minSampleSplit and currDepth < self.maxDepth :
            bestSplit = self.getBestSplit(dataset, numSamples, numFeatures)
            if bestSplit['infoGain'] > 0 :
                # recur left
                leftSubtree = self.buildTree(bestSplit["dataset_left"], currDepth+1)
                # recur right
                rightSubtree = self.buildTree(bestSplit["dataset_right"], currDepth+1)
                return Node(bestSplit["featureIndex"], bestSplit["threshold"], leftSubtree, rightSubtree, bestSplit["infoGain"])

        leaf_val = self.calculateLeafValue(Y)
        return Node (value=leaf_val)

    
    def getBestSplit(self, dataset, numSamples, numFeatures):
        
        # dict to store the best split
        bestSplit = {}
        maxInfoGain = -float("inf")
        
        # loop over all features
        for featureIndex in range(numFeatures):
            featureValues = dataset[:, featureIndex]
            possibleThresholds = np.unique(featureValues)

            for threshold in possibleThresholds:
                # get current split
                datasetLeft, datasetRight = self.split(dataset, featureIndex, threshold)
                # check if childs are not null
                if len(datasetLeft)>0 and len(datasetRight)>0:
                    y, left_y, right_y = dataset[:, -1], datasetLeft[:, -1], datasetRight[:, -1]
                    
                    # get information gain
                    currInfoGain = self.informationGain(y, left_y, right_y, "gini")
                    
                    # update the best split if needed
                    if currInfoGain>maxInfoGain:
                        bestSplit["feature_index"] = featureIndex
                        bestSplit["threshold"] = threshold
                        bestSplit["datasetLeft"] = datasetLeft
                        bestSplit["datasetRight"] = datasetRight
                        bestSplit["infoGain"] = currInfoGain
                        maxInfoGain = currInfoGain
                        
        return bestSplit
    
    # split dataset based on feature index and threshold
    def split(self, dataset, featureIndex, threshold):
        
        datasetLeft = np.array([row for row in dataset if row[featureIndex]<=threshold])
        datasetRight = np.array([row for row in dataset if row[featureIndex]>threshold])
        return datasetLeft, datasetRight
    
    # compute information gain
    def informationGain(self, parent, lChild, rChild, mode="entropy"):
        
        weightLeft = len(lChild) / len(parent)
        weightRight = len(rChild) / len(parent)
        if mode=="gini":
            gain = self.giniIndex(parent) - (weightLeft*self.giniIndex(lChild) + weightRight*self.giniIndex(rChild))
        else:
            gain = self.entropy(parent) - (weightLeft*self.entropy(lChild) + weightRight*self.entropy(rChild))
        return gain
    
    def entropy(self, y):
        
        classLabels = np.unique(y)
        entropy = 0
        for cls in classLabels:
            pCls = len(y[y == cls]) / len(y)
            entropy += -pCls * np.log2(pCls)
        return entropy
    
    def giniIndex(self, y):
        
        classLabels = np.unique(y)
        gini = 0
        for cls in classLabels:
            pCls = len(y[y == cls]) / len(y)
            gini += pCls**2
        return 1 - gini
        
    def calculateLeafValue(self, Y):
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.buildTree(dataset)
    
    def predict(self, X):
        
        preditions = [self.makePrediction(x, self.root) for x in X]
        return preditions
    
    def makePrediction(self, x, tree):
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.makePrediction(x, tree.left)
        else:
            return self.makePrediction(x, tree.right)