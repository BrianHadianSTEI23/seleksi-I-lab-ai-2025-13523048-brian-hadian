import pandas as pd
from src.utils.manhattan import manhattanDistance
from src.utils.euclidean import euclideanDistance
from src.utils.minkowski import minkowskiDistance

def knn(file : str, n : int, distance : int, algorithm : int, input : pd.Series) :

    # opem file
    df = pd.read_csv(file)

    # algorithm check
    if algorithm == 1 : # manhattan distance
        # get the dict
        dictionary : pd.DataFrame = manhattanDistance(df, input)
    elif algorithm == 2 : # euclidean distance
        # get the dict
        dictionary : pd.DataFrame = euclideanDistance(df, input)
    elif algorithm == 3 :
        # get the dict
        dictionary : pd.DataFrame = minkowskiDistance(df, input)
        
    # get the n minimum value from dict based on distance
    filtered = dictionary[dictionary["distances"] < distance]
    neighbors = filtered.sort_values(by="distances").head(n)
    return neighbors