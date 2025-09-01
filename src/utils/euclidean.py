import pandas as pd

def euclideanDistance(df : pd.DataFrame, val2 : pd.Series) -> pd.DataFrame :
    distances = df.apply(lambda row: ((row - val2) ** 2).sum(), axis=1)
    result = df.copy()
    result["distances"] = distances
    return result
