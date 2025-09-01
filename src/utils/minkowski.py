import pandas as pd

def minkowskiDistance(df : pd.DataFrame, val2 : pd.Series, p : int) -> pd.DataFrame :
    distances = df.apply(lambda row: (((row - val2) ** p).sum())**(1/p), axis=1)
    result = df.copy()
    result["distances"] = distances
    return result
