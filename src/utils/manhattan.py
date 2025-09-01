
import pandas as pd

def manhattanDistance(df : pd.DataFrame, val2 : pd.Series) -> pd.DataFrame :
    distances = df.apply(lambda row: (row - val2).abs().sum(), axis=1)
    result = df.copy()
    result["distances"] = distances
    return result