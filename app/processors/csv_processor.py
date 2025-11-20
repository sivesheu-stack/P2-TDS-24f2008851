import pandas as pd
import numpy as np


def load_csv(path: str) -> pd.DataFrame:
    """
    Loads CSV file into a pandas DataFrame with automatic dtype inference.
    """
    return pd.read_csv(path)


def summarize_csv(df: pd.DataFrame):
    """
    Returns useful summary stats:
    - Column names
    - Row count
    - Numerical column summary (mean, sum, etc.)
    """
    return {
        "columns": df.columns.tolist(),
        "numeric_summary": df.describe(include=[np.number]).to_dict()
    }
