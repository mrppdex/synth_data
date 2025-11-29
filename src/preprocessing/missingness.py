import pandas as pd
import numpy as np

class MissingnessHandler:
    """
    Handles missing values in the dataset.
    """
    def __init__(self, strategy='fill_categorical'):
        print("Initializing MissingnessHandler (v2 - Fixed PyArrow)")
        self.strategy = strategy
        self.fill_values = {}

    def fit(self, df: pd.DataFrame):
        """
        Learn fill values or patterns.
        """
        for col in df.columns:
            dtype_str = str(df[col].dtype).lower()
            val = None
            
            # If column is all null, we can't really determine a fill value easily.
            # If we assume it's categorical, we fill 'MISSING'.
            # But if it's numeric/date, we shouldn't fill 'MISSING'.
            # PyArrow types might be 'null' or 'int64' (all nulls).
            
            if df[col].isna().all():
                 # Skip all-null columns
                 continue

            if pd.api.types.is_numeric_dtype(df[col]) and 'date' not in dtype_str and 'time' not in dtype_str:
                val = df[col].median()
            elif pd.api.types.is_datetime64_any_dtype(df[col]) or 'timestamp' in dtype_str or 'datetime' in dtype_str:
                val = df[col].median()
            else:
                # Categorical, string, object, etc.
                val = 'MISSING'
            
            # Only add if val is not null (NaN/NaT)
            if val is not None and not pd.isna(val):
                self.fill_values[col] = val
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missingness handling.
        """
        df_copy = df.copy()
        for col, value in self.fill_values.items():
            if col in df_copy.columns:
                 df_copy[col] = df_copy[col].fillna(value)
        return df_copy

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Revert missingness handling if possible (e.g. convert 'MISSING' back to NaN).
        """
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].replace('MISSING', np.nan)
        return df_copy
