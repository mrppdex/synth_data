import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.missingness import MissingnessHandler

def test_missingness_pyarrow():
    print("Testing MissingnessHandler with PyArrow strings...")
    
    # Create data with pyarrow backend
    data = {
        'STR_COL': ['A', 'B', 'A', None],
        'NUM_COL': [1, 2, 3, None],
        'DATE_COL': pd.to_datetime(['2021-01-01', '2021-01-02', None, '2021-01-03'])
    }
    df = pd.DataFrame(data).convert_dtypes(dtype_backend='pyarrow')
    
    print("Dtypes:")
    print(df.dtypes)
    
    handler = MissingnessHandler()
    
    try:
        handler.fit(df)
        print("Fit successful.")
        print("Fill Values:", handler.fill_values)
        
        df_transformed = handler.transform(df)
        print("Transform successful.")
        print(df_transformed)
        
        # Verify fills
        assert df_transformed['STR_COL'].iloc[3] == 'MISSING'
        assert df_transformed['NUM_COL'].iloc[3] == 2.0 # Median of 1, 2, 3
        # Date median might vary slightly depending on implementation but should be filled
        assert not pd.isna(df_transformed['DATE_COL'].iloc[2])
        
        print("SUCCESS: MissingnessHandler handles PyArrow strings and dates!")
        
        print("\nTesting All-Null Column...")
        data_null = {'NULL_COL': [None, None, None]}
        df_null = pd.DataFrame(data_null).convert_dtypes(dtype_backend='pyarrow')
        handler.fit(df_null)
        print("Fill Values for Null:", handler.fill_values)
        assert 'NULL_COL' not in handler.fill_values
        df_null_trans = handler.transform(df_null)
        assert df_null_trans['NULL_COL'].isna().all()
        print("SUCCESS: All-null column ignored correctly.")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_missingness_pyarrow()
