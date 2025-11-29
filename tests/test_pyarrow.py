import sys
import os
print(f"Python Executable: {sys.executable}")
print(f"Sys Path: {sys.path}")
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.mapping import OneToOneMapper

def test_pyarrow_backend():
    print("Testing PyArrow Backend Compatibility...")
    
    # Create data with pyarrow backend
    data = {
        'A': ['foo', 'bar', 'foo', 'bar'],
        'B': [1, 2, 1, 2]
    }
    df = pd.DataFrame(data).convert_dtypes(dtype_backend='pyarrow')
    
    print("Dtypes:")
    print(df.dtypes)
    
    mapper = OneToOneMapper()
    
    try:
        # Collapse
        df_collapsed = mapper.collapse(df, ['A', 'B'], 'AB_Collapsed')
        print("Collapse successful.")
        print(df_collapsed.head())
        print(df_collapsed.dtypes)
        
        # Restore
        df_restored = mapper.restore(df_collapsed)
        print("Restore successful.")
        print(df_restored.head())
        print(df_restored.dtypes)
        
        # Check equality (values)
        # Note: restored types might differ (object vs string[pyarrow]) depending on implementation
        # We care about values.
        
        assert df_restored['A'].astype(str).equals(df['A'].astype(str))
        assert df_restored['B'].astype(int).equals(df['B'].astype(int))
        print("Values match!")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pyarrow_backend()
