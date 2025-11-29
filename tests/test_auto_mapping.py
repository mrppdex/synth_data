import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.mapping import OneToOneMapper

def test_auto_mapping():
    print("Testing Auto-Mapping Logic...")
    
    data = {
        'USUBJID': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
        'ARM': ['A', 'B', 'A', 'B', 'A', 'B'],
        'ARMCD': ['1', '2', '1', '2', '1', '2'], # Perfectly correlated with ARM
        'SITEID': ['01', '01', '02', '02', '03', '03'], # Site 01 has both A and B. Site 02 has both.
        'VISIT': ['V1', 'V1', 'V1', 'V1', 'V1', 'V1'],
        'VISITNUM': [1, 1, 1, 1, 1, 1] # Perfectly correlated with VISIT
    }
    
    df = pd.DataFrame(data)
    
    mapper = OneToOneMapper()
    mapper.fit(df)
    
    print("Detected Mappings:")
    for group, info in mapper.collapsed_columns.items():
        print(f"Group: {group}, Key: {info['key']}, Dependents: {info['dependents']}")
        
    # We expect ARM <-> ARMCD and VISIT <-> VISITNUM to be detected
    # The key choice is arbitrary (based on sort order), but one should be key and other dependent.
    
    detected_cols = set()
    for info in mapper.collapsed_columns.values():
        detected_cols.add(info['key'])
        detected_cols.update(info['dependents'])
        
    assert 'ARM' in detected_cols
    assert 'ARMCD' in detected_cols
    assert 'VISIT' in detected_cols
    assert 'VISITNUM' in detected_cols
    # SITEID might be a key (if it determines VISIT), but should not be a dependent of ARM or USUBJID
    # Check that SITEID is not in dependents list of any group
    all_dependents = set()
    for info in mapper.collapsed_columns.values():
        all_dependents.update(info['dependents'])
        
    assert 'SITEID' not in all_dependents
    
    print("Collapsing...")
    df_collapsed = mapper.collapse(df)
    print(df_collapsed.head())
    
    # Check that dependents are dropped
    for info in mapper.collapsed_columns.values():
        for dep in info['dependents']:
            assert dep not in df_collapsed.columns
            
    print("Restoring...")
    df_restored = mapper.restore(df_collapsed)
    print(df_restored.head())
    
    # Check equality
    # Note: Types might change to object/string
    assert df_restored['ARMCD'].astype(str).equals(df['ARMCD'].astype(str))
    assert df_restored['VISITNUM'].astype(str).equals(df['VISITNUM'].astype(str))
    
    print("SUCCESS: Auto-mapping works!")

if __name__ == "__main__":
    test_auto_mapping()
