import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.mapping import OneToOneMapper
from src.preprocessing.missingness import MissingnessHandler
from src.models.subject_level import SubjectLevelSynthesizer

def test_pipeline():
    print("Creating dummy data...")
    # Create dummy DM-like data
    data = {
        'USUBJID': [f'SUBJ{i:03d}' for i in range(100)],
        'AGE': np.random.randint(18, 65, 100),
        'SEX': np.random.choice(['M', 'F'], 100),
        'ARM': np.random.choice(['Placebo', 'Active'], 100),
        'ARMCD': [] # Dependent on ARM
    }
    
    # Create 1:1 relationship
    arm_map = {'Placebo': 'PBO', 'Active': 'ACT'}
    data['ARMCD'] = [arm_map[arm] for arm in data['ARM']]
    
    df = pd.DataFrame(data)
    
    # Introduce some missingness
    df.loc[0:5, 'AGE'] = np.nan
    
    print("Original Data Head:")
    print(df.head())
    
    # 1. Preprocessing: Collapse 1:1
    print("\n--- Preprocessing ---")
    mapper = OneToOneMapper()
    mapper.fit(df)
    df_collapsed = mapper.collapse(df)
    
    if mapper.collapsed_columns:
        print(f"Collapsed: {mapper.collapsed_columns.keys()}")
    else:
        print("WARNING: No columns collapsed!")
    
    # 2. Preprocessing: Missingness
    missing_handler = MissingnessHandler()
    df_processed = missing_handler.fit(df_collapsed).transform(df_collapsed)
    
    print("Processed Data Head:")
    print(df_processed.head())
    
    # 3. Model Training
    print("\n--- Training ---")
    model = SubjectLevelSynthesizer(epochs=10)
    model.train(df_processed)
    
    # 4. Generation
    print("\n--- Generation ---")
    synth_df_processed = model.generate(50)
    
    # 5. Restore
    print("\n--- Restoring ---")
    synth_df = mapper.restore(synth_df_processed)
    
    print("Synthetic Data Head:")
    print(synth_df.head())
    
    # Verification
    # Check if ARM and ARMCD are consistent
    print("\n--- Verification ---")
    consistent = True
    for _, row in synth_df.iterrows():
        if row['ARM'] == 'Placebo' and row['ARMCD'] != 'PBO':
            consistent = False
        if row['ARM'] == 'Active' and row['ARMCD'] != 'ACT':
            consistent = False
            
    if consistent:
        print("SUCCESS: 1:1 Relationship preserved!")
    else:
        print("FAILURE: 1:1 Relationship broken!")

if __name__ == "__main__":
    test_pipeline()
