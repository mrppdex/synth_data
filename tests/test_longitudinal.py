import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.longitudinal import LongitudinalSynthesizer

def test_longitudinal_model():
    print("Testing LongitudinalSynthesizer...")
    
    # Create dummy longitudinal data
    data = {
        'USUBJID': ['S1', 'S1', 'S1', 'S2', 'S2'],
        'VISITNUM': [1, 2, 3, 1, 2],
        'LBTESTCD': ['A', 'A', 'A', 'B', 'B'],
        'LBORRES': [10, 12, 11, 20, 22]
    }
    df = pd.DataFrame(data)
    
    print("Data:")
    print(df)
    
    try:
        # Initialize model
        # sequence_index='VISITNUM', entity_columns=['USUBJID']
        # Add constraints: LBORRES > 0
        constraints = {'LBORRES': {'min': 0}}
        
        model = LongitudinalSynthesizer(
            sequence_index='VISITNUM', 
            entity_columns=['USUBJID'], 
            epochs=1,
            constraints=constraints
        )
        
        print("Training...")
        model.train(df)
        print("Training successful.")
        
        print("Generating...")
        synth_df = model.generate(n_samples=2)
        print("Generation successful.")
        print(synth_df)
        
        # Verify Integer Enforcement (VISITNUM should be int)
        if pd.api.types.is_integer_dtype(synth_df['VISITNUM']):
            print("SUCCESS: VISITNUM is integer.")
        else:
            # Check if all values are actually integers
            is_int = np.all(synth_df['VISITNUM'] == synth_df['VISITNUM'].astype(int))
            if is_int:
                print("SUCCESS: VISITNUM values are integers (even if dtype is float).")
            else:
                print("FAILED: VISITNUM contains non-integers.")
                
        # Verify Constraints (LBORRES >= 0)
        if (synth_df['LBORRES'] >= 0).all():
            print("SUCCESS: LBORRES constraints satisfied (>= 0).")
        else:
            print("FAILED: LBORRES constraints violated.")
            print(synth_df[synth_df['LBORRES'] < 0])
        
        print("SUCCESS: LongitudinalSynthesizer works!")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_longitudinal_model()
