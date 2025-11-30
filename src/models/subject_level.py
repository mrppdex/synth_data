from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd

class SubjectLevelSynthesizer:
    """
    Wrapper for CTGAN to generate subject-level data (e.g., DM domain).
    """
    def __init__(self, epochs=300, constraints=None):
        self.metadata = SingleTableMetadata()
        self.model = None
        self.epochs = epochs
        self.constraints = constraints if constraints else []

    def train(self, df: pd.DataFrame):
        """
        Train the CTGAN model.
        """
        self.metadata.detect_from_dataframe(df)
        self.model = CTGANSynthesizer(self.metadata, epochs=self.epochs)
        self.model.fit(df)

    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data.
        """
        synth_data = self.model.sample(num_rows=n_samples)
        
        # Apply constraints
        if self.constraints:
            synth_data = self._apply_constraints(synth_data)
            
        return synth_data

    def _apply_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply constraints to the generated DataFrame.
        """
        import numpy as np
        
        for c in self.constraints:
            target = c['target']
            op = c['op']
            ref_type = c['ref_type']
            ref = c['ref']
            
            if target not in df.columns:
                continue
                
            # Get target values
            target_vals = df[target]
            
            # Get reference values
            if ref_type == 'column':
                if ref not in df.columns:
                    continue
                ref_vals = df[ref]
            else:
                ref_vals = ref
                
            try:
                if op == '>':
                    if pd.api.types.is_integer_dtype(target_vals):
                        if ref_type == 'column':
                            df[target] = np.maximum(target_vals, ref_vals + 1)
                        else:
                            df[target] = np.maximum(target_vals, float(ref) + 1)
                    else:
                        df[target] = np.maximum(target_vals, ref_vals)
                        
                elif op == '>=':
                    df[target] = np.maximum(target_vals, ref_vals)
                    
                elif op == '<':
                    if pd.api.types.is_integer_dtype(target_vals):
                        if ref_type == 'column':
                            df[target] = np.minimum(target_vals, ref_vals - 1)
                        else:
                            df[target] = np.minimum(target_vals, float(ref) - 1)
                    else:
                        df[target] = np.minimum(target_vals, ref_vals)
                        
                elif op == '<=':
                    df[target] = np.minimum(target_vals, ref_vals)
                    
                elif op == '==':
                    df[target] = ref_vals
                    
            except Exception:
                pass
                
        return df

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = CTGANSynthesizer.load(path)
