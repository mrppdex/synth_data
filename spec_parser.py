import pandas as pd
import json
import os

class SpecParser:
    def __init__(self, spec_dir="specs"):
        self.spec_dir = spec_dir

    def parse_excel_spec(self, filename):
        """
        Reads an Excel ADaM specification and converts it to a structured dictionary.
        Assumes columns: 'Variable', 'Label', 'Type', 'Length', 'Codelist', 'Derivation'.
        """
        filepath = os.path.join(self.spec_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Spec file not found: {filepath}")

        # Read Excel - assuming 'Variables' sheet or first sheet contains the metadata
        try:
            df = pd.read_excel(filepath)
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return None
        
        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        
        # Required columns mapping
        required_cols = {
            'variable': 'name',
            'type': 'type'
        }
        
        # Verify required columns exist
        for req in required_cols:
            if req not in df.columns:
                print(f"Warning: Missing required column '{req}' in spec.")
                # Try to fuzzy match or continue?
        
        variables = {}
        for _, row in df.iterrows():
            var_name = row.get('variable')
            if pd.isna(var_name):
                continue
                
            var_def = {
                "name": var_name,
                "label": row.get('label', ''),
                "type": row.get('type', 'Char'),
                "length": row.get('length', None),
                "codelist": row.get('codelist', None),
                "derivation": row.get('derivation', '')
            }
            
            # Clean up codelist if it is a string representation of a list
            if isinstance(var_def['codelist'], str) and ',' in var_def['codelist']:
                var_def['codelist'] = [x.strip() for x in var_def['codelist'].split(',')]
            elif pd.isna(var_def['codelist']):
                var_def['codelist'] = None

            variables[var_name] = var_def
            
        return {
            "dataset": filename.split('.')[0].upper(), # inferred from filename
            "variables": variables
        }

    def save_context_map(self, context_map, output_path):
        with open(output_path, 'w') as f:
            json.dump(context_map, f, indent=2)

if __name__ == "__main__":
    # Test execution
    parser = SpecParser()
    # Mock usage:
    # context = parser.parse_excel_spec("ADAE.xlsx")
    # print(json.dumps(context, indent=2))
