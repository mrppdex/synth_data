import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Validator:
    @staticmethod
    def validate_integrity(df, spec, dataset_type):
        """
        Validates logical integrity of the dataset specific to its type.
        """
        issues = []
        if df.empty:
            return ["Dataset is empty"]
            
        # 1. Temporal Integrity (ASTDT <= AENDT)
        if 'ASTDT' in df.columns and 'AENDT' in df.columns:
            # Ensure dates are datetime
            try:
                # Convert to datetime if not already, assume ISO8601 YYYY-MM-DD
                start = pd.to_datetime(df['ASTDT'], errors='coerce')
                end = pd.to_datetime(df['AENDT'], errors='coerce')
                
                mask = (start > end) & (start.notna()) & (end.notna())
                violators = df[mask]
                if not violators.empty:
                    issues.append(f"Found {len(violators)} rows where ASTDT > AENDT")
            except Exception as e:
                issues.append(f"Date validation error: {e}")

        # 2. Check if generated USUBJID is present
        if 'USUBJID' not in df.columns:
            issues.append("USUBJID column missing")
            
        return issues

    @staticmethod
    def validate_cdisc(df, spec):
        """
        Validates against the specification metadata (Type, Length).
        """
        issues = []
        spec_vars = spec.get('variables', {})
        
        for col in df.columns:
            if col not in spec_vars:
                # Unexpected column, maybe okay?
                continue
                
            var_def = spec_vars[col]
            val_type = var_def.get('type')
            max_len = var_def.get('length')
            
            # Type Check (Basic)
            # if val_type == 'Num', check if numeric
            if val_type == 'Num':
                if not pd.to_numeric(df[col], errors='coerce').notna().all():
                     issues.append(f"Column {col} contains non-numeric values but spec says Num")
            
            # Length Check (for char)
            if val_type == 'Char' and max_len:
                # Check max length of string representation
                lengths = df[col].astype(str).str.len()
                if (lengths > max_len).any():
                     issues.append(f"Column {col} has values exceeding length {max_len}")
                     
        return issues
