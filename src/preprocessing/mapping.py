import pandas as pd
import json

class OneToOneMapper:
    """
    Identifies and collapses columns with 1:1 relationships into a single column
    to preserve consistency during synthetic data generation.
    """
    def __init__(self):
        self.mapping_dict = {}
        self.collapsed_columns = {}

    def fit(self, df: pd.DataFrame):
        """
        Automatically detect columns that have 1:1 or 1:N relationships.
        We look for columns B that are fully determined by column A (A -> B).
        If A -> B, we can drop B and store the mapping A -> B.
        
        However, the user requirement is to "replace them by only one column".
        If A <-> B (1:1), we can collapse to one.
        If A -> B (1:N from B's perspective, or functional dependency A->B), we can keep A and map B.
        
        Strategy:
        1. Identify categorical columns.
        2. Find pairs (A, B) where A determines B.
        3. Group transitive dependencies? 
           For simplicity: Greedy approach.
           Find all B that are determined by some A.
           If A -> B, add B to "to_drop" and store mapping A -> B.
           If A <-> B, pick one as key.
        """
        # We only look at categorical/object/string columns for now, or low cardinality numerics
        # For performance on large datasets, we should be careful.
        # We'll use a heuristic: if nunique is relatively small compared to len(df), check.
        
        candidates = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category' or df[c].dtype == 'string' or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() < 1000)]
        
        # Calculate unique counts once
        uniques = {c: df[c].nunique() for c in candidates}
        
        # Sort candidates by unique count (descending) - higher entropy columns are likely keys
        candidates = sorted(candidates, key=lambda c: uniques[c], reverse=True)
        
        processed = set()
        
        for i, col_a in enumerate(candidates):
            if col_a in processed:
                continue
                
            # Find all columns determined by col_a
            determined_cols = []
            for col_b in candidates:
                if col_a == col_b or col_b in processed:
                    continue
                
                # Check if A -> B
                # A determines B if for every value of A, there is only 1 unique value of B.
                # We can check this by grouping by A and counting unique B.
                # Optimization: If nunique(A) == nunique(A, B), then A -> B.
                
                # We need to handle NaNs carefully. usually we treat NaN as a value or ignore.
                # Let's treat as value.
                
                # Using string conversion for safety with pyarrow/mixed types
                try:
                    # Quick check: nunique(A) must be >= nunique(B) for A -> B
                    if uniques[col_a] < uniques[col_b]:
                        continue
                        
                    # Heuristic: Do not collapse if A is a primary key (unique per row) 
                    # AND B is not (i.e., B is an attribute).
                    # We want to model attributes, not IDs.
                    # If both are unique (1:1 IDs), we can collapse.
                    is_a_pk = (uniques[col_a] == len(df))
                    is_b_pk = (uniques[col_b] == len(df))
                    
                    if is_a_pk and not is_b_pk:
                        continue
                        
                    # Check combined unique count
                    # This can be expensive.
                    # combined_nunique = df[[col_a, col_b]].astype(str).drop_duplicates().shape[0]
                    # if combined_nunique == uniques[col_a]:
                    #    determined_cols.append(col_b)
                    
                    # Alternative: groupby check (might be faster or slower depending on implementation)
                    # max_b_per_a = df.groupby(col_a)[col_b].nunique().max()
                    # if max_b_per_a == 1:
                    #     determined_cols.append(col_b)
                    
                    # Let's use the combined nunique approach, it's vectorizable
                    # We use a sample if df is huge? No, must be exact.
                    
                    # Create a temporary frame for just these two to avoid copying huge df
                    # Use drop_duplicates on just these two
                    pair_unique = df[[col_a, col_b]].drop_duplicates()
                    if len(pair_unique) == uniques[col_a]:
                         determined_cols.append(col_b)
                         
                except Exception:
                    continue

            if determined_cols:
                # We found a group where col_a determines all columns in determined_cols
                # We can collapse them or just keep col_a and map the rest.
                # The user said "replace them by only one column".
                # So we will treat [col_a] + determined_cols as a group.
                
                group_name = f"{col_a}_group"
                # We store the mapping for all determined cols based on col_a
                
                # Build mapping
                # We need a dictionary: { val_a: { 'col_b': val_b, 'col_c': val_c ... } }
                
                # Get unique combinations
                cols_in_group = [col_a] + determined_cols
                unique_combos = df[cols_in_group].drop_duplicates()
                
                mapping = {}
                for _, row in unique_combos.iterrows():
                    key = str(row[col_a])
                    mapping[key] = row[determined_cols].to_dict()
                
                self.mapping_dict[group_name] = mapping
                self.collapsed_columns[group_name] = {'key': col_a, 'dependents': determined_cols}
                
                processed.add(col_a)
                processed.update(determined_cols)

    def collapse(self, df: pd.DataFrame, group_cols: list[str] = None, new_col_name: str = None) -> pd.DataFrame:
        """
        Collapses columns based on detected dependencies.
        If group_cols is provided (legacy manual mode), it uses that.
        Otherwise it uses the internal detected mappings.
        """
        df_copy = df.copy()
        
        if group_cols:
            # Legacy manual mode (keeping for backward compat or specific overrides)
            # ... (existing logic could go here but we'll focus on the auto mode)
            pass

        # Apply auto-detected mappings
        for group_name, info in self.collapsed_columns.items():
            key_col = info['key']
            dep_cols = info['dependents']
            
            # We keep the key_col, and drop the dep_cols.
            # We assume key_col is sufficient to regenerate dep_cols.
            # We rename key_col to something indicating it's a group? 
            # Or just keep it as is? User said "replace them by only one column".
            # Keeping the key column name is usually cleaner, but maybe we tag it.
            
            # Check if columns exist (might have been dropped or renamed)
            if key_col not in df_copy.columns:
                continue
            
            cols_to_drop = [c for c in dep_cols if c in df_copy.columns]
            if cols_to_drop:
                df_copy = df_copy.drop(columns=cols_to_drop)
                
        return df_copy

    def restore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Restores the dependent columns.
        """
        df_copy = df.copy()
        
        for group_name, info in self.collapsed_columns.items():
            key_col = info['key']
            dep_cols = info['dependents']
            mapping = self.mapping_dict[group_name]
            
            if key_col not in df_copy.columns:
                continue
                
            # Restore each dependent column
            for dep_col in dep_cols:
                # Map values
                # We cast key to string to match dictionary keys
                df_copy[dep_col] = df_copy[key_col].astype(str).map(lambda x: mapping.get(x, {}).get(dep_col, None))
                
        return df_copy

