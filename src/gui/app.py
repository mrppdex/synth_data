import streamlit as st
import pandas as pd
import os
import sys
import pyreadstat
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.mapping import OneToOneMapper
from src.preprocessing.missingness import MissingnessHandler
from src.models.subject_level import SubjectLevelSynthesizer
from src.models.longitudinal import LongitudinalSynthesizer
from src.evaluation.diagnostics import evaluate_data_quality, plot_column_distribution, plot_correlation_heatmap
from sdv.metadata import SingleTableMetadata

st.set_page_config(page_title="Synthetic CDISC Data Generator", layout="wide")

st.title("Synthetic CDISC SDTM Data Generator")

# Sidebar for configuration
st.sidebar.header("Configuration")
domain_type = st.sidebar.selectbox("Domain Type", ["Subject-Level (e.g., DM)", "Longitudinal (e.g., LB, AE)"])

# File Upload
uploaded_file = st.file_uploader("Upload Training Data (CSV or SAS7BDAT)", type=["csv", "sas7bdat"])

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file, dtype_backend='pyarrow')
    elif file.name.endswith('.sas7bdat'):
        # pyreadstat reads into pandas, we then convert to pyarrow backend
        df, meta = pyreadstat.read_sas7bdat(file.name) # Streamlit file object might need temp save for pyreadstat if it doesn't support bytes
        # Actually pyreadstat needs a path usually, but let's check if we can pass bytes. 
        # If not, we save to temp.
        # For simplicity in this env, let's assume we save it.
        with open("temp_upload.sas7bdat", "wb") as f:
            f.write(file.getbuffer())
        df, meta = pyreadstat.read_sas7bdat("temp_upload.sas7bdat")
        return df.convert_dtypes(dtype_backend='pyarrow')
    return None

def ensure_sdv_compatible(df):
    """
    Converts DataFrame columns to SDV-compatible types.
    Handles PyArrow types and ensures strings are object dtype.
    """
    df = df.copy()
    for col in df.columns:
        # Convert PyArrow types
        if isinstance(df[col].dtype, pd.ArrowDtype):
             if 'string' in str(df[col].dtype):
                 df[col] = df[col].astype(object)
             elif 'int' in str(df[col].dtype):
                 if df[col].isnull().any():
                     df[col] = df[col].astype('float64') # Use float for nullable ints to support NaNs
                 else:
                     df[col] = df[col].astype('int64') # Force standard int
             elif 'float' in str(df[col].dtype):
                 df[col] = df[col].astype('float64')
             elif 'timestamp' in str(df[col].dtype):
                 df[col] = df[col].astype('datetime64[ns]')
             else:
                 df[col] = df[col].astype(object)
        
        # Also ensure standard object for any other string-likes (including numpy U/S)
        if df[col].dtype.kind in ('U', 'S'):
            df[col] = df[col].astype(object)
            
    # Ensure string-like objects are actually strings (not mixed) if possible, or left as object
    # SDV prefers 'object' dtype for categorical/text.
    df = df.astype({c: 'object' for c in df.select_dtypes(include=['string']).columns})
    
    return df

if uploaded_file:
    df = load_data(uploaded_file)
    
    st.write("### Original Data Preview")
    st.dataframe(df.head())

    # Auto-detect *DTC columns and force datetime BEFORE Schema Editor
    # This ensures the editor sees them as Datetime and allows user to override if needed.
    dtc_cols = [c for c in df.columns if c.endswith('DTC')]
    if dtc_cols:
        st.info(f"Detected potential datetime columns: {dtc_cols}. Converting to datetime...")
        for col in dtc_cols:
            try:
                # We use errors='coerce' to handle partial/invalid dates by turning them into NaT
                # This handles "not present at all" or "missing day/month" (if invalid format)
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                st.warning(f"Could not convert {col} to datetime: {e}")

    # Schema Editor
    mapper = st.session_state.get('mapper', None)
    
    st.subheader("Schema Editor")
    st.write("Review and modify detected column types.")
    
    # Create a DataFrame for schema editing
    schema_data = []
    for col in df.columns:
        # Detect current type
        current_type = "String"
        if pd.api.types.is_integer_dtype(df[col]):
            current_type = "Integer"
        elif pd.api.types.is_float_dtype(df[col]):
            # Check if it's actually an integer (with NaNs)
            clean_series = df[col].dropna()
            # Use isclose for robustness against floating point artifacts
            # Cast to float64 to avoid "mod not implemented" error with PyArrow
            if not clean_series.empty and np.all(np.isclose(clean_series.astype('float64') % 1, 0)):
                current_type = "Integer"
            else:
                current_type = "Float"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            current_type = "Date"
            
        # Heuristic for Code columns (*CD, *CN) -> Default to String if they look categorical
        # Also check for low cardinality floats/ints that might be categorical codes
        is_code_col = col.endswith('CD') or col.endswith('CN') or col.endswith('CODE')
        low_cardinality = df[col].nunique() < 50
        
        if is_code_col:
            current_type = "String"
        elif (current_type == "Float" or current_type == "Integer") and low_cardinality:
             # Low cardinality numerics are likely categorical codes.
             # Defaulting to String ensures they are treated as discrete categories.
             current_type = "String"
            
        schema_data.append({
            "Column": col,
            "Data Type": current_type,
        })
    
    schema_df = pd.DataFrame(schema_data)
    
    edited_schema = st.data_editor(
        schema_df,
        column_config={
            "Data Type": st.column_config.SelectboxColumn(
                "Data Type",
                options=["Integer", "Float", "String", "Date", "Sequence"],
                required=True
            )
        },
        disabled=["Column"],
        hide_index=True,
        key="schema_editor"
    )
    
    # Apply Schema Overrides
    typed_df = df.copy()
    explicit_int_cols = []
    sequence_cols = []
    
    for index, row in edited_schema.iterrows():
        col = row['Column']
        dtype = row['Data Type']
        
        try:
            if dtype == "Integer":
                if typed_df[col].isnull().any():
                     typed_df[col] = typed_df[col].astype('float64')
                else:
                     typed_df[col] = typed_df[col].astype('int64')
                explicit_int_cols.append(col)
                
            elif dtype == "Float":
                typed_df[col] = typed_df[col].astype('float64')
                
            elif dtype == "Date":
                typed_df[col] = pd.to_datetime(typed_df[col], errors='coerce')
                
            elif dtype == "Sequence":
                # Ensure it's numeric for safety, though it will be re-generated
                typed_df[col] = pd.to_numeric(typed_df[col], errors='coerce')
                sequence_cols.append(col)
                
            elif dtype == "String":
                # Smart conversion: 
                # 1. If it's float, check if integer-like.
                # 2. If it's object/string, it might be "1.0". Try converting to numeric first.
                
                # Create a temporary series converted to numeric to check
                temp_series = pd.to_numeric(typed_df[col], errors='coerce')
                
                # If conversion resulted in valid numbers (not all NaN if original wasn't all empty)
                if not temp_series.dropna().empty:
                     # Check if integer-like
                     # Cast to float64 (numpy) to avoid "mod not implemented" error with PyArrow types
                     clean = temp_series.dropna().astype('float64')
                     if np.all(np.isclose(clean % 1, 0)):
                         # It is integer-like (e.g. 1.0, 2.0 or "1.0", "2.0")
                         # Use the numeric series, convert to Int64, then string
                         typed_df[col] = temp_series.astype('Int64').astype(str)
                         
                         # Restore NaNs (Int64 handles them, but astype(str) makes them '<NA>')
                         typed_df[col] = typed_df[col].replace('<NA>', np.nan)
                     else:
                         # It's a real float (1.5), keep as string representation of float?
                         # Or just keep original?
                         # If we want "String", we just cast to str.
                         typed_df[col] = typed_df[col].astype(str)
                else:
                    # Not numeric, just cast to str
                    typed_df[col] = typed_df[col].astype(str)
                    
                # Clean up stringified NaNs
                typed_df[col] = typed_df[col].replace(['nan', '<NA>', 'None'], np.nan)
                
        except Exception as e:
            st.warning(f"Could not convert {col} to {dtype}: {e}")

    # Update df reference
    df = typed_df

    # Training Configuration (Moved before Preprocessing to allow user selection to influence mapping)
    sequence_index = None
    entity_columns = None
    constraints = []
    epochs = 100 # Default
    
    # General Configuration
    st.subheader("Model Configuration")
    epochs = st.slider("Training Epochs", min_value=10, max_value=500, value=100, step=10)

    if domain_type == "Longitudinal (e.g., LB, AE)":
        st.subheader("Longitudinal Settings")
        # Use df.columns (pre-processing) to ensure user can select any column
        # Support multiple columns for sorting (e.g. Date + Seq)
        sequence_index = st.multiselect("Sequence Index (Time/Order Columns)", df.columns, placeholder="Select columns defining order (e.g., EXSTDT, EXSEQ)")
        
        # Default entity columns to USUBJID if present
        default_entities = [c for c in df.columns if 'USUBJID' in c]
        entity_columns = st.multiselect("Entity Columns (Subject ID)", df.columns, default=default_entities)
        
    # Constraints (Shared for both)
    st.subheader("Constraints")
    
    # Preprocessing Options
    st.sidebar.subheader("Preprocessing")
    st.sidebar.info("1:1 relationships are automatically detected and collapsed.")
    
    # Preprocessing Step
    mapper = OneToOneMapper()
    missing_handler = MissingnessHandler()
    
    processed_df = df.copy()
    
    # Handle Constant Columns (Single Value)
    # User requested to remove them from preprocessing/modeling and restore them later.
    # This prevents them from being collapsed or modeled unnecessarily.
    constant_cols = [c for c in processed_df.columns if processed_df[c].nunique(dropna=False) <= 1]
    # Store constant values (take first non-null if possible, else None)
    constant_values = {}
    for c in constant_cols:
        # Get unique values
        uniques = processed_df[c].unique()
        # If there's a value (even if it's just NaN), take it.
        # But if it's all NaN, nunique is 0 (by default dropna=True) or 1 (dropna=False).
        # We used dropna=False, so nunique <= 1 covers all-NaN too.
        val = uniques[0] if len(uniques) > 0 else None
        constant_values[c] = val
        
    # Drop constant columns from processing
    if constant_cols:
        st.info(f"Detected {len(constant_cols)} constant columns (will be restored after generation): {constant_cols}")
        processed_df = processed_df.drop(columns=constant_cols)
        # Store for restoration
        st.session_state['constant_values'] = constant_values
    else:
        st.session_state['constant_values'] = {}

    # Auto-detect and collapse
    # Strict Protection: Ensure Entity and Sequence columns are NEVER collapsed or used as keys
    # We want to generate attributes (like Treatment) for new subjects, not map them from non-existent IDs.
    protected_cols = []
    if entity_columns:
        protected_cols.extend(entity_columns)
    if sequence_index:
        protected_cols.extend(sequence_index)
        
    # Also exclude potential IDs just in case
    potential_ids = [c for c in processed_df.columns if 'SUBJID' in c or 'ID' in c or c.endswith('ID')]
    protected_cols.extend(potential_ids)
    protected_cols = list(set(protected_cols)) # Deduplicate
    
    # Run detection (exclude protected cols from being keys)
    mapper.fit(processed_df, exclude_keys=protected_cols)
    
    # Interactive Preprocessing: Allow user to review/modify groups
    if mapper.collapsed_columns:
        st.write("### Detected 1:1/1:N Relationships")
        st.info("Uncheck any groups you do NOT want to collapse. You can also edit the Key or Dependent columns.")
        
        # Prepare data for editor
        group_data = []
        for group_name, info in mapper.collapsed_columns.items():
            key_col = info['key']
            deps = info['dependents']
            
            # Determine default state
            should_collapse = True
            
            # Criteria for NOT collapsing (default unchecked):
            # 1. Sequence columns (*SEQ, *NUM)
            # 2. Date/Time columns (*DT, *DTC) - as per user request (DY is allowed to collapse)
            # 3. Protected columns (shouldn't be here if exclude_keys worked, but check deps)
            for d in deps:
                if d in protected_cols:
                    should_collapse = False
                    break
                if d.endswith('SEQ') or d.endswith('NUM') or d.endswith('DT') or d.endswith('DTC'):
                    should_collapse = False
                    break
            
            group_data.append({
                "Collapse": should_collapse,
                "Key Column": key_col,
                "Dependent Columns": ", ".join(deps),
                "Group Name": group_name
            })
            
        group_df = pd.DataFrame(group_data)
        
        edited_groups = st.data_editor(
            group_df,
            column_config={
                "Collapse": st.column_config.CheckboxColumn("Collapse?", default=True),
                "Key Column": st.column_config.TextColumn("Key Column", disabled=False), # Allow editing
                "Dependent Columns": st.column_config.TextColumn("Dependent Columns", disabled=False), # Allow editing
                "Group Name": st.column_config.TextColumn("Group Name", disabled=True),
            },
            hide_index=True,
            key="group_editor"
        )
        
        # Apply user edits
        new_collapsed_columns = {}
        for index, row in edited_groups.iterrows():
            if row['Collapse']:
                group_name = row['Group Name']
                key_col = row['Key Column']
                # Parse dependents string back to list
                deps_str = row['Dependent Columns']
                deps = [d.strip() for d in deps_str.split(',') if d.strip()]
                
                # Re-generate map for this group based on user's edited key and dependents
                if key_col in processed_df.columns: # Ensure the key column still exists
                     # Create new mapping
                     # Verify dependent columns exist in the current processed_df
                     valid_deps = [d for d in deps if d in processed_df.columns]
                     
                     if valid_deps:
                         # Ensure the key column is not in valid_deps to avoid self-mapping issues
                         valid_deps = [d for d in valid_deps if d != key_col]
                         
                         # Only proceed if there are actual dependents to map
                         if valid_deps:
                             unique_combos = processed_df[[key_col] + valid_deps].drop_duplicates()
                             mapping = {}
                             for _, r in unique_combos.iterrows():
                                 k = str(r[key_col])
                                 mapping[k] = r[valid_deps].to_dict()
                             
                             # Update mapper's internal mapping_dict and collapsed_columns
                             mapper.mapping_dict[group_name] = mapping
                             new_collapsed_columns[group_name] = {'key': key_col, 'dependents': valid_deps}
                         else:
                             st.warning(f"Group '{group_name}': No valid dependent columns found after editing. This group will not be collapsed.")
                     else:
                         st.warning(f"Group '{group_name}': No valid dependent columns found after editing. This group will not be collapsed.")
                else:
                    st.warning(f"Group '{group_name}': Key column '{key_col}' not found in data. This group will not be collapsed.")
        
        mapper.collapsed_columns = new_collapsed_columns
        
    processed_df = mapper.collapse(processed_df)
    
    if mapper.collapsed_columns:
        st.success(f"Automatically collapsed {len(mapper.collapsed_columns)} column groups: {list(mapper.collapsed_columns.keys())}")
        
    processed_df = missing_handler.fit(processed_df).transform(processed_df)
    
    # Ensure selected columns are in processed_df (in case they were collapsed - though we try to exclude them now)
    # If they were somehow collapsed despite exclusion (or if logic failed), restore them.
    cols_to_restore = []
    if sequence_index: # sequence_index is now a list
        for col in sequence_index:
            if col not in processed_df.columns:
                cols_to_restore.append(col)
    if entity_columns:
        for col in entity_columns:
            if col not in processed_df.columns:
                cols_to_restore.append(col)
                
    for col in cols_to_restore:
        st.info(f"Restoring {col} (was collapsed) for training.")
        processed_df[col] = df[col]
        # Simple missing value imputation for restored columns
        if pd.api.types.is_numeric_dtype(processed_df[col]):
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        else:
            processed_df[col] = processed_df[col].fillna('MISSING')

    st.write("### Processed Data for Training")
    st.dataframe(processed_df.head())
    st.subheader("Constraints")
    
    if 'constraints' not in st.session_state:
        st.session_state['constraints'] = []
        
    # Constraint Builder
    with st.expander("Add New Constraint", expanded=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
        
        with c1:
            # Ensure entity_columns and sequence_index are not in target_col options
            excluded_cols = (entity_columns or []) + ([sequence_index] if sequence_index else [])
            target_col = st.selectbox("Target Column", [c for c in processed_df.columns if c not in excluded_cols])
            
        with c2:
            operator = st.selectbox("Operator", [">", ">=", "<", "<=", "==", "!="])
            
        with c3:
            ref_type = st.selectbox("Ref Type", ["Value", "Column"])
            
        with c4:
            if ref_type == "Value":
                ref_val = st.text_input("Value", "0")
            else:
                ref_val = st.selectbox("Ref Column", [c for c in processed_df.columns if c != target_col])
        
        if st.button("Add Constraint"):
            # Validate value if needed
            final_ref = ref_val
            if ref_type == "Value":
                try:
                    final_ref = float(ref_val)
                except ValueError:
                    pass # Keep as string if not a valid float
            
            new_constraint = {
                'target': target_col,
                'op': operator,
                'ref_type': ref_type.lower(),
                'ref': final_ref
            }
            st.session_state['constraints'].append(new_constraint)
            st.success(f"Added: {target_col} {operator} {final_ref}")
            
    # List Constraints
    if st.session_state['constraints']:
        st.write("Active Constraints:")
        for i, c in enumerate(st.session_state['constraints']):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.code(f"{c['target']} {c['op']} {c['ref']} ({c['ref_type']})")
            with col2:
                if st.button("Remove", key=f"del_{i}"):
                    st.session_state['constraints'].pop(i)
                    st.rerun()
                    
    constraints = st.session_state['constraints']

    # Training Button
    if st.button("Train Model"):
        # Validation before spinner
        if domain_type == "Longitudinal (e.g., LB, AE)" and (not sequence_index or not entity_columns):
            st.error("Please specify Sequence Index and Entity Columns for longitudinal training.")
        else:
            with st.spinner("Training model... This may take a while."):
                # Convert back to standard numpy/pandas types if SDV doesn't support pyarrow fully yet
                train_df = ensure_sdv_compatible(processed_df)
                
                if domain_type == "Subject-Level (e.g., DM)":
                    model = SubjectLevelSynthesizer(epochs=epochs, constraints=constraints)
                    model.train(train_df)
                    st.session_state['model'] = model
                    st.session_state['mapper'] = mapper
                    st.session_state['missing_handler'] = missing_handler
                    st.session_state['real_data'] = df
                    st.success("Model trained successfully!")
                else:
                    # Longitudinal Model
                    # Initialize session state for training if not present
                    if 'is_training' not in st.session_state:
                        st.session_state['is_training'] = False
                        st.session_state['current_epoch'] = 0
                        st.session_state['loss_history'] = []
                    
                    # Start Training
                    if not st.session_state['is_training']:
                        st.write("Debug: explicit_int_cols passed to model:", explicit_int_cols)
                        model = LongitudinalSynthesizer(
                            sequence_index=sequence_index, 
                            entity_columns=entity_columns, 
                            epochs=epochs, 
                            constraints=constraints,
                            int_cols=explicit_int_cols,
                            sequence_cols=sequence_cols
                        )
                        
                        # Prepare (Preprocess, Init)
                        model.prepare_training(train_df)
                        
                        st.session_state['model'] = model
                        st.session_state['mapper'] = mapper
                        st.session_state['missing_handler'] = missing_handler
                        st.session_state['real_data'] = df
                        st.session_state['is_training'] = True
                        st.session_state['current_epoch'] = 0
                        st.session_state['loss_history'] = []
                        st.rerun()
                        
    # Training Loop (Runs if is_training is True)
    if st.session_state.get('is_training'):
        st.subheader("Training Progress")
        
        # Stop Button
        if st.button("Stop Training"):
            st.session_state['is_training'] = False
            st.warning("Training stopped by user. You can now generate data with the model trained so far.")
            st.rerun()
            
        model = st.session_state['model']
        current_epoch = st.session_state['current_epoch']
        loss_history = st.session_state['loss_history']
        
        # Train one epoch
        avg_loss = model.train_epoch()
        
        # Update State
        current_epoch += 1
        loss_history.append(avg_loss)
        st.session_state['current_epoch'] = current_epoch
        st.session_state['loss_history'] = loss_history
        
        # Display Progress
        progress = current_epoch / epochs
        st.progress(progress)
        st.write(f"Epoch {current_epoch}/{epochs} - Loss: {avg_loss:.4f}")
        st.line_chart(loss_history)
        
        # Check completion
        if current_epoch >= epochs:
            st.session_state['is_training'] = False
            st.success("Training completed successfully!")
        else:
            st.rerun()

    # Generation
    if 'model' in st.session_state:
        st.divider()
        st.header("Generate Synthetic Data")
        n_samples = st.number_input("Number of samples", min_value=1, value=len(df))
        
        if st.button("Generate"):
            model = st.session_state['model']
            mapper = st.session_state['mapper']
            missing_handler = st.session_state['missing_handler']
            
            synth_df = model.generate(n_samples)
            
            # Restore
            synth_df = mapper.restore(synth_df)
            
            if missing_handler:
                synth_df = missing_handler.inverse_transform(synth_df)
            
            # Restore Constant Columns
            if 'constant_values' in st.session_state:
                for col, val in st.session_state['constant_values'].items():
                    synth_df[col] = val
            
            st.session_state['synth_data'] = synth_df
            st.success("Data generated!")
            st.rerun()

    # Evaluation
    if 'model' in st.session_state:
        # No "Generate & Evaluate" button anymore.
        pass

    # Evaluation (Original section, now modified to only show if synth_data exists)
    if 'synth_data' in st.session_state:
        real_data = st.session_state['real_data']
        synth_data = st.session_state['synth_data']
        
        # Create Metadata for Evaluation
        # Ensure real_data is compatible with SDV metadata detection (no PyArrow/Unicode)
        eval_real_data = ensure_sdv_compatible(real_data)
        
        # Check sequence index if available
        if 'model' in st.session_state and hasattr(st.session_state['model'], 'sequence_index'):
            seq_col = st.session_state['model'].sequence_index
            if seq_col:
                # Handle list or string
                seq_cols_check = seq_col if isinstance(seq_col, list) else [seq_col]
                for sc in seq_cols_check:
                    if sc not in synth_data.columns:
                        st.error(f"CRITICAL: {sc} missing from synthetic data!")
            
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(eval_real_data)
        
        tab1, tab2, tab3 = st.tabs(["Data Preview", "Visual Diagnostics", "Quality Report"])
        
        with tab1:
            st.write("### Generated Data")
            st.dataframe(synth_data.head())
            
            # Export Options
            c1, c2 = st.columns(2)
            with c1:
                csv = synth_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name='synthetic_data.csv',
                    mime='text/csv',
                )
            with c2:
                # Parquet requires pyarrow
                try:
                    import io
                    buffer = io.BytesIO()
                    synth_data.to_parquet(buffer, index=False)
                    st.download_button(
                        label="Download as Parquet",
                        data=buffer.getvalue(),
                        file_name='synthetic_data.parquet',
                        mime='application/octet-stream',
                    )
                except ImportError:
                    st.warning("PyArrow not installed. Parquet export unavailable.")
            
        with tab2:
            col_to_plot = st.selectbox("Select column to visualize", real_data.columns)
            # Ensure types match for plotting
            try:
                # Use compatible data for plotting too
                fig = plot_column_distribution(eval_real_data, synth_data, col_to_plot, metadata=metadata)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot {col_to_plot}: {e}")
            
            st.subheader("Correlation Heatmaps")
            c1, c2 = st.columns(2)
            with c1:
                st.write("Original")
                st.plotly_chart(plot_correlation_heatmap(eval_real_data, "Original Correlation"), use_container_width=True)
            with c2:
                st.write("Synthetic")
                st.plotly_chart(plot_correlation_heatmap(synth_data, "Synthetic Correlation"), use_container_width=True)

        with tab3:
            st.write("Quality report generation can be slow.")
            if st.button("Run Quality Report"):
                with st.spinner("Generating Quality Report..."):
                    try:
                        report = evaluate_data_quality(eval_real_data, synth_data, metadata)
                        st.write(f"### Overall Score: {report.get_score():.2%}")
                        st.write("#### Properties:")
                        st.dataframe(report.get_properties())
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
