import streamlit as st
import pandas as pd
import os
import sys
import pyreadstat

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
    st.subheader("Schema Editor")
    st.write("Review and modify detected data types.")
    
    col_types = df.dtypes.astype(str).to_dict()
    
    # Create a form for schema editing
    with st.form("schema_editor"):
        cols = st.columns(3)
        new_types = {}
        for i, (col, dtype) in enumerate(col_types.items()):
            with cols[i % 3]:
                # Simplified type selection
                current_type = "Numeric" if "int" in dtype or "float" in dtype else "Categorical"
                if "date" in dtype or "time" in dtype:
                    current_type = "Datetime"
                
                selected_type = st.selectbox(f"{col}", ["Numeric", "Categorical", "Datetime"], index=["Numeric", "Categorical", "Datetime"].index(current_type))
                new_types[col] = selected_type
        
        apply_schema = st.form_submit_button("Apply Schema Changes")
    
    if apply_schema:
        # Apply type conversions
        for col, type_choice in new_types.items():
            try:
                if type_choice == "Numeric":
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif type_choice == "Categorical":
                    df[col] = df[col].astype(str)
                elif type_choice == "Datetime":
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                st.error(f"Could not convert {col} to {type_choice}: {e}")
        st.success("Schema updated!")
        st.dataframe(df.dtypes.astype(str))

    # Preprocessing Options
    st.sidebar.subheader("Preprocessing")
    st.sidebar.info("1:1 relationships are automatically detected and collapsed.")
    
    # Preprocessing Step
    mapper = OneToOneMapper()
    missing_handler = MissingnessHandler()
    
    processed_df = df.copy()
    
    # Auto-detect and collapse
    mapper.fit(processed_df)
    processed_df = mapper.collapse(processed_df)
    
    if mapper.collapsed_columns:
        st.success(f"Automatically collapsed {len(mapper.collapsed_columns)} column groups: {list(mapper.collapsed_columns.keys())}")
        
    processed_df = missing_handler.fit(processed_df).transform(processed_df)
    
    st.write("### Processed Data for Training")
    st.dataframe(processed_df.head())

    # Training Configuration
    sequence_index = None
    entity_columns = None
    constraints = {}
    
    if domain_type == "Longitudinal (e.g., LB, AE)":
        st.subheader("Longitudinal Model Configuration")
        sequence_index = st.selectbox("Sequence Index (Time Column)", processed_df.columns, index=None, placeholder="Select time/sequence column (e.g., VISITNUM, LBDTC)")
        # Default entity columns to USUBJID if present
        default_entities = [c for c in processed_df.columns if 'USUBJID' in c]
        entity_columns = st.multiselect("Entity Columns (Subject ID)", processed_df.columns, default=default_entities)
        
        # Constraints
        st.markdown("#### Constraints")
        enforce_positive = st.checkbox("Enforce Positive Values for Numeric Columns", value=True, help="Ensure all generated numeric values are >= 0.")
        if enforce_positive:
             # Identify numeric columns
             num_cols = [c for c in processed_df.columns if pd.api.types.is_numeric_dtype(processed_df[c]) and c not in (entity_columns or []) and c != sequence_index]
             for col in num_cols:
                 constraints[col] = {'min': 0}

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
                    model = SubjectLevelSynthesizer(epochs=100)
                    model.train(train_df)
                    st.session_state['model'] = model
                    st.session_state['mapper'] = mapper
                    st.session_state['missing_handler'] = missing_handler
                    st.session_state['real_data'] = df
                    st.success("Model trained successfully!")
                else:
                    # Longitudinal Model
                    model = LongitudinalSynthesizer(sequence_index=sequence_index, entity_columns=entity_columns, epochs=100, constraints=constraints)
                    
                    # Progress Callback
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    chart_placeholder = st.empty()
                    loss_history = []
                    
                    def update_progress(epoch, loss):
                        progress = epoch / 100
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch}/100 - Loss: {loss:.4f}")
                        loss_history.append(loss)
                        chart_placeholder.line_chart(loss_history)
                        
                    model.train(train_df, progress_callback=update_progress)
                    
                    st.session_state['model'] = model
                    st.session_state['mapper'] = mapper
                    st.session_state['missing_handler'] = missing_handler
                    st.session_state['real_data'] = df
                    st.success("Model trained successfully!")

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
            
            st.session_state['synth_data'] = synth_df
            st.success("Data generated!")

    # Evaluation
    if 'synth_data' in st.session_state:
        st.divider()
        st.header("Evaluation")
        
        real_data = st.session_state['real_data']
        synth_data = st.session_state['synth_data']
        
        # Create Metadata for Evaluation
        # Ensure real_data is compatible with SDV metadata detection (no PyArrow/Unicode)
        eval_real_data = ensure_sdv_compatible(real_data)
        
        st.write("Debug: Real Data Columns:", eval_real_data.columns.tolist())
        st.write("Debug: Synth Data Columns:", synth_data.columns.tolist())
        
        # Check sequence index if available
        if 'model' in st.session_state and hasattr(st.session_state['model'], 'sequence_index'):
            seq_col = st.session_state['model'].sequence_index
            if seq_col:
                if seq_col in eval_real_data.columns:
                    st.write(f"Debug: Real {seq_col} dtype: {eval_real_data[seq_col].dtype}")
                if seq_col in synth_data.columns:
                    st.write(f"Debug: Synth {seq_col} dtype: {synth_data[seq_col].dtype}")
                else:
                    st.error(f"CRITICAL: {seq_col} missing from synthetic data!")
            
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(eval_real_data)
        
        tab1, tab2, tab3 = st.tabs(["Data Preview", "Visual Diagnostics", "Quality Report"])
        
        with tab1:
            st.subheader("Synthetic Data")
            st.dataframe(synth_data.head())
            
            # Download
            csv = synth_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "synthetic_data.csv", "text/csv")
            
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
