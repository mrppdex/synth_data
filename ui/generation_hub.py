import streamlit as st
import pandas as pd
import json
import os
import time
from generator import QwenGenerator

# Cache the model loader to avoid reloading on every rerun
@st.cache_resource
def load_model(model_id, load_in_4bit):
    return QwenGenerator(model_id=model_id, load_in_4bit=load_in_4bit)

def render_generation_hub():
    st.title("⚡ Generation Hub")
    
    # Retrieve config
    config = st.session_state.get('config', {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "temperature": 0.7,
        "max_tokens": 512,
        "load_in_4bit": False
    })
    
    st.write(f"**Target Model:** `{config['model_id']}`")
    
    if st.button("Initialize Engine"):
        with st.spinner("Loading Model... (This may take a minute)"):
            try:
                gen = load_model(config['model_id'], config.get('load_in_4bit', False))
                st.session_state['generator'] = gen
                st.success("Engine Initialized!")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    st.markdown("---")
    
    # Generation Controls
    if os.path.exists("specs") and os.path.exists("output/anchors.json"):
        specs = [f for f in os.listdir("specs") if f.endswith(".xlsx")]
        selected_spec_file = st.selectbox("Select Target Dataset", specs)
        
        if st.button("Generate Dataset"):
            if 'generator' not in st.session_state:
                st.error("Please Initialize Engine first.")
                return

            gen = st.session_state['generator']
            
            # Load Spec & Anchors
            # (In a real app, use the Parser properly here)
            # For PoC, assuming spec filename matches spec_parser logic or we re-instantiate parser
            from spec_parser import SpecParser
            parser = SpecParser(spec_dir="specs")
            spec_context = parser.parse_excel_spec(selected_spec_file)
            
            with open("output/anchors.json") as f:
                anchors = json.load(f)
            
            # Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Custom generation loop to update UI
            rows = []
            total = len(anchors)
            
            st.write(f"Generating {total} rows...")
            
            dataset_name = spec_context.get('dataset', 'DATA')
            
            for i, anchor in enumerate(anchors):
                status_text.text(f"Processing Subject {anchor.get('USUBJID', 'Unknown')} ({i+1}/{total})")
                
                messages = gen.construct_messages(
                    anchor, 
                    spec_context, 
                    row_instruction=f"Generate {dataset_name} row."
                )
                
                # We reuse the generate_row logic but might want to pass dynamic config like temp
                row_data = gen.generate_row(messages)
                
                if row_data:
                    if "USUBJID" not in row_data:
                        row_data["USUBJID"] = anchor["USUBJID"]
                    rows.append(row_data)
                
                progress_bar.progress((i + 1) / total)
            
            st.success("Generation Complete!")
            
            # Show Results
            df = pd.DataFrame(rows)
            st.dataframe(df)
            
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                csv,
                f"{dataset_name}.csv",
                "text/csv",
                key='download-csv'
            )
            
    else:
        st.warning("Please upload a Spec and generate Anchors in Data Studio first.")
