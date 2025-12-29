import streamlit as st
import pandas as pd
import os
from spec_parser import SpecParser

def render_data_studio():
    st.title("📂 Data Studio")
    st.markdown("Import specifications and reference data.")

    tab1, tab2 = st.tabs(["Specifications (Excel)", "Subject Anchors"])

    with tab1:
        st.subheader("ADaM Specifications")
        uploaded_spec = st.file_uploader("Upload Spec (.xlsx)", type=["xlsx"])
        
        if uploaded_spec:
            # Save uploaded file
            if not os.path.exists("specs"):
                os.makedirs("specs")
            
            save_path = os.path.join("specs", uploaded_spec.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_spec.getbuffer())
            st.success(f"Saved {uploaded_spec.name}")

        # List existing specs
        if os.path.exists("specs"):
            files = [f for f in os.listdir("specs") if f.endswith(".xlsx")]
            selected_spec = st.selectbox("Select Specification to Preview", files)
            
            if selected_spec:
                parser = SpecParser(spec_dir="specs")
                context = parser.parse_excel_spec(selected_spec)
                if context:
                    st.json(context)
                else:
                    st.error("Failed to parse spec.")

    with tab2:
        st.subheader("Subject Anchors")
        st.markdown("Load pre-generated or upload anchor data (CSV/JSON).")
        
        # Load existing anchors if available
        if os.path.exists("output/anchors.json"):
            st.info("Found existing anchors.json")
            if st.button("Load Anchors"):
                anchors = pd.read_json("output/anchors.json")
                st.dataframe(anchors)
