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
        st.subheader("Population Architect")
        st.markdown("Design your synthetic cohort demographics.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### ⚙️ Configuration")
            n_subjects = st.slider("Number of Subjects", 10, 1000, 50)
            age_range = st.slider("Age Range", 0, 100, (18, 65))
            
            st.markdown("Start Date")
            start_date = st.date_input("Study Reference Date", value=pd.to_datetime("2023-01-01"))
            
            st.markdown("Study Arms")
            # Tag input simulation
            default_arms = "Placebo, Active Low, Active High"
            arms_input = st.text_area("Arms (comma separated)", value=default_arms)
            arms_list = [a.strip() for a in arms_input.split(",") if a.strip()]

        with col2:
            st.markdown("#### 📊 Demographics Preview")
            gender_balance = st.slider("Gender Balance (% Female)", 0, 100, 50)
            
            # Preview generation logic
            if st.button("Generate Anchors"):
                from anchor_generator import AnchorGenerator
                gen = AnchorGenerator()
                
                config = {
                    "n_subjects": n_subjects,
                    "age_range": age_range,
                    "gender_ratio": {"F": gender_balance/100, "M": 1 - (gender_balance/100)},
                    "arms": arms_list,
                    "start_date": str(start_date)
                }
                
                anchors = gen.generate_anchors(config)
                
                # Save
                gen.save_anchors(anchors)
                st.success(f"Generated {len(anchors)} subjects!")
                
                # Visualization
                df_anchors = pd.DataFrame(anchors)
                
                st.markdown("##### Age Distribution")
                st.bar_chart(df_anchors['AGE'].value_counts())
                
                st.markdown("##### Arm Allocation")
                st.bar_chart(df_anchors['ARM'].value_counts())
                
                with st.expander("View Raw Data"):
                    st.dataframe(df_anchors)
