import streamlit as st
import logging
from ui.data_studio import render_data_studio
from ui.config_lab import render_config_lab
from ui.generation_hub import render_generation_hub

# Page Config
st.set_page_config(
    page_title="Synth Data Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for "Brilliant" Look
st.markdown("""
<style>
    /* Global Clean Dark Theme Enhancements */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Cards / Containers */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #1f242d;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        box_shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Buttons */
    .stButton>button {
        color: white;
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.sidebar.title("🧬 Synth Data Engine")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Data Studio", "Training Lab", "Generation Hub"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("v2.0-qwen | Apache 2.0")

    if page == "Data Studio":
        render_data_studio()
    elif page == "Training Lab":
        render_config_lab()
    elif page == "Generation Hub":
        render_generation_hub()

if __name__ == "__main__":
    main()
