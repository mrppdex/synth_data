import streamlit as st

def render_config_lab():
    st.title("🧪 Training Lab")
    st.markdown("Configure the Generator Model and Prompt Strategy.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Settings")
        
        model_id = st.text_input("Model ID", value="Qwen/Qwen2.5-3B-Instruct")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7)
        max_tokens = st.number_input("Max New Tokens", value=512)
        quantization = st.checkbox("4-bit Quantization (Requires CUDA)", value=False)
        
        if st.button("Save Configuration"):
            st.session_state['config'] = {
                "model_id": model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "quantization": quantization
            }
            st.success("Configuration Saved to Session State")

    with col2:
        st.subheader("System Prompt Preview")
        st.markdown("""
        The system prompt enforces the JSON output format and role.
        
        ```text
        Role: CDISC ADaM Synthetic Data Generator
        Constraint: Output must be valid JSON matching the Specification variables.
        Do not output markdown code blocks. Output ONLY the raw JSON object.
        ```
        """)
        
        st.text_area("Custom Instructions (Appended)", placeholder="e.g. Ensure all dates are in ISO8601 format...")
