import streamlit as st

st.title("🧪 Streamlit Test")
st.write("If you can see this, Streamlit is working!")
st.success("✅ Basic Streamlit functionality is OK")

# Test imports without loading anything
try:
    import torch
    st.write(f"✅ PyTorch: {torch.__version__}")
except Exception as e:
    st.error(f"❌ PyTorch issue: {e}")

try:
    import transformers
    st.write(f"✅ Transformers: {transformers.__version__}")
except Exception as e:
    st.error(f"❌ Transformers issue: {e}")

st.write("If you see this, the app is working fine!") 