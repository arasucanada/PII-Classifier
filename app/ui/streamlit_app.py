
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import streamlit as st
from app.services.pii_service import analyze_text

st.set_page_config(page_title="PII Detection POC", layout="wide")

st.title("🔐 PII Detection POC")

text = st.text_area("Enter text", height=150)

if st.button("Analyze"):
    with st.spinner("Running PII models..."):
        result = analyze_text(text)

    st.subheader("Roblox (Sequence Classifier)")
    st.json(result["roblox"])

    st.subheader("Piiranha (Token Classifier)")
    st.text(result["piiranha"]["masked_text"])
    st.json(result["piiranha"]["entities"])
