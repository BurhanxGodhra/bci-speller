import streamlit as st

st.set_page_config(page_title="BCI Speller", layout="wide")
st.title("Dual-Paradigm BCI Speller")
st.write(
    "P300 + SSVEP brain-computer interface speller with online artifact rejection, "
    "LLM predictive text, and OS-level output."
)
st.write("Use the sidebar to explore:")
st.markdown(
    "- **Research Demo** — benchmark results from public BCI datasets (not personalized)\n"
    "- **Your Setup** — connect your own headset and calibrate a personal model"
)
