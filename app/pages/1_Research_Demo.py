import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from components.stimulus import render_ssvep

st.set_page_config(page_title="Research Demo", layout="wide")
st.title("Research Demo")
st.info(
    "Results below are from public research datasets (BNCI2014009, Nakanishi2015), "
    "not from a personal calibration. See **P300 Setup** or **SSVEP Setup** to train/validate on your own data."
)

st.header("P300 speller — xDAWN + shrinkage LDA")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "88.4%", "±1.8%")
col2.metric("ROC-AUC", "0.918", "±0.013")
col3.metric("Dataset", "BNCI2014009")
st.caption("5-fold stratified CV, subject 1, 1728 epochs (16.7% target rate)")

st.header("SSVEP — CCA / FBCCA / CNN baseline")
df = pd.DataFrame({"Method": ["CCA", "FBCCA", "CNN (supervised)"], "Accuracy": [0.704, 0.778, 0.074]})
st.bar_chart(df.set_index("Method"))
st.caption(
    "Nakanishi2015, subject 1, 12-class, 180 total epochs. CNN underperforms — too few "
    "training examples per class (~10) for supervised deep learning at this sample size. "
    "CCA/FBCCA need no training data at all."
)

st.header("Offline replay -> online decoder -> OS keypress injection (FBCCA)")
col1, col2, col3 = st.columns(3)
col1.metric("Selections", "15")
col2.metric("Accuracy", "86.67%")
col3.metric("ITR", "38.36 bits/min")
st.caption("Wolpaw ITR, held-out labeled trials, 4s per selection")

with st.expander("See the stimulus engine in action"):
    targets = [
        {"label": "SPACE", "freq_hz": 12.0, "x": 50, "y": 20, "w": 150, "h": 60},
        {"label": "BACK", "freq_hz": 15.0, "x": 220, "y": 20, "w": 150, "h": 60},
        {"label": "SUGGEST1", "freq_hz": 20.0, "x": 390, "y": 20, "w": 150, "h": 60},
        {"label": "SUGGEST2", "freq_hz": 30.0, "x": 560, "y": 20, "w": 150, "h": 60},
    ]
    render_ssvep(targets, key="demo_ssvep_preview")
