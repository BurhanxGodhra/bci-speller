import streamlit as st
from components.stimulus import render_ssvep, render_p300

st.set_page_config(layout="wide")
st.title("Stimulus component test")

mode = st.radio("Mode", ["ssvep", "p300"])

if mode == "ssvep":
    targets = [
        {"label": "SPACE", "freq_hz": 12.0, "x": 50, "y": 20, "w": 150, "h": 60},
        {"label": "BACK", "freq_hz": 15.0, "x": 220, "y": 20, "w": 150, "h": 60},
        {"label": "SUGGEST1", "freq_hz": 20.0, "x": 390, "y": 20, "w": 150, "h": 60},
        {"label": "SUGGEST2", "freq_hz": 30.0, "x": 560, "y": 20, "w": 150, "h": 60},
    ]
    render_ssvep(targets, key="ssvep_test")
else:
    grid = [list("ABCDEF"), list("GHIJKL"), list("MNOPQR"),
            list("STUVWX"), list("YZ0123"), list("456789")]
    result = render_p300(grid, key="p300_test")
    if result:
        st.write(f"Received {len(result['flash_events'])} flash events")
        st.json(result["flash_events"][:5])
