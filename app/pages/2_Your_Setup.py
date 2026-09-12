import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from calibration.session import CalibrationRunner, GRID
from calibration.epoching import extract_epochs
from paradigms.p300.classifier import evaluate_pipeline
from app.components.stimulus import render_p300_controlled

st.set_page_config(page_title="Your Setup", layout="wide")
st.title("Your Setup — P300 Calibration")
st.caption("Requires an LSL EEG stream running (real headset or hardware/playback.py --source synthetic)")

target_letter = st.selectbox("Attend to this letter during calibration:", [c for row in GRID for c in row])
n_reps = st.slider("Repetitions", 5, 20, 10)

if "runner" not in st.session_state:
    st.session_state.runner = None

if st.button("Start Calibration", disabled=st.session_state.runner is not None):
    runner = CalibrationRunner(target_letter, n_repetitions=n_reps)
    runner.start()
    st.session_state.runner = runner
    st.rerun()

runner = st.session_state.runner

if runner is not None:
    render_p300_controlled(GRID, active_row=runner.active_row, active_col=runner.active_col, key="calib_stim")

    if not runner.is_done:
        time.sleep(0.1)
        st.rerun()
    else:
        if runner.error:
            st.error(f"Calibration failed: {runner.error}")
        else:
            eeg, eeg_ts, events, sample_rate = runner.result
            if eeg.shape[1] == 0:
                st.error("No EEG data recorded — check your LSL stream is running.")
            else:
                X, y = extract_epochs(eeg, eeg_ts, events, sample_rate=sample_rate)
                st.write(f"Extracted {len(X)} epochs ({int(y.sum())} target / {int((y == 0).sum())} non-target)")

                if y.sum() < 3 or (y == 0).sum() < 3:
                    st.warning("Not enough epochs per class to train — increase repetitions.")
                else:
                    results = evaluate_pipeline(X, y, n_filters=3, n_splits=min(5, int(y.sum())))
                    col1, col2 = st.columns(2)
                    col1.metric("Accuracy", f"{results['accuracy_mean']:.1%}")
                    col2.metric("ROC-AUC", f"{results['auc_mean']:.3f}")

        if st.button("Reset"):
            st.session_state.runner = None
            st.rerun()
