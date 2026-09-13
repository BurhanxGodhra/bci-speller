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

st.markdown(
    "**What this does:** the grid below will flash rows and columns of letters in "
    "random order. Your brain produces a distinct signal roughly 300ms after you see "
    "a flash you're specifically watching for. Recording enough of these lets us train "
    "a personal classifier that can later tell which letter you're attending to."
)
st.markdown(
    "**What you do:** pick a letter below, then during the run, keep your eyes fixed "
    "on that letter the entire time — don't look around the grid. You'll see it flash "
    "along with others; just keep watching it."
)
st.caption("Requires an LSL EEG stream running (real headset or hardware/playback.py --source synthetic)")

target_letter = st.selectbox("Which letter will you attend to?", [c for row in GRID for c in row])
n_reps = st.slider("Repetitions", 5, 20, 10, help="More repetitions = more data = usually better accuracy, but longer session")

if "runner" not in st.session_state:
    st.session_state.runner = None

if st.button("Start Calibration", disabled=st.session_state.runner is not None):
    runner = CalibrationRunner(target_letter, n_repetitions=n_reps)
    runner.start()
    st.session_state.runner = runner
    st.rerun()

runner = st.session_state.runner

if runner is not None:
    if not runner.is_done:
        st.info(f" Keep your eyes on **{runner.target_letter}** — don't look at any other letter until this finishes.")
        st.progress(runner.flashes_done / runner.total_flashes)
        st.caption(f"Flash {runner.flashes_done} of {runner.total_flashes}")

    render_p300_controlled(GRID, active_row=runner.active_row, active_col=runner.active_col, key="calib_stim")

    if not runner.is_done:
        time.sleep(0.1)
        st.rerun()
    else:
        st.success("Recording finished. Analyzing your data below.")
        if runner.error:
            st.error(f"Calibration failed: {runner.error}")
        else:
            eeg, eeg_ts, events, sample_rate = runner.result
            if eeg.shape[1] == 0:
                st.error("No EEG data recorded — check your LSL stream is running.")
            else:
                X, y = extract_epochs(eeg, eeg_ts, events, sample_rate=sample_rate)
                st.write(f"Extracted {len(X)} epochs ({int(y.sum())} where '{runner.target_letter}' flashed, "
                         f"{int((y == 0).sum())} where it didn't)")

                if y.sum() < 3 or (y == 0).sum() < 3:
                    st.warning("Not enough epochs per class to train — try again with more repetitions.")
                else:
                    from calibration.base_model import build_or_load_prior_model
                    from calibration.blending import evaluate_blended

                    hal_channels = ["O1", "Oz", "O2", "Cz", "Pz"]
                    with st.spinner("Loading/training prior model from public data..."):
                        prior_bundle = build_or_load_prior_model(hal_channels)
                    channel_indices = [hal_channels.index(ch) for ch in prior_bundle["channels"]]

                    blend_results = evaluate_blended(X, y, prior_bundle, channel_indices,
                                                      n_splits=min(5, int(y.sum())))

                    st.subheader("Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Personal model only**")
                        st.metric("Accuracy", f"{blend_results['personal_accuracy']:.1%}")
                        st.metric("ROC-AUC", f"{blend_results['personal_auc']:.3f}")
                    with col2:
                        st.markdown("**Warm-started (blended with public data)**")
                        st.metric("Accuracy", f"{blend_results['blended_accuracy']:.1%}")
                        st.metric("ROC-AUC", f"{blend_results['blended_auc']:.3f}")

                    with st.expander("What do these numbers mean?"):
                        st.markdown(
                            "- **Accuracy** — how often the model correctly told target flashes apart from "
                            "non-target flashes, on data it wasn't trained on.\n"
                            "- **ROC-AUC** — a more reliable measure than accuracy here, since target flashes "
                            "are rare (~1 in 6). 0.5 = no better than guessing, 1.0 = perfect. Real signal "
                            "usually shows up as AUC clearly above 0.5.\n"
                            "- **Warm-started** blends your personal model with one pretrained on public "
                            "research data — meant to help when you don't have much personal data yet. "
                            "With enough of your own data, it converges to the personal-only result "
                            "(you'll see this if the two columns match)."
                        )
                        st.caption(f"Personal model weight in the blend: {blend_results['weight_used']:.2f} "
                                   f"(0 = fully relying on public data, 1 = fully your own)")
                        st.caption(f"Public prior model was restricted to channels overlapping your setup: "
                                   f"{prior_bundle['channels']}")

        if st.button("Reset"):
            st.session_state.runner = None
            st.rerun()
