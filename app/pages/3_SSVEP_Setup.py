import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from calibration.ssvep_session import SSVEPCalibrationRunner, SSVEP_TARGETS
from calibration.ssvep_epoching import extract_ssvep_trials
from paradigms.ssvep.fbcca import FBCCAClassifier
from app.components.stimulus import render_ssvep

st.set_page_config(page_title="SSVEP Setup", layout="wide")
st.title("Your Setup — SSVEP Validation")

st.markdown(
    "**What this does:** the 4 boxes below flicker at different speeds. Staring at one "
    "produces a matching rhythm in your brain's visual cortex. This checks whether we "
    "can actually detect that rhythm on your headset — it's a quick check, not a "
    "training step (unlike P300, this method needs no training data at all)."
)
st.markdown(
    "**What you do:** one box at a time will be marked **ATTEND**. Stare directly at "
    "that box, and try not to blink or look away, until it moves to the next one."
)
st.caption("Requires an LSL EEG stream running (real headset or hardware/playback.py --source synthetic)")

trial_seconds = st.slider("Seconds per target", 2.0, 8.0, 4.0)
n_repeats = st.slider("Repeats per target", 1, 3, 1)

if "ssvep_runner" not in st.session_state:
    st.session_state.ssvep_runner = None

if st.button("Start Validation", disabled=st.session_state.ssvep_runner is not None):
    runner = SSVEPCalibrationRunner(trial_seconds=trial_seconds, n_repeats=n_repeats)
    runner.start()
    st.session_state.ssvep_runner = runner
    st.rerun()

runner = st.session_state.ssvep_runner

if runner is not None:
    if not runner.is_done and runner.current_target_idx is not None:
        current_label = SSVEP_TARGETS[runner.current_target_idx]["label"]
        elapsed = time.time() - runner.trial_start_time if runner.trial_start_time else 0
        remaining = max(0.0, runner.trial_seconds - elapsed)
        st.info(f" Stare at **{current_label}** — {remaining:.1f}s remaining")
        st.progress((runner.trial_index + 1) / runner.total_trials)
        st.caption(f"Trial {runner.trial_index + 1} of {runner.total_trials}")
    elif not runner.is_done:
        st.info("Get ready for the next target...")

    targets = []
    for i, t in enumerate(SSVEP_TARGETS):
        is_current = runner.current_target_idx == i
        targets.append({
            "label": ("ATTEND: " if is_current else "") + t["label"],
            "freq_hz": t["freq_hz"],
            "x": 50 + i * 170, "y": 20, "w": 150, "h": 60,
        })
    render_ssvep(targets, key="ssvep_calib_stim")

    if not runner.is_done:
        time.sleep(0.2)
        st.rerun()
    else:
        st.success("Validation finished. Analyzing your data below.")
        if runner.error:
            st.error(f"Validation failed: {runner.error}")
        elif runner.eeg.shape[1] == 0:
            st.error("No EEG data recorded — check your LSL stream is running.")
        else:
            freqs = [t["freq_hz"] for t in SSVEP_TARGETS]
            segments, true_freqs = extract_ssvep_trials(runner.eeg, runner.eeg_ts, runner.trials)

            if not segments:
                st.warning("No valid trials extracted — check EEG stream stayed connected throughout.")
            else:
                classifier = FBCCAClassifier(freqs, runner.sample_rate)
                correct = 0
                rows = []
                for seg, true_f in zip(segments, true_freqs):
                    pred_idx = classifier.predict(seg[None, ...])[0]
                    pred_f = freqs[pred_idx]
                    correct += int(pred_f == true_f)
                    rows.append({"attended": true_f, "predicted": pred_f, "correct": pred_f == true_f})

                accuracy = correct / len(segments)
                st.metric("Per-trial accuracy", f"{accuracy:.1%}", f"{correct}/{len(segments)} trials")
                st.dataframe(rows)

                chance = 1 / len(freqs)
                with st.expander("What does this number mean?"):
                    st.markdown(
                        f"With {len(freqs)} possible targets, random guessing gets **{chance:.0%}** "
                        f"accuracy. Anything meaningfully above that means FBCCA is actually picking "
                        f"up your steady-state response.\n\n"
                        f"If you're testing against the synthetic mock stream (not a real headset), "
                        f"~{chance:.0%} is the *expected, correct* result — the mock stream has no real "
                        f"periodic signal for FBCCA to find. This only becomes a meaningful accuracy "
                        f"check when run against real EEG."
                    )

        if st.button("Reset"):
            st.session_state.ssvep_runner = None
            st.rerun()
