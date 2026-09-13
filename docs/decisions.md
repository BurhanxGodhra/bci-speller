# Engineering Decision Log

Itemized record of the non-obvious calls made during this build, and why. Numbered in roughly chronological order.

---

**D-001 — Substituting `BNCI2014009` for "BCI Competition III Dataset II"**

The original spec called for the classic BCI Competition III P300 speller dataset. It isn't packaged under that name in MOABB. `BNCI2014009` uses the same experimental paradigm (row/column flashing matrix, 6-letter grid) and is a well-supported MOABB dataset, so it was used instead. If the literal competition files are ever needed, MOABB has a separate raw-download path that would need to be wired in.

---

**D-002 — Shrinkage LDA over plain LDA for P300**

P300 epoch counts per subject are small relative to feature dimensionality after xDAWN flattening. Plain LDA's covariance estimate overfits in that regime. `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` uses Ledoit-Wolf shrinkage to regularize toward a well-conditioned target — standard practice for this exact data regime, not a novel choice, but a deliberate one.

---

**D-003 — Score-level fusion for warm-start blending, not feature-level**

The public prior model and the personal model are trained on different channel counts (prior restricted to whatever public-dataset channels overlap the HAL montage; personal uses the full HAL set). Trying to share xDAWN spatial filters across mismatched channel spaces isn't well-defined. Blending each model's independent scalar `decision_function` output sidesteps the mismatch entirely — each model only ever sees its own channels, and the outputs are combined as a weighted sum of two per-sample real numbers, which places no constraint on their internal channel dimensionality.

---

**D-004 — Z-score by std only, not by mean, when blending decision scores**

First implementation subtracted each model's training-score mean before scaling by std, on the assumption this was standard normalization. It isn't correct here: LDA's actual decision threshold is 0, and subtracting the training mean shifts the effective threshold to wherever that mean happens to sit — which is generally *not* 0, since decision-function score distributions aren't centered at the boundary in general. Confirmed via a direct check: at blend weight 1.0, blended output should be mathematically identical to personal-only. It wasn't, until mean-centering was dropped and only std-scaling kept.

---

**D-005 — Resample to the prior model's timepoint count, not just match channels**

Channel-overlap checking alone wasn't sufficient to make the two models compatible for score fusion — `BNCI2014009`'s native ~257Hz sample rate gives a different timepoint count per 0.8s epoch than personal calibration epochs extracted at 250Hz, and xDAWN's flattened feature count depends on both channel count and timepoint count. Fixed by resampling the picked-channel personal segment to the prior model's exact `n_times` via `scipy.signal.resample` before scoring — not before training the personal model itself, only at the point of scoring against the prior.

---

**D-006 — Browser stimulus timing derived from elapsed time, not frame count**

SSVEP flicker state is computed as `floor(t_seconds * 2 * freq) % 2` using `performance.now()`, rather than counting rendered frames. Frame-count-based toggling silently assumes a stable frame rate; if the browser's actual `requestAnimationFrame` cadence varies even slightly, frame-counted flicker drifts from the intended frequency over time. Time-based state computation is self-correcting regardless of actual frame delivery rate.

---

**D-007 — Dropped `streamlit-component-lib` CDN dependency for the stimulus component**

Initial implementation loaded `streamlit-component-lib` from unpkg via a `<script>` tag, on the assumption it would expose a global `Streamlit` object like most CDN-distributed libraries do. It doesn't — the package is built for bundler consumption, not direct script-tag use, and the global was never defined, throwing `ReferenceError: Streamlit is not defined`. Replaced with a direct implementation of Streamlit's component `postMessage` protocol (`streamlit:componentReady`, `streamlit:setFrameHeight`, `streamlit:setComponentValue`), removing the external dependency entirely.

---

**D-008 — Background thread + rerun-polling for live component updates, not repeated calls in one script run**

Calling a keyed Streamlit component function repeatedly inside a single blocking Python loop (the initial P300 calibration wizard design) throws `StreamlitDuplicateElementKey` — Streamlit's execution model registers an element once per script run, and doesn't support re-registering the same key mid-run to simulate animation. Correct pattern: run the actual timing-critical work (flash scheduling, LSL marker pushing, EEG recording) in a background thread that only updates plain shared attributes; have the Streamlit script itself rerun on a fixed short interval (`time.sleep(0.1); st.rerun()`), calling the component exactly once per rerun with whatever the current state is. Marker timestamps come from the background thread's own clock, independent of UI rerun cadence — visual smoothness and epoch-timing correctness are fully decoupled.

---

**D-009 — P300 uses Python-driven flash timing; SSVEP does not**

P300 needs precise, low-jitter flash-onset timestamps for accurate epoching (the ERP is a transient, time-locked response). The browser's self-scheduling JS timer, reporting flash events back to Python in batches, introduces variable websocket round-trip latency into that timestamp — acceptable for a visual demo, not acceptable for real epoch alignment. P300 calibration therefore has Python decide every flash and push the LSL marker itself, with the browser purely rendering whatever state it's told. SSVEP's steady-state response builds up continuously over 1–2 seconds, so second-level marker precision is sufficient; the SSVEP validation session keeps the simpler design (Python marks trial start/end only, JS self-schedules the actual flicker).

---

**D-010 — Pin Python to 3.11 for this project, explicitly**

`torch==2.4.1` has no published wheel for Python 3.13. A `.venv` created without specifying the interpreter picked up whatever `python3` resolved to system-wide, which had moved to 3.13 between the start of the project and a later session, breaking a previously-working `pip install -r requirements.txt`. Fixed by creating the venv explicitly with `python3.11 -m venv .venv` rather than relying on the default `python3` resolution.

---

**D-011 — NSScreen query over software frame-timing for authoritative refresh rate**

Timing a small windowed SDL2 surface (even with `Renderer(vsync=True)`) is not reliable on macOS — it reported physically impossible refresh rates (96Hz, then 131Hz) on a MacBook Air with a fixed 60Hz panel. Queried `NSScreen.maximumFramesPerSecond()` via `pyobjc` instead, which correctly reports both fixed-rate panels and ProMotion's variable-rate ceiling. Software-measured frame timing was kept only as a clearly-labeled, non-authoritative sanity check.
