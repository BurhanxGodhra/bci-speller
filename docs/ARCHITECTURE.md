# Architecture

This is organized by subsystem, each with what it does and why it's built the way it is. For the chronological story of how it got to this state (including the bugs), see `PROJECT_NARRATIVE.md`. For the itemized rationale behind specific non-obvious calls, see `decisions.md`.

## Hardware Abstraction Layer (`hardware/`)

Everything above this layer — paradigm code, UI, calibration — only ever sees numpy arrays of a fixed channel set at a fixed sample rate. The HAL's entire job is making that true regardless of what's actually plugged in.

- **`drivers/lsl_discovery.py`** — wraps `pylsl.resolve_byprop('type', 'EEG')`, LSL's canonical device-agnostic discovery mechanism. Any correctly-configured source (OpenBCI's LSL bridge, `muse-lsl`, our own `playback.py`) announces itself this way regardless of host/port, so the rest of the system never hardcodes a device name.
- **`drivers/channel_mapper.py`** — reconciles a device's actual electrode labels against the channels paradigms need (`config.yaml`'s `required_channels`), using a per-device fallback table for headsets that physically lack an electrode (e.g. Muse has no true occipital channel; `config.yaml` declares which electrode substitutes for it, and the mapper logs a warning whenever a fallback fires).
- **`drivers/resampler.py`** — harmonizes sample rate via `scipy.signal.resample_poly` (polyphase filtering with proper anti-aliasing), not naive decimation or linear interpolation, which would smear the high-frequency content SSVEP detection depends on.
- **`drivers/artifact_filter.py`** — three checks per chunk (absolute amplitude, peak-to-peak, sliding-window variance), any one of which rejects the whole chunk. Wired into `hal.py`'s `stream()` as an optional filter, so paradigm code can opt into clean-only data without touching HAL internals.
- **`playback.py`** — publishes a synthetic LSL stream with the same channel/rate contract as a real device, so every other subsystem can be developed and tested without hardware. Everything in this repo's verified results either uses this or a public dataset — see **Limitations** in `README.md`.

## Paradigm modules (`paradigms/`)

P300 and SSVEP are siblings, not a shared "signal processing" module — their math is different enough (ERP time-locking vs. frequency-domain correlation) that forcing them into one abstraction would just produce an if/else maze.

- **`p300/`** — `epoching.py` loads MOABB data (optionally restricted to a channel subset, used by the warm-start prior model — see below); `xdawn.py` wraps `pyriemann.estimation.Xdawn` as an sklearn-compatible transformer; `classifier.py` chains it into a shrinkage-LDA pipeline with stratified cross-validation.
- **`ssvep/`** — `cca.py` builds sine/cosine reference signals per candidate frequency and computes canonical correlation; `fbcca.py` extends this with a filter bank over harmonic sub-bands, weighted by band index. Both are zero-shot — no `.fit()` step exists because none is needed, which is the entire point of CCA-family methods for SSVEP.

## Calibration (`calibration/`)

This is where "the algorithm works on a benchmark dataset" and "a real person could use this" diverge, and where most of the actual engineering problems in this project turned up.

**LSL marker synchronization.** `lsl_markers.py` wraps a `pylsl` Markers-type outlet. `recorder.py` runs a background thread continuously pulling raw (unresampled) EEG with per-sample LSL timestamps into a growing buffer. Because both the marker outlet and the EEG inlet share the same machine's LSL clock domain, a marker pushed at time `t` can be matched against the EEG buffer via `np.searchsorted` on timestamps — no explicit clock synchronization protocol needed, LSL handles that internally.

**Why P300 and SSVEP calibration have different timing architectures.** P300 is a transient, time-locked response — accurate epoching needs each flash's onset timestamp accurate to well under 100ms. The browser can't be trusted to self-report that precisely (websocket round-trip jitter), so `session.py` runs the flash schedule *in Python*, on a background thread, pushing the LSL marker itself at the exact moment it decides to flash — the browser only ever renders whatever state Python tells it to (`p300_controlled` mode in the stimulus component; see below). SSVEP's steady-state response builds up continuously over 1–2 seconds, so second-level precision is enough; `ssvep_session.py` only marks trial start/end, and lets the browser self-schedule the actual flicker.

**Warm-start blending.** `base_model.py` trains a P300 prior model on public data, restricted to whatever channels overlap the personal calibration montage (discovered automatically, not assumed — currently `Oz`/`Cz`/`Pz`, 3 of 5). `blending.py` combines the prior model's and the personal model's `decision_function` outputs as a weighted sum, not at the feature/spatial-filter level — this sidesteps the channel-count mismatch between the two models entirely, since each only ever scores its own channels independently. Two real numerical issues had to be fixed here for the blend to be mathematically sound at all: a timepoint-count mismatch from differing native sample rates (fixed via resampling before scoring, D-005), and a decision-threshold shift from mean-centering during z-score normalization (fixed by scaling with std only, D-004). Both are documented with the actual before/after numbers in `decisions.md` and `BENCHMARKS.md`.

## Browser stimulus component (`app/components/stimulus/`)

A static HTML/JS component registered via `streamlit.components.v1.declare_component`, communicating with Python through Streamlit's actual component protocol — implemented directly via `postMessage` (`streamlit:componentReady`, `streamlit:setFrameHeight`, `streamlit:setComponentValue`) rather than the `streamlit-component-lib` npm package, which isn't built for direct script-tag use (D-007).

Three render modes:
- **`ssvep`** — canvas-based flicker, state computed from elapsed `performance.now()` time (`floor(t * 2 * freq) % 2`), not frame count, so it doesn't drift if the browser's actual frame delivery rate varies (D-006).
- **`p300`** — self-scheduling flash sequence with its own internal row/col scheduler, batching flash events back to Python every 40 flashes. Used for the visual demo on the Research Demo page, where precise epoch-worthy timestamps aren't needed.
- **`p300_controlled`** — no internal scheduling at all; draws whatever `active_row`/`active_col` Python's most recent render call specified. This is what the calibration wizard uses, paired with the background-thread architecture described below.

**Why the calibration wizard uses a background thread + rerun-polling pattern, not repeated component calls.** Streamlit registers a keyed element once per script execution — calling the same component multiple times inside one blocking loop throws `StreamlitDuplicateElementKey` (this was the actual first-draft bug, D-008). The fix: the timing-critical work (flash scheduling, marker pushing, EEG recording) runs in a background thread updating plain Python attributes; the Streamlit script itself reruns on a ~100ms cadence via `time.sleep(0.1); st.rerun()`, calling the component exactly once per rerun with the current state. This decouples epoch-timing correctness (comes from the background thread's own clock) from UI redraw smoothness (comes from however fast Streamlit happens to rerun) — visual choppiness under this pattern doesn't affect the calibration data's validity.

## Streamlit app structure (`app/`)

Two pages deliberately kept visually and functionally separate: **Research Demo** (read-only, pretrained-on-public-data numbers, explicitly labeled as not personalized) and **P300 Setup** / **SSVEP Setup** (the calibration wizards, operating on the user's own connected stream). The mock LSL stream can be started/stopped from inside either setup page (`calibration/mock_stream_control.py`, a thin `subprocess.Popen` wrapper) rather than requiring a second terminal — mirroring the pattern used in the author's prior BCI project (`mi-bci-pipeline`), where the app manages its own underlying stream subprocess.

## Closed-loop output (`os_hooks/`)

`keypress_injector.py` wraps `pyautogui` for OS-level text injection into whatever window has focus. `itr.py` implements the standard Wolpaw information-transfer-rate formula. Both are deliberately simple, stateless wrappers — the actual interesting engineering in this project is upstream of this point (signal processing, calibration, timing), not in the OS injection itself.
