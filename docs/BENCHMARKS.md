# Benchmarks

Every number here comes from an actual run logged during this build — none of it is estimated or interpolated. Where a run only used one subject from a public dataset, that's stated explicitly; single-subject numbers should not be read as claims about how the pipeline performs across a population.

**Environment:** MacBook Air (Apple Silicon, fixed 60Hz display, no ProMotion), macOS, Python 3.11, `torch==2.4.1`, `scikit-learn`/`mne`/`moabb` unpinned (see `requirements.txt` and `docs/decisions.md` D-010 for why).

---

## P300 — xDAWN + shrinkage-LDA

**Dataset:** `BNCI2014009` (Aricò et al., 2014), subject 1 only, loaded via MOABB's `P300` paradigm.
**Data:** 1728 epochs, 16 channels, 206 timepoints/epoch (~0.8s @ ~257Hz). 288 target / 1440 non-target (16.7% target rate — a classifier that always predicts non-target gets 83.3% accuracy for free, which is why AUC is the metric that actually matters here, not the accuracy number below).
**Evaluation:** 5-fold stratified cross-validation (stratified specifically because of the class imbalance — a plain K-fold risks a fold with almost no target examples).

| Fold | Accuracy | ROC-AUC |
|---|---|---|
| 0 | 0.861 | 0.896 |
| 1 | 0.893 | 0.927 |
| 2 | 0.893 | 0.929 |
| 3 | 0.864 | 0.911 |
| 4 | 0.907 | 0.929 |
| **Mean ± SD** | **0.884 ± 0.018** | **0.918 ± 0.013** |

No fold is an outlier — accuracy stays within a 4.6-point band across all five, which is what you'd want to see before trusting the mean.

**Balanced accuracy and confusion matrix** (summed across all 5 folds, `scripts/train_p300.py`):

| Metric | Value |
|---|---|
| Balanced accuracy | 0.783 ± 0.031 |

|  | Predicted NonTarget | Predicted Target |
|---|---|---|
| **Actual NonTarget** | 1345 | 95 |
| **Actual Target** | 106 | 182 |

Sensitivity (catching real target flashes) = 182/288 = **63.2%**. Specificity (correctly rejecting non-targets) = 1345/1440 = **93.4%**. Balanced accuracy (0.783) sitting well above chance (0.5) and meaningfully below raw accuracy (0.884) is the expected pattern for imbalanced data — it's the number to trust over raw accuracy.

**Character-level decode**: row/column vote aggregation into an actual letter decision (`calibration/character_decode.py`), evaluated by splitting a calibration session in half by repetition — first half trains, second half is a genuinely held-out spelling attempt. Two runs against the synthetic mock stream, both target letter chosen deliberately to avoid the top-left corner (which would make a default-to-index-0 bug indistinguishable from a real decode):

| Session | Flash-level AUC | Repetitions | Decode result |
|---|---|---|---|
| Target "A" | 0.593 | 1-5 | Correctly decoded 'A' at every repetition count |
| Target "E" | 0.627 | 1-5 | Predicted '1', 'C', 'I', 'K', 'K' — never correct, and unstable across repetition counts |

Both AUCs sit only modestly above chance (0.5) — there's essentially no real P300 signal in synthetic noise, as expected. The "A" result isn't evidence the decoder works; it's a coincidental hit at long odds (each session has roughly a 1-in-36 chance of the classifier's spurious noise-driven boundary happening to line up with the true target). The "E" result is the more informative one: predictions genuinely *vary* across repetition counts rather than converging on a single fixed (wrong) answer, which confirms the vote-aggregation logic is behaving correctly and responding to input — it rules out a "silently defaults to one letter regardless of input" bug, which was the first hypothesis tested here. What it does *not* do is demonstrate working character-level decoding, since there's no real signal in synthetic data for it to lock onto. That test can only happen on real EEG.

---

## SSVEP — CCA vs. FBCCA vs. CNN

**Dataset:** `Nakanishi2015` (Nakanishi et al., 2015), subject 1 only.
**Data:** 180 epochs, 8 channels, 12-class (frequencies 9.25–14.75Hz in 0.5Hz steps).
**Split:** 70/30 train/test, stratified.

| Method | Accuracy | Notes |
|---|---|---|
| CCA | 70.4% | zero-shot, no training data used |
| **FBCCA** | **77.8%** | zero-shot, harmonic filter-bank weighting |
| CNN (supervised) | 7.4% | ~10 training examples/class — not enough data, included to show why CCA/FBCCA are standard for SSVEP |

Chance level for 12-class is 8.3%. The CNN's 7.4% is at chance, which is the expected outcome given the sample size, not a broken implementation — see `docs/PROJECT_NARRATIVE.md`.

**Accuracy vs. calibration budget** (`scripts/train_ssvep_budget.py`) — does more training data close the gap to FBCCA? Fixed 30% held-out test set across all budgets for a fair comparison; `Nakanishi2015` subject 1 caps out at 10 examples/class available for training after that split.

| Training examples/class | CNN accuracy | FBCCA (reference, zero-shot) |
|---|---|---|
| 2 | 9.3% | 77.8% |
| 4 | 16.7% | 77.8% |
| 6 | 7.4% | 77.8% |
| 8 | 20.4% | 77.8% |
| 10 (max available) | 11.1% | 77.8% |

No monotonic improvement — accuracy is noisy and stays near the 8.3% chance floor at every budget tested, including the dataset's maximum. The honest conclusion isn't "CNN needs slightly more data to catch up" — it's that **this dataset's entire available calibration budget doesn't get CNN meaningfully off the ground**, let alone close to FBCCA's zero-shot 77.8%. That's a specific, falsifiable finding about this dataset's size, not a hand-wave about small samples in general.

---

## Closed loop — FBCCA + OS keypress injection

**Data:** 15 held-out labeled `Nakanishi2015` trials, classified live and typed via `pyautogui` into a focused text field.
**Timing:** 4 seconds per selection (fixed, matching the trial duration in the source dataset).

| Metric | Value |
|---|---|
| Selections | 15 |
| Correct | 13 |
| Accuracy | 86.67% |
| ITR (Wolpaw formula) | 38.36 bits/min |

ITR calculated as `bits_per_trial(N=12, p=0.8667) × (60 / 4)`. For reference, published FBCCA-based SSVEP spellers report roughly 20–60 bits/min depending on subject and trial length — 38.36 sits comfortably inside that range.

---

## Calibration pipeline correctness (not a classification benchmark)

These runs used the synthetic mock LSL stream (`hardware/playback.py`), which has no real evoked response in it. The point of these numbers is to confirm the marker-to-EEG timing pipeline is correct, not to measure classification quality.

**P300 wizard, target letter "A", 10 repetitions:**

| Metric | Value |
|---|---|
| Epochs extracted | 119 |
| Target epochs | 19 (expected: ~20, since row+col both hit for a corner letter) |
| Non-target epochs | 100 (expected: exactly 100 — 10 reps × 10 non-target flashes) |
| Accuracy | 84.1% |
| ROC-AUC | 0.545 |

The exact match on non-target count (100/100 predicted) is the real evidence here — it confirms every flash marker landed in the recorded EEG at the position the schedule predicts. AUC near chance is the *correct* outcome on synthetic noise, not a failure.

**Warm-start blending, weight-saturation check:**

At `personal_weight = 1.0` (achieved once target-class epochs exceed the ramp threshold), blended output must be mathematically identical to personal-only. Confirmed after fixing two bugs (timepoint mismatch, mean-centering in z-scoring — see `docs/decisions.md` D-004, D-005):

| | Personal only | Blended |
|---|---|---|
| Accuracy | 82.5% | 82.5% |
| ROC-AUC | 0.360 | 0.360 |

Exact match, as required.

**SSVEP validation, 4 targets, 2 repeats (8 trials), synthetic stream:**

| Metric | Value |
|---|---|
| Per-trial accuracy | 25.0% (2/8 correct) |
| Chance level (4 classes) | 25.0% |

Right at chance, which is the expected, correct result on a stream with no real periodic signal to detect.
