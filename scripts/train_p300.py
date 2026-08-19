"""
Phase 3 checkpoint: loads real P300 data via MOABB, trains xDAWN+LDA, reports
cross-validated accuracy/AUC.

First run downloads the dataset (~internet required, cached to ~/mne_data afterward).

Run:
    python scripts/train_p300.py
    python scripts/train_p300.py --subjects 1 2 3 --n-filters 6
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

from paradigms.p300.epoching import load_p300_epochs
from paradigms.p300.classifier import evaluate_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BNCI2014009")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--n-filters", type=int, default=4)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading {args.dataset}, subjects={args.subjects}...")
    data = load_p300_epochs(dataset_name=args.dataset, subject_ids=args.subjects)

    print(f"\nData shape: X={data.X.shape}, y={data.y.shape}")
    print(f"Target rate: {100 * data.y.mean():.1f}%")

    print(f"\nRunning {args.n_splits}-fold cross-validated xDAWN+LDA "
          f"(n_filters={args.n_filters})...")
    results = evaluate_pipeline(data.X, data.y, n_filters=args.n_filters, n_splits=args.n_splits)

    print("\n=== SUMMARY ===")
    print(f"Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
    print(f"ROC-AUC:  {results['auc_mean']:.3f} ± {results['auc_std']:.3f}")
    print(f"({results['n_folds']}-fold cross-validation)")

    if results["auc_mean"] < 0.6:
        print("\n⚠️  AUC is close to chance (0.5) — worth checking data loading / class "
              "balance before proceeding, though single-subject P300 AUC in the 0.6-0.75 "
              "range is not unusual before any hyperparameter tuning.")
    else:
        print("\n✅ Classifier is learning a real signal above chance — Phase 3 checkpoint met.")
