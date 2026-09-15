"""
Multi-budget SSVEP comparison: how does supervised CNN accuracy scale with
training examples per class, compared against zero-shot FBCCA (which needs
none)? Answers "when would calibration-based deep learning overtake zero-shot
FBCCA" rather than a single CNN number at whatever data happened to be available.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from paradigms.ssvep.epoching import load_ssvep_epochs
from paradigms.ssvep.fbcca import FBCCAClassifier
from paradigms.ssvep.cnn_baseline import train_cnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Nakanishi2015")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--sfreq", type=float, default=256.0)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6, 8])
    args = parser.parse_args()

    X, y, freqs, meta = load_ssvep_epochs(args.dataset, args.subjects)
    n_classes = len(freqs)
    print(f"X={X.shape}, classes={n_classes}")

    # Fixed held-out test set, same across every budget for a fair comparison
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    fbcca_acc = (FBCCAClassifier(freqs, args.sfreq).predict(X_test) == y_test).mean()
    print(f"\nFBCCA (zero-shot, no training data): {fbcca_acc:.3f}")

    max_per_class = min(np.bincount(y_train_full))
    print(f"Max available training examples/class: {max_per_class}")

    print(f"\n{'Budget/class':<15}{'CNN accuracy':<15}{'FBCCA (reference)':<20}")
    for budget in args.budgets:
        if budget > max_per_class:
            print(f"{budget:<15}(skipped -- exceeds {max_per_class} available)")
            continue

        idx_subset = []
        for c in range(n_classes):
            class_idx = np.where(y_train_full == c)[0]
            rng = np.random.default_rng(42)
            idx_subset.extend(rng.choice(class_idx, size=budget, replace=False))
        idx_subset = np.array(idx_subset)

        _, cnn_acc = train_cnn(X_train_full[idx_subset], y_train_full[idx_subset],
                                X_test, y_test, n_classes=n_classes)
        print(f"{budget:<15}{cnn_acc:<15.3f}{fbcca_acc:<20.3f}")

    print(f"\nFull training set ({len(X_train_full)} examples, "
          f"~{len(X_train_full) // n_classes}/class):")
    _, cnn_full_acc = train_cnn(X_train_full, y_train_full, X_test, y_test, n_classes=n_classes)
    print(f"CNN: {cnn_full_acc:.3f}  |  FBCCA: {fbcca_acc:.3f}")
