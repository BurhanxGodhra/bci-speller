import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from sklearn.model_selection import train_test_split

from paradigms.ssvep.epoching import load_ssvep_epochs
from paradigms.ssvep.cca import CCAClassifier
from paradigms.ssvep.fbcca import FBCCAClassifier
from paradigms.ssvep.cnn_baseline import train_cnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Nakanishi2015")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--sfreq", type=float, default=256.0)
    args = parser.parse_args()

    X, y, freqs, meta = load_ssvep_epochs(args.dataset, args.subjects)
    print(f"X={X.shape}, classes={len(freqs)}, freqs={freqs}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    cca_acc = (CCAClassifier(freqs, args.sfreq).predict(X_test) == y_test).mean()
    fbcca_acc = (FBCCAClassifier(freqs, args.sfreq).predict(X_test) == y_test).mean()
    _, cnn_acc = train_cnn(X_train, y_train, X_test, y_test, n_classes=len(freqs))

    print(f"\nCCA accuracy:   {cca_acc:.3f}")
    print(f"FBCCA accuracy: {fbcca_acc:.3f}")
    print(f"CNN accuracy:   {cnn_acc:.3f}")
