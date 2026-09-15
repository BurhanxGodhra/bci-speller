import numpy as np

from paradigms.p300.classifier import build_xdawn_lda_pipeline


def decode_character(events, decision_scores, n_rows=6, n_cols=6):
    """Aggregates flash-level decision scores into row/column votes by summation
    (standard P300 speller decoding), returns the predicted (row, col)."""
    row_scores = np.zeros(n_rows)
    col_scores = np.zeros(n_cols)
    for (t, row, col, is_target), score in zip(events, decision_scores):
        if row is not None:
            row_scores[row] += score
        elif col is not None:
            col_scores[col] += score
    return int(np.argmax(row_scores)), int(np.argmax(col_scores))


def evaluate_character_accuracy(X, y, events, target_row, target_col,
                                 flashes_per_rep=12, n_filters=3):
    """
    Splits the calibration session by repetition (not by shuffled epoch) into a
    train half and a held-out test half, so character decoding is evaluated on
    genuinely unseen flashes from the same session -- not just unseen epochs
    mixed in from repetitions the model already saw parts of.

    Returns a list of (n_reps_used, correct) using 1..N repetitions worth of the
    held-out half, cumulatively -- the standard "accuracy vs. repetitions" curve
    reported in P300 speller literature.
    """
    n_total_reps = len(events) // flashes_per_rep
    n_train_reps = n_total_reps // 2
    train_cutoff = n_train_reps * flashes_per_rep

    if n_train_reps < 2 or (n_total_reps - n_train_reps) < 1:
        raise ValueError(f"Need at least ~4 repetitions to split train/test; got {n_total_reps}")

    X_train, y_train = X[:train_cutoff], y[:train_cutoff]
    X_test, events_test = X[train_cutoff:], events[train_cutoff:]

    pipeline = build_xdawn_lda_pipeline(n_filters=n_filters)
    pipeline.fit(X_train, y_train)
    test_scores = pipeline.decision_function(X_test)

    n_test_reps = len(events_test) // flashes_per_rep
    results = []
    for k in range(1, n_test_reps + 1):
        cutoff = k * flashes_per_rep
        pred_row, pred_col = decode_character(events_test[:cutoff], test_scores[:cutoff])
        correct = (pred_row == target_row) and (pred_col == target_col)
        results.append((k, correct, pred_row, pred_col))

    return results
