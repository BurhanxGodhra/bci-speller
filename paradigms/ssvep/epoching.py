import numpy as np


def load_ssvep_epochs(dataset_name="Nakanishi2015", subject_ids=[1]):
    from moabb.datasets import Nakanishi2015
    from moabb.paradigms import SSVEP

    dataset = {"Nakanishi2015": Nakanishi2015}[dataset_name]()
    paradigm = SSVEP()
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject_ids)

    freqs = sorted(set(float(l) for l in labels))
    y = np.array([freqs.index(float(l)) for l in labels])
    return X, y, freqs, meta
