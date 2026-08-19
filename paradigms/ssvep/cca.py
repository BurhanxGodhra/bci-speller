import numpy as np
from sklearn.cross_decomposition import CCA


def make_reference_signals(freqs, n_harmonics, n_samples, sfreq):
    t = np.arange(n_samples) / sfreq
    refs = {}
    for f in freqs:
        sigs = []
        for h in range(1, n_harmonics + 1):
            sigs.append(np.sin(2 * np.pi * f * h * t))
            sigs.append(np.cos(2 * np.pi * f * h * t))
        refs[f] = np.array(sigs).T
    return refs


class CCAClassifier:
    def __init__(self, freqs, sfreq, n_harmonics=3):
        self.freqs = freqs
        self.sfreq = sfreq
        self.n_harmonics = n_harmonics

    def predict(self, X):
        n_samples = X.shape[-1]
        refs = make_reference_signals(self.freqs, self.n_harmonics, n_samples, self.sfreq)

        preds = []
        for epoch in X:
            scores = []
            for f in self.freqs:
                cca = CCA(n_components=1)
                cca.fit(epoch.T, refs[f])
                x_c, y_c = cca.transform(epoch.T, refs[f])
                scores.append(np.corrcoef(x_c.T, y_c.T)[0, 1])
            preds.append(int(np.argmax(scores)))
        return np.array(preds)
