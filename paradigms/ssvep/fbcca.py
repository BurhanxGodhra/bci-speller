import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA

from paradigms.ssvep.cca import make_reference_signals


def filter_bank(X, sfreq, n_bands=5, band_width=6, high=80):
    nyq = sfreq / 2
    bands = []
    for i in range(n_bands):
        low = band_width * (i + 1)
        b, a = butter(4, [low / nyq, min(high, nyq - 1) / nyq], btype="band")
        bands.append(filtfilt(b, a, X, axis=-1))
    return bands


class FBCCAClassifier:
    def __init__(self, freqs, sfreq, n_harmonics=3, n_bands=5, weight_a=1.25, weight_b=0.25):
        self.freqs = freqs
        self.sfreq = sfreq
        self.n_harmonics = n_harmonics
        self.n_bands = n_bands
        self.band_weights = np.array([(b + 1) ** -weight_a + weight_b for b in range(n_bands)])

    def predict(self, X):
        n_samples = X.shape[-1]
        refs = make_reference_signals(self.freqs, self.n_harmonics, n_samples, self.sfreq)

        preds = []
        for epoch in X:
            bands = filter_bank(epoch, self.sfreq, self.n_bands)
            scores = np.zeros(len(self.freqs))
            for b_idx, band_signal in enumerate(bands):
                for f_idx, f in enumerate(self.freqs):
                    cca = CCA(n_components=1)
                    cca.fit(band_signal.T, refs[f])
                    x_c, y_c = cca.transform(band_signal.T, refs[f])
                    r = np.corrcoef(x_c.T, y_c.T)[0, 1]
                    scores[f_idx] += self.band_weights[b_idx] * (r ** 2) * np.sign(r)
            preds.append(int(np.argmax(scores)))
        return np.array(preds)
