import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from hardware.drivers.artifact_filter import ArtifactRejector

if __name__ == "__main__":
    rejector = ArtifactRejector()
    rng = np.random.default_rng(0)

    clean = rng.normal(0, 5e-6, size=(5, 250))
    blink = clean.copy()
    blink[:, 100:110] += 200e-6
    emg = clean.copy()
    emg[2, :] += rng.normal(0, 50e-6, size=250)

    print("clean chunk:", rejector.is_clean(clean))
    print("blink chunk:", rejector.is_clean(blink))
    print("emg chunk:  ", rejector.is_clean(emg))
