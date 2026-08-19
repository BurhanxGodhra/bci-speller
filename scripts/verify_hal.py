import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np

from hardware.hal import HAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    print("Discovering LSL stream (make sure hardware/playback.py is running in another terminal)...")
    hal = HAL.from_config("hardware/config/config.yaml")

    print(f"\nMapped channels: {hal.mapping.required_channels}")
    print(f"Fallback substitutions used: {hal.mapping.fallback_used or 'none'}")
    print(f"Native rate: {hal.native_sample_rate}Hz -> Target rate: {hal.target_sample_rate}Hz")

    print("\nPulling 5 chunks...")
    n_chunks = 0
    for chunk in hal.stream(timeout=2.0):
        print(f"chunk {n_chunks}: shape={chunk.shape}, "
              f"mean={np.mean(chunk):.2e}, std={np.std(chunk):.2e}")
        n_chunks += 1
        if n_chunks >= 5:
            break

    print("\nHAL pipeline verified")
