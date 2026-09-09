import math


def bits_per_trial(n_classes: int, accuracy: float) -> float:
    p = min(max(accuracy, 1e-6), 1 - 1e-6)
    if n_classes <= 1:
        return 0.0
    bits = math.log2(n_classes) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n_classes - 1))
    return max(bits, 0.0)


def itr_bits_per_min(n_classes: int, accuracy: float, seconds_per_trial: float) -> float:
    return bits_per_trial(n_classes, accuracy) * (60.0 / seconds_per_trial)
