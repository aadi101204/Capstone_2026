import numpy as np

def sequence(r, x0, length, map_type="logistic"):
    """
    Generates a chaotic sequence based on the specified map type.
    """
    seq = np.zeros(length)
    seq[0] = x0
    for i in range(1, length):
        if map_type == "logistic":
            seq[i] = r * seq[i-1] * (1 - seq[i-1]) 
        elif map_type == "tent":
            seq[i] = r * seq[i-1] if seq[i-1] < 0.5 else r * (1 - seq[i-1])
        elif map_type == "cosine":
            seq[i] = r * (1 - np.cos(np.pi * seq[i-1])) / 2.0
        else:
            raise ValueError(f"Unknown map_type: {map_type}")
        seq[i] = np.clip(seq[i], 1e-6, 1-1e-6)
    return seq
