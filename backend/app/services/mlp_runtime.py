"""Portable inference of exported PyTorch Linear/ReLU weights."""
import numpy as np


def predict_bundle(bundle, differences, best_of=3):
    if best_of not in (3, 5):
        raise ValueError('best_of must be 3 or 5')
    raw = np.asarray(differences, dtype=np.float64).reshape(-1, 8)
    probabilities = []
    for member in bundle['members']:
        directional = []
        for sign in (1, -1):
            x = np.column_stack([(sign * raw - member['mean']) / member['scale'],
                                 np.full(len(raw), float(best_of == 5))]).astype(np.float32)
            for i, (weight, bias) in enumerate(member['layers']):
                x = x @ weight.T + bias
                if i < 2:
                    x = np.maximum(x, 0)
            z = x[:, 0].astype(np.float64)
            directional.append(np.exp(-np.logaddexp(0, -z)))
        probabilities.append((directional[0] + 1 - directional[1]) / 2)
    return np.mean(probabilities, axis=0)
