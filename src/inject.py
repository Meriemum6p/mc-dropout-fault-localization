"""Pannes artificielles : la seule facon d'avoir une verite terrain par modalite sur SKAB."""
import numpy as np
from .data import MOD_NAMES, SLICES

FAULTS = ("offset", "drift", "noise", "stuck", "shuffle")
SEVERITIES = (1, 2, 3, 5)


def channels(mods, rng, scope):
    ch = []
    for m in mods:
        a, b = SLICES[m]
        ch += [int(rng.integers(a, b))] if scope == "one_sensor" else list(range(a, b))
    return ch


def inject(W, mods, fault, sev, rng, scope="whole"):
    """W : (N, 60, 8) en unites normalisees. La panne couvre toute la fenetre."""
    W = W.copy()
    N, T, _ = W.shape
    for i in range(N):
        for c in channels(mods, rng, scope):
            sign = rng.choice([-1.0, 1.0])
            if fault == "offset":
                W[i, :, c] += sign * sev
            elif fault == "drift":
                W[i, :, c] += sign * sev * np.linspace(0, 1, T)
            elif fault == "noise":
                W[i, :, c] += sev * rng.normal(size=T)
            elif fault == "stuck":
                W[i, :, c] = W[i, 0, c]
            elif fault == "shuffle":
                W[i, :, c] = W[i, rng.permutation(T), c]
    return W
