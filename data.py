"""Chargement et preparation de SKAB (Katser & Kozitsin, 2020).

Regle de base : une experience entiere va soit dans l'entrainement, soit dans
le test. Le debut de chaque experience (avant toute anomalie) sert de reference.
"""
import glob
import os
import numpy as np
import pandas as pd

# Les 8 capteurs de SKAB, regroupes en 5 modalites physiques.
MODALITIES = {
    "vibration": ["Accelerometer1RMS", "Accelerometer2RMS"],
    "electrique": ["Current", "Voltage"],
    "temperature": ["Temperature", "Thermocouple"],
    "pression": ["Pressure"],
    "debit": ["Volume Flow RateRMS"],
}
MOD_NAMES = list(MODALITIES)
SENSORS = [s for m in MOD_NAMES for s in MODALITIES[m]]
SLICES = {}
_i = 0
for _m in MOD_NAMES:
    SLICES[_m] = (_i, _i + len(MODALITIES[_m]))
    _i += len(MODALITIES[_m])

K_CALIB = 100   # secondes de reference au debut de chaque experience (premiere anomalie >= 104 s)
WINDOW = 60     # longueur d'une fenetre : 60 s
STRIDE = 5      # pas entre deux fenetres : 5 s

# Familles, d'apres data/README.md de SKAB.
OTHER_FAMILY = {
    1: "fuite", 2: "fuite", 3: "fuite", 4: "fuite",
    5: "balourd", 6: "balourd", 7: "balourd", 8: "balourd", 9: "balourd",
    10: "ajout_eau", 11: "ajout_eau",
    12: "cavitation", 13: "cavitation",
    14: "eau_chaude",
}

# Hypotheses physiques fixees AVANT de regarder les resultats.
# Elles ne viennent pas de SKAB : a faire valider par un expert du procede.
EXPECTED = {
    "valve1": {"debit", "pression"},
    "valve2": {"debit", "pression"},
    "fuite": {"debit", "pression"},
    "balourd": {"vibration"},
    "ajout_eau": {"debit", "pression"},
    "cavitation": {"debit", "pression", "vibration"},
    "eau_chaude": {"temperature"},
}


def load_experiments(root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
        rel = os.path.relpath(f, root).replace(os.sep, "/")   # meme nom sous Windows, Mac et Linux
        d = pd.read_csv(f, sep=";", parse_dates=["datetime"]).sort_values("datetime")
        group = rel.split("/")[0]
        num = int(os.path.splitext(os.path.basename(f))[0]) if group != "anomaly-free" else -1
        family = OTHER_FAMILY[num] if group == "other" else group
        y = d["anomaly"].values.astype(int) if "anomaly" in d else np.zeros(len(d), int)
        out.append(dict(name=rel.replace(".csv", ""), family=family,
                        X=d[SENSORS].values.astype(float), y=y))
    return out


def split_anomaly_free(exp, chunk=1000):
    """Coupe le long enregistrement normal en morceaux traites comme des experiences."""
    n = len(exp["X"])
    return [dict(name=f"{exp['name']}#{i}", family="normal",
                 X=exp["X"][s:s + chunk], y=exp["y"][s:s + chunk])
            for i, s in enumerate(range(0, n - chunk + 1, chunk))]


def folds(experiments, k=5, seed=0):
    """K plis stratifies par famille : une experience n'apparait que dans un seul test."""
    rng = np.random.default_rng(seed)
    by = {}
    for e in experiments:
        by.setdefault(e["family"], []).append(e["name"])
    assign, i = {}, 0
    for fam in sorted(by):
        names = list(by[fam])
        rng.shuffle(names)
        for n in names:
            assign[n] = i % k
            i += 1
    return assign


def calibrate(X):
    """Retire le niveau propre a l'experience, mesure sur ses K premieres secondes."""
    return X - np.median(X[:K_CALIB], axis=0)


def normalize(e, s):
    """Position et echelle propres a chaque experience, mesurees sur ses K premieres secondes.
    SKAB melange plusieurs regimes (debit de 32 ou de 125 L/min) : une echelle unique les comparerait mal.
    Le plancher 0,5*s evite une echelle minuscule pour les capteurs lents comme la temperature."""
    sk = np.maximum(e["X"][:K_CALIB].std(axis=0), 0.5 * s)
    return calibrate(e["X"]) / sk


def windows(Z, y=None, stride=STRIDE, W=WINDOW, start=0, stop=None):
    stop = len(Z) if stop is None else stop
    idx = np.arange(start, stop - W + 1, stride)
    if len(idx) == 0:
        return np.empty((0, W, Z.shape[1])), np.empty(0, int)
    w = np.stack([Z[i:i + W] for i in idx])
    lab = np.zeros(len(idx), int) if y is None else y[idx + W - 1]  # etat a la fin de la fenetre
    return w, lab


def normal_end(e):
    """Indice de la premiere anomalie, ou fin de l'experience s'il n'y en a pas."""
    return int(np.argmax(e["y"] == 1)) if e["y"].any() else len(e["y"])


def robust_scale(exps):
    """Echelle par capteur : MAD des points normaux calibres, en intra-experience."""
    v = np.concatenate([calibrate(e["X"])[K_CALIB:normal_end(e)] for e in exps if normal_end(e) > K_CALIB])
    s = 1.4826 * np.median(np.abs(v - np.median(v, axis=0)), axis=0)
    # La pression est tres discretisee (pas de 0,328 bar) : sa MAD vaut 0, on prend alors l'ecart-type.
    return np.where(s < 1e-9, v.std(axis=0), s)
