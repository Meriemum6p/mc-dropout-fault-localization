"""Jeu Hydraulic Systems (Helwig, Pignanelli & Schuetze, 2015 ; UCI, DOI 10.24432/C5CW21 ; licence CC BY 4.0).

Chaque ligne d'un fichier = un cycle de 60 s. On ramene tout a une mesure par seconde.
Les grandeurs virtuelles CE, CP et SE sont ecartees : elles sont calculees, pas mesurees.

Pour creer le fichier compact a partir des fichiers d'origine :
    python -m src.hydraulique creer DOSSIER_D_ORIGINE
Pour verifier le fichier compact contre une copie officielle (UCI ou Kaggle) :
    python -m src.hydraulique verifier DOSSIER_D_ORIGINE
"""
import hashlib
import os
import sys
import numpy as np
import pandas as pd

MODALITES = {
    "vibration": ["VS1"],
    "electrique": ["EPS1"],     # puissance du moteur (W)
    "temperature": ["TS1", "TS2", "TS3", "TS4"],
    "pression": ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6"],
    "debit": ["FS1", "FS2"],
}
MOD_NOMS = list(MODALITES)
CAPTEURS = [c for m in MOD_NOMS for c in MODALITES[m]]
TRANCHES = {}
_i = 0
for _m in MOD_NOMS:
    TRANCHES[_m] = (_i, _i + len(MODALITES[_m]))
    _i += len(MODALITES[_m])
FICHIER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "donnees_hydraulique_1hz.npz")

# Etat sain de chaque composant, d'apres la documentation du jeu.
SAIN = dict(cooler=100, valve=100, pump=0, accumulator=130)
# Hypotheses physiques fixees AVANT de regarder les resultats. A faire valider par un expert.
ATTENDU = {
    "cooler": {"temperature"},                # refroidissement reduit : l'huile chauffe
    "valve": {"pression", "debit"},           # retard de commutation
    "pump": {"debit", "pression"},            # fuite interne
    "accumulator": {"pression"},              # pression de gonflage reduite
}


def lire_capteur(dossier, nom):
    a = pd.read_csv(os.path.join(dossier, nom + ".txt"), sep="\t", header=None, dtype=np.float64).values
    assert a.shape[0] == 2205 and a.shape[1] in (60, 600, 6000), (nom, a.shape)
    pas = a.shape[1] // 60
    return a.reshape(2205, 60, pas).mean(2)   # moyenne sur chaque seconde


def lire_origine(dossier):
    X = np.stack([lire_capteur(dossier, c) for c in CAPTEURS], 2).astype(np.float32)
    p = pd.read_csv(os.path.join(dossier, "profile.txt"), sep="\t", header=None).values.astype(np.int32)
    assert p.shape == (2205, 5)
    return X, p


def creer(dossier):
    X, p = lire_origine(dossier)
    np.savez_compressed(FICHIER, X=X, profil=p, capteurs=np.array(CAPTEURS))
    print("cree", FICHIER, X.shape, round(os.path.getsize(FICHIER) / 1e6, 1), "Mo", "empreinte", empreinte(X, p))


def empreinte(X, p):
    return hashlib.sha256(np.round(X.astype(np.float64), 2).tobytes() + p.tobytes()).hexdigest()[:16]


def verifier(dossier):
    X, p = lire_origine(dossier)
    Y = charger()
    ok = np.allclose(X, Y["X"], atol=1e-3) and np.array_equal(p, Y["profil"])
    print("IDENTIQUE a la copie fournie" if ok else "DIFFERENT : ne pas utiliser, prevenez-moi")


def charger():
    if not os.path.exists(FICHIER):
        sys.exit("Fichier manquant : donnees_hydraulique_1hz.npz")
    z = np.load(FICHIER)
    assert list(z["capteurs"]) == CAPTEURS
    return dict(X=z["X"].astype(np.float64), profil=z["profil"])


def etats(p):
    """Renvoie l'etat de chaque cycle : sain, stable, et composant unique en defaut."""
    c = dict(cooler=p[:, 0], valve=p[:, 1], pump=p[:, 2], accumulator=p[:, 3])
    defaut = {k: c[k] != SAIN[k] for k in SAIN}
    n = sum(defaut.values())
    stable = p[:, 4] == 0
    seul = np.array(["sain"] * len(p), dtype=object)
    seul[n > 1] = "multiple"
    for k in SAIN:
        seul[(n == 1) & defaut[k]] = k
    return dict(sain=(n == 0), stable=stable, famille=seul, defaut=(n > 0))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "creer":
        creer(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "verifier":
        verifier(sys.argv[2])
    else:
        print(__doc__)
