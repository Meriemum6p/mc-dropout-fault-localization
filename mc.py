"""MC Dropout : on repete la prediction T fois et on mesure l'accord entre les passages.

U (incertitude) = variance entre passages, moyennee sur les points de la fenetre.
E (erreur)      = ecart entre l'entree et la moyenne des passages.
"""
import numpy as np
from .data import MOD_NAMES, SLICES


def to_mods(W):
    """(N, 60, 8) -> liste de 5 tableaux aplatis, un par modalite."""
    return [W[:, :, SLICES[m][0]:SLICES[m][1]].reshape(len(W), -1) for m in MOD_NAMES]


class Running:
    """Moyenne et variance cumulees : aucun passage n'est stocke."""

    def __init__(self):
        self.n, self.s1, self.s2 = 0, 0.0, 0.0

    def add(self, x):
        self.n += 1
        self.s1 = self.s1 + x
        self.s2 = self.s2 + x * x

    def mean(self):
        return self.s1 / self.n

    def var(self):
        if self.n < 2:
            return np.zeros_like(self.s1)
        return np.maximum(self.s2 / self.n - self.mean() ** 2, 0) * self.n / (self.n - 1)


def _batched(N, batch):
    for i in range(0, N, batch):
        yield i, min(N, i + batch)


def mc_multi(model, xs, T, batch=2048):
    """Tout le dropout actif. Une seule passe donne un echantillon pour chaque branche et pour la fusion."""
    model.all_on()
    M, N = model.M, len(xs[0])
    out = {k: np.zeros((N, M)) for k in ("E_loc", "U_loc", "E_fus", "U_fus")}
    for a, b in _batched(N, batch):
        x = [v[a:b] for v in xs]
        rl, rf = [Running() for _ in range(M)], [Running() for _ in range(M)]
        for _ in range(T):
            loc, fus = model.forward(x)
            for m in range(M):
                rl[m].add(loc[m]); rf[m].add(fus[m])
        for m in range(M):
            out["E_loc"][a:b, m] = ((x[m] - rl[m].mean()) ** 2).mean(1)
            out["U_loc"][a:b, m] = rl[m].var().mean(1)
            out["E_fus"][a:b, m] = ((x[m] - rf[m].mean()) ** 2).mean(1)
            out["U_fus"][a:b, m] = rf[m].var().mean(1)
    return out


def mc_branch(model, xs, T, batch=2048):
    """Un seul encodeur aleatoire a la fois : quelle part du doute de la fusion vient de lui ?
    Cout : M x T passages. Proposition du projet, inspiree de la decomposition de variance."""
    M, N = model.M, len(xs[0])
    U = np.zeros((N, M))
    for m in range(M):
        model.set_dropout((m,), False)
        for a, b in _batched(N, batch):
            x = [v[a:b] for v in xs]
            r = [Running() for _ in range(M)]
            for _ in range(T):
                _, fus = model.forward(x)
                for k in range(M):
                    r[k].add(fus[k])
            U[a:b, m] = np.mean([r[k].var().mean(1) for k in range(M)], axis=0)
    return U


def mc_fusion_only(model, xs, T, batch=2048):
    """Seule la fusion est aleatoire : part du doute qui ne vient d'aucun encodeur."""
    model.set_dropout((), True)
    M, N = model.M, len(xs[0])
    U = np.zeros(N)
    for a, b in _batched(N, batch):
        x = [v[a:b] for v in xs]
        r = [Running() for _ in range(M)]
        for _ in range(T):
            _, fus = model.forward(x)
            for k in range(M):
                r[k].add(fus[k])
        U[a:b] = np.mean([r[k].var().mean(1) for k in range(M)], axis=0)
    return U


def mc_simple(model, x, T, batch=2048):
    """Autoencodeur simple. Renvoie la moyenne et la variance par dimension."""
    model.set_dropout(True)
    mu, var = np.zeros_like(x), np.zeros_like(x)
    for a, b in _batched(len(x), batch):
        r = Running()
        for _ in range(T):
            r.add(model.forward(x[a:b]))
        mu[a:b], var[a:b] = r.mean(), r.var()
    return mu, var
