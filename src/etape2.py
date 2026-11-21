"""Etape 2 : les memes tests pour le MC Dropout, le clustering et les temoins simples.
Tout est appris sur des fenetres normales. Les etiquettes d'anomalie ne servent qu'a evaluer."""
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from .mc import Running, mc_multi
from .metrics import RobustZ
from .model import MultiEncoderAE, fit

T_TRAVAIL = 50
P_DROP = 0.1
FAULTS_AMPLITUDE = ("offset", "drift", "noise")
INTENSITES = (1, 2, 3, 5)


def en_modalites(W, tranches):
    return [W[:, :, a:b].reshape(len(W), -1) for a, b in tranches]


def injecter(W, ab, panne, intensite, rng, echelle):
    """Meme definition des pannes que src/inject.py. `echelle` traduit 1 unite de variation normale, par canal."""
    W = W.copy()
    N, T, _ = W.shape
    for i in range(N):
        for c in range(ab[0], ab[1]):
            signe = rng.choice([-1.0, 1.0])
            if panne == "offset":
                W[i, :, c] += signe * intensite * echelle[c]
            elif panne == "drift":
                W[i, :, c] += signe * intensite * echelle[c] * np.linspace(0, 1, T)
            elif panne == "noise":
                W[i, :, c] += intensite * echelle[c] * rng.normal(size=T)
            elif panne == "stuck":
                W[i, :, c] = W[i, 0, c]
            elif panne == "shuffle":
                W[i, :, c] = W[i, rng.permutation(T), c]
    return W


def conditions():
    c = [(f, s) for f in FAULTS_AMPLITUDE for s in INTENSITES]
    return c + [("stuck", 0), ("shuffle", 0)]


class Ensemble:
    """Tous les modeles compares, entraines sur les memes fenetres normales."""

    def __init__(self, tranches, L, seed, epochs=40, avec_cnn=True, T=T_TRAVAIL):
        self.tr, self.L, self.seed, self.epochs, self.avec_cnn, self.T = tranches, L, seed, epochs, avec_cnn, T
        self.dims = [L * (b - a) for a, b in tranches]
        self.M = len(tranches)

    def fit(self, W_train, W_val):
        t0 = time.time()
        xs = en_modalites(W_train, self.tr)
        n = len(W_train)
        self.mlp = MultiEncoderAE(self.dims, p=P_DROP, seed=self.seed)
        fit(self.mlp, lambda b: [x[b] for x in xs], n, epochs=self.epochs, seed=self.seed)
        if self.avec_cnn:
            self.cnn = MultiEncoderAE(self.dims, p=P_DROP, seed=self.seed + 1, temporel=self.L)
            fit(self.cnn, lambda b: [x[b] for x in xs], n, epochs=self.epochs, seed=self.seed + 1)
        self.t_reseaux = time.time() - t0
        self.gabarit = W_train.mean(0)
        self._fit_stats_temps(W_train)
        self._fit_clusters(W_train, xs)
        brut = self._scores_bruts(W_val)
        self.Z = {k: RobustZ().fit(v) for k, v in brut.items()}
        self.seuils = {}
        z = self.scores(W_val, brut)
        for k, v in z.items():
            self.seuils[k] = float(np.percentile(v.mean(1), 99))
        self.seuils["km_global_dist"] = float(np.percentile(self.km_global.transform(W_val.reshape(len(W_val), -1)).min(1), 99))
        self.t_total = time.time() - t0
        return self

    # ---------- temoins simples
    def _stats_temps(self, W):
        """Variation du signal et variation d'une seconde a l'autre, par modalite."""
        std = np.stack([W[:, :, a:b].std(1).mean(1) for a, b in self.tr], 1)
        dif = np.stack([np.abs(np.diff(W[:, :, a:b], axis=1)).mean((1, 2)) for a, b in self.tr], 1)
        return std, dif

    def _fit_stats_temps(self, W):
        std, dif = self._stats_temps(W)
        self.z_std, self.z_dif = RobustZ().fit(std), RobustZ().fit(dif)

    # ---------- clustering
    def _fit_clusters(self, W, xs):
        s = self.seed
        plat = W.reshape(len(W), -1)
        self.km_global = KMeans(16, n_init=5, random_state=s).fit(plat)
        self.km_mod = {k: [KMeans(k, n_init=5, random_state=s + m).fit(xs[m]) for m in range(self.M)] for k in (2, 4, 8, 16, 32)}
        r = np.random.default_rng(s + 77)
        self.km_ens = []
        for m in range(self.M):
            membres = []
            for b in range(10):
                i = r.integers(0, len(W), len(W))
                membres.append(KMeans(8, n_init=1, random_state=s + 100 * m + b).fit(xs[m][i]))
            self.km_ens.append(membres)
        self.gmm = [GaussianMixture(4, covariance_type="diag", reg_covar=1e-3, random_state=s + m, max_iter=100).fit(xs[m]) for m in range(self.M)]
        self.mlp.set_dropout((), False)
        self.kmz = [KMeans(8, n_init=5, random_state=s + m).fit(self.mlp.enc[m].forward(xs[m])) for m in range(self.M)]

    def _kmz_mc(self, xs):
        """K-means sur le resume de chaque encodeur. Le MC Dropout fait bouger ce resume : c'est le doute du clustering."""
        self.mlp.all_on()
        N = len(xs[0])
        dist, var, inst = np.zeros((N, self.M)), np.zeros((N, self.M)), np.zeros((N, self.M))
        for m in range(self.M):
            for a in range(0, N, 2048):
                x = xs[m][a:a + 2048]
                r = Running()
                votes = np.zeros((len(x), self.kmz[m].n_clusters))
                for _ in range(self.T):
                    d = self.kmz[m].transform(self.mlp.enc[m].forward(x))
                    r.add(d.min(1))
                    votes[np.arange(len(x)), d.argmin(1)] += 1
                dist[a:a + 2048, m], var[a:a + 2048, m] = r.mean(), r.var()
                inst[a:a + 2048, m] = 1 - votes.max(1) / self.T
        return dist, var, inst

    def _scores_bruts(self, W):
        xs = en_modalites(W, self.tr)
        S = {"RAW": np.stack([(x ** 2).mean(1) for x in xs], 1)}
        g = en_modalites(self.gabarit[None], self.tr)
        S["TPL"] = np.stack([((x - gm) ** 2).mean(1) for x, gm in zip(xs, g)], 1)
        std, dif = self._stats_temps(W)
        S["TEMP"] = (np.abs(self.z_std(std)) + np.abs(self.z_dif(dif))) / 2
        for nom, modele in (("mlp", self.mlp), ("cnn", getattr(self, "cnn", None))):
            if modele is None:
                continue
            r = mc_multi(modele, xs, self.T)
            for k in ("E_loc", "E_fus", "U_loc", "U_fus"):
                S[f"{nom}_{k}"] = r[k]
        plat = W.reshape(len(W), -1)
        c = self.km_global.cluster_centers_[self.km_global.transform(plat).argmin(1)].reshape(W.shape)
        S["km_global"] = np.stack([((W[:, :, a:b] - c[:, :, a:b]) ** 2).mean((1, 2)) for a, b in self.tr], 1)
        for k, lst in self.km_mod.items():
            nom = "km_mod" if k == 8 else f"km_mod_k{k}"
            S[nom] = np.stack([km.transform(x).min(1) ** 2 / x.shape[1] for km, x in zip(lst, xs)], 1)
        moy, var = [], []
        for membres, x in zip(self.km_ens, xs):
            d = np.stack([km.transform(x).min(1) ** 2 / x.shape[1] for km in membres])
            moy.append(d.mean(0)); var.append(d.var(0, ddof=1))
        S["km_ens"], S["km_ens_var"] = np.stack(moy, 1), np.stack(var, 1)
        S["gmm_mod"] = np.stack([-g.score_samples(x) / x.shape[1] for g, x in zip(self.gmm, xs)], 1)
        S["kmz_dist"], S["kmz_var"], S["kmz_instab"] = self._kmz_mc(xs)
        return S

    def scores(self, W, brut=None):
        brut = self._scores_bruts(W) if brut is None else brut
        return {k: self.Z[k](v) for k, v in brut.items()}

    def score_km_global_dist(self, W):
        return self.km_global.transform(W.reshape(len(W), -1)).min(1)
