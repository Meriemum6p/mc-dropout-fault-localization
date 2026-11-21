"""Modeles : autoencodeur multi-encodeurs (propose) et autoencodeur simple (temoins).

Le dropout est place dans les encodeurs et la fusion. Les decodeurs restent
deterministes : c'est un MC Dropout PARTIEL. Gal & Ghahramani (2016) placent le
dropout avant chaque couche de poids, donc leur justification theorique ne couvre
pas strictement ce choix.
"""
import numpy as np
from .nn import Dense, ReLU, Dropout, Seq, Adam, Conv1D, Flatten, Unflatten


class SimpleAE:
    """Un seul encodeur. Sert pour un capteur seul ou pour la fusion precoce."""

    def __init__(self, d, h=64, z=16, p=0.1, seed=0):
        r = np.random.default_rng(seed)
        self.enc = Seq(Dense(d, h, r), ReLU(), Dropout(p, r), Dense(h, z, r))
        self.dec = Seq(Dense(z, h, r), ReLU(), Dense(h, d, r))
        self.d, self.rng = d, r

    def seqs(self):
        return [self.enc, self.dec]

    def set_dropout(self, on):
        for l in self.enc.drops():
            l.active = on

    def forward(self, x):
        return self.dec.forward(self.enc.forward(x))

    def train_batch(self, x):
        o = self.forward(x)
        g = 2 * (o - x) / self.d
        self.enc.backward(self.dec.backward(g))
        return np.mean((o - x) ** 2)


class MultiEncoderAE:
    """Un encodeur par modalite, un decodeur local par modalite, puis une fusion."""

    def __init__(self, dims, h=64, z=16, hf=64, zf=32, p=0.1, seed=0, temporel=None, partout=False):
        """temporel = longueur de la fenetre : les encodeurs deviennent des reseaux convolutifs 1D.
        partout = dropout avant chaque couche de poids, entree et decodeurs compris (Gal & Ghahramani, 2016)."""
        r = np.random.default_rng(seed)
        self.dims, self.M = dims, len(dims)
        if temporel:
            self.enc = [self._encodeur_temporel(d, temporel, z, p, r) for d in dims]
        elif partout:
            self.enc = [Seq(Dropout(p, r), Dense(d, h, r), ReLU(), Dropout(p, r), Dense(h, z, r)) for d in dims]
        else:
            self.enc = [Seq(Dense(d, h, r), ReLU(), Dropout(p, r), Dense(h, z, r)) for d in dims]
        if partout:
            self.loc = [Seq(Dropout(p, r), Dense(z, h, r), ReLU(), Dropout(p, r), Dense(h, d, r)) for d in dims]
        else:
            self.loc = [Seq(Dense(z, h, r), ReLU(), Dense(h, d, r)) for d in dims]
        self.fus = Seq(Dropout(p, r), Dense(z * self.M, hf, r), ReLU(), Dropout(p, r), Dense(hf, zf, r), ReLU())
        if partout:
            self.fdec = [Seq(Dropout(p, r), Dense(zf, h, r), ReLU(), Dropout(p, r), Dense(h, d, r)) for d in dims]
        else:
            self.fdec = [Seq(Dense(zf, h, r), ReLU(), Dense(h, d, r)) for d in dims]
        self.z = z

    @staticmethod
    def _encodeur_temporel(d, L, z, p, r, f1=8, f2=16, k=5, s=2):
        c = d // L
        n1 = (L - k) // s + 1
        n2 = (n1 - k) // s + 1
        return Seq(Unflatten(L, c), Conv1D(c, f1, k, s, r), ReLU(), Dropout(p, r),
                   Conv1D(f1, f2, k, s, r), ReLU(), Dropout(p, r), Flatten(), Dense(n2 * f2, z, r))

    def seqs(self):
        return self.enc + self.loc + [self.fus] + self.fdec

    def set_dropout(self, encoders=(), fusion=False):
        """Choisit ou le hasard est actif : quels encodeurs, et la fusion ou non."""
        for m in range(self.M):
            for l in self.enc[m].drops():
                l.active = m in encoders
        for l in self.fus.drops():
            l.active = fusion
        for s in self.loc + self.fdec:      # n'agit que si partout=True
            for l in s.drops():
                l.active = bool(fusion and len(list(encoders)) == self.M)

    def all_on(self):
        self.set_dropout(list(range(self.M)), True)

    def forward(self, xs):
        zs = [self.enc[m].forward(xs[m]) for m in range(self.M)]
        loc = [self.loc[m].forward(zs[m]) for m in range(self.M)]
        f = self.fus.forward(np.concatenate(zs, 1))
        fus = [self.fdec[m].forward(f) for m in range(self.M)]
        return loc, fus

    def train_batch(self, xs):
        loc, fus = self.forward(xs)
        loss, gz = 0.0, [0.0] * self.M
        gf = 0.0
        for m in range(self.M):
            d = self.dims[m]
            gz[m] = self.loc[m].backward(2 * (loc[m] - xs[m]) / (d * self.M))
            gf = gf + self.fdec[m].backward(2 * (fus[m] - xs[m]) / (d * self.M))
            loss += (np.mean((loc[m] - xs[m]) ** 2) + np.mean((fus[m] - xs[m]) ** 2)) / self.M
        g = self.fus.backward(gf)
        for m in range(self.M):
            s = self.z * m
            self.enc[m].backward(gz[m] + g[:, s:s + self.z])
        return loss


def fit(model, get_batch_input, n, epochs=40, batch=128, lr=1e-3, wd=1e-5, seed=0):
    """Entrainement sur des fenetres normales uniquement."""
    r = np.random.default_rng(seed + 1000)
    opt = Adam(model.seqs(), lr=lr, weight_decay=wd)
    if isinstance(model, MultiEncoderAE):
        model.all_on()
    else:
        model.set_dropout(True)
    hist = []
    for ep in range(epochs):
        order, tot, c = r.permutation(n), 0.0, 0
        for i in range(0, n, batch):
            b = order[i:i + batch]
            tot += model.train_batch(get_batch_input(b)) * len(b)
            c += len(b)
            opt.step(len(b))
        hist.append(tot / c)
    return hist
