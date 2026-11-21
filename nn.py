"""Mini-reseau de neurones en NumPy : couches denses, ReLU, dropout, Adam."""
import numpy as np


class Dense:
    def __init__(self, a, b, rng):
        self.W = rng.normal(0, np.sqrt(2.0 / a), (a, b))   # initialisation de He
        self.b = np.zeros(b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.gW = self.x.T @ g
        self.gb = g.sum(0)
        return g @ self.W.T

    def params(self):
        return [("W", self.W, "gW"), ("b", self.b, "gb")]


class Conv1D:
    """Filtre qui glisse le long du temps : les memes poids lisent chaque petit morceau du signal.
    Entree et sortie : (lot, temps, canaux)."""

    def __init__(self, c_in, c_out, k, stride, rng):
        self.W = rng.normal(0, np.sqrt(2.0 / (k * c_in)), (k, c_in, c_out))
        self.b = np.zeros(c_out)
        self.k, self.s = k, stride

    def forward(self, x):
        win = np.lib.stride_tricks.sliding_window_view(x, self.k, axis=1)[:, ::self.s]   # (N, nW, C, k)
        self.win, self.n_t = win, x.shape[1]
        return np.tensordot(win, self.W, axes=([2, 3], [1, 0])) + self.b

    def backward(self, g):
        self.gW = np.tensordot(self.win, g, axes=([0, 1], [0, 1])).transpose(1, 0, 2)   # (k, C, O)
        self.gb = g.sum((0, 1))
        out = np.zeros((g.shape[0], self.n_t, self.W.shape[1]))
        debut = np.arange(g.shape[1]) * self.s
        for j in range(self.k):
            out[:, debut + j, :] += g @ self.W[j].T
        return out

    def params(self):
        return [("W", self.W, "gW"), ("b", self.b, "gb")]


class Unflatten:
    """Remet une fenetre aplatie sous la forme (lot, temps, canaux)."""

    def __init__(self, L, C):
        self.L, self.C = L, C

    def forward(self, x):
        return x.reshape(len(x), self.L, self.C)

    def backward(self, g):
        return g.reshape(len(g), -1)

    def params(self):
        return []


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, g):
        return g.reshape(self.shape)

    def params(self):
        return []


class ReLU:
    def forward(self, x):
        self.m = x > 0
        return x * self.m

    def backward(self, g):
        return g * self.m

    def params(self):
        return []


class Dropout:
    """Dropout de Bernoulli. `active` reste controlable a la prediction : c'est le MC Dropout."""

    def __init__(self, p, rng):
        self.p, self.rng, self.active = p, rng, False

    def forward(self, x):
        if not self.active or self.p == 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, g):
        return g if self.mask is None else g * self.mask

    def params(self):
        return []


class Seq:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, g):
        for l in reversed(self.layers):
            g = l.backward(g)
        return g

    def dense(self):
        """Toutes les couches qui ont des poids a apprendre."""
        return [l for l in self.layers if l.params()]

    def drops(self):
        return [l for l in self.layers if isinstance(l, Dropout)]


class Adam:
    def __init__(self, seqs, lr=1e-3, weight_decay=1e-5, b1=0.9, b2=0.999):
        self.items = [(l, name, arr, g) for s in seqs for l in s.dense() for name, arr, g in l.params()]
        self.lr, self.wd, self.b1, self.b2, self.t = lr, weight_decay, b1, b2, 0
        self.m = [np.zeros_like(a) for _, _, a, _ in self.items]
        self.v = [np.zeros_like(a) for _, _, a, _ in self.items]

    def step(self, n):
        """n = taille du lot. La penalite lambda*||W||^2 correspond a l'equation 1 de Gal & Ghahramani."""
        self.t += 1
        for i, (l, name, a, gname) in enumerate(self.items):
            g = getattr(l, gname) / n
            if name == "W":
                g = g + 2 * self.wd * a
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            a -= self.lr * mh / (np.sqrt(vh) + 1e-8)
