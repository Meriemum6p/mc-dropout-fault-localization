import numpy as np


class RobustZ:
    """Rend comparables des scores de modalites differentes, a partir de fenetres normales.
    Sans cette etape, une modalite a grande variance gagnerait toujours."""

    def fit(self, S):
        self.med = np.median(S, axis=0)
        self.mad = 1.4826 * np.median(np.abs(S - self.med), axis=0)
        self.mad = np.where(self.mad < 1e-12, np.maximum(S.std(axis=0), 1e-12), self.mad)
        return self

    def __call__(self, S):
        return (S - self.med) / self.mad


def ranks_desc(z):
    return (-z).argsort(axis=1).argsort(axis=1)   # 0 = modalite la plus suspecte


def auc(y, s):
    """AUROC par les rangs, avec ex aequo."""
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auprc(y, s):
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, s)


def f1_far_mar(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, fp, fn, tn = (y & pred).sum(), (~y & pred).sum(), (y & ~pred).sum(), (~y & ~pred).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return f1, fp / max(fp + tn, 1), fn / max(fn + tp, 1)


def boot_ci(v, n=5000, seed=0):
    """Intervalle a 95 % en retirant des experiences, jamais des fenetres : elles sont correlees."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = np.random.default_rng(seed)
    b = r.choice(v, (n, len(v))).mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
