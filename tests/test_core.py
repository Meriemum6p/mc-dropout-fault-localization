"""Verifications de base : gradients, MC Dropout, injection."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.model import MultiEncoderAE, SimpleAE
from src.nn import Dense


def loss_multi(m, xs):
    loc, fus = m.forward(xs)
    return sum((np.mean((loc[i] - xs[i]) ** 2) + np.mean((fus[i] - xs[i]) ** 2)) / m.M for i in range(m.M))


def test_gradient_multi():
    r = np.random.default_rng(0)
    m = MultiEncoderAE([6, 3], h=5, z=2, hf=5, zf=3, p=0.0, seed=1)
    xs = [r.normal(size=(4, 6)), r.normal(size=(4, 3))]
    # Biais non nuls : sinon une couche ReLU entierement eteinte place la suivante pile sur le point
    # anguleux de ReLU (x = 0), ou la derivee n'existe pas et la comparaison numerique n'a pas de sens.
    for s in m.seqs():
        for l in s.dense():
            l.b += r.normal(0, 0.05, l.b.shape)
    m.train_batch(xs)
    n, worst = 0, 0.0
    for s in m.seqs():
        for l in s.dense():
            for name, a, g in l.params():
                an = getattr(l, g) / 4
                for _ in range(4):
                    i = tuple(r.integers(0, d) for d in a.shape)
                    old = a[i]
                    a[i] = old + 1e-5; up = loss_multi(m, xs)
                    a[i] = old - 1e-5; lo = loss_multi(m, xs)
                    a[i] = old
                    num = (up - lo) / 2e-5
                    worst = max(worst, abs(num - an[i]) / (abs(num) + abs(an[i]) + 1e-8))
                    n += 1
    assert worst < 1e-4, worst
    print("gradient multi-encodeurs OK", n, "verifications, ecart max", worst)


def test_gradient_simple():
    r = np.random.default_rng(0)
    m = SimpleAE(5, h=4, z=2, p=0.0, seed=2)
    x = r.normal(size=(3, 5))
    for s in m.seqs():
        for l in s.dense():
            l.b += r.normal(0, 0.05, l.b.shape)
    m.train_batch(x)
    worst = 0.0
    for s in m.seqs():
        for l in s.dense():
            for name, a, g in l.params():
                an = getattr(l, g) / 3
                i = tuple(r.integers(0, d) for d in a.shape)
                old = a[i]
                a[i] = old + 1e-5; up = np.mean((m.forward(x) - x) ** 2)
                a[i] = old - 1e-5; lo = np.mean((m.forward(x) - x) ** 2)
                a[i] = old
                num = (up - lo) / 2e-5
                worst = max(worst, abs(num - an[i]) / (abs(num) + abs(an[i]) + 1e-8))
    assert worst < 1e-4, worst
    print("gradient simple OK, ecart max", worst)


def test_mc_modes():
    r = np.random.default_rng(0)
    m = MultiEncoderAE([6, 3], h=5, z=2, hf=5, zf=3, p=0.3, seed=1)
    xs = [r.normal(size=(4, 6)), r.normal(size=(4, 3))]
    m.set_dropout((), False)
    a = m.forward(xs); b = m.forward(xs)
    assert all(np.array_equal(a[1][i], b[1][i]) for i in range(2)), "dropout coupe : sorties identiques"
    m.set_dropout((0,), False)
    a = m.forward(xs); b = m.forward(xs)
    assert not np.array_equal(a[0][0], b[0][0]), "encodeur 0 actif : sa sortie locale varie"
    assert np.array_equal(a[0][1], b[0][1]), "encodeur 1 coupe : sa sortie locale ne varie pas"
    assert not np.array_equal(a[1][1], b[1][1]), "le hasard de l'encodeur 0 traverse la fusion"
    print("modes MC OK")


if __name__ == "__main__":
    test_gradient_multi(); test_gradient_simple(); test_mc_modes()



def test_metrics_and_inject():
    from src.metrics import auc, RobustZ, boot_ci
    from src.inject import inject
    from src.data import SLICES, MOD_NAMES
    r = np.random.default_rng(0)
    y = r.integers(0, 2, 20000)
    assert abs(auc(y, r.random(20000)) - 0.5) < 0.02, "score aleatoire : AUROC proche de 0,5"
    assert auc(np.array([0, 0, 1, 1]), np.array([1, 2, 3, 4])) == 1.0
    z = r.normal(size=(20000, 5))
    assert abs((z.argmax(1) == r.integers(0, 5, 20000)).mean() - 0.2) < 0.02, "hasard : Top-1 proche de 20 %"
    W = r.normal(size=(4, 60, 8))
    for f in ("offset", "drift", "noise", "stuck", "shuffle"):
        B = inject(W, ["electrique"], f, 3, r)
        a, b = SLICES["electrique"]
        keep = [c for c in range(8) if not a <= c < b]
        assert np.array_equal(B[:, :, keep], W[:, :, keep]), "seule la modalite visee change"
        assert not np.array_equal(B[:, :, a:b], W[:, :, a:b])
    B = inject(W, ["electrique"], "shuffle", 0, r)
    assert np.allclose(np.sort(B, axis=1), np.sort(W, axis=1)), "le melange garde les memes valeurs"
    print("mesures et injection OK")


def test_folds_no_leak():
    from src.data import load_experiments, folds
    import os
    root = os.environ.get("SKAB_DATA", "SKAB/data")
    if not os.path.isdir(root):
        return
    lab = [e for e in load_experiments(root) if e["family"] != "anomaly-free"]
    assert len(lab) == 34
    for s in range(5):
        a = folds(lab, 5, s)
        assert set(a) == {e["name"] for e in lab} and set(a.values()) == set(range(5)), "chaque experience a un seul pli"
    print("plis OK")


if __name__ == "__main__":
    test_metrics_and_inject(); test_folds_no_leak()


def test_gradient_temporel():
    r = np.random.default_rng(3)
    m = MultiEncoderAE([40, 20], h=5, z=3, hf=6, zf=3, p=0.0, seed=4, temporel=20)
    xs = [r.normal(size=(3, 40)), r.normal(size=(3, 20))]
    for s in m.seqs():
        for l in s.dense():
            l.b += r.normal(0, 0.05, l.b.shape)
    m.train_batch(xs)
    worst, n = 0.0, 0
    for s in m.seqs():
        for l in s.dense():
            for name, a, g in l.params():
                an = getattr(l, g) / 3
                for _ in range(5):
                    i = tuple(r.integers(0, d) for d in a.shape)
                    old = a[i]
                    a[i] = old + 1e-5; up = loss_multi(m, xs)
                    a[i] = old - 1e-5; lo = loss_multi(m, xs)
                    a[i] = old
                    num = (up - lo) / 2e-5
                    worst = max(worst, abs(num - an[i]) / (abs(num) + abs(an[i]) + 1e-8))
                    n += 1
    assert worst < 1e-4, worst
    print("gradient temporel OK", n, "verifications, ecart max", worst)


def test_ordre_compte():
    """Le reseau temporel doit reagir a l'ordre des valeurs."""
    r = np.random.default_rng(5)
    m = MultiEncoderAE([20], h=5, z=3, hf=6, zf=3, p=0.0, seed=6, temporel=20)
    x = r.normal(size=(1, 20))
    a = m.enc[0].forward(x).copy()
    b = m.enc[0].forward(x[:, r.permutation(20)])
    assert not np.allclose(a, b)
    print("ordre pris en compte OK")


if __name__ == "__main__":
    test_gradient_temporel(); test_ordre_compte()


def test_dropout_partout():
    r = np.random.default_rng(8)
    m = MultiEncoderAE([6, 3], h=5, z=2, hf=5, zf=3, p=0.3, seed=1, partout=True)
    xs = [r.normal(size=(4, 6)), r.normal(size=(4, 3))]
    m.set_dropout((), False)
    a = m.forward(xs); b = m.forward(xs)
    assert np.array_equal(a[1][0], b[1][0]), "tout coupe : sorties identiques"
    m.all_on()
    assert all(l.active for s in m.seqs() for l in s.drops()), "tout actif, decodeurs compris"
    m2 = MultiEncoderAE([6, 3], h=5, z=2, hf=5, zf=3, p=0.0, seed=1, partout=True)
    m2.train_batch(xs)
    assert np.isfinite(m2.enc[0].dense()[0].gW).all()
    print("dropout partout OK")


if __name__ == "__main__":
    test_dropout_partout()
