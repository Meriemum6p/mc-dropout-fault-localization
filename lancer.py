"""Lance le projet en une seule commande.

    python lancer.py test       verifie que tout marche (quelques minutes)
    python lancer.py complet    refait toute l'experience (environ 1 heure, reprise possible)
    python lancer.py comparer   compare vos chiffres aux chiffres de reference

Etape 2 (clustering, modele temporel, jeu hydraulique) :
    python lancer.py test2      verifie que l'etape 2 marche (environ 5 minutes)
    python lancer.py etape2     refait toute l'etape 2 (environ 1 heure, reprise possible)

Robustesse et article :
    python lancer.py robustesse refait le test des reglages du dropout (environ 1 heure, reprise possible)
    python lancer.py article    refait les figures et les tableaux de l'article avec VOS resultats

Donnees et modeles :
    python lancer.py donnees    telecharge les deux jeux depuis leurs sources officielles et les verifie
    python lancer.py modele     entraine et sauvegarde un modele (SKAB, reseau dense, pli 0, graine 0)
"""
import csv
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request
import zipfile

ICI = os.path.dirname(os.path.abspath(__file__))
COMMIT = "b2c0d46c2971dcbfe71e26087b6d231998bb91c2"   # version exacte de SKAB utilisee
URL = f"https://github.com/waico/SKAB/archive/{COMMIT}.zip"
TEST, MES, REF = "resultats_test", "mes_resultats", "resultats_reference"
TOLERANCE = 0.03  # 3 points d'ecart acceptes : l'ordre des calculs varie un peu selon le PC

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def stop(msg):
    print("\n" + "=" * 70 + "\nPROBLEME\n" + "=" * 70 + "\n" + msg + "\n")
    sys.exit(1)


def verifier_python():
    if sys.version_info < (3, 9):
        stop(f"Votre Python est trop ancien ({sys.version.split()[0]}). Installez Python 3.9 ou plus recent.")
    manquants = []
    for nom, pip in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib")):
        try:
            __import__(nom)
        except ImportError:
            manquants.append(pip)
    if manquants:
        stop("Il manque : " + ", ".join(manquants) + "\nLancez cette commande, puis recommencez :\n\n    python -m pip install -r requirements.txt")
    print(f"[ok] Python {sys.version.split()[0]} et les bibliotheques sont prets")


def chercher_donnees():
    voulu = {"anomaly-free", "valve1", "valve2", "other"}
    for racine, dossiers, _ in os.walk(ICI):
        if racine[len(ICI):].count(os.sep) > 4:
            dossiers[:] = []
            continue
        if voulu <= set(dossiers):
            return racine
    return None


def telecharger(zip_path):
    contextes = [None]
    try:
        import certifi
        contextes.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    premiere = None
    for ctx in contextes:
        try:
            with urllib.request.urlopen(URL, context=ctx, timeout=120) as r, open(zip_path, "wb") as f:
                while True:
                    bloc = r.read(1 << 20)
                    if not bloc:
                        return
                    f.write(bloc)
        except Exception as e:
            premiere = premiere or e
    raise premiere


def obtenir_donnees():
    d = chercher_donnees()
    if d is None:
        print("Telechargement des donnees SKAB depuis GitHub (environ 30 Mo)...")
        zip_path = os.path.join(ICI, "skab.zip")
        try:
            telecharger(zip_path)
            with zipfile.ZipFile(zip_path) as z:
                # On n'extrait que les donnees : certains chemins du depot sont trop longs pour Windows.
                z.extractall(os.path.join(ICI, "donnees"), [n for n in z.namelist() if "/data/" in n])
            os.remove(zip_path)
        except Exception as e:
            stop(f"Le telechargement automatique a echoue ({e}).\n\nFaites-le a la main :\n"
                 f"1. Ouvrez ce lien dans votre navigateur :\n   {URL}\n"
                 f"2. Decompressez le fichier dans ce dossier :\n   {ICI}\n3. Relancez la meme commande.")
        d = chercher_donnees()
    if d is None:
        stop("Je ne trouve pas les donnees SKAB (dossiers valve1, valve2, other, anomaly-free).")
    n = sum(f.endswith(".csv") for _, _, fs in os.walk(d) for f in fs)
    if n != 35:
        stop(f"Le dossier {d} contient {n} fichiers CSV au lieu de 35. Supprimez-le et relancez.")
    print(f"[ok] Donnees SKAB trouvees : 35 fichiers dans {os.path.relpath(d, ICI)}")
    return d


def environnement(data):
    env = os.environ.copy()
    # Un seul fil de calcul : les chiffres sont plus stables d'un lancement a l'autre.
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["SKAB_DATA"] = data
    return env


def etape(titre, args, journal, env, sortie=None):
    """Lance un script, garde tout dans un journal, et n'affiche que l'avancement."""
    print(f"\n>> {titre}", flush=True)
    os.makedirs(os.path.dirname(journal), exist_ok=True)
    t0 = time.time()
    with open(journal, "w", encoding="utf-8") as log:
        p = subprocess.Popen([sys.executable, "-u"] + args, cwd=ICI, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        lignes = []
        for ligne in p.stdout:
            log.write(ligne)
            lignes = (lignes + [ligne])[-15:]
            if ligne.startswith(("Pli ", "pli ", "repetition ")):
                print(f"   {ligne.strip()}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        p.wait()
    if p.returncode != 0:
        stop("Une etape s'est arretee sur une erreur. Copiez-moi ces lignes :\n\n" + "".join(lignes) + f"\nJournal complet : {journal}")
    if sortie:
        with open(journal, encoding="utf-8") as src, open(sortie, "w", encoding="utf-8") as dst:
            dst.write(src.read())
    print(f"   termine en {(time.time() - t0) / 60:.1f} min", flush=True)


def lire(dossier, nom):
    p = os.path.join(ICI, dossier, "tableaux", nom)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def valeur(dossier, nom, cle, val, colonne="moy"):
    for r in lire(dossier, nom):
        if r.get(cle) == val and r.get(colonne) not in (None, ""):
            return float(r[colonne])
    return None


# (nom simple, fichier, colonne de la ligne, ligne)
CHIFFRES = [
    ("EXPLIQUER : la bonne modalite est classee premiere (pannes d'intensite 3 et 5)", None, None, None),
    ("Doute avant la fusion (U local)", "attribution_sev3plus.csv", "method", "U_loc"),
    ("Doute apres la fusion (U fusion)", "attribution_sev3plus.csv", "method", "U_fus"),
    ("Doute d'un seul encodeur (U branche)", "attribution_sev3plus.csv", "method", "U_branch"),
    ("Erreur apres la fusion (E fusion)", "attribution_sev3plus.csv", "method", "E_fus"),
    ("Ecart simple, sans reseau (ecart brut)", "attribution_sev3plus.csv", "method", "RAW"),
    ("DETECTER : AUROC (0,5 = hasard, 1 = parfait)", None, None, None),
    ("Doute avant la fusion (U local)", "detection.csv", "method", "U_loc"),
    ("Doute apres la fusion (U fusion)", "detection.csv", "method", "U_fus"),
    ("Erreur apres la fusion (E fusion)", "detection.csv", "method", "E_fus"),
    ("Ecart simple, sans reseau (ecart brut)", "detection.csv", "method", "RAW"),
    ("Clustering (K-means)", "detection.csv", "method", "KMeans"),
    ("TESTS COMPLEMENTAIRES", None, None, None),
    ("Doute sans l'effet de l'amplitude", "redondance.csv", "method", "U_sans_amplitude"),
    ("Accord entre passages (AUROC)", "confiance.csv", "mesure", "auc_votes"),
    ("Ecart entre les deux premieres modalites (AUROC)", "confiance.csv", "mesure", "auc_ecart"),
]


def afficher(dossier, avec_reference):
    print("\n" + "=" * 78)
    print(f"{'Mesure':<52}{'Vous':>8}" + (f"{'Reference':>11}{'OK ?':>7}" if avec_reference else ""))
    print("=" * 78)
    ok, total, titre = 0, 0, None
    for nom, fichier, cle, ligne in CHIFFRES:
        if fichier is None:
            titre = nom
            continue
        v = valeur(dossier, fichier, cle, ligne)
        if v is None:
            continue
        if titre:
            print(f"\n{titre}")
            titre = None
        txt = f"  {nom:<50}{v:>8.3f}".replace(".", ",")
        if avec_reference:
            r = valeur(REF, fichier, cle, ligne)
            bon = abs(v - r) <= TOLERANCE
            ok, total = ok + bon, total + 1
            txt += f"{r:>11.3f}".replace(".", ",") + f"{'oui' if bon else 'NON':>7}"
        print(txt)
    if avec_reference:
        tv, tr = t_minimum(dossier), t_minimum(REF)
        if tv is not None:
            print(f"\nCHOIX DE T\n  {'T minimum pour 95 % de classements identiques':<50}{tv:>8}{tr:>11}{'oui' if tv == tr else 'proche' if abs(tv - tr) <= 20 else 'NON':>7}")
    print("=" * 78)
    return ok, total


def t_minimum(dossier):
    for r in lire(dossier, "T_minimum.csv"):
        if r["score"] == "U_loc" and r["M"] == "5" and float(r["niveau"]) == 0.95 and r["T_min"]:
            return int(float(r["T_min"]))
    return None


def mode_test():
    print("TEST RAPIDE\n")
    verifier_python()
    data = obtenir_donnees()
    env = environnement(data)
    d = os.path.join(ICI, TEST, "journaux")
    etape("Verification du code", ["tests/test_core.py"], os.path.join(d, "verifications.txt"), env)
    etape("Petite experience (7 experiences de SKAB sur 34)", ["run_pilot.py", "--data", data, "--out", TEST, "--seed", "0", "--only_fold", "0"], os.path.join(d, "experience.txt"), env)
    etape("Calcul des chiffres", ["analyze.py", TEST], os.path.join(d, "analyse.txt"), env, os.path.join(ICI, TEST, "analyse.txt"))
    etape("Creation des figures", ["make_figures.py", TEST], os.path.join(d, "figures.txt"), env)
    afficher(TEST, False)
    print("\nTEST REUSSI : le projet fonctionne sur votre PC.")
    print("Ces chiffres ne portent que sur 7 experiences : ne les utilisez pas dans l'article.")
    print(f"Vos figures sont dans le dossier {TEST}/figures.")
    print("\nEtape suivante :  python lancer.py complet")


def mode_complet():
    print("EXPERIENCE COMPLETE. Si elle s'arrete, relancez la meme commande : elle reprend ou elle en etait.\n")
    verifier_python()
    data = obtenir_donnees()
    env = environnement(data)
    d = os.path.join(ICI, MES, "journaux")
    plan = [(f"Experience principale, graine {s + 1} sur 5", ["run_pilot.py", "--seed", str(s)], f"config_s{s}.json") for s in range(5)]
    plan += [(f"Choix de T, modele {s + 1} sur 3", ["run_T.py", "--seed", str(s)], f"cout_s{s}.json") for s in range(3)]
    plan += [(f"Test de l'amplitude, graine {s + 1} sur 2", ["run_redondance.py", "--seed", str(s)], f"amplitude_s{s}.csv") for s in range(2)]
    plan += [(f"Test de la fiabilite, graine {s + 1} sur 2", ["run_confiance.py", "--seed", str(s)], f"confiance_courbe_s{s}.csv") for s in range(2)]
    for i, (titre, args, fin) in enumerate(plan, 1):
        if os.path.exists(os.path.join(ICI, MES, fin)):
            print(f"[deja fait] {titre}")
            continue
        etape(f"Etape {i} sur {len(plan)} : {titre}", args + ["--data", data, "--out", MES], os.path.join(d, fin + ".txt"), env)
    etape("Calcul des chiffres", ["analyze.py", MES], os.path.join(d, "analyse.txt"), env, os.path.join(ICI, MES, "analyse.txt"))
    etape("Creation des figures", ["make_figures.py", MES], os.path.join(d, "figures.txt"), env)
    mode_comparer()


CHIFFRES2 = [
    ("SKAB : la bonne famille est classee premiere", None, None),
    ("K-means par famille", "skab_top1_principal.csv", "km_mod"),
    ("Ecart simple", "skab_top1_principal.csv", "RAW"),
    ("Doute avant la fusion", "skab_top1_principal.csv", "mlp_U_loc"),
    ("Erreur apres la fusion", "skab_top1_principal.csv", "mlp_E_fus"),
    ("Doute avant la fusion, reseau temporel", "skab_top1_principal.csv", "cnn_U_loc"),
    ("Erreur apres la fusion, reseau temporel", "skab_top1_principal.csv", "cnn_E_fus"),
    ("Doute apres la fusion", "skab_top1_principal.csv", "mlp_U_fus"),
    ("HYDRAULIQUE : la bonne famille est classee premiere", None, None),
    ("K-means par famille", "hydraulique_top1_principal.csv", "km_mod"),
    ("Doute avant la fusion", "hydraulique_top1_principal.csv", "mlp_U_loc"),
    ("Doute avant la fusion, reseau temporel", "hydraulique_top1_principal.csv", "cnn_U_loc"),
    ("Ecart simple", "hydraulique_top1_principal.csv", "RAW"),
    ("Erreur apres la fusion, reseau temporel", "hydraulique_top1_principal.csv", "cnn_E_fus"),
    ("Erreur apres la fusion", "hydraulique_top1_principal.csv", "mlp_E_fus"),
    ("Doute apres la fusion", "hydraulique_top1_principal.csv", "mlp_U_fus"),
    ("SKAB : detection (AUROC)", None, None),
    ("K-means global", "skab_detection.csv", "km_global_dist"),
    ("Doute avant la fusion", "skab_detection.csv", "mlp_U_loc"),
    ("Ecart simple", "skab_detection.csv", "RAW"),
]


def afficher2(dossier, avec_reference):
    print("\n" + "=" * 78)
    print(f"{'Mesure':<52}{'Vous':>8}" + (f"{'Reference':>11}{'OK ?':>7}" if avec_reference else ""))
    print("=" * 78)
    ok, total, titre = 0, 0, None
    for nom, fichier, ligne in CHIFFRES2:
        if fichier is None:
            titre = nom
            continue
        v = valeur(dossier, fichier, "methode", ligne)
        if v is None:
            continue
        if titre:
            print(f"\n{titre}")
            titre = None
        txt = f"  {nom:<50}{v:>8.3f}".replace(".", ",")
        if avec_reference:
            r = valeur(os.path.join(REF, "etape2"), fichier, "methode", ligne)
            bon = abs(v - r) <= TOLERANCE
            ok, total = ok + bon, total + 1
            txt += f"{r:>11.3f}".replace(".", ",") + f"{'oui' if bon else 'NON':>7}"
        print(txt)
    print("=" * 78)
    return ok, total


def mode_test2():
    print("TEST RAPIDE DE L'ETAPE 2\n")
    verifier_python()
    data = obtenir_donnees()
    env = environnement(data)
    out = os.path.join(TEST, "etape2")
    d = os.path.join(ICI, out, "journaux")
    etape("Verification du code", ["tests/test_core.py"], os.path.join(d, "verifications.txt"), env)
    for t, titre in (("skab", "SKAB"), ("hyd_capteurs", "Hydraulique, pannes de capteurs"), ("hyd_composants", "Hydraulique, pannes de composants")):
        etape(f"Petit essai : {titre}", ["run_etape2.py", "--test", t, "--seed", "0", "--un_pli", "--data", data, "--out", out], os.path.join(d, t + ".txt"), env)
    etape("Calcul des chiffres", ["analyze_etape2.py", out], os.path.join(d, "analyse.txt"), env, os.path.join(ICI, out, "analyse.txt"))
    etape("Creation des figures", ["make_figures_etape2.py", out], os.path.join(d, "figures.txt"), env)
    afficher2(out, False)
    print("\nTEST REUSSI : l'etape 2 fonctionne sur votre PC.")
    print("Ces chiffres portent sur une petite partie des donnees : ne les utilisez pas dans l'article.")
    print("\nEtape suivante :  python lancer.py etape2")


def mode_etape2():
    print("ETAPE 2 COMPLETE. Si elle s'arrete, relancez la meme commande : elle reprend ou elle en etait.\n")
    verifier_python()
    data = obtenir_donnees()
    env = environnement(data)
    out = os.path.join(MES, "etape2")
    d = os.path.join(ICI, out, "journaux")
    plan = [(t, s) for t in ("skab", "hyd_capteurs", "hyd_composants") for s in range(3)]
    noms = dict(skab="SKAB", hyd_capteurs="Hydraulique, pannes de capteurs", hyd_composants="Hydraulique, pannes de composants")
    for i, (t, s) in enumerate(plan, 1):
        titre = f"{noms[t]}, graine {s + 1} sur 3"
        if os.path.exists(os.path.join(ICI, out, f"fin_{t}_s{s}.json")):
            print(f"[deja fait] {titre}")
            continue
        etape(f"Etape {i} sur {len(plan)} : {titre}", ["run_etape2.py", "--test", t, "--seed", str(s), "--data", data, "--out", out], os.path.join(d, f"{t}_s{s}.txt"), env)
    etape("Calcul des chiffres", ["analyze_etape2.py", out], os.path.join(d, "analyse.txt"), env, os.path.join(ICI, out, "analyse.txt"))
    etape("Creation des figures", ["make_figures_etape2.py", out], os.path.join(d, "figures.txt"), env)
    ok, total = afficher2(out, True)
    print(f"\n{ok} chiffres sur {total} sont a moins de 0,03 de la reference.")
    print("Resultats retrouves." if ok == total else "Certains chiffres s'ecartent. Envoyez-moi ce tableau avant de les utiliser.")
    print(f"Tous vos tableaux sont dans {out}/tableaux et vos figures dans {out}/figures.")


def mode_robustesse():
    print("TEST DE ROBUSTESSE. Si il s'arrete, relancez la meme commande : il reprend ou il en etait.\n")
    verifier_python()
    data = obtenir_donnees()
    env = environnement(data)
    out = os.path.join(MES, "robustesse")
    d = os.path.join(ICI, out, "journaux")
    plan = [(j, s) for j in ("skab", "hydraulique") for s in range(2)]
    for i, (j, s) in enumerate(plan, 1):
        titre = f"{'SKAB' if j == 'skab' else 'Hydraulique'}, graine {s + 1} sur 2"
        if os.path.exists(os.path.join(ICI, out, f"fin_{j}_s{s}.json")):
            print(f"[deja fait] {titre}")
            continue
        etape(f"Etape {i} sur {len(plan)} : {titre}", ["run_robustesse.py", "--jeu", j, "--seed", str(s), "--data", data, "--out", out], os.path.join(d, f"{j}_s{s}.txt"), env)
    etape("Calcul des chiffres", ["analyze_robustesse.py", out], os.path.join(d, "analyse.txt"), env, os.path.join(ICI, out, "analyse.txt"))
    ok = total = 0
    print(f"\n{'Jeu':<13}{'Reglage':<16}{'Methode':<9}{'Vous':>8}{'Reference':>11}")
    for j in ("skab", "hydraulique"):
        mes = lire(os.path.join(MES, "robustesse"), f"{j}_robustesse.csv")
        ref = lire(os.path.join(REF, "robustesse"), f"{j}_robustesse.csv")
        for cfg, m in (("partiel_p0.1", "U_loc"), ("partout_p0.1", "U_loc"), ("partiel_p0.1", "U_fus"), ("W60", "km_mod")):
            a = [float(r["moy"]) for r in mes if r["config"] == cfg and r["methode"] == m]
            b = [float(r["moy"]) for r in ref if r["config"] == cfg and r["methode"] == m]
            if a and b:
                total += 1
                ok += abs(a[0] - b[0]) <= TOLERANCE
                print(f"{j:<13}{cfg:<16}{m:<9}{a[0]:>8.3f}{b[0]:>11.3f}")
    print(f"\n{ok} chiffres sur {total} sont a moins de 0,03 de la reference.")
    print(f"Le detail est dans {out}/analyse.txt")
    print("\nEtape suivante :  python lancer.py article")


def mode_article():
    """Refait les figures et les tableaux de l'article a partir de VOS resultats, dans mon_article/."""
    print("FIGURES ET TABLEAUX DE L'ARTICLE AVEC VOS RESULTATS\n")
    manque = [n for n, p in (("complet", os.path.join(MES, "tableaux", "detection.csv")),
                             ("etape2", os.path.join(MES, "etape2", "tableaux", "skab_top1_principal.csv")),
                             ("robustesse", os.path.join(MES, "robustesse", "tableaux", "skab_robustesse.csv")))
             if not os.path.exists(os.path.join(ICI, p))]
    if manque:
        stop("Il manque des resultats. Lancez d'abord :  " + "  puis  ".join(f"python lancer.py {m}" for m in manque))
    env = environnement("")
    cible = os.path.join(ICI, "mon_article")
    if not os.path.exists(cible):
        shutil.copytree(os.path.join(ICI, "article"), cible)
    d = os.path.join(ICI, MES, "journaux_article")
    etape("Figures de l'article (PDF et SVG)", ["make_figures_revision.py", "--resultats", MES, "--sortie", os.path.join("mon_article", "figures")], os.path.join(d, "figures_en.txt"), env)
    etape("Figures en francais (PNG)", ["make_figures_article.py", "--resultats", MES, "--sortie", os.path.join(MES, "figures_presentation"), "--langue", "fr", "--format", "png"], os.path.join(d, "figures_fr.txt"), env)
    etape("Tableaux LaTeX", ["make_tables_article.py", "--resultats", MES, "--sortie", os.path.join("mon_article", "tables")], os.path.join(d, "tableaux.txt"), env)
    print("\nTermine. Le dossier mon_article contient l'article avec VOS figures et VOS tableaux.")
    print("Attention : le texte de main.tex cite les chiffres de reference. Verifiez-les si vos chiffres different.")
    print("Pour Overleaf : compressez le dossier mon_article en .zip, puis New Project > Upload Project.")


def mode_comparer():
    if not lire(MES, "detection.csv"):
        stop("Vous n'avez pas encore de resultats. Lancez d'abord :  python lancer.py complet")
    ok, total = afficher(MES, True)
    print(f"\n{ok} chiffres sur {total} sont a moins de 0,03 de la reference.")
    if ok == total:
        print("Resultats retrouves : vous pouvez les presenter a votre encadrante.")
    else:
        print("Certains chiffres s'ecartent. Envoyez-moi ce tableau avant de les utiliser.")
    print(f"Tous vos tableaux sont dans {MES}/tableaux et vos figures dans {MES}/figures.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "test":
        mode_test()
    elif mode == "complet":
        mode_complet()
    elif mode == "comparer":
        mode_comparer()
    elif mode == "test2":
        mode_test2()
    elif mode == "etape2":
        mode_etape2()
    elif mode == "robustesse":
        mode_robustesse()
    elif mode == "article":
        mode_article()
    elif mode == "donnees":
        verifier_python()
        subprocess.run([sys.executable, os.path.join(ICI, "telecharger_donnees.py")], cwd=ICI, env=environnement(""))
    elif mode == "modele":
        verifier_python()
        data = obtenir_donnees()
        subprocess.run([sys.executable, os.path.join(ICI, "entrainer_modele.py"), "--jeu", "skab"], cwd=ICI, env=environnement(data))
    else:
        print(__doc__)
