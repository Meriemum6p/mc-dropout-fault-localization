# Does Monte Carlo Dropout Uncertainty Localize Faults in Multimodal Industrial Data?

Code, result tables and manuscript sources for a controlled evaluation of MC dropout uncertainty as a way to
identify the faulty sensor family in multi-sensor industrial data, compared with simple baselines
(raw deviation, per-family k-means, temporal statistics).

**Main finding.** Before fusion, per-family MC dropout uncertainty localizes injected faults well, but it largely
follows signal amplitude and does not beat per-family k-means on either dataset. After fusion, it stays close
to chance. Details are in `article/main.tex`.

A step-by-step guide in French is in [`GUIDE_FR.md`](GUIDE_FR.md).

## Quick start

```bash
conda create -n mcdropout python=3.11 numpy pandas scipy scikit-learn matplotlib certifi -y
conda activate mcdropout
python lancer.py test        # checks the installation (a few minutes; downloads SKAB)
python lancer.py donnees     # downloads both datasets from their official sources and verifies them
```

Without conda: `python -m pip install -r requirements.txt`. No GPU is needed: the networks are written in NumPy.

## Reproducing the paper

| Command | What it does | Output |
|---|---|---|
| `python lancer.py complet` | Stage 1 on SKAB: detection, localization, number of passes, amplitude test (5 seeds) | `mes_resultats/` |
| `python lancer.py etape2` | Stage 2 on SKAB and the hydraulic dataset: clustering baselines, 1D-CNN, real degradations (3 seeds) | `mes_resultats/etape2/` |
| `python lancer.py robustesse` | Dropout rate, dropout placement and window length (2 seeds) | `mes_resultats/robustesse/` |
| `python lancer.py article` | Rebuilds the figures (PDF, SVG) and LaTeX tables from your results | `mon_article/` |
| `python lancer.py comparer` | Compares your numbers with the reference tables (tolerance 0.03) | console |
| `python lancer.py modele` | Trains one model and saves its weights, configuration and seeds | `modeles/` |

Each long command can be interrupted and restarted; it resumes where it stopped. The reference tables used in
the paper are in `resultats_reference/`. Numbers may differ in the last digits across machines and library versions.

## Data

The datasets are not owned by this project.

| Dataset | Official source | Licence | In this repository |
|---|---|---|---|
| SKAB (Katser and Kozitsin, 2020) | https://github.com/waico/SKAB, commit `b2c0d46`; DOI 10.34740/KAGGLE/DSV/1693952 | GPL-3.0 | Not included; downloaded by `lancer.py` |
| Condition monitoring of hydraulic systems (Helwig, Pignanelli and Schütze, 2015) | https://archive.ics.uci.edu/dataset/447; DOI 10.24432/C5CW21 | CC BY 4.0 | `donnees_hydraulique_1hz.npz`: the 14 physical sensors averaged to 1 Hz, plus the condition labels |

`python lancer.py donnees` downloads the official hydraulic files and checks that they match the compact file.
See [`NOTICE.md`](NOTICE.md) for attributions.

## No data leakage

A whole SKAB experiment, or a whole block of 25 consecutive hydraulic cycles, is assigned to training, validation
or test, never to two of them. Models, normalization and score calibration use normal training and validation
windows only. Anomaly labels are used for evaluation only.

## Repository layout

| Path | Content |
|---|---|
| `lancer.py` | Single entry point for all commands |
| `src/` | Data loading, NumPy networks (dense, 1D convolution, dropout, Adam), MC dropout, fault injection, metrics |
| `run_*.py` | Experiments |
| `analyze*.py`, `make_tables_article.py`, `make_figures_revision.py` | Tables and figures from the result files |
| `telecharger_donnees.py`, `entrainer_modele.py`, `extraire_exemple.py` | Data download and check, model saving, example of the illustration figure |
| `tests/` | Unit tests (numerical gradient checks, MC dropout, fault injection): `python -m pytest tests` |
| `resultats_reference/` | Result tables behind the paper |
| `article/` | LaTeX sources of the manuscript (Elsevier `elsarticle` class) |
| `HYPOTHESES_ETAPE2.md` | Decision rule and hypotheses written before stage 2 |

File and function names are in French; score names are explained in `GUIDE_FR.md`.

## Limitations

Injected faults are not real degradations; both datasets come from laboratory testbeds; the expected sensor family
of each real fault is a hypothesis that no domain expert has validated; one fusion architecture and one small
temporal network were tested.

## Licence and citation

Code: MIT licence (see `LICENSE`). Data and third-party material keep their own licences (see `NOTICE.md`).
To cite this work, see `CITATION.cff`.

## Use of AI assistance

The code and the manuscript were prepared with the assistance of a generative AI tool (Claude, Anthropic).
The author reviewed the content and is responsible for it.
