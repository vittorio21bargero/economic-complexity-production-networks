"""
Figures for the README (English labels), saved as PNG in docs/figures/.

They reuse results already produced by the notebooks:
- results/stability/C/extended_fitness_with_imports/   (notebooks/data_generation/C_stability.ipynb)
- results/econometria/regressioni/regressioni.csv       (notebooks/econometria/econometria_paesi.ipynb)
and recompute Extended Fitness with and without the import filter for a single year
(as in notebooks/econometria/effetto_import.ipynb), which needs the data in data/.

Run from the repository root:  python scripts/readme_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting import set_style, figsize, load_stability, _plot_corr_lines, _corr_legend, PALETTE, SEQ_CMAP, SINGLE, HALF
from src.algorithms import compute_alpha, compute_extended_fitness_no_imports
from src.algorithms_with_imports import compute_extended_fitness_with_imports

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
set_style()


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


def rewiring():
    """Rank stability of product and input complexity when the inputs of each product are rewired."""
    df = load_stability("C", "extended_fitness_with_imports", resolutions=(4,))[4]
    fig, ax = plt.subplots(figsize=figsize(*HALF))
    present = _plot_corr_lines(ax, df["eta"], df, ["Q", "CQ", "Delta"])
    ax.set_xlabel(r"fraction of rewired input links $\eta$")
    ax.set_ylabel(r"$r\,(X_\eta, X_0)$")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1.05)
    _corr_legend(ax, present, loc="lower left")
    fig.tight_layout()
    save(fig, "rewiring_null_model")


def growth_r2():
    """Extra R^2 over initial GDP when one complexity metric is added to growth regressions."""
    res = pd.read_csv(ROOT / "results" / "econometria" / "regressioni" / "regressioni.csv")
    names = ["Extended Fitness", "Extended ECI", "Fitness", "ECI", "N1 medio"]
    ticks = ["Extended\nFitness", "Extended\nECI", "Fitness", "ECI", "mean\n" + r"$N_1$"]
    colors = {5: SEQ_CMAP(0.25), 10: SEQ_CMAP(0.8)}
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=figsize(*SINGLE))
    for j, T in enumerate((5, 10)):
        s = res[res["T"] == T].set_index("modello").reindex(names)
        ax.bar(x + (j - 0.5) * 0.36, s["dR2"], width=0.36, color=colors[T], label=f"$T = {T}$ years")
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_ylabel("increase in $R^2$ over\ninitial GDP per capita only")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "growth_regressions_dR2")


def import_filter(year=2012, res=4, n_label=6):
    """Extended Fitness with vs without the import filter, for one year."""
    data = ROOT / "data" / "HS02_all_years" / f"{res}_digits" / "alligned" / "binarized"
    iso3 = pd.read_csv(ROOT / "data" / "raw" / "BACI" / "country_codes_V202601.csv").set_index("country_code")["country_iso3"]
    C = pd.read_parquet(data / "C" / f"C_{res}_digits_HS02.parquet")
    Me = pd.read_parquet(data / "Mcp_export" / f"Mcp_export_alligned_Y{year}.parquet")
    Mi = pd.read_parquet(data / "Mcp_import" / f"Mcp_import_alligned_Y{year}.parquet")
    alpha = compute_alpha(C)  # a = 1

    _, F, *_ = compute_extended_fitness_with_imports(Me, Mi, C.T, alpha=alpha, tensor_window=3, max_iter=5000,
                                                     tol=1e-6, track_convergence=False, save_delta=False)
    _, F0, *_ = compute_extended_fitness_no_imports(Me, C.T, alpha=alpha, max_iter=5000, tol=1e-6,
                                                    track_convergence=False)
    d = pd.DataFrame({"x": np.log(F0), "y": np.log(F / F0)})
    d.index = Me.index.map(iso3)

    fig, ax = plt.subplots(figsize=figsize(*SINGLE))
    ax.scatter(d["x"], d["y"], s=10, color=PALETTE[4], alpha=0.7, lw=0)
    ax.axhline(0, color="0.5", lw=0.8, ls=":")
    for c, r in d.nsmallest(n_label, "y").iterrows():  # most penalised countries
        ax.annotate(c, (r["x"], r["y"]), xytext=(3, 2), textcoords="offset points", fontsize=7, color="0.2")
    ax.set_xlabel(r"$\ln F$ without import filter (Extended Fitness, " + f"{year})")
    ax.set_ylabel(r"$\ln (F_{\mathrm{with\ filter}} / F_{\mathrm{without\ filter}})$")
    fig.tight_layout()
    save(fig, f"import_filter_{year}")


if __name__ == "__main__":
    rewiring()
    growth_r2()
    import_filter()
