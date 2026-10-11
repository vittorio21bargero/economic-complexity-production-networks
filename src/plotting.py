"""
Stile comune e funzioni di plot per i notebook in notebooks/plotting.

Stile "da articolo": niente titoli (la spiegazione va nella didascalia),
niente griglia, box con tick verso l'interno in basso e a sinistra, legenda
dentro gli assi, un pannello per figura (in LaTeX si affiancano con subfigure),
font Computer Modern
(come il testo LaTeX della tesi), figure dimensionate sulla larghezza del testo
e salvate in PDF vettoriale (+ PNG di anteprima).

Per cambiare l'aspetto di TUTTI i plot basta modificare PALETTE, TEXTWIDTH e STYLE.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# Parametri di stile (da cambiare qui, valgono per tutti i plot)
# --------------------------------------------------------------------------- #
# Larghezza del testo della tesi in pollici (A4, margini standard ~ 15 cm).
# In LaTeX: \the\textwidth (in pt) / 72.27
TEXTWIDTH = 5.9

# Palette colorblind-safe (Okabe-Ito). Ogni grandezza ha SEMPRE lo stesso
# colore in tutte le figure della tesi.
PALETTE = {
    "Q":     "#0072B2",   # blu
    "CQ":    "#D55E00",   # vermiglio
    "Delta": "#009E73",   # verde
    "F":     "#CC79A7",   # porpora
    "S":     "#E69F00",   # arancio
    "N1":    "#000000",   # numero di input
    "Qin":   "#8C8C8C",   # complessita' media degli input
    4:       "#0072B2",   # risoluzione 4 digits
    6:       "#D55E00",   # risoluzione 6 digits
}
RES_MARKER = {4: "o", 6: "s"}

# Mappa sequenziale per parametri ordinati (es. alpha): viridis senza gli estremi
# troppo chiari/scuri.
_viridis = plt.get_cmap("viridis")
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", _viridis(np.linspace(0.0, 0.85, 256)))

# Simboli matematici per le grandezze salvate nei parquet
SYMBOLS = {"Q": r"$Q$", "CQ": r"$QC$", "Delta": r"$\Delta$", "F": r"$F$", "S": r"$S$",
           "N1": r"$N_1$", "Qin": r"$\langle Q \rangle_{\mathrm{in}}$"}

STYLE = {
    # font: Computer Modern come LaTeX (senza bisogno di LaTeX installato)
    # cmr10 non ha le lettere accentate: matplotlib (>= 3.6) usa STIX solo per i caratteri mancanti
    "font.family": ["cmr10", "STIXGeneral"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # assi: box completo, tick verso l'interno solo in basso e a sinistra;
    # minor tick solo sugli assi logaritmici (li gestisce matplotlib)
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    # linee e marker sottili
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
    "lines.markeredgewidth": 0.9,
    # legenda senza cornice
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "legend.borderaxespad": 0.4,
    # output
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}


def set_style():
    plt.rcParams.update(STYLE)
    # fontTools avvisa sui timestamp del font cmr10 a ogni salvataggio in PDF: innocuo
    logging.getLogger("fontTools").setLevel(logging.ERROR)


def figsize(width_frac=1.0, aspect=0.62):
    """(larghezza, altezza) in pollici: width_frac = frazione di \\textwidth."""
    w = TEXTWIDTH * width_frac
    return (w, w * aspect)


# Dimensioni standard:
# - SINGLE: figura da sola nella pagina
# - HALF:   figura pensata per essere affiancata a un'altra (\subfigure{0.49	extwidth})
SINGLE = (0.7, 0.72)
HALF = (0.49, 0.8)


def common_ylim(frames, columns, pad=0.04):
    """Limiti y comuni a piu' figure da affiancare (stesso asse => confronto onesto)."""
    vals = np.concatenate([df[c].to_numpy() for df in frames for c in columns if c in df])
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    d = (hi - lo) * pad or 0.01
    return lo - d, hi + d


def save_fig(fig, path):
    """Salva la figura in PDF (per LaTeX)."""
    path = Path(path).with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))


def panel_label(ax, text, loc="upper left"):
    """Etichetta di pannello, es. '(a)', dentro gli assi."""
    x, ha = (0.04, "left") if "left" in loc else (0.96, "right")
    y, va = (0.95, "top") if "upper" in loc else (0.05, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va)


def two_part_legend(ax, color_items, style_items, discrete=False, **kw):
    """
    Legenda in due blocchi: colore = grandezza, stile = metodo.
    color_items: [(label, color)] oppure [(label, color, marker)],
    style_items: [(label, linestyle, marker_filled)]
    discrete=False: x continua -> solo linee (piena/tratteggiata);
    discrete=True:  x discreta -> solo marker (pieno/vuoto); il colore e' un rettangolo,
    o il marker indicato (es. RES_MARKER per le risoluzioni).
    """
    if discrete:
        handles = [Line2D([], [], ls="none", marker=it[2], color=it[1]) if len(it) > 2
                   else Patch(color=it[1]) for it in color_items]
        handles += [Line2D([], [], ls="none", marker="o", color="0.25",
                           mfc="0.25" if filled else "white", mec="0.25")
                    for _, _, filled in style_items]
    else:
        handles = [Line2D([], [], color=c, lw=1.3) for _, c in color_items]
        handles += [Line2D([], [], color="0.25", ls=ls) for _, ls, _ in style_items]
    labels = [it[0] for it in color_items] + [l for l, _, _ in style_items]
    return ax.legend(handles, labels, **kw)


# --------------------------------------------------------------------------- #
# Correlazioni (stabilita', numero di input)
# --------------------------------------------------------------------------- #
METHODS = (("pearson", "-", True, "Pearson"), ("spearman", "--", False, "Spearman"))


def _plot_corr_lines(ax, x, df, quantities, discrete=False):
    """
    Colore = grandezza. x continua (discrete=False): solo linee, piena = Pearson,
    tratteggiata = Spearman. x discreta (discrete=True): solo marker, pieno = Pearson,
    vuoto = Spearman.
    """
    present = []
    for q in quantities:
        color = PALETTE[q]
        drawn = False
        for m, ls, filled, _ in METHODS:
            col = f"{m}_{q}"
            if col not in df:
                continue
            if discrete:
                ax.plot(x, df[col], ls="none", marker="o", color=color,
                        mfc=color if filled else "white", mec=color)
            else:
                ax.plot(x, df[col], ls=ls, color=color)
            drawn = True
        if drawn:
            present.append(q)
    return present


def _corr_legend(ax, present, discrete=False, **kw):
    """Legenda compatta su due colonne: grandezze | metodi."""
    kw = {"ncol": 2, "handlelength": 1.2 if discrete else 1.8, "columnspacing": 1.0,
          "labelspacing": 0.3, **kw}
    two_part_legend(ax,
                    [(SYMBOLS.get(q, q), PALETTE[q]) for q in present],
                    [(lab, ls, filled) for _, ls, filled, lab in METHODS], discrete=discrete, **kw)


def load_stability(target, algo, resolutions=(4, 6)):
    """
    Carica le tabelle di stabilita' da results/stability/{target}/{algo}/.
    target: "C" | "Mcp_export" | "Mcp_import". Ritorna {res: DataFrame}.
    """
    folder = ROOT / "results" / "stability" / target / algo
    out = {}
    for res in resolutions:
        f = folder / f"{target}_stability_{res}_digits.parquet"
        if f.exists():
            out[res] = pd.read_parquet(f)
        else:
            print(f"[manca] {f.relative_to(ROOT)}")
    return out


def check_eta0(tables, quantities, tol=1e-6):
    """A eta = 0 tutte le correlazioni devono valere 1, altrimenti i dati vanno rigenerati."""
    ok = True
    for res, df in tables.items():
        row = df.loc[np.isclose(df["eta"], 0)]
        if row.empty:
            continue
        for q in quantities:
            for m, *_ in METHODS:
                col = f"{m}_{q}"
                if col in row and abs(row[col].iloc[0] - 1) > tol:
                    print(f"[ATTENZIONE] {res} digits: {col}(eta=0) = {row[col].iloc[0]:.3f} != 1")
                    ok = False
    return ok


def plot_stability(df, quantities, ylim=None, save_path=None, legend_loc="best", legend_kw=None):
    """
    Correlazione con il caso imperturbato (eta = 0) al variare di eta, per UNA risoluzione.
    Passare lo stesso ylim (vedi common_ylim) alle figure da affiancare.
    """
    fig, ax = plt.subplots(figsize=figsize(*HALF))
    present = _plot_corr_lines(ax, df["eta"], df, quantities)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$r\,(X_\eta,\, X_0)$")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(np.linspace(0, 1, 6))
    if ylim is not None:
        ax.set_ylim(*ylim)
    _corr_legend(ax, present, **{"loc": legend_loc, **(legend_kw or {})})
    fig.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    plt.show()
    return fig


def plot_corr_vs_order(df, quantities, ylim=None, save_path=None, legend_loc="best"):
    """Correlazione fra numero di input di ordine k e le grandezze date, al variare di k."""
    fig, ax = plt.subplots(figsize=figsize(*HALF))
    present = _plot_corr_lines(ax, df["ordine"], df, quantities, discrete=True)
    ax.set_xticks(df["ordine"])
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$r\,(N_k,\, X)$")
    if ylim is not None:
        ax.set_ylim(*ylim)
    _corr_legend(ax, present, discrete=True, loc=legend_loc)
    fig.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    plt.show()
    return fig


def plot_corr_vs_a(df, quantities, ylabel, ylim=None, save_path=None, legend_loc="lower right", legend_kw=None):
    """
    Correlazioni al variare di a = rho * alpha (asse log), con linea verticale in a = 1.
    df deve avere la colonna "a" e le colonne f"{pearson|spearman}_{q}".
    """
    fig, ax = plt.subplots(figsize=figsize(*HALF))
    ax.axvline(1, color="0.6", lw=0.8, ls=":", zorder=0)
    present = _plot_corr_lines(ax, df["a"], df, quantities)
    ax.set_xscale("log")
    ax.set_xlabel(r"$a = \rho\,\alpha$")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _corr_legend(ax, present, **{"loc": legend_loc, "ncol": 1, **(legend_kw or {})})
    fig.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    plt.show()
    return fig
