"""Percorsi e I/O dei dati per l'econometria.

    results/econometria/
        metriche/{res}_digits/{algo}/{kind}/Y{year}.parquet      algo in ALGOS, kind in KINDS
        misure_esterne/country/Y{year}.parquet                  indipendenti dalla risoluzione
        misure_esterne/product/misure_esterne_hs{res}.parquet   statiche (senza anno)
        raw/                                                    file scaricati (PWT, World Bank, HDR, OECD)

File delle metriche (formato largo, una riga per paese/prodotto):
    country: year, country, iso3, <variabili>
    product: year, product, <variabili>
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "econometria"
METRICS_DIR = BASE / "metriche"
EXTERNAL_DIR = BASE / "misure_esterne"
RAW_DIR = BASE / "raw"

ALGOS = ("eci", "fitness", "extended_eci", "extended_fitness")
KINDS = ("country", "product")
RESOLUTIONS = (2, 4, 6)
YEARS = tuple(range(2002, 2025))

VARIABLES = {
    ("eci", "country"): ["ECI"],
    ("eci", "product"): ["PCI"],
    ("fitness", "country"): ["F"],
    ("fitness", "product"): ["Q"],
    ("extended_eci", "country"): ["F", "S"],
    ("extended_eci", "product"): ["Q", "CQ", "Delta"],
    ("extended_fitness", "country"): ["F", "S"],
    ("extended_fitness", "product"): ["Q", "CQ", "Delta"],
}
KEY = {"country": "country", "product": "product"}


def _check(res, algo, kind):
    if res not in RESOLUTIONS:
        raise ValueError(f"risoluzione {res} non in {RESOLUTIONS}")
    if algo not in ALGOS:
        raise ValueError(f"algoritmo {algo!r} non in {ALGOS}")
    if kind not in KINDS:
        raise ValueError(f"tipo {kind!r} non in {KINDS}")


# ---------------------------------------------------------------- metriche

def metric_path(res, algo, kind, year) -> Path:
    _check(res, algo, kind)
    return METRICS_DIR / f"{res}_digits" / algo / kind / f"Y{year}.parquet"


def available_years(res, algo, kind):
    d = metric_path(res, algo, kind, 0).parent
    return sorted(int(p.stem[1:]) for p in d.glob("Y*.parquet"))


def save_metric(df: pd.DataFrame, res, algo, kind, year):
    """Salva la tabella larga di un anno; controlla le colonne attese."""
    cols = ["year", KEY[kind]] + (["iso3"] if kind == "country" else []) + VARIABLES[(algo, kind)]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti per {algo}/{kind}: {sorted(missing)}")
    path = metric_path(res, algo, kind, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].reset_index(drop=True).to_parquet(path)
    return path


def load_metric(res, algo, kind, years=None) -> pd.DataFrame:
    """Tabella larga di un algoritmo, concatenata sugli anni (tutti quelli disponibili se years=None)."""
    years = available_years(res, algo, kind) if years is None else list(years)
    if not years:
        raise FileNotFoundError(f"nessun anno per {res} digits / {algo} / {kind}")
    return pd.concat([pd.read_parquet(metric_path(res, algo, kind, y)) for y in years], ignore_index=True)


def load_metrics_long(res, kind, algos=ALGOS, years=None) -> pd.DataFrame:
    """Tutti gli algoritmi in formato lungo: year, <chiave>, [iso3], metric, variable, value."""
    ids = ["year", KEY[kind]] + (["iso3"] if kind == "country" else [])
    frames = []
    for algo in algos:
        w = load_metric(res, algo, kind, years)
        frames.append(w.melt(id_vars=ids, value_vars=VARIABLES[(algo, kind)],
                             var_name="variable", value_name="value").assign(metric=algo))
    return pd.concat(frames, ignore_index=True)[ids + ["metric", "variable", "value"]]


# ---------------------------------------------------------- misure esterne

def external_country_path(year) -> Path:
    return EXTERNAL_DIR / "country" / f"Y{year}.parquet"


def external_product_path(res) -> Path:
    return EXTERNAL_DIR / "product" / f"misure_esterne_hs{res}.parquet"


def load_external_country(years=None) -> pd.DataFrame:
    """Misure esterne dei paesi (PIL, popolazione, HDI, ...): year, iso3, <misure>."""
    d = EXTERNAL_DIR / "country"
    years = sorted(int(p.stem[1:]) for p in d.glob("Y*.parquet")) if years is None else list(years)
    return pd.concat([pd.read_parquet(external_country_path(y)) for y in years], ignore_index=True)


def load_external_product(res) -> pd.DataFrame:
    """Misure esterne dei prodotti (Lall, ATP, R&D, NBER), indicizzate per codice HS a `res` cifre."""
    return pd.read_parquet(external_product_path(res))
