"""
OLS con errori standard clusterizzati, senza dipendere da statsmodels
(la versione installata e' incompatibile con scipy).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class OLSResult:
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    rsquared: float
    nobs: int
    resid: pd.Series


def design(d: pd.DataFrame, regressors, fixed_effects=()):
    """Matrice di disegno: costante + regressori + dummy (prima categoria esclusa) per ogni effetto fisso."""
    X = pd.DataFrame({"const": 1.0}, index=d.index)
    for r in regressors:
        X[r] = d[r].astype(float)
    for fe in fixed_effects:
        X = X.join(pd.get_dummies(d[fe], prefix=fe, drop_first=True, dtype=float))
    return X


def ols(d: pd.DataFrame, y: str, regressors, fixed_effects=(), cluster=None) -> OLSResult:
    """
    OLS di y sui regressori (+ effetti fissi).
    cluster: nome della colonna dei gruppi -> errori standard cluster-robust con correzione CR1
    (G/(G-1) * (N-1)/(N-K), come Stata); None -> errori standard robusti HC1.
    I p-value usano una t di Student con G-1 gradi di liberta' (N-K senza cluster).
    """
    X = design(d, regressors, fixed_effects)
    Y = d[y].astype(float).to_numpy()
    Xv = X.to_numpy()
    n, k = Xv.shape
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ Y
    u = Y - Xv @ beta

    if cluster is not None:
        groups = pd.factorize(d[cluster])[0]
        G = groups.max() + 1
        meat = np.zeros((k, k))
        for g in range(G):
            s = Xv[groups == g].T @ u[groups == g]
            meat += np.outer(s, s)
        corr = G / (G - 1) * (n - 1) / (n - k)
        dof = G - 1
    else:
        meat = (Xv * u[:, None] ** 2).T @ Xv
        corr = n / (n - k)
        dof = n - k
    V = corr * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    r2 = 1 - (u @ u) / np.sum((Y - Y.mean()) ** 2)

    idx = X.columns
    return OLSResult(pd.Series(beta, idx), pd.Series(se, idx), pd.Series(p, idx), float(r2), n,
                     pd.Series(u, index=d.index))


def residualize(d: pd.DataFrame, y: str, regressors, fixed_effects=()):
    """Residui di y su regressori + effetti fissi (per gli added-variable plot)."""
    return ols(d, y, regressors, fixed_effects).resid
