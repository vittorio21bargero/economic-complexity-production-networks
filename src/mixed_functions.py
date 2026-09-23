#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# 0) VECCHIO ALGORITMO (2022_HS4_analysis)

def compute_extended_fitness_April(
    Mcp: pd.DataFrame,
    C: pd.DataFrame,
    alpha: float,
    max_iter: int = 100000,
    tol: float = 1e-15,
    track: bool = True,
    save_delta: bool = False,
    delta_window: int | None = None,
):
   

    # Gestione INPUT
    M  = Mcp.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = M.shape
    
    # 1) INIZIALIZZAZIONE
    F = np.ones(n_c)
    Q = np.ones(n_p)
    
    # Salvataggio dei Delta ai passi precedenti
    history    = [] if track else None
    delta_hist = deque(maxlen=delta_window) if save_delta else None
    converged  = False
    
    
    
    i = 0
    for i in range(max_iter):
       
        F_old = F.copy()
        Q_old = Q.copy()
        
        
        #  2.1) EQUAZIONI SISTEMA DINAMICO
        F     = M @ Q_old                       
        Delta = 1.0 / (M.T @ (1.0 / F_old))
        mean_Delta = np.mean(Delta)
        Delta = Delta / mean_Delta
        CQ    = alpha * C @ Q_old                      
        Q     = CQ + Delta                   
       
        #  2.2) NORMALIZZAZIONE
        mean_Q = np.mean(Q)
        
        F     = F / np.mean(F)
        CQ    = CQ / mean_Q
        Delta = Delta /  mean_Q
        Q     = Q / mean_Q
        
        
        
        if track:
            history.append((np.mean(np.abs(F - F_old)), np.mean(np.abs(Q - Q_old))))
        if save_delta:
            for d in delta_hist:
                d /= meanQ
            delta_hist.append(Delta.copy())
        
        
        
        # 3) CONVERGENZA
        if np.all(np.abs(F - F_old) < tol) and np.all(np.abs(Q - Q_old) < tol):
            converged = True
            
            
            break
    fitness    = pd.Series(F,     index=Mcp.index,   name="Fitness")
    CQ         = pd.Series(CQ,    index=Mcp.columns, name="input_complexity")
    Delta      = pd.Series(Delta, index=Mcp.columns, name="intrinsic_complexity")
    complexity = pd.Series(Q,     index=Mcp.columns, name="total_complexity")
    
    
    if track or save_delta:
        info = {"n_iter": i, "converged": converged}
        if track:
            info["history"] = np.array(history)
        if save_delta:
            # shape (n_salvate, n_p): righe = iterazioni, dalla più vecchia tenuta alla più recente
            info["delta_history"] = np.array(delta_hist)
        return complexity, fitness, CQ, Delta, info
    
    
    return complexity, fitness, CQ, Delta


# 2.1.1.0) FITNESS con C (à la Vito con i delta)

# matrice C: SI
# delta: SI
# import: NO

def compute_extended_fitness_with_delta(
    Mcp: pd.DataFrame,
    C: pd.DataFrame,
    phi,
    theta,
    delta: float = 1e-2,      # <-- ecco il parametro, molto minore di 1
    max_iter: int = 1000,
    tol: float = 1e-6
):
    M = Mcp.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = M.shape

    # 1) INIZIALIZZAZIONE
    F = np.ones(n_c)
    Q = np.ones(n_p)

    # storia della convergenza step-by-step
    # storia della convergenza step-by-step
    history = {"F_diff": [], "Q_diff": [], "F_mean_diff": [], "Q_mean_diff": []}

    # 2) SISTEMA DINAMICO
    for it in range(max_iter):
        F_old = F.copy()
        Q_old = Q.copy()

        # equazione 1 + regolarizzazione delta
        F = phi(M, Q) + delta

        # equazione 2 + regolarizzazione delta
        Delta = theta(F, M)
        CQ = C @ Q                  # contributo input
        Q = CQ + Delta + delta      # update totale + delta


        # registro la variazione massima a questo step
        f_diff = np.max(np.abs(F - F_old))
        q_diff = np.max(np.abs(Q - Q_old))
        history["F_diff"].append(f_diff)
        history["Q_diff"].append(q_diff)
        history["F_mean_diff"].append(np.mean(np.abs(F - F_old)))
        history["Q_mean_diff"].append(np.mean(np.abs(Q - Q_old)))

        # convergenza
        if f_diff < tol and q_diff < tol:
            break

    fitness    = pd.Series(F,     index=Mcp.index,   name="Fitness")
    CQ         = pd.Series(CQ,    index=Mcp.columns, name="input_complexity")
    Delta      = pd.Series(Delta, index=Mcp.columns, name="intrinsic_complexity")
    complexity = pd.Series(Q,     index=Mcp.columns, name="total_complexity")

    return complexity, fitness, CQ, Delta, history
