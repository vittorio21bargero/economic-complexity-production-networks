#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os
import re  # per estrarre l'anno dal nome file
from scipy.sparse.linalg import eigs
from collections import deque
import sys
from pathlib import Path


ROOT = Path().resolve().parent
sys.path.insert(0, str(ROOT))


from src.C_properties import is_C_nilpotente, autovettore_and_autovalore_principale

# In[4]:



def compute_alpha(C: pd.DataFrame):
    """
    Calcola l'alpha da usare nell'algoritmo.
    - Se C e' nilpotente  -> rho = degree_medio(C)
    - Altrimenti          -> rho = autovettore_principale(C)
    Restituisce alpha = 1 / rho.
    """
    if is_C_nilpotente(C):
        rho = indegree_medio(C)
    else:
        rho, _ = autovettore_and_autovalore_principale(C)

    return 1.0 / rho








def compute_eci(Mcp: pd.DataFrame):
    """
    Calcola ECI e PCI usando il metodo dell'autovettore.

    Input:
    - Mcp 
     
    Output:
    - eci : vettore complessità dei paesi (countries)
    - pci : vettore complessità dei prodotti (products)
    
    """
    # 0) Gestione input
    M = Mcp.values.astype(float)

    # 1) Algoritmo
    
    # 1.1) Calcolo diversity and ubiquity (fattore di normalizzazione di questo algoritmo)
    k_c = M.sum(axis=1)  # diversity countries
    k_p = M.sum(axis=0)  # ubiquity products
    # Evito divisioni per zero
    k_c[k_c == 0] = 1
    k_p[k_p == 0] = 1

    # 1.2) Matrice normalizzata di similitudine paese-paese
    M_tilde = (M / k_c[:, None]) @ (M.T / k_p[:,None ])   # None: serve per permettermi di moltiplicare matrice e vettori correttamente 

    # 1.3) AUTOVALORI e AUTOVETTORI
    eigvals, eigvecs = np.linalg.eig(M_tilde)

    # Ranko gli AUTOVALORI (dal pià grande al più piccolo)
    idx = np.argsort(eigvals)[::-1]   # idx contiene la posizione del ranking (ordine decrescente) degli autovalori ( 0 al più grande)
    eigvals = eigvals[idx]

    # Ordino gli AUTOVETTORI (da quello corrispondente all'autovalore massimo a quello corrispondente al minimo) 
    eigvecs = eigvecs[:, idx] #riordino allo stesso modo gli autovettori

    # A) ECI ---> mi interessa il 2° autovettore
    eci = np.real(eigvecs[:, 1])
    eci = (eci - np.mean(eci)) / np.std(eci)    # Standardizzazione eci (media 0, varianza 1)

    # B) PCI
    # Proiezione sui prodotti
    pci = (M.T / kc[None, :]) @ eci
    pci = (pci - np.mean(pci)) / np.std(pci)    # Standardizzazione pci (media 0, varianza 1)

   
    eci = pd.Series(eci, index=Mcp.index, name="ECI")
    pci = pd.Series(pci, index=Mcp.columns, name="PCI")

    return eci, pci


    




def compute_fitness(Mcp: pd.DataFrame, max_iter: int = 1000, tol: float = 1e-6,
                    track_convergence: bool = True):
    """
    Calcola Fitness e Complessità dei prodotti (Tacchella).

    Input:
    - Mcp
    - max_iter  
    - tol : tolleranza per convergenza
    - track : bool
        Se True, salva la storia della convergenza (delta medi per iterazione)

    Output:
    - fitness : Fitness dei paesi
    - complexity : Complessità dei prodotti

    - info_convergence
         
    """
    M = Mcp.values.astype(float)
    n_c, n_p = M.shape

    # inizializzazione: complessità 1 a tutti countries e products
    F = np.ones(n_c)
    Q = np.ones(n_p)

    history = [] if track else None
    converged = False
    i = 0

    for i in range(max_iter):
        # salvo i valori dello step precedente per verificare la convergenza
        F_old = F.copy()
        Q_old = Q.copy()

        # aggiornamento Fitness
        F = M @ Q
        
        # aggiornamento Complessità (non lineare!)
        Q = 1 / (M.T @ (1 / F))
        Q[np.isinf(Q)] = 0.0
        
        # normalizzazione (evita divergenze numeriche)
        F = F / np.mean(F)
        Q = Q / np.mean(Q)

        # delta calcolati una sola volta e riusati
        dF = np.abs(F - F_old)
        dQ = np.abs(Q - Q_old)

        if track_convergence:
            history.append((dF.mean(), dQ.mean()))

        # controllo convergenza
        if np.all(dF < tol) and np.all(dQ < tol):
            converged = True
            break

    fitness = pd.Series(F, index=Mcp.index, name="Fitness")
    complexity = pd.Series(Q, index=Mcp.columns, name="Complexity")

    if track:
        info_convergence = {
            "history": np.array(history),
            "n_iter": i,
            "converged": converged,
        }
        return fitness, complexity, info_convergence

    return fitness, complexity





def compute_extended_fitness_no_imports(
    Mcp: pd.DataFrame,
    C: pd.DataFrame,
    alpha: float,
    max_iter: int = 100000,
    tol: float = 1e-15,
    track_convergence: bool = True,
    save_delta: bool = False,
    delta_window: int | None = None,
):
    """
    Fitness estesa con valore aggiunto e matrice di composizione C (nilpotente).
    Sistema:
        F     = M @ Q                 (phi,   eq. Tacchella standard)
        Delta = 1 / (M.T @ (1/F))     (theta, eq. Tacchella standard)
        Q     = C @ Q + Delta
 
    Input:
    - Mcp : matrice binaria country-product
    - C : matrice binaria di composizione
    - alpha : parametro di sconto
 
    - track_convergence: variabile booleana per vedere o meno come la funzione
      converge passo dopo passo
    - save_delta: variabile booleana per salvare o meno i delta (normalizzati)
      ai passi precedenti
    - delta_window: numero di passi precedenti in cui salvare i delta
 
    Output:
    - complexity : vettore product complexity
    - fitness : vettore country fitness
 
    NB su save_delta: i Delta salvati accumulano una normalizzazione per ogni
    iterazione successiva alla loro creazione, quindi il Delta a distanza k
    dall'iterazione corrente risulta diviso per k+1 fattori mean(Q).
    """
    
    # 0) Gestione INPUT
    M = Mcp.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = M.shape
 
    # 1) INIZIALIZZAZIONE
    F = np.ones(n_c)
    Q = np.ones(n_p)
 
    # *) SALVATAGGI OPZIONALI
 
    # *.a) CONVERGENZA
    history_convergence = [] if track_convergence else None
    # *.b) DELTA
    # maxlen è la lunghezza massima di deque oltre la quale se aggiungo un
    # elemento da una parte ne scarto uno dall'altra (esattamente quello che voglio)
    delta_hist = deque(maxlen=delta_window) if save_delta else None
 
    # 2) RUN DEL SISTEMA DINAMICO
    i = 0
    for i in range(max_iter):
        F_old = F.copy()
        Q_old = Q.copy()
 
        #  2.1) EQUAZIONI SISTEMA DINAMICO
        F = M @ Q_old
 
        Delta = 1.0 / (M.T @ (1.0 / F_old))
        Delta[np.isinf(Delta)] = 0.0
 
        mean_Delta = np.mean(Delta)
        Delta = Delta / mean_Delta
 
        CQ = alpha * C @ Q_old
        Q  = CQ + Delta
 
        #  2.2) NORMALIZZAZIONE
        mean_Q = np.mean(Q)
 
        F     = F / np.mean(F)
        CQ    = CQ / mean_Q
        Delta = Delta / mean_Q
        Q     = Q / mean_Q
 
        # *.a) CONVERGENZA
        if track_convergence:
            history_convergence.append(
                (np.mean(np.abs(F - F_old)), np.mean(np.abs(Q - Q_old)))
            )
        # *.b) DELTA
        if save_delta:
            # accumula una normalizzazione in più sui delta già salvati,
            # poi appendi il nuovo (che ha già la sua unica normalizzazione)
            for d in delta_hist:
                d /= mean_Q
            delta_hist.append(Delta.copy())
 
        # 3) CONVERGENZA
        if np.all(np.abs(F - F_old) < tol) and np.all(np.abs(Q - Q_old) < tol):
            break
 
    # Risultati ottenuti a convergenza
    fitness    = pd.Series(F,     index=Mcp.index,   name="Fitness")
    CQ         = pd.Series(CQ,    index=Mcp.columns, name="input_complexity")
    Delta      = pd.Series(Delta, index=Mcp.columns, name="intrinsic_complexity")
    complexity = pd.Series(Q,     index=Mcp.columns, name="total_complexity")
 
    # *) Risultati opzionali (salvati in info)
    info_convergence = None
    delta_history = None
 
    if track_convergence:
        info_convergence = {"n_iter": i, "history": np.array(history_convergence)}
    if save_delta:
        delta_history = np.array(delta_hist)
 
    if track_convergence or save_delta:
        return complexity, fitness, CQ, Delta, info_convergence, delta_history
 
    return complexity, fitness, CQ, Delta





def compute_extended_eci_no_imports(
    Mcp: pd.DataFrame,
    C: pd.DataFrame,
    alpha: float,
    max_iter: int = 100000,
    tol: float = 1e-15,
    track_convergence: bool = True,
    save_delta: bool = False,
    delta_window: int | None = None,
):
    """
    ECI esteso con matrice di composizione C, senza filtro per gli import.
    Sistema:
        F     = D^-1 @ M @ Q            (phi,   metodo delle riflessioni)
        Delta = U^-1 @ M.T @ F          (theta, metodo delle riflessioni)
        Q     = alpha * C @ Q + Delta
 
    dove D = diag(diversification) e U = diag(ubiquity).
 
    Input:
    - Mcp : matrice binaria country-product
    - C : matrice binaria di composizione (da passare gia' trasposta, come per
      la versione Fitness: l'algoritmo fa C @ Q e deve sommare sugli input)
    - alpha : parametro di sconto
 
    - track_convergence: variabile booleana per vedere o meno come la funzione
      converge passo dopo passo
    - save_delta: variabile booleana per salvare o meno i delta (normalizzati)
      ai passi precedenti
    - delta_window: numero di passi precedenti in cui salvare i delta
 
    Output:
    - complexity : vettore product complexity
    - fitness : vettore country ECI
 
    NB su save_delta: i Delta salvati accumulano una normalizzazione per ogni
    iterazione successiva alla loro creazione, quindi il Delta a distanza k
    dall'iterazione corrente risulta diviso per k+1 fattori mean(Q).
 
    NB: essendo l'ECI lineare, a convergenza Q e' l'autovettore dominante di
    (alpha*C + U^-1 M.T D^-1 M) e mean_Q il corrispondente autovalore.
    """
 
    # 0) Gestione INPUT
    M = Mcp.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = M.shape
 
    # 0.1) Diversification e ubiquity
    diversification = M.sum(axis=1)
    ubiquity        = M.sum(axis=0)
 
    D_inv = np.divide(1.0, diversification,
                      out=np.zeros_like(diversification), where=diversification != 0)
    U_inv = np.divide(1.0, ubiquity,
                      out=np.zeros_like(ubiquity), where=ubiquity != 0)
 
    # 1) INIZIALIZZAZIONE
    F = np.ones(n_c)
    Q = np.ones(n_p)
 
    # *) SALVATAGGI OPZIONALI
 
    # *.a) CONVERGENZA
    history_convergence = [] if track_convergence else None
    # *.b) DELTA
    delta_hist = deque(maxlen=delta_window) if save_delta else None
 
    # 2) RUN DEL SISTEMA DINAMICO
    i = 0
    for i in range(max_iter):
        F_old = F.copy()
        Q_old = Q.copy()
 
        #  2.1) EQUAZIONI SISTEMA DINAMICO
        F = D_inv * (M @ Q_old)
 
        Delta = U_inv * (M.T @ F_old)
 
        mean_Delta = np.mean(Delta)
        Delta = Delta / mean_Delta
 
        CQ = alpha * (C @ Q_old)
        Q  = CQ + Delta
 
        #  2.2) NORMALIZZAZIONE
        mean_Q = np.mean(Q)
 
        F     = F / np.mean(F)
        CQ    = CQ / mean_Q
        Delta = Delta / mean_Q
        Q     = Q / mean_Q
 
        # *.a) CONVERGENZA
        if track_convergence:
            history_convergence.append(
                (np.mean(np.abs(F - F_old)), np.mean(np.abs(Q - Q_old)))
            )
        # *.b) DELTA
        if save_delta:
            # accumula una normalizzazione in piu' sui delta gia' salvati,
            # poi appendi il nuovo (che ha gia' la sua unica normalizzazione)
            for d in delta_hist:
                d /= mean_Q
            delta_hist.append(Delta.copy())
 
        # 3) CONVERGENZA
        if np.all(np.abs(F - F_old) < tol) and np.all(np.abs(Q - Q_old) < tol):
            break
 
    # Risultati ottenuti a convergenza
    fitness    = pd.Series(F,     index=Mcp.index,   name="ECI")
    CQ         = pd.Series(CQ,    index=Mcp.columns, name="input_complexity")
    Delta      = pd.Series(Delta, index=Mcp.columns, name="intrinsic_complexity")
    complexity = pd.Series(Q,     index=Mcp.columns, name="total_complexity")
 
    # *) Risultati opzionali (salvati in info)
    info_convergence = None
    delta_history = None
 
    if track_convergence:
        info_convergence = {"n_iter": i, "mean_Q": mean_Q,
                            "history": np.array(history_convergence)}
    if save_delta:
        delta_history = np.array(delta_hist)
 
    if track_convergence or save_delta:
        return complexity, fitness, CQ, Delta, info_convergence, delta_history
 
    return complexity, fitness, CQ, Delta










































def compute_Q_medio_extended_fitness(
    Mcp: pd.DataFrame,
    C: pd.DataFrame,
    alpha: float,
    max_iter: int = 100000,
    tol: float = 1e-15,
    ):
    """
    history qui mi serve per riportare i valori di Q medio step by step
    
    """
    # 0) Gestione INPUT
    M  = Mcp.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = M.shape

    # 1) INIZIALIZZAZIONE
    F = np.ones(n_c)
    Q = np.ones(n_p)

    # *) SALVATAGGI OPZIONALI

    # *.a) Q MEDIO
    history    = [] 
   

    # 2) RUN DEL SISTEMA DINAMICO
    i = 0
    for i in range(max_iter):
        F_old = F.copy()
        Q_old = Q.copy()

        #  2.1) EQUAZIONI SISTEMA DINAMICO
        F     = M @ Q_old

        Delta = 1.0 / (M.T @ (1.0 / F_old))
        Delta[np.isinf(Delta)] = 0.0
        
        mean_Delta = np.mean(Delta)
        Delta = Delta / mean_Delta

        CQ    = alpha * C @ Q_old
        Q     = CQ + Delta

        #  2.2) NORMALIZZAZIONE
        mean_Q = np.mean(Q)

        F     = F / np.mean(F)
        CQ    = CQ / mean_Q
        Delta = Delta / mean_Q
        Q     = Q / mean_Q
        
       
        history.append(mean_Q)
       

        # 3) CONVERGENZA
        if np.all(np.abs(F - F_old) < tol) and np.all(np.abs(Q - Q_old) < tol):
            break


    return mean_Q , history



# In[ ]:

# In[5]:


