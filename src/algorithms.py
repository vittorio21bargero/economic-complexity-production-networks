#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os
import re  # per estrarre l'anno dal nome file
from scipy.sparse.linalg import eigs


# In[4]:

#  1) ECI 

#  1.1) ECI standard 

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


    


 


def compute_extended_eci(
    Mcp: pd.DataFrame,
    C: pd.DataFrame,
    alpha: float,
    max_iter: int = 100000,
    tol: float = 1e-15,
    track: bool = True,
    save_delta: bool = False,
    delta_window: int | None = None,
):
    """
    ECI esteso con "valore aggiunto" e matrice di composizione C (nilpotente).
    Sistema:
       
    
    Input:
    - Mcp : matrice binaria country-product
    - C : matrice binaria di composizione
    - alpha : parametro di sconto
    
    - track: variabile booleana per vedere o meno come la funzione converge passo dopo passo
    - save_delta: variabile booleana per salvare o meno i delta (normalizzati) ai passi precedenti
    - delta_window: numero di passi precedenti in cui salvare i delta
    
    Output:
    - pci : vettore product complexity
    - eci : vettore country fitness

    NB su save_delta: i Delta salvati accumulano una normalizzazione per ogni
    iterazione successiva alla loro creazione, quindi il Delta a distanza k
    dall'iterazione corrente risulta diviso per k+1 fattori mean(Q).
    
    """
    # 0) Gestione INPUT
    Mcp  = Mcp.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = M.shape

    # 1) INIZIALIZZAZIONE
    F = np.ones(n_c)
    Q = np.ones(n_p)

    # *) SALVATAGGI OPZIONALI

    # *.a) CONVERGENZA
    history    = [] if track else None
    # *.b) DELTA
    # maxlen è la lunghezza massima di deque oltre la quale se aggiungo un elemento da una parte ne scarto uno dall'altra (esattamente quello che voglio)
    delta_hist = deque(maxlen=delta_window) if save_delta else None

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
        
        # *.a)
        if track:
            history.append((np.mean(np.abs(F - F_old)), np.mean(np.abs(Q - Q_old))))
        # *.b)
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
    info = {"n_iter": i}
    
    # *) Risultati (salvati in info)
    if track or save_delta:
        # *.a)
        if track:
            # due vettori con F  e Q
            info["history"] = np.array(history)
        # *.b)
        if save_delta:
            # matrice dei Delta: delta_hist[delta_window,n_p]
            info["delta_history"] = np.array(delta_hist) 
        return complexity, fitness, CQ, Delta, info

    return complexity, fitness, CQ, Delta


def compute_extended_eci_with_imports(
    Mcp_export: pd.DataFrame,
    Mcp_import: pd.DataFrame,
    C: pd.DataFrame,
    delta_window: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-6,
):
    """
    ECI esteso con import (stessa logica tensoriale di
    compute_extended_fitness_with_imports, ma con le equazioni ECI:
        F     = D^-1 (M_export Q - import_penalty)
        Delta = U^-1 M_export^T F
        Q     = C Q + Delta
    
    """

    # 0) Gestione INPUT
    
    # 0.1) Matrici
    # qui prende float ma nel caso binarizzato in realtà si tratta di int
    M_export = Mcp_export.values.astype(float)
    M_import = Mcp_import.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = Mcp_export.shape

    # 0.2) Calcolo diversity e ubiquity 
    diversification = M_export.sum(axis=1)
    ubiquity = M_export.sum(axis=0)

    D_inv = np.divide(1.0, diversification,
                       out=np.zeros_like(diversification), where=diversification != 0)
    U_inv = np.divide(1.0, ubiquity,
                       out=np.zeros_like(ubiquity), where=ubiquity != 0)

    
    # 1) Creazione Tensori
    
    # 1.a) Tensore con C
    C_tensor = np.zeros((n_p, n_p, delta_window))
    C_k = C.copy()
    for k in range(delta_window):
        C_tensor[:, :, k] = C_k
        C_k = C_k @ C
    
    # 1.b) Tensore con Mcp (export e import)
    M_import_tensor = np.repeat(M_import[:, :, np.newaxis], delta_window, axis=2)
    M_export_tensor = np.repeat(M_export[:, :, np.newaxis], delta_window, axis=2)

    
    # 2) Algoritmo
    
    # 2.1) Inizializzazione
    F     = np.ones(n_c)
    Q     = np.ones(n_p)
    Delta = np.ones((n_c, n_p, delta_window))
    
    # 2.2) Sistema dinamico
    for _ in range(max_iter):
        F_old = F.copy()
        Q_old = Q.copy()

        # 2.2.1) ECI 
        # 2.2.1.a) Calcolo import penalty
        k = 0
        eci_imported_input_penalty = 0
        for k in range(delta_window):

            # FILTRO 1: import
            import_filtered = Delta[:, :, k] * M_import_tensor[:, :, k]
            # Per ogni prodotto (e paese e ordine di distanza) prendo i delta importati
            input_imported = import_filtered @ C_tensor[:, :, k]
            # FILTRO 2: export
            export_filtered = (input_imported * M_export_tensor[:, :, k])
            # export_filtered: mi dice per ogni prodotto p ESPORTATO dal paese c la somma delle complessità degli input importati
            # per calcolare lo sconto alla fitness devo sommare:
            # a) su tutti i prodotti
            export_filtered = export_filtered.sum(axis=1)
            # b) su tutti gli ordini di distanza
            eci_imported_input_penalty += export_filtered
        
        F = D_inv * (M_export @ Q_old - eci_imported_input_penalty)

        # 2.2.2) Complessità intrinseca Delta
        Delta_new = U_inv * (M_export.T @ F_old)
        mean_Delta_new = np.mean(Delta_new)
        Delta_new = Delta_new / mean_Delta_new
        
        # 2.2.3) Complessità totale Q
        CQ = C @ Q
        Q  = CQ + Delta_new
        
        # 2.3) Normalizzazione
        mean_Q    = np.mean(Q)
        F         = F / mean_Q
        CQ        = CQ / mean_Q
        Q         = Q / mean_Q
        
        
        # 2.4) Procedura Delta

        # shifto (lasciando vuota la posizione 0) e normalizzo
        Delta[:, :, 1:] = Delta[:, :, :-1] / mean_Q

        # fillo la posizione 0 con il valore di delta_new
        Delta[:, :, 0] = Delta_new / mean_Q
       # 3) Convergenza
        if np.all(np.abs(F - F_old) < tol) and np.all(np.abs(Q - Q_old) < tol):
            break
    fitness              = pd.Series(F,         index=Mcp_export.index,   name="Fitness")
    complexity           = pd.Series(Q,         index=Mcp_export.columns, name="Complexity")
    input_complexity     = pd.Series(CQ,        index=Mcp_export.columns, name="Input Complexity")
    intrinsic_complexity = pd.Series(Delta_new, index=Mcp_export.columns, name="Intrinsic Complexity")

    return complexity, fitness, input_complexity, intrinsic_complexity

#-----------------------------------------------------------------------------------------------------------------------------------





# 2) FITNESS

# 2.0.0.0) 

# matrice C: NO
# delta: NO
# import: NO

def compute_fitness(Mcp: pd.DataFrame, max_iter: int = 1000, tol: float = 1e-6,
                    track: bool = True):
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

        if track:
            history.append((dF.mean(), dQ.mean()))

        # controllo convergenza
        if np.all(dF < tol) and np.all(dQ < tol):
            converged = True
            break

    fitness = pd.Series(F, index=Mcp.index, name="Fitness")
    complexity = pd.Series(Q, index=Mcp.columns, name="Complexity")

    if track:
        info = {
            "history": np.array(history),
            "n_iter": i,
            "converged": converged,
        }
        return fitness, complexity, info

    return fitness, complexity






# 2.1.0.0) 

# matrice C: SI
# delta: NO
# import: NO 

import numpy as np
import pandas as pd
from collections import deque   # se non già importato


def compute_extended_fitness(
    Mcp: pd.DataFrame,
    C: pd.DataFrame,
    alpha: float,
    max_iter: int = 100000,
    tol: float = 1e-15,
    track: bool = True,
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
    
    - track: variabile booleana per vedere o meno come la funzione converge passo dopo passo
    - save_delta: variabile booleana per salvare o meno i delta (normalizzati) ai passi precedenti
    - delta_window: numero di passi precedenti in cui salvare i delta
    
    Output:
    - complexity : vettore product complexity
    - fitness : vettore country fitness

    NB su save_delta: i Delta salvati accumulano una normalizzazione per ogni
    iterazione successiva alla loro creazione, quindi il Delta a distanza k
    dall'iterazione corrente risulta diviso per k+1 fattori mean(Q).
    
    """
    # 0) Gestione INPUT
    M  = Mcp.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = M.shape

    # 1) INIZIALIZZAZIONE
    F = np.ones(n_c)
    Q = np.ones(n_p)

    # *) SALVATAGGI OPZIONALI

    # *.a) CONVERGENZA
    history    = [] if track else None
    # *.b) DELTA
    # maxlen è la lunghezza massima di deque oltre la quale se aggiungo un elemento da una parte ne scarto uno dall'altra (esattamente quello che voglio)
    delta_hist = deque(maxlen=delta_window) if save_delta else None

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
        
        # *.a)
        if track:
            history.append((np.mean(np.abs(F - F_old)), np.mean(np.abs(Q - Q_old))))
        # *.b)
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
    info = {"n_iter": i}
    
    # *) Risultati (salvati in info)
    if track or save_delta:
        # *.a)
        if track:
            # due vettori con F  e Q
            info["history"] = np.array(history)
        # *.b)
        if save_delta:
            # matrice dei Delta: delta_hist[delta_window,n_p]
            info["delta_history"] = np.array(delta_hist) 
        return complexity, fitness, CQ, Delta, info

    return complexity, fitness, CQ, Delta



# 2.1.0.1) fitness con filtro import
# matrice C: SI
# delta: NO
# import: SI

def compute_extended_fitness_with_imports(
    Mcp_export: pd.DataFrame,
    Mcp_import: pd.DataFrame,
    C: pd.DataFrame,
    delta_window: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-6,
):
    
    
    # 0) Gestione INPUT

    # 0.1) Matrici
    # qui prende float ma nel caso binarizzato in realtà si tratta di int
    M_export = Mcp_export.values.astype(float)
    M_import = Mcp_import.values.astype(float)
    C = C.values.astype(float)

    n_c, n_p = Mcp_export.shape

    # 1) Creazione Tensori

    # 1.a) Tensore con C
    
    C_tensor = np.zeros((n_p, n_p, delta_window))

    C_k = C.copy()
    for k in range(delta_window):
        C_tensor[:, :, k] = C_k
        C_k = C_k @ C

    # 1.b) Tensore con Mcp (export e import)
    M_import_tensor = np.repeat(M_import[:, :, np.newaxis], delta_window, axis=2)
    M_export_tensor = np.repeat(M_export[:, :, np.newaxis], delta_window, axis=2)
    
    # 2) Algoritmo

    # 2.1) Inizializzazione
    F     = np.ones(n_c)
    Q     = np.ones(n_p)
    Delta = np.ones((n_c, n_p, delta_window))

    # 2.2) Sistema dinamico
    for _ in range(max_iter):
        F_old = F.copy()
        Q_old = Q.copy()
    
    
        # 2.2.1) Fitness F

        # 2.2.1.a) Calcolo import penalty
        k = 0
        fitness_imported_input_penalty = 0
        for k in range(delta_window):
        
            # FILTRO 1: import
            import_filtered = Delta[:, :, k] * M_import_tensor[:, :, k]

            # Per ogni prodotto (e paese e ordine di distanza) prendo i delta importati
            input_imported = import_filtered @ C_tensor[:, :, k]

            # FILTRO 2: export  
            export_filtered = (input_imported * M_export_tensor[:, :, k])

            # export_filtered: mi dice per ogni prodotto p ESPORTATO dal paese c la somma delle complessità degli input importati
            # per calcolare lo sconto alla fitness devo sommare:
            # a) su tutti i prodotti
            export_filtered = export_filtered.sum(axis = 1)
            # b) su tutti gli ordini di distanza
            fitness_imported_input_penalty += export_filtered


        F = M_export @ Q_old - fitness_imported_input_penalty

        
        # 2.2.2) Complessità intrinseca Delta
        Delta_new = 1.0 / (M_export.T @ (1.0 / F_old)) 
        Delta_new[np.isinf(Delta_new)] = 0.0

        mean_Delta_new = np.mean(Delta_new)
        Delta_new = Delta_new / mean_Delta_new

        # 2.2.3) Complessità totale Q
        CQ = C @ Q
        Q  = CQ + Delta_new

        # 2.3) Normalizzazione
        mean_Q    = np.mean(Q)
        F         = F / mean_Q  
        CQ        = CQ / mean_Q
        Q         = Q / mean_Q

        # 2.4) Procedura Delta
        
        # shifto (lasciando vuota la posizione 0) e normalizzo 
        Delta[:, :, 1:] = Delta[:, :, :-1] / mean_Q
    
        # fillo la posizione 0 con il valore di delta_new 
        Delta[:, :, 0] = Delta_new / mean_Q



       # 3) Convergenza
        if np.all(np.abs(F - F_old) < tol) and np.all(np.abs(Q - Q_old) < tol):
            break

    fitness           = pd.Series(F,         index=Mcp_export.index,   name="Fitness")
    complexity        = pd.Series(Q,         index=Mcp_export.columns, name="Complexity")
    input_complexity  = pd.Series(CQ,        index=Mcp_export.columns, name="Input Complexity")
    intrinsic_complexity  = pd.Series(Delta_new, index=Mcp_export.columns, name="Intrinsic Complexity")
   
    return complexity, fitness, input_complexity, intrinsic_complexity





# In[ ]:

# In[5]:


