#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import deque




import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import deque


def compute_extended_fitness_with_imports(
    Mcp_export: pd.DataFrame,
    Mcp_import: pd.DataFrame,
    C: pd.DataFrame,
    alpha: float,
    tensor_window: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-12,
    track_convergence: bool = True,
    save_delta: bool = False,
    delta_window: int | None = None,
):
    """
    Fitness estesa con import/export e matrice di composizione C.

    Input:
    - alpha: 
    - tensor_window: profondita' del tensore usato nel calcolo della penalita'
      da input importati (ordini di distanza in C). 
    - delta_window 
    - track_convergence
    - save_delta

    """

    # 0) Gestione INPUT

    # 0.1) Matrici
    M_export = Mcp_export.values.astype(float)
    M_import = Mcp_import.values.astype(float)
    C = C.values.astype(float)

    n_c, n_p = Mcp_export.shape

    # 1) Creazione Tensori

    # 1.a) Matrice di propagazione B.
    # La penalita' e' un prodotto vettore-riga per matrice,
    #     sum_p' (Delta ⊙ M^I)_{c p'} (C^m)_{p' p},
    # mentre l'aggiornamento di Q e' un prodotto matrice-vettore, C @ Q: le due
    # contrazioni percorrono la matrice in direzioni opposte. Poiche' a questa
    # funzione si passa C gia' trasposta, la matrice da usare a destra nella
    # penalita' e' B = C.T, cioe' la C originale come letta dal file.
    # Le potenze di C NON vengono mai materializzate: lo schema di Horner
    # (piu' sotto) moltiplica per B un passo alla volta. Si evitano cosi' un
    # tensore (tensor_window, n_p, n_p) - 2 GB a HS6 con finestra 10 - il
    # relativo precalcolo, e la perdita di precisione dovuta a entrate ~rho^k.
    B = sp.csr_matrix(np.ascontiguousarray(C.T))

    # 2) Algoritmo

    # 2.1) Inizializzazione
    F     = np.ones(n_c)
    Q     = np.ones(n_p)
    # una slice in piu': C^(k+1) va accoppiato con Delta[:, :, k+1],
    # che ha accumulato k+2 normalizzazioni (la teoria vuole C^m con m+1)
    Delta = np.zeros((tensor_window + 1, n_c, n_p))

    # *) SALVATAGGI OPZIONALI

    # *.a) CONVERGENZA
    history_convergence = [] if track_convergence else None
    # *.b) DELTA
    delta_hist = deque(maxlen=delta_window) if save_delta else None

    # 2.2) Sistema dinamico
    i = 0
    for i in range(max_iter):
        F_old = F.copy()
        Q_old = Q.copy()

        # 2.2.1) Fitness F

        # 2.2.1.a) Calcolo import penalty (schema di Horner)
        #
        #   S = sum_(m=1..K) alpha^m (Delta[m] * M_import) B^m
        #     = alpha ( X_1 + alpha ( X_2 + ... + alpha X_K B ) B ) B
        #
        # con X_m = Delta[m] * M_import (prodotto elemento-per-elemento).
        # Ogni passo e' una sola moltiplicazione per B: nessuna potenza viene
        # costruita. Delta[m] porta gia' le m+1 normalizzazioni per mean(Q)
        # che competono all'ordine m.
        K = tensor_window

        # FILTRO 1 (import): applicato a ogni ordine separatamente
        R = Delta[K] * M_import                      # X_K
        for m in range(K - 1, 0, -1):
            R = R @ B                                # un passo a valle nella rete
            R = Delta[m] * M_import + alpha * R      # aggiungo l'import di ordine m
        S = alpha * np.asarray(R @ B)                # (n_c, n_p)

        # FILTRO 2 (export): identico per tutti gli ordini, quindi si applica
        # una volta sola alla fine; poi si somma su tutti i prodotti.
        import_penalty = (S * M_export).sum(axis=1)

        F = M_export @ Q_old - import_penalty

        # La penalita' non deve superare la fitness lorda: se accade, il
        # regime e' sbagliato (tipicamente alpha troppo grande) e tutto cio'
        # che segue (1/F) perde significato.
        # Con lo sfasamento corretto la penalita' e' vincolata: sottrae la sola
        # frazione importata dei contributi di ordine >=1 a Q, quindi
        # penalty_c < sum_p M^E_cp Q_p e F resta positiva. Una F<=0 segnala
        # un errore di scala, non un regime.
        if np.any(F <= 0):
            raise ValueError(
                f"iterazione {i}: {int((F <= 0).sum())} paesi con fitness non positiva. "
                "La penalita' da import supera la fitness lorda: controllare alpha, "
                "l'accoppiamento fra potenze di C e slice di Delta, e M_import."
            )

        # 2.2.2) Complessita' intrinseca Delta
        with np.errstate(divide="ignore", invalid="ignore"):
            Delta_new = 1.0 / (M_export.T @ (1.0 / F_old))
        Delta_new[~np.isfinite(Delta_new)] = 0.0

        mean_Delta_new = np.mean(Delta_new)
        Delta_new = Delta_new / mean_Delta_new

        # 2.2.3) Complessita' totale Q
        CQ = alpha * (C @ Q_old)
        Q  = CQ + Delta_new

        # 2.3) Normalizzazione
        mean_Q = np.mean(Q)
        Delta_out = Delta_new / mean_Q
        CQ        = CQ / mean_Q
        Q         = Q / mean_Q
        F         = F / np.mean(F)

        # 2.4) Procedura Delta

        # shifto (lasciando vuota la posizione 0) e normalizzo
        Delta[1:] = Delta[:-1] / mean_Q

        # fillo la posizione 0 con il valore di delta_new
        Delta[0]  = Delta_out

        # *.a) CONVERGENZA
        if track_convergence:
            history_convergence.append(
                (np.mean(np.abs(F - F_old)), np.mean(np.abs(Q - Q_old)))
            )

        # *.b) DELTA (storico per output, indipendente dal tensore interno)
        if save_delta:
            for d in delta_hist:
                d /= mean_Q
            delta_hist.append(Delta[0].copy())

        # 3) Convergenza (criterio relativo)
        if (np.max(np.abs(Q - Q_old) / np.maximum(Q, 1e-300)) < tol
                and np.max(np.abs(F - F_old) / np.maximum(F, 1e-300)) < tol):
            break

    fitness              = pd.Series(F,         index=Mcp_export.index,   name="Fitness")
    import_penalty       = pd.Series(import_penalty, index=Mcp_export.index,   name="Import penalty")
    complexity           = pd.Series(Q,         index=Mcp_export.columns, name="Complexity")
    input_complexity     = pd.Series(CQ,        index=Mcp_export.columns, name="Input Complexity")
    intrinsic_complexity = pd.Series(Delta_out, index=Mcp_export.columns, name="Intrinsic Complexity")

    # *) Risultati opzionali
    info_convergence = None
    delta_history = None

    if track_convergence:
        info_convergence = {"n_iter": i, "mean_Q": mean_Q,
                            "history": np.array(history_convergence)}
    if save_delta:
        delta_history = np.array(delta_hist)

    if track_convergence or save_delta:
        return (complexity, fitness, input_complexity, intrinsic_complexity, import_penalty,
                info_convergence, delta_history)

    return complexity, fitness, input_complexity, intrinsic_complexity, import_penalty







def compute_extended_eci_with_imports(
    Mcp_export: pd.DataFrame,
    Mcp_import: pd.DataFrame,
    C: pd.DataFrame,
    alpha: float,
    tensor_window: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-12,
    track_convergence: bool = True,
    save_delta: bool = False,
    delta_window: int | None = None,
):
    """
    ECI esteso con import (stessa logica tensoriale di
    compute_extended_fitness_with_imports, ma con le equazioni ECI):
        F     = D^-1 (M_export Q - import_penalty)
        Delta = U^-1 M_export^T F
        Q     = alpha * C Q + Delta

    NB parametri:
    - alpha: parametro di sconto che pesa il contributo di C @ Q
      nell'aggiornamento di Q. Lo STESSO sconto va applicato alla penalita' da
      input importati: la slice k del tensore C contiene C^(k+1) e va quindi
      pesata con alpha^(k+1), altrimenti si sottrae un termine calcolato su una
      rete non scontata mentre il resto usa una Q scontata.
    - tensor_window: profondita' del tensore usato nel calcolo della penalita'
      da input importati (ordini di distanza in C). NON e' lo storico dei
      Delta calcolati iterazione dopo iterazione.
    - delta_window: se save_delta=True, numero di iterazioni passate di Delta
      (quello effettivamente usato in Delta[:, :, 0], cioe' gia' normalizzato
      per mean_Q) da conservare per l'output/ispezione.
    - track_convergence: se True, salva ad ogni iterazione la coppia
      (mean |F - F_old|, mean |Q - Q_old|).
    - save_delta: se True, salva uno storico (separato dal tensore interno
      Delta) dei Delta_new normalizzati, utile per ispezione a posteriori.

    NB su save_delta: come nelle altre funzioni, i Delta salvati accumulano una
    normalizzazione aggiuntiva per ogni iterazione successiva alla loro
    creazione.
    """

    # 0) Gestione INPUT

    # 0.1) Matrici
    M_export = Mcp_export.values.astype(float)
    M_import = Mcp_import.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = Mcp_export.shape

    # 0.2) Calcolo diversity e ubiquity
    diversification = M_export.sum(axis=1)
    ubiquity        = M_export.sum(axis=0)

    D_inv = np.divide(1.0, diversification,
                      out=np.zeros_like(diversification), where=diversification != 0)
    U_inv = np.divide(1.0, ubiquity,
                      out=np.zeros_like(ubiquity), where=ubiquity != 0)

    # 1) Creazione Tensori

    # 1.a) Matrice di propagazione B.
    # La penalita' e' un prodotto vettore-riga per matrice,
    #     sum_p' (Delta ⊙ M^I)_{c p'} (C^m)_{p' p},
    # mentre l'aggiornamento di Q e' un prodotto matrice-vettore, C @ Q: le due
    # contrazioni percorrono la matrice in direzioni opposte. Poiche' a questa
    # funzione si passa C gia' trasposta, la matrice da usare a destra nella
    # penalita' e' B = C.T, cioe' la C originale come letta dal file.
    # Le potenze di C NON vengono mai materializzate: lo schema di Horner
    # (piu' sotto) moltiplica per B un passo alla volta. Si evitano cosi' un
    # tensore (tensor_window, n_p, n_p) - 2 GB a HS6 con finestra 10 - il
    # relativo precalcolo, e la perdita di precisione dovuta a entrate ~rho^k.
    B = sp.csr_matrix(np.ascontiguousarray(C.T))

    # 2) Algoritmo

    # 2.1) Inizializzazione
    F = np.ones(n_c)
    Q = np.ones(n_p)
    # zeri, non uni: alla prima iterazione non esiste ancora storia di Delta,
    # quindi la penalita' deve essere nulla e non un valore arbitrario.
    # Una slice in piu': C^(k+1) va accoppiato con Delta[:, :, k+1], che ha
    # accumulato k+2 normalizzazioni (la teoria vuole C^m con m+1).
    Delta = np.zeros((tensor_window + 1, n_c, n_p))

    # *) SALVATAGGI OPZIONALI

    # *.a) CONVERGENZA
    history_convergence = [] if track_convergence else None
    # *.b) DELTA
    delta_hist = deque(maxlen=delta_window) if save_delta else None

    # 2.2) Sistema dinamico
    i = 0
    for i in range(max_iter):
        F_old = F.copy()
        Q_old = Q.copy()

        # 2.2.1) ECI

        # 2.2.1.a) Calcolo import penalty (schema di Horner)
        #
        #   S = sum_(m=1..K) alpha^m (Delta[m] * M_import) B^m
        #     = alpha ( X_1 + alpha ( X_2 + ... + alpha X_K B ) B ) B
        #
        # con X_m = Delta[m] * M_import (prodotto elemento-per-elemento).
        # Ogni passo e' una sola moltiplicazione per B: nessuna potenza viene
        # costruita. Delta[m] porta gia' le m+1 normalizzazioni per mean(Q)
        # che competono all'ordine m.
        K = tensor_window

        # FILTRO 1 (import): applicato a ogni ordine separatamente
        R = Delta[K] * M_import                      # X_K
        for m in range(K - 1, 0, -1):
            R = R @ B                                # un passo a valle nella rete
            R = Delta[m] * M_import + alpha * R      # aggiungo l'import di ordine m
        S = alpha * np.asarray(R @ B)                # (n_c, n_p)

        # FILTRO 2 (export): identico per tutti gli ordini, quindi si applica
        # una volta sola alla fine; poi si somma su tutti i prodotti.
        import_penalty = (S * M_export).sum(axis=1)

        F = D_inv * (M_export @ Q_old - import_penalty)

        # La penalita' sottrae la sola frazione importata dei contributi di
        # ordine >=1 a Q, quindi e' minore della somma lorda e F resta positiva.
        # Una F<=0 segnala un errore di scala, non un regime.
        if np.any(F <= 0):
            raise ValueError(
                f"iterazione {i}: {int((F <= 0).sum())} paesi con ECI non positivo. "
                "La penalita' da import supera il termine lordo: controllare alpha, "
                "l'accoppiamento fra potenze di C e slice di Delta, e M_import."
            )

        # 2.2.2) Complessita' intrinseca Delta
        Delta_new = U_inv * (M_export.T @ F_old)
        mean_Delta_new = np.mean(Delta_new)
        Delta_new = Delta_new / mean_Delta_new

        # 2.2.3) Complessita' totale Q
        CQ = alpha * (C @ Q_old)
        Q  = CQ + Delta_new

        # 2.3) Normalizzazione
        mean_Q    = np.mean(Q)
        Delta_out = Delta_new / mean_Q
        CQ        = CQ / mean_Q
        Q         = Q / mean_Q
        F         = F / np.mean(F)

        # 2.4) Procedura Delta

        # shifto (lasciando vuota la posizione 0) e normalizzo
        Delta[1:] = Delta[:-1] / mean_Q

        # fillo la posizione 0 con il valore di delta_new
        Delta[0]  = Delta_out

        # *.a) CONVERGENZA
        if track_convergence:
            history_convergence.append(
                (np.mean(np.abs(F - F_old)), np.mean(np.abs(Q - Q_old)))
            )

        # *.b) DELTA (storico per output, indipendente dal tensore interno)
        if save_delta:
            for d in delta_hist:
                d /= mean_Q
            delta_hist.append(Delta[0].copy())

        # 3) Convergenza (criterio relativo)
        if (np.max(np.abs(Q - Q_old) / np.maximum(Q, 1e-300)) < tol
                and np.max(np.abs(F - F_old) / np.maximum(F, 1e-300)) < tol):
            break

    fitness              = pd.Series(F,         index=Mcp_export.index,   name="ECI")
    import_penalty       = pd.Series(import_penalty, index=Mcp_export.index,   name="Import penalty")
    complexity           = pd.Series(Q,         index=Mcp_export.columns, name="Complexity")
    input_complexity     = pd.Series(CQ,        index=Mcp_export.columns, name="Input Complexity")
    intrinsic_complexity = pd.Series(Delta_out, index=Mcp_export.columns, name="Intrinsic Complexity")

    # *) Risultati opzionali
    info_convergence = None
    delta_history = None

    if track_convergence:
        info_convergence = {"n_iter": i, "mean_Q": mean_Q,
                            "history": np.array(history_convergence)}
    if save_delta:
        delta_history = np.array(delta_hist)

    if track_convergence or save_delta:
        return (complexity, fitness, input_complexity, intrinsic_complexity, import_penalty,
                info_convergence, delta_history)

    return complexity, fitness, input_complexity, intrinsic_complexity, import_penalty,