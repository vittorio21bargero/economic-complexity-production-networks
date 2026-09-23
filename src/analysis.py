#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd


def peso_K_ordini(
    complexity,
    delta_history,
    C: pd.DataFrame,
    alpha: float,
    K: int = 5,
):
    """
    Calcola quanto incide in media ogni ordine k sulla complessita' totale dei
    prodotti.

    Input:
    - complexity: Series con la total_complexity calcolata dall'algoritmo
    - delta_history: output `delta_history` dell'algoritmo. Nelle funzioni
      con import ha shape (n_salvate, n_c, n_p): le righe sono identiche fra
      paesi (il Delta e' clonato lungo l'asse dei paesi), quindi si prende una
      riga qualsiasi. Nelle funzioni senza import ha shape (n_salvate, n_p).
    - C: matrice di composizione NON trasposta, con la stessa convenzione usata
      nella chiamata all'algoritmo (a cui si passa C.T).
    - alpha: DEVE essere lo stesso alpha usato nel run che ha prodotto
      delta_history, altrimenti i pesi non hanno significato.
    - K: numero di ordini da calcolare. Serve K <= len(delta_history), quindi
      il run va fatto con save_delta=True e delta_window >= K.

    Output:
    - pesi_df: DataFrame con "ordine", "peso_medio", "peso_cumulativo".

    NB: non serve alcuna divisione aggiuntiva per mean(Q). I Delta salvati
    hanno gia' accumulato una normalizzazione per ogni iterazione successiva
    alla loro creazione, quindi delta_history[-(k+1)] porta esattamente i k+1
    fattori mean(Q) che competono all'ordine k.
    """
    dh = np.asarray(delta_history)
    if dh.ndim == 3:                 # (n_salvate, n_c, n_p) -> righe identiche
        dh = dh[:, 0, :]
    n_salvate, n_p = dh.shape

    if K > n_salvate:
        raise ValueError(
            f"K={K} ma delta_history contiene solo {n_salvate} iterazioni. "
            f"Rilancia l'algoritmo con save_delta=True e delta_window >= {K}."
        )

    Cm = C.values.astype(float)
    C_code = Cm.T                    # la stessa matrice passata all'algoritmo

    Delta_matrix = np.zeros((n_p, K))
    for k in range(K):
        # -(k+1): la riga -1 e' il Delta piu' recente, la -2 quello di un passo
        # prima, ecc. Il Delta a distanza k ha gia' k+1 normalizzazioni.
        v = dh[-(k + 1)].copy()
        for _ in range(k):           # applica C_code k volte, senza matrix_power
            v = C_code @ v
        Delta_matrix[:, k] = (alpha ** k) * v

    mean_Q = float(np.mean(np.asarray(complexity)))
    peso_medio_ordine = Delta_matrix.mean(axis=0) / mean_Q

    pesi_df = pd.DataFrame({
        "ordine": range(K),
        "peso_medio": peso_medio_ordine,
    })
    pesi_df["peso_cumulativo"] = pesi_df["peso_medio"].cumsum()

    return pesi_df, Delta_matrix