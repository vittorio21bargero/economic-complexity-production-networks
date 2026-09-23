#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.sparse.csgraph import connected_components


def autovettore_and_autovalore_principale(C_df):
    """rho e autovettore principale (destro) di C, senza calcolare tutto lo spettro."""
    A = sp.csr_matrix(np.asarray(C_df.values if hasattr(C_df, 'values') else C_df, dtype=float))
    val, vec = eigs(A, k=1, which='LM')
    rho = float(np.abs(val[0]))
    v = np.abs(np.real(vec[:, 0]))
    return rho, v / v.mean()          # normalizzato a media 1, si potrebbe anche evitare ma ci sta


def indegree_medio(C_df):
    """
    Calcola l'indegree medio della matrice C.
    
    """
    C = C_df.values
    
    return C.sum(axis=1).mean()




def is_C_nilpotente(C: pd.DataFrame):
    """
    C e' nilpotente <=> il grafo diretto associato e' aciclico.


    Input:
    - C : matrice di composizione

    Output:
    - bool: mi dice se la matrice è nilpotente o meno
    
    """
    
    M = C.values.astype(int) 

    if np.any(np.diag(M) != 0):        # nel caso C contenga un self loop allora non è nilpotente 
        return False                   # questo non possiamo piu vederlo dopo

    A = sp.csr_matrix(M == 1)  # mi interessa solo dove ci sono gli archi

    # numero di componenti fortemente connesse
    n_comp, _ = connected_components(A, directed=True, connection='strong') 

    # se ogni nodo è una componente fortemente connessa allora la matrice è nilpotente
    return n_comp == A.shape[0]


