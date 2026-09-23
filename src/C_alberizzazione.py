#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

def eades_order(C):
    """
    Da capire: per ora teniamoci il fatto che rende C nilpotente
    """
    
    A = (C == 1)
    n = A.shape[0]
    alive = np.ones(n, bool)
    indeg  = A.sum(1).astype(float)   # A[i,:] = input di i
    outdeg = A.sum(0).astype(float)   # A[:,i] = prodotti di cui i e' input
    s_head, s_tail = [], []
    while alive.any():
        moved = True
        while moved:
            moved = False
            for v in np.where(alive & (outdeg == 0))[0]:   # pozzi -> coda
                if alive[v]:
                    s_tail.append(v); alive[v] = False
                    indeg[A[:, v]] -= 1; outdeg[A[v, :]] -= 1; moved = True
            for v in np.where(alive & (indeg == 0))[0]:    # sorgenti -> testa
                if alive[v]:
                    s_head.append(v); alive[v] = False
                    indeg[A[:, v]] -= 1; outdeg[A[v, :]] -= 1; moved = True
        if alive.any():
            score = np.where(alive, outdeg - indeg, -np.inf)
            v = int(np.argmax(score))
            s_head.append(v); alive[v] = False
            indeg[A[:, v]] -= 1; outdeg[A[v, :]] -= 1
    return np.array(s_head + s_tail[::-1])


def make_C_nilpotente(C: pd.DataFrame):
    """
    Da capire: per ora teniamoci il fatto che rende C nilpotente
    """
    
    C_mat = C.values.astype(int)
    n = C.shape[0]
    order = eades_order(C_mat)
    pos = np.empty(n, int); pos[order] = np.arange(n)
    keep = pos[None, :] < pos[:, None]          # keep[i,j] = (pos[j] < pos[i])
    removed = (C_mat == 1) & (~keep)
    C_nilpotente = C_mat * keep
    C_nilpotente_df = pd.DataFrame(C_nilpotente, index=C.index, columns=C.columns) 
    return C_nilpotente_df