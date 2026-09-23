#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import numpy as np

def C_perturbation(C: pd.DataFrame, eta: float, seed: int | None = None):
    """
    LINK REWIRING
    Prende una matrice C e per ogni prodotto (colonna) ridireziona (rewire)
    una frazione eta dei suoi link in input verso altri prodotti scelti a caso.

    Input:
    - C: matrice di composizione (convenzione Cp'p: riga p' input di colonna p)
    - eta: frazione dei link in input da rewirare, per ogni prodotto (0 <= eta <= 1)
    - seed: opzionale, per riproducibilità

    Output:
    - C_rewired: nuova matrice con i link ridirezionati
    
    """
    # 0) GESTIONE 
    
    # 0.1) RANDOM SEED
    rng = np.random.default_rng(seed)
    
    # 0.2) VARIE
    C_mat = C.values.astype(float).copy()
    n = C_mat.shape[0]
    C_rewired = C_mat.copy()

    # 1) ALGORITMO 
    
    # iterazione sulle colonne
    for p in range(C_mat.shape[1]):
        colonna = C_rewired[:, p]

        # 1.a) indici di riga degli input attuali (righe p' con link = 1)
        input_attuali = np.where(colonna != 0)[0]
        n_link = len(input_attuali)

        # 1.b) numero di link da riwire
        n_rewire = int(np.floor(eta * n_link))
        if n_rewire == 0:
            continue

        # 1.c) scelta link da rimuovere
        da_rimuovere = rng.choice(input_attuali, size=n_rewire, replace=False) # rng choice (array, size = n, replace = False): estrae n elementi a caso da un array (False ---> senza rimpiazzo)


        # 1.d) scelta link da aggiungere
        
        # candidati per i nuovi link: righe che NON sono già input di p + p stesso
        gia_input = set(np.where(colonna != 0)[0])
        candidati = [i for i in range(n) if i not in gia_input and i != p]

        nuovi_input = rng.choice(candidati, size=n_rewire, replace=False)

        # 1.e) applicazione del rewiring
        C_rewired[da_rimuovere, p] = 0
        C_rewired[nuovi_input, p] = 1


    return pd.DataFrame(C_rewired, index=C.index, columns=C.columns)



def Mcp_perturbation(Mcp: pd.DataFrame, eta, seed=None):
    """
    Flippa ogni entrata della matrice Mcp con probabilita' eta. 
    Stile Vito nel suo paper : "A New and Stable Estimation Method of Country
    Economic Fitness and Product Complexity"
    - Il flip è perciò simmetrico ---> se aumento eta aumenta anche la densità della matrice

    Input:
    - Mcp : export o import
    - eta : probabilità di un entrata di flippare
    - seed : per riproducibilità

    Output:
    - Mcp_flipped
    
    """
    M = Mcp.values.astype(int) 
    rng = np.random.default_rng(seed)
    # Indici della matrice da flippare
    flip = rng.random(M.shape) < eta # True se il valore random è minore di eta
    
    M_flipped = M.copy()
    M_flipped[flip] = 1.0 - M_flipped[flip] # =1 se Mcp = 0  | =0 se Mcp = 1
    
    return pd.DataFrame(M_flipped, index=Mcp.index, columns=Mcp.columns) 

