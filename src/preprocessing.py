#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import os
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root



#  1) BACI / Mcp processing

# 1.0) cartella zip (CEPII) ---> BACI (import + export) per ogni anno

def process_BACI_raw_folder(input_zip: str, output_folder: str):
    """
    Input:
    - input_zip: percorso del file .zip contenente i csv BACI (colonne t, i, j, k, v)
    - output_folder: cartella dove salvare i parquet

    Output:
    - Per ogni csv dentro lo zip crea DUE parquet:
        * BACI_export_Y<anno>.parquet  -> aggregato per esportatore (i)
        * BACI_import_Y<anno>.parquet  -> aggregato per importatore (j)
    
    
    """
    os.makedirs(output_folder, exist_ok=True)

    with zipfile.ZipFile(input_zip, 'r') as z:
        for filename in z.namelist():
            if not filename.endswith(".csv"):
                continue

            # legge il csv direttamente dallo zip, senza estrarlo, inoltre converto subiot la colonna dei prodotti in formato stringa
            with z.open(filename) as f:
                df = df = pd.read_csv(f, encoding='utf-8-sig', dtype={'k': str})

            # estrai anno dal nome file, es: "BACI_HS92_Y2024_V202601.csv"
            match = re.search(r'Y(\d{4})', filename)
            if match:
                year = int(match.group(1))
            else:
                print(f"Anno non trovato in {filename}, salto file")
                continue

            df['t'] = year

            # estrai nomenclatura dal nome file, es: "HS17", "HS92"
            match_hs = re.search(r'(HS\d{2})', filename)
            nomenclature = match_hs.group(1) if match_hs else "HSNA"

            # --- aggregazione EXPORT: per esportatore (i) ---
            df_export = (
                df.groupby(['i', 'k', 't'], as_index=False)['v']
                  .sum()
                  .rename(columns={'i': 'country'})
            )
            export_path = os.path.join(output_folder, f"BACI_export_{nomenclature}_Y{year}.parquet")
            df_export.to_parquet(export_path, index=False)

            # --- aggregazione IMPORT: per importatore (j) ---
            df_import = (
                df.groupby(['j', 'k', 't'], as_index=False)['v']
                  .sum()
                  .rename(columns={'j': 'country'})
            )
            import_path = os.path.join(output_folder, f"BACI_import_{nomenclature}_Y{year}.parquet")
            df_import.to_parquet(import_path, index=False)

            print(f"{os.path.basename(filename)}: salvati export e import per anno {year}")


# 1.1.a) BACI HS6 to BACI HS4 (singolo file parquet)

def BACI_HS6_to_HS4(BACI_HS6: pd.DataFrame):
    """
    Converte un DataFrame BACI da HS6 a HS4 (prime 4 cifre del codice k).

    Input:
    - BACI_HS6: dataframe BACI HS6 (export o import)

    Output:
    -BACI_HS4: dataframe BACI HS6 (export o import)
    
    """
    
    df_HS6 = BACI_HS6.copy()
    # individua automaticamente la colonna del paese ('country', 'i' o 'j')
    country_col = next(c for c in ['country', 'i', 'j'] if c in df.columns)
    # prime 4 cifre del codice prodotto (zfill per non perdere gli zeri iniziali)
    df_HS4['k'] = df_HS6['k'].astype(str).str.zfill(6).str[:4]
    # riaggrega sommando v
    df_HS4 = df.groupby([country_col, 't', 'k'], as_index=False)['v'].sum()
    BACI_HS4 = df_HS4
    
    return BACI_HS4





# 1.1.b) BACI HS6 to BACI HS4 (intera cartella)

def process_BACI_HS6_folder_to_HS4(input_folder: str, output_folder: str):
    """
    Input:
    - input_folder: cartella con i parquet BACI a 6 digit
    - output_folder: cartella dove salvare i parquet BACI a 4 digit
    
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".parquet"):
            continue

        in_path = os.path.join(input_folder, filename)
        df = pd.read_parquet(in_path)

        df_hs4 = BACI_HS6_to_HS4(df)

        out_path = os.path.join(output_folder, filename)
        df_hs4.to_parquet(out_path, index=False)
        print(f"{filename}: convertito HS6 -> HS4")




# 1.2.a) BACI  ---> Mcp binarizzato (singolo file)

def build_Mcp_from_BACI(
    BACI_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Input:
    - df: dataframe BACI relativo a UN SOLO anno, con export/import per prodotto per ogni nazione 

    Output:
    - Matrice Mcp (country x product), valori 0/1
    
    """
    df = BACI_df.copy()

    country_tot = df.groupby('country')['v'].transform('sum')
    product_tot = df.groupby('k')['v'].transform('sum')
    total = df['v'].sum()

    df['RCA'] = (df['v'] / country_tot) / (product_tot / total)
    df['Mcp'] = (df['RCA'] >= 1).astype(int)

    matrix = df.pivot_table(
        index= 'country',
        columns= 'k',
        values='Mcp',
        fill_value=0
    ).astype(int)
    return matrix



# 1.2.b) BACI  ---> Mcp binarizzato (singolo file)

def build_Mcp_non_binarized_from_BACI(
    BACI_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Input:
    - BACI_df: dataframe BACI relativo a UN SOLO anno, con export/import per prodotto per ogni nazione 

    Output:
    - Matrice Mcp (country x product), valori reali
    
    """
    df = BACI_df.copy()

    country_tot = df.groupby('country')['v'].transform('sum')
    product_tot = df.groupby('k')['v'].transform('sum')
    total = df['v'].sum()

    df['Mcp'] = (df['v'] / country_tot) / (product_tot / total)
    

    matrix = df.pivot_table(
        index= 'country',
        columns= 'k',
        values='Mcp',
        fill_value=0
    ).astype(float)
    return matrix




# 1.2) BACI ---> Mcp (intera cartella)

def Mcp_from_BACI_pipeline(
    BACI_folder: str,
    Mcp_folder: str,
) :
    """
    Converte tutti i file di una cartella in Mcp
    
    Input:
    - BACI_folder: cartella con i file BACI
    - Mcp_folder: cartella in cui salvare i file Mcp
    
    """
    os.makedirs(Mcp_folder, exist_ok=True)

    for filename in os.listdir(BACI_folder):
        if not filename.endswith(".parquet"):
            continue

        in_path = os.path.join(BACI_folder, filename)
        BACI_df = pd.read_parquet(in_path)

        Mcp = build_Mcp_from_BACI(BACI_df)

        # Sostituisco il prefisso BACI -> Mcp nel nome file
        out_filename = "Mcp" + filename[len("BACI"):]
        
        out_path = os.path.join(Mcp_folder, out_filename)
        Mcp.to_parquet(out_path, index=False)
        




# 2) AIPNET / C processing


# 2.1) AIPNET to C adiancency matrix

def AIPNET_to_C(AIPNET_df: pd.DataFrame, 
                upstream_product_column_name: str, 
                downstream_product_column_name: str, 
                N) -> pd.DataFrame:

    """
    - Questa funzione mi permette di convertire i dati AIPNET in una matrice di composizione binaria C_pp' (p è input di p')
    - Da applicare a ciascuno dei 10 database AIPNET (4/6-digits x HS02/07/12/17/22)
     
    Input:
    - AIPNET_df: Dataframe con i dati AIPNET 
    - upstream_product_column_name : nome della colonna coi prodotti upstream
    - downstream_product_column_name: nome della colonna coi prodotti downstream
    - N: numero di cifre della classificazione HS che voglio ---> mi serve per mantenere lo 0 davanti in caso di necessità

    Output:
    - C: matrice di composizione
    
    """
    
    # creo una copia  
    C = AIPNET_df.copy()
    
    # converto gli indici delle due colonne in stringhe a n cifre (nel caso fillo con lo 0 davanti)
    C[upstream_product_column_name] = C[upstream_product_column_name].astype(str).str.zfill(N)  
    C[downstream_product_column_name] = C[downstream_product_column_name].astype(str).str.zfill(N)
    
    # creo la matrice di adiacenza (il numero di righe e colonne sarà diverso in quanto non vengono considerati prodotti puramente upstream/downstream)
    C = pd.crosstab(C[upstream_product_column_name], C[downstream_product_column_name])
    
    # aggiungo anche prodotti puramente upstream/downstream
    all_codes = sorted(set(C.index) | set(C.columns))
    C = C.reindex(index=all_codes, columns=all_codes, fill_value=0)

    return C






#3) ALLINEAMENTO MATRICI C E Mcp per poter runnare gli algoritmi extended
    
#3.1) Funzione per paragonare i codici prodotto di M e C (singoli file)

def M_and_C_different_products(M: pd.DataFrame, C: pd.DataFrame,
                             verbose: bool = True) -> dict:
    """
    Questa funzione confronta i codici prodotto della matrice M e C

    Input:
    - M, C          : le due matrici (DataFrame) che vado a comparare
    - name_M, name_C: nomi descrittivi per la stampa
    
    Output :
    - n_codes_A, n_codes_B, n_common: informazioni che possono servire
    - only_M : codici presenti in M ma non in C
    - only_C : codici presenti in C ma non in M
    - common : codici in comune
    
    """
    
    # 1) CODICI PRODOTTO
    M_codes =  set(pd.unique(M.columns.astype(str)))
    C_codes =  set(pd.unique(C.columns.astype(str)))

    # 2) CONFRONTO CODICI PRODOTTO
    common = M_codes & C_codes
    only_M = M_codes - C_codes   
    only_C = C_codes - M_codes
     

    
    # INTERRUTTORE di stampa (opzionale)
    if verbose:
        print(f"  [Mcp  vs  C]")
        print(f"     Codici in Mcp: {len(M_codes)}  |  Codici in C: {len(C_codes)}")
        print(f"     In comune: {len(common)}")
        print(f"     Solo in Mcp (mancanti in C): {len(only_M)} -> {sorted(only_M)}")
        print(f"     Solo in C (mancanti in Mcp): {len(only_C)} -> {sorted(only_C)}")

    
    return {
        "numero_codici_M": len(M_codes),
        "numero_codici_C": len(C_codes),
        "numero_prodotti_comuni": len(common),
        "prodotti_in_comune": common,
        "only_M": only_M,
        "only_C": only_C,
        
    }
    
#3.2) Calcolo impatto economico codici che devo escludere da Mcp

def missing_in_C_codes_economic_impact(missing, BACI_df ):
    """
    Input:
    - missing: lista di codici presenti in BACI ma non in AIPNET
    - BACI_df: dataframe con i dati di export aggregati

    Output:
    - Impatto economico dei codici non presenti in AIPNET 

    """
    
    # filtro i prodotti di interesse
    
    BACI_df_diff = BACI_df[BACI_df["k"].isin(missing)]  # isin() --- true or false   controlla quali righe hanno quei codici prodotto

    # export totale dei codici selezionati
    export_diff = BACI_df_diff["v"].sum()

    # export totale complessivo
    export_totale = BACI_df["v"].sum()
 
    # quota percentuale
    quota = export_diff / export_totale

    print("Export codici selezionati:", export_diff)
    print("Export totale:", export_totale)
    print("Quota:", quota)

    return export_diff, export_totale, quota




def pipeline (M: pd.DataFrame, C: pd.DataFrame, BACI: pd.DataFrame)-> pd.DataFrame :
    """
    Questa funzione crea la matrice Mcp allineata con C per il singolo anno.

    Input:
    - M, C: 
    - BACI: 
    
    Output :
    - Mcp_alligned: matrice Mcp su cui posso runnare gli algoritmi extended
    
    """
    

    # 1) PRODOTTI DIVERSI TRA C e M
    results = M_and_C_different_products(M, C, verbose = True) 
    # 1.a) Prodotti solo in M ---> sono quelli che dovrò eliminare e di cui calcolo l'impatto economico
    only_M = results["only_M"]
    # 1.b) Prodotti su calcolo Mcp (i prodotti in comune tra C  e Mcp) 
    M_and_C_common_codes = results["prodotti_in_comune"]

    # 2) IMPATTO ECONOMICO CODICI CHE DEVO SCARTARE
    missing_in_C_codes_economic_impact(only_M, BACI)
    
    # 3) MATRICE Mcp ALLINEATA 
    
    # 3.1) costruisco Mcp a partire da BACI con i codici in comune
    BACI_alligned = BACI[BACI["k"].isin(M_and_C_common_codes)]
    Mcp = build_Mcp_from_BACI(BACI_alligned)   # !!!!!

    # 3.2) Aggiungo le colonne (di 0) con i only_C per completare l'allineamento
    only_C = results["only_C"]
    all_cols = list(Mcp.columns) + [p for p in only_C ]
    Mcp_alligned = Mcp.reindex(columns= C.columns, fill_value=0)
    
    return Mcp_alligned




def pipeline_completa(
    M: pd.DataFrame,
    C: pd.DataFrame,
    BACI: pd.DataFrame,
    binarized: bool = True
) -> pd.DataFrame:
    """
    Questa funzione crea la matrice Mcp allineata con C per il singolo anno.
    
    Input:
    - M, C, BACI: matrici di input
    - binarized: se True usa build_Mcp_from_BACI (versione binarizzata),
                 se False usa build_Mcp_non_binarized_from_BACI (versione non binarizzata)
    
    Output:
    - Mcp_alligned: matrice Mcp su cui posso runnare gli algoritmi extended
    """
    
    # 1) PRODOTTI DIVERSI TRA C e M
    results = M_and_C_different_products(M, C, verbose=True)
    # 1.a) Prodotti solo in M ---> sono quelli che dovrò eliminare e di cui calcolo l'impatto economico
    only_M = results["only_M"]
    # 1.b) Prodotti su calcolo Mcp (i prodotti in comune tra C e Mcp)
    M_and_C_common_codes = results["prodotti_in_comune"]
    
    # 2) IMPATTO ECONOMICO CODICI CHE DEVO SCARTARE
    missing_in_C_codes_economic_impact(only_M, BACI)
    
    # 3) MATRICE Mcp ALLINEATA
    
    # 3.1) costruisco Mcp a partire da BACI con i codici in comune
    BACI_alligned = BACI[BACI["k"].isin(M_and_C_common_codes)]
    
    if binarized:
        Mcp = build_Mcp_from_BACI(BACI_alligned)
    else:
        Mcp = build_Mcp_non_binarized_from_BACI(BACI_alligned)
    
    # 3.2) Aggiungo le colonne (di 0) con i only_C per completare l'allineamento
    only_C = results["only_C"]
    all_cols = list(Mcp.columns) + [p for p in only_C ]
    Mcp_alligned = Mcp.reindex(columns=C.columns, fill_value=0)
    
    return Mcp_alligned





def run_pipeline_all_years(
    M_folder: Path,
    BACI_folder: Path,
    C: pd.DataFrame
) -> dict:
    """
    Esegue pipeline() per ogni anno disponibile.
    Input:
    - M_folder, BACI_folder: cartelle con le matrici
    - C: dataframe con la matrice C (unica)
    
    Output :
    - Salva la matrice Mcp_alligned per ogni anno in una cartella

    NOTA BENE:
    A seconda del caso vanno modificate le seguenti parti:
    - BACI_path
    - salvataggio
    
    """
    results = {}


    # Cerco dentro la cartella Mcp tutti i file che finiscono in .parquet e uso sorted per ordinarli (comodo)
    for M_path in sorted(M_folder.glob("*.parquet")):

        # Cerco i 4 numeri dopo Y 
        match = re.search(r"Y(\d{4})", M_path.name)
        
        # group (0) mi darebbe anche la Y
        year = int(match.group(1))
        
        BACI_path = BACI_folder / f"BACI_export_HS02_Y{year}.parquet"

        M = pd.read_parquet(M_path)
        BACI = pd.read_parquet(BACI_path)

        Mcp = pipeline(M, C, BACI)
        
        # salvataggio (opzionale)
        Mcp.to_parquet(ROOT / "data" / "HS02_all_years" / "6_digits" / "Mcp_export_non_binarized_alligned" / f"Mcp_export_alligned_Y{year}.parquet")


    return Mcp

def run_pipeline_all_years_completa(
    M_folder: Path,
    BACI_folder: Path,
    C: pd.DataFrame,
    output_folder: Path,
    trade_type: str = "export",
    binarized: bool = True
) -> dict:
    """
    Esegue pipeline() per ogni anno disponibile.
    
    Input:
    - M_folder, BACI_folder: cartelle con le matrici
    - C: dataframe con la matrice C (unica)
    - output_folder: cartella dove salvare i file di output
    - trade_type: "export" o "import"
    
    Output:
    - Salva la matrice Mcp_alligned per ogni anno nella cartella specificata
    - Ritorna un dizionario {anno: Mcp}
    """
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Cerco dentro la cartella M tutti i file che finiscono in .parquet e li ordino
    for M_path in sorted(M_folder.glob("*.parquet")):
        # Cerco i 4 numeri dopo Y
        match = re.search(r"Y(\d{4})", M_path.name)
        
        year = int(match.group(1))
        
        BACI_path = BACI_folder / f"BACI_{trade_type}_HS02_Y{year}.parquet"
        
        M = pd.read_parquet(M_path)
        BACI = pd.read_parquet(BACI_path)
        
        Mcp = pipeline_completa(M, C, BACI, binarized = binarized)
        
        # salvataggio
        save_path = output_folder / f"Mcp_{trade_type}_alligned_Y{year}.parquet"
        Mcp.to_parquet(save_path)
        
        results[year] = Mcp
    
    return results
   
