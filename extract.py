import pandas as pd
import os

def extract_data(base_folder="DATA"):
    '''
    Extrait les données depuis les CSV stockés dans le dossier DATA
    '''
    dfs_caract = []
    dfs_lieux = []
    dfs_usagers = []
    dfs_vehicules = []

    for dossier in os.listdir(base_folder):
        chemin_dossier = os.path.join(base_folder, dossier)
        if not os.path.isdir(chemin_dossier):
            continue

        for fichier in os.listdir(chemin_dossier):
            chemin_fichier = os.path.join(chemin_dossier, fichier)
            if not fichier.endswith(".csv"):
                continue
            df = pd.read_csv(chemin_fichier, sep=';')
            df['annee'] = dossier
            if fichier.startswith("caract"):
                dfs_caract.append(df)
            elif fichier.startswith("lieux"):
                dfs_lieux.append(df)
            elif fichier.startswith("usagers"):
                dfs_usagers.append(df)
            elif fichier.startswith("vehicules"):
                dfs_vehicules.append(df)

    df_caract = pd.concat(dfs_caract, ignore_index=True) if dfs_caract else pd.DataFrame()
    df_lieux = pd.concat(dfs_lieux, ignore_index=True) if dfs_lieux else pd.DataFrame()
    df_usagers = pd.concat(dfs_usagers, ignore_index=True) if dfs_usagers else pd.DataFrame()
    df_vehicules = pd.concat(dfs_vehicules, ignore_index=True) if dfs_vehicules else pd.DataFrame()

    return df_caract, df_lieux, df_usagers, df_vehicules
