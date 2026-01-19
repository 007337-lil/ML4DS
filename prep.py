import pandas as pd
import numpy as np
import os

sexe_map = {
    1: 'Homme', 
    2: 'Femme'
}
catu_map = {
    1: 'Conducteur', 
    2: 'Passager', 
    3: 'Piéton'
}
trajet_map = {
    -1: 'Autre', 
    0: 'Autre', 
    1: 'Travail / Ecole', 
    2: 'Travail / Ecole', 
    3: 'Autre', 
    4: 'Autre', 
    5: 'Promenade', 
    9: 'Autre'
}
catv_map = {
    1: 'Vélo', 
    80: 'VAE'
}
agg_map = {
    1: 'Hors agglomération', 
    2: 'En agglomération'
}
col_map = {
    -1: "Autre",                  
     1: "Frontale",               
     2: "Arrière",           
     3: "Côté",            
     4: "Autre",   
     5: "Autre",
     6: "Autre",                    
     7: "Sans collision"            
}
int_map = {
    1: "Hors intersection",         
    2: "X",       
    3: "T",
    4: "Autre",             
    5: "Autre",      
    6: "Giratoire",
    7: "Autre",
    8: "Autre",
    9: "Autre"
}
catr_map = {
    1: "Autre",
    2: "Autre",
    3: "Route départementale",
    4: "Voie communale",
    5: "Autre",
    6: "Autre",      
    7: "Autre",           
    9: "Autre"
}
grav_map = {
    1: "Indemne",
    2: "Tué",
    3: "Blessé grave",
    4: "Blessé léger"
}
def data_prep(df_caract, df_lieux, df_usagers, df_vehicules):
    '''
    Renvoie les données préparées et nettoyées pour la modélisation.

    Paramètres:
    - df_caract (pd.DataFrame) : table caractéristique
    - df_lieux (pd.DataFrame) : table lieux
    - df_usagers (pd.DataFrame) : table usagers
    - df_vehicules (pd.DataFrame) : tables véhicules

    Retourne:
    - cyclistes_final (pd.DataFrame) : table finale
    '''
    # Fusion des DataFrames usagers et véhicules pour identifier les cyclistes
    usagers_vehicules = pd.merge(
        df_usagers,
        df_vehicules,
        on=['Num_Acc', 'id_vehicule'],
        how='inner'
    )
    # Filtrage des cyclistes:
    # catv: 1: vélo, 80: vélo électrique
    # catu: 1: conducteur, 2: passager
    cyclistes = usagers_vehicules.loc[
        (usagers_vehicules['catv'].isin([1, 80])) &
        (usagers_vehicules['catu'].isin([1, 2]))
    ]
    # Fusion avec les caractéristiques des accidents
    cyclistes2 = pd.merge(
        cyclistes,
        df_caract,
        on='Num_Acc',
        how='inner'
    )
    # Sélection des variables
    cyclistes2 = cyclistes2[[
        'Num_Acc', 'id_usager', 'id_vehicule', 'sexe', 
        'an_nais', 'trajet', 'catu', 'secu1', 'secu2', 
        'secu3', 'catv', 'obs', 'obsm', 'manv',
        'an', 'mois', 'hrmn', 'jour', 'col', 'int',
        'agg', 'grav'
    ]]
    # Fusion avec les lieux de l'accident
    columns_to_keep = ['Num_Acc', 'catr', 'vosp', 'surf', 'vma']
    cyclistes3 = pd.merge(
        cyclistes2, 
        df_lieux[columns_to_keep], 
        on='Num_Acc', 
        how='inner'
    )
    # Gestion des doublons en gardant les informations les plus précises
    cyclistes3 = (
        cyclistes3
        .sort_values(['vosp'], ascending=False)
        .drop_duplicates(subset=['Num_Acc', 'id_usager'], keep='first')
    )
    cyclistes_final = cyclistes3.copy()
    # Nettoyage et transformation des données
    # Application des regroupements
    cyclistes_final['sexe'] = cyclistes_final['sexe'].map(sexe_map)
    cyclistes_final['catu'] = cyclistes_final['catu'].map(catu_map)
    cyclistes_final['trajet'] = cyclistes_final['trajet'].map(trajet_map)
    cyclistes_final['catv'] = cyclistes_final['catv'].map(catv_map)
    cyclistes_final['col'] = cyclistes_final['col'].replace(col_map)
    cyclistes_final['int'] = cyclistes_final['int'].replace(int_map)
    cyclistes_final['catr'] = cyclistes_final['catr'].replace(catr_map)
    cyclistes_final['grav'] = cyclistes_final['grav'].replace(grav_map)
    cyclistes_final['agg'] = cyclistes_final['agg'].replace(agg_map)
    # Transformation de l'année de naissance en âge
    cyclistes_final['age'] = cyclistes_final['an'] - cyclistes_final['an_nais']
    cyclistes_final['age'] = cyclistes_final['age'].astype('Int64')
    cyclistes_final = cyclistes_final.drop(columns=['an_nais'])
    # Transformation des variables secu1, secu2, secu3 en binaire 
    cyclistes_final['protection'] = np.where(cyclistes_final[['secu1', 'secu2', 'secu3']].isin([-1, 0]).all(axis=1), 0, 1)
    cyclistes_final.drop(columns=['secu1', 'secu2', 'secu3'], inplace=True)
    # Transformation des variables obs et obsm en binaire
    cyclistes_final['obs'] = np.where(cyclistes_final['obs'].isin([-1, 0]), 0, 1)
    cyclistes_final['obsm'] = np.where(cyclistes_final['obsm'].isin([-1, 0]), 0, 1)
    # Transformation de la variable vosp en binaire 
    cyclistes_final['vosp'] = np.where(cyclistes_final['vosp'].isin([-1, 0]), 0, 1)
    # Transformation de la variable manv en binaire
    cyclistes_final['manv'] = np.where(cyclistes_final['manv'].isin([-1, 0, 1, 2]), 0, 1)
    # Transformation de la variable surf en binaire 
    cyclistes_final['surf'] = np.where(cyclistes_final['surf'].isin([-1, 1]), 0, 1)
    # Création d'une variable date grâce aux variables an, mois, jour, hrmn
    cyclistes_final['datetime'] = pd.to_datetime(
    cyclistes_final['an'].astype(str) + '-' + 
    cyclistes_final['mois'].astype(str).str.zfill(2) + '-' + 
    cyclistes_final['jour'].astype(str).str.zfill(2) + ' ' + 
    cyclistes_final['hrmn']
    )
    cyclistes_final.drop(columns=['an', 'mois', 'jour', 'hrmn'], inplace=True)
    # Création de la variable jour_semaine : Semaine ou Week-end
    cyclistes_final['jour_semaine'] = cyclistes_final['datetime'].dt.day_name()
    cyclistes_final['jour_semaine'] = np.where(
        cyclistes_final['jour_semaine'].isin(['Saturday', 'Sunday']),
        'Week-end',
        'Semaine'
    )
    # Création de la variable moment_journee
    cyclistes_final['moment_journee'] = pd.cut(
        cyclistes_final['datetime'].dt.hour,
        bins=[-1, 5, 12, 17, 21, 24],
        labels=['Nuit', 'Matin', 'Après-midi', 'Soir', 'Nuit'],
        ordered=False
    )
    # On supprime les quelques lignes manquantes pour le sexe et l'âge
    cyclistes_final = cyclistes_final.dropna(subset=['sexe', 'age'])
    cyclistes_final.drop(columns=['Num_Acc', 'id_usager', 'id_vehicule', 'datetime'], inplace=True)
    cyclistes_final.reset_index(drop=True, inplace=True)
    # Passage des variables en catégories
    cat_features = [col for col in cyclistes_final.columns.tolist() if col not in ['vma', 'age', 'grav', 'grav_ord']]
    cyclistes_final[cat_features] = cyclistes_final[cat_features].astype('category')
    cyclistes_final['grav'] = cyclistes_final['grav'].astype('category') 
    # Ordre pour la variable cible
    ordre_grav = ['Indemne', 'Blessé léger', 'Blessé grave', 'Tué']
    cyclistes_final['grav'] = pd.Categorical(
        cyclistes_final['grav'],
        categories=ordre_grav,
        ordered=True
    )
    cyclistes_final['grav_ord'] = cyclistes_final['grav'].cat.codes
    
    return cyclistes_final