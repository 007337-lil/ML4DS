import pandas as pd
import numpy as np
import os

sexe_map = {1: 'Homme', 2: 'Femme'}
catu_map = {1: 'Conducteur', 2: 'Passager', 3: 'Piéton'}
trajet_map = {-1: 'Autre', 0: 'Autre', 1: 'Travail', 2: 'Ecole', 3: 'Courses', 4: 'Profession', 5: 'Promenade', 9: 'Autre'}
catv_map = {1: 'Vélo', 80: 'VAE'}
agg_map = {1: 'Hors agglomération', 2: 'En agglomération'}
obsm_map = {
    -1: "Aucun",   
     0: "Aucun",   
     1: "Piéton",
     2: "Véhicule",
     4: "Véhicule",  
     5: "Animal",   
     6: "Animal",   
     9: "Autre"      
}
atm_map = {
    -1: "Normale",        
     1: "Normale",        
     2: "Pluie",          
     3: "Pluie",          
     4: "Neige",
     5: "Couvert",
     6: "Tempête",
     7: "Temps éblouissant", 
     8: "Couvert",
     9: "Autre"
}
col_map = {
    -1: "Autre",                  
     1: "Frontale",               
     2: "Arrière",           
     3: "Côté",            
     4: "Chaine (3+ véhicules)",   
     5: "Multiples (3+ véhicules)",
     6: "Autre",                    
     7: "Sans collision"            
}
int_map = {
    1: "Hors intersection",         
    2: "X",       
    3: "T",
    4: "Y",             
    5: "4+ branches",      
    6: "Giratoire",
    7: "Place",
    8: "Passage à niveau",
    9: "Autre"
}
catr_map = {
    1: "Autoroute",
    2: "Route nationale",
    3: "Route départementale",
    4: "Voie communale",
    5: "Hors réseau public",
    6: "Parking",      
    7: "Réseau urbain",           
    9: "Autre"
}
vosp_map = {
    -1: "Aucun",              
     0: "Aucun",             
     1: "Piste cyclable",   
     2: "Bande cyclable",    
     3: "Voie réservée"      
}
manv_map = {
    # Autre / inconnu
    -1: "Autre / inconnu",
     0: "Autre / inconnu",
    26: "Autre / inconnu",
    # Trajectoire stable
     1: "Trajectoire stable",
     2: "Trajectoire stable",
    # Changement / insertion
     3: "Changement / insertion",
     9: "Changement / insertion",
    11: "Changement / insertion",
    12: "Changement / insertion",
    # Virage / dépassement
    10: "Virage / dépassement",
    13: "Virage / dépassement",
    14: "Virage / dépassement",
    15: "Virage / dépassement",
    16: "Virage / dépassement",
    17: "Virage / dépassement",
    18: "Virage / dépassement",
    # Manœuvres dangereuses / atypiques
     4: "Manœuvre dangereuse",
     5: "Manœuvre dangereuse",
     6: "Manœuvre dangereuse",
     7: "Manœuvre dangereuse",
     8: "Manœuvre dangereuse",
    19: "Manœuvre dangereuse",
    20: "Manœuvre dangereuse",
    21: "Manœuvre dangereuse",
}
infra_map = {
    -1: "Aucun / autre",
     0: "Aucun / autre",
     1: "Souterrain / tunnel",
     2: "Pont / autopont",
     3: "Bretelle",
     4: "Voie ferrée",
     5: "Carrefour aménagé",
     6: "Zone piétonne",
     7: "Zone de péage",
     8: "Chantier",
     9: "Aucun / autre"
}
surf_map = {
    -1: "Normale",
     1: "Normale",
     2: "Mouillée / autre",
     3: "Mouillée / autre",
     4: "Mouillée / autre",
     5: "Mouillée / autre",
     6: "Mouillée / autre",
     7: "Mouillée / autre",
     8: "Mouillée / autre",
     9: "Mouillée / autre"
}
grav_map = {
    1: "Indemne",
    2: "Tué",
    3: "Blessé grave",
    4: "Blessé léger"
}
def data_prep(df_caract, df_lieux, df_usagers, df_vehicules):
    '''
    Prépare et nettoie les données
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
    cyclistes2 = cyclistes2[[
        'Num_Acc', 'id_usager', 'id_vehicule', 'sexe', 
        'an_nais', 'trajet', 'catu', 'secu1', 'secu2', 
        'secu3', 'etatp', 'catv', 'obs', 'obsm', 'manv',
        'an', 'mois', 'hrmn', 'jour', 'atm', 'col', 'int',
        'agg', 'grav'
    ]]
    # Fusion avec les lieux de l'accident
    columns_to_keep = ['Num_Acc', 'catr', 'vosp', 'infra', 'surf', 'vma']
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
    cyclistes_final['sexe'] = cyclistes_final['sexe'].map(sexe_map)
    cyclistes_final['catu'] = cyclistes_final['catu'].map(catu_map)
    cyclistes_final['trajet'] = cyclistes_final['trajet'].map(trajet_map)
    cyclistes_final['catv'] = cyclistes_final['catv'].map(catv_map)
    cyclistes_final['atm'] = cyclistes_final['atm'].replace(atm_map)
    cyclistes_final['col'] = cyclistes_final['col'].replace(col_map)
    cyclistes_final['int'] = cyclistes_final['int'].replace(int_map)
    cyclistes_final['catr'] = cyclistes_final['catr'].replace(catr_map)
    cyclistes_final['vosp'] = cyclistes_final['vosp'].replace(vosp_map)
    cyclistes_final['manv'] = cyclistes_final['manv'].replace(manv_map)
    cyclistes_final['infra'] = cyclistes_final['infra'].replace(infra_map)
    cyclistes_final['surf'] = cyclistes_final['surf'].replace(surf_map)
    cyclistes_final['grav'] = cyclistes_final['grav'].replace(grav_map)
    cyclistes_final['agg'] = cyclistes_final['agg'].replace(agg_map)

    cyclistes_final['age'] = cyclistes_final['an'] - cyclistes_final['an_nais']
    cyclistes_final['age'] = cyclistes_final['age'].astype('Int64')
    cyclistes_final = cyclistes_final.drop(columns=['an_nais'])

    cyclistes_final['protection'] = np.where(cyclistes_final[['secu1', 'secu2', 'secu3']].isin([-1, 0]).all(axis=1), 0, 1)
    cyclistes_final['casque'] = np.where(cyclistes_final[['secu1', 'secu2', 'secu3']].isin([2]).any(axis=1), 1, 0)
    cyclistes_final['gilet'] = np.where(cyclistes_final[['secu1', 'secu2', 'secu3']].isin([4]).any(axis=1), 1, 0)
    cyclistes_final['gants'] = np.where(cyclistes_final[['secu1', 'secu2', 'secu3']].isin([6]).any(axis=1), 1, 0)
    cyclistes_final.drop(columns=['secu1', 'secu2', 'secu3'], inplace=True)

    cyclistes_final['obsm'] = cyclistes_final['obsm'].replace(obsm_map)
    cyclistes_final['obsm'] = np.where(cyclistes_final['catu'] == 'Piéton', 'Vélo', cyclistes_final['obsm'])
    cyclistes_final['obsm'] = np.where(cyclistes_final['obs'].isin([-1, 0]), cyclistes_final['obsm'], 'Obstacle fixe')
    cyclistes_final.rename(columns={'obsm': 'obstacle'}, inplace=True)
    cyclistes_final.drop(columns=['obs', 'etatp'], inplace=True)

    cyclistes_final['datetime'] = pd.to_datetime(
    cyclistes_final['an'].astype(str) + '-' + 
    cyclistes_final['mois'].astype(str).str.zfill(2) + '-' + 
    cyclistes_final['jour'].astype(str).str.zfill(2) + ' ' + 
    cyclistes_final['hrmn']
    )
    cyclistes_final.drop(columns=['an', 'mois', 'jour', 'hrmn'], inplace=True)
    cyclistes_final['jour_semaine'] = cyclistes_final['datetime'].dt.day_name()
    cyclistes_final['moment_journee'] = pd.cut(
        cyclistes_final['datetime'].dt.hour,
        bins=[-1, 5, 12, 17, 21, 24],
        labels=['Nuit', 'Matin', 'Après-midi', 'Soir', 'Nuit'],
        ordered=False
    )

    cyclistes_final = cyclistes_final.dropna(subset=['sexe', 'age'])
    cyclistes_final.drop(columns=['Num_Acc', 'id_usager', 'id_vehicule'], inplace=True)
    cyclistes_final.reset_index(drop=True, inplace=True)
    return cyclistes_final