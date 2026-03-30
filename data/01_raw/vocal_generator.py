import pandas as pd
import numpy as np
import random

def generateur_vocal_format_long(nb_patients=1000, nom_fichier="vocal_dataset.csv"):
    intensites = np.arange(0, 105, 5) # 0, 5, 10... 100 dB
    data = []

    # Définition des 5 catégories et de leurs plages de SRT50 (Seuil de Réception Vocale)
    categories_params = {
        'Normo-entendant': (0, 20),
        'Surdité Légère': (21, 40),
        'Surdité Moyenne': (41, 70),
        'Surdité Sévère': (71, 90),
        'Surdité Profonde': (91, 110)
    }

    for p_id in range(nb_patients):
        # 1. Sélection aléatoire d'une catégorie
        cat = random.choice(list(categories_params.keys()))
        plage = categories_params[cat]
        
        # 2. Paramètres influencés par la catégorie
        srt50_sans = round(random.uniform(plage[0], plage[1]), 2)
        type_s = random.choice(['perception', 'transmission'])
        
        # Le gain prévu est souvent plus important sur les surdités fortes
        gain_base = 0 if cat == 'Normo-entendant' else random.uniform(10, 30)
        gain_prevu = round(gain_base, 2)
        
        # Le score max (plateau) baisse souvent dans les surdités de perception sévères
        if type_s == 'perception' and srt50_sans > 70:
            score_max = random.uniform(50, 75)
        else:
            score_max = 100 if type_s == 'transmission' else random.uniform(85, 95)
            
        pente = random.uniform(0.1, 0.2)

        for aided in [0, 1]: # 0 = sans appareil, 1 = avec
            # Si normo-entendant, l'appareil ne change quasiment rien ou n'est pas porté
            current_srt = srt50_sans - (gain_prevu if aided == 1 else 0)
            
            for db in intensites:
                # Calcul du score (Sigmoïde)
                score = score_max / (1 + np.exp(-pente * (db - current_srt)))
                # Ajout de bruit réaliste
                score = np.clip(score + random.uniform(-2, 2), 0, 100)
                
                data.append({
                    'patient_id': p_id,
                    'categorie_surdite': cat, # Nouvelle colonne demandée
                    'is_aided': aided,
                    'type_surdite': type_s,
                    'intensity_db': db,
                    'recognition_score': round(score, 1),
                    'true_srt50': srt50_sans,
                    'true_gain_db': gain_prevu
                })

    df = pd.DataFrame(data)
    df.to_csv(nom_fichier, index=False)
    print(f"Fichier généré : {nom_fichier} avec 5 catégories.")

generateur_vocal_format_long()