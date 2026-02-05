import pandas as pd
import numpy as np
import time
import random


def generate_thresholds_by_profile(profile, frequencies):
    if profile == 'normal':
        return [random.randint(-10, 20) for _ in frequencies]
    elif profile == 'mild':
        return [random.randint(20, 40) for _ in frequencies]
    elif profile == 'moderate':
        return [random.randint(40, 60) for _ in frequencies]
    elif profile == 'severe':
        return [random.randint(60, 80) for _ in frequencies]
    elif profile == 'profound':
        return [random.randint(80, 120) for _ in frequencies]
    elif profile == 'slope':
        return [random.randint(0, 25) + i*10 for i, _ in enumerate(frequencies)]
    elif profile == 'reverse':
        return [random.randint(70, 90) - i*5 for i, _ in enumerate(frequencies)]
    else:
        # Default case, should not happen but just in case
        return [random.randint(0, 100) for _ in frequencies]


def calculate_improvements(profile, thresholds):
    if profile in ['normal', 'mild']:
        return np.clip(thresholds - np.random.randint(15, 30, size=len(thresholds)), 0, 120).tolist()
    elif profile in ['moderate', 'severe', 'profound']:
        return np.clip(thresholds - np.random.randint(5, 15, size=len(thresholds)), 0, 120).tolist()
    elif profile == 'slope':
        return np.clip(thresholds - np.random.randint(10, 20, size=len(thresholds)), 0, 120).tolist()
    elif profile == 'reverse':
        # For reverse loss, improvements might be less predictable
        return np.clip(thresholds - np.random.randint(0, 10, size=len(thresholds)), 0, 120).tolist()
    else:
        return thresholds  # No improvement as a fallback


def generate_audiograms_with_profiled_improvement(exam_count: int, csv_filename: str, init_freq: int = 125) -> None:
    frequencies = [init_freq * (2 ** i) for i in range(7)]
    headers = [f"exam_{freq}_Hz" for freq in frequencies]
    headers_before = ["before_" + header for header in headers]
    headers_after = ["after_" + header for header in headers]
    columns = headers_before + headers_after

    data = []

    for _ in range(exam_count):
        profile = random.choice(['normal', 'mild', 'moderate', 'severe', 'profound', 'slope', 'reverse'])
        thresholds = generate_thresholds_by_profile(profile, frequencies)

        improvements = calculate_improvements(profile, thresholds)

        if random.random() < 0.1:  # 10% de chance par audiogramme
            outlier_position = random.randint(0, len(thresholds) - 1)
            # Choisir le type d'aberration : très haute ou très basse perte
            if random.random() < 0.5:
                thresholds[outlier_position] = random.randint(120, 150)  # Perte très élevée
            else:
                thresholds[outlier_position] = random.randint(-10, 0)  # Audition exceptionnellement bonne

        data.append(thresholds + improvements)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(csv_filename, index=False)

    print(f"Audiogrammes générés et sauvegardés dans '{csv_filename}'.")


def add_realism_to_data(csv_filename: str) -> None:
    # Charger les données
    df = pd.read_csv(csv_filename)
    
    # Convertir toutes les colonnes en type 'object' pour permettre différents types de données
    for col in df.columns:
        df[col] = df[col].astype('object')

    # Déterminer le nombre de modifications à apporter
    num_rows, num_cols = df.shape
    num_changes = random.randint(np.round(num_rows*0.3), num_rows)  # Nombre aléatoire de lignes à modifier

    for _ in range(num_changes):
        row_index = random.randint(0, num_rows - 1)  # Sélectionner une ligne aléatoire
        col_index = random.choice(df.columns)  # Sélectionner une colonne aléatoire

        # Type de modification aléatoire: float, NaN, ou lettre
        modification_type = random.choice(['float', 'NaN', 'letter'])

        if modification_type == 'float':
            # Remplacer par un float aléatoire
            df.at[row_index, col_index] = float(np.round(random.uniform(0, 100), 2))
        elif modification_type == 'NaN':
            # Insérer une valeur NaN
            df.at[row_index, col_index] = np.nan
        elif modification_type == 'letter':
            # Remplacer par une lettre aléatoire, conversion explicite en str
            df.at[row_index, col_index] = str(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

    # Sauvegarder le DataFrame modifié dans le fichier CSV
    df.to_csv(csv_filename, index=False)


start = time.time()
# Number of exam desired
exam_count = 10000
# Name of output file
filename = "tonal_exams.csv"

generate_audiograms_with_profiled_improvement(exam_count, filename)
add_realism_to_data(filename)

end = time.time()

print(f'Total time: {round(end-start, 3)}')