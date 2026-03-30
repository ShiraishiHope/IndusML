"""
Pipeline d'inférence vocal — nodes.py
src/audio_prediction/pipelines/inference_vocal/nodes.py
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List

# Doit être identique à l'ordre utilisé pendant l'entraînement
LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


def prepare_vocal_input(df: pd.DataFrame) -> np.ndarray:
    """
    Transforme le DataFrame d'entrée en array numpy (1, 21, 2).

    Le CSV vocal_inference_input.csv contient :
      - score_0, score_5, ..., score_100  → canal 0 (recognition_score)
      - true_srt50                         → canal 1 (répété sur 21 timesteps)

    Retourne un array de shape (1, 21, 2).
    """
    score_cols = [f"score_{lvl}" for lvl in LEVELS]

    # Vérification des colonnes
    missing = [c for c in score_cols + ["true_srt50"] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans vocal_inference_input : {missing}")

    scores = df[score_cols].values.astype(np.float32)           # (1, 21)
    srt50  = df["true_srt50"].values.astype(np.float32)         # (1,)

    # Répéter srt50 sur les 21 timesteps → (1, 21)
    srt50_repeated = np.repeat(srt50[:, np.newaxis], len(LEVELS), axis=1)

    # Stack sur l'axe canal → (1, 21, 2)
    X = np.stack([scores, srt50_repeated], axis=-1)

    return X


def predict_vocal(model: tf.keras.Model, X: np.ndarray) -> pd.DataFrame:
    """
    Effectue la prédiction avec le CNN vocal.

    Entrée  : X de shape (1, 21, 2)
    Sortie  : DataFrame avec colonnes pred_0, pred_5, ..., pred_100
    """
    # predictions shape : (1, 21)
    predictions = model.predict(X)

    pred_cols = [f"pred_{lvl}" for lvl in LEVELS]
    df_output = pd.DataFrame(predictions, columns=pred_cols)

    # Clip entre 0 et 100 — scores d'intelligibilité
    df_output = df_output.clip(0, 100)

    return df_output