"""
Nodes pour l'inférence avec le modèle CNN.
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple


def validate_prediction_input(
    df: pd.DataFrame,
    input_columns: List[str]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Valide les données d'entrée pour la prédiction.
    """
    errors = []
    valid_indices = []
    
    missing_cols = [col for col in input_columns if col not in df.columns]
    if missing_cols:
        return pd.DataFrame(), [{"error": f"Colonnes manquantes: {missing_cols}"}]
    
    for idx, row in df.iterrows():
        row_errors = []
        is_valid = True
        
        for col in input_columns:
            value = row[col]
            
            if pd.isna(value):
                row_errors.append(f"{col}: valeur manquante")
                is_valid = False
            elif isinstance(value, str):
                row_errors.append(f"{col}: valeur non numérique '{value}'")
                is_valid = False
            else:
                try:
                    num_val = float(value)
                    if num_val < -20 or num_val > 150:
                        row_errors.append(f"{col}: hors limites ({num_val})")
                        is_valid = False
                except (ValueError, TypeError):
                    row_errors.append(f"{col}: conversion impossible")
                    is_valid = False
        
        if is_valid:
            valid_indices.append(idx)
        else:
            errors.append({"row_index": int(idx), "errors": row_errors})
    
    valid_df = df.loc[valid_indices, input_columns].copy()
    for col in input_columns:
        valid_df[col] = pd.to_numeric(valid_df[col])
    
    return valid_df, errors


def predict(
    model: tf.keras.Model,
    X: pd.DataFrame,
    output_columns: List[str]
) -> pd.DataFrame:
    """
    Effectue les prédictions avec le modèle CNN.
    """
    if X.empty:
        return pd.DataFrame(columns=output_columns)
    
    # Reshape pour Conv1D: (samples, 7, 1)
    X_array = X.values.astype(np.float32)
    X_cnn = X_array.reshape((X_array.shape[0], X_array.shape[1], 1))
    
    predictions = model.predict(X_cnn)
    return pd.DataFrame(predictions, columns=output_columns, index=X.index)