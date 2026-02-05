"""
Nodes pour le traitement des données audiométriques.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Any


def validate_data(df: pd.DataFrame, input_columns: List[str], output_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Valide et nettoie les données d'entrée.
    Identifie les lignes avec des valeurs invalides (NaN, lettres, hors limites).
    
    Args:
        df: DataFrame brut
        input_columns: Colonnes d'entrée attendues
        output_columns: Colonnes de sortie attendues
    
    Returns:
        Tuple contenant le DataFrame nettoyé et un rapport de validation
    """
    all_columns = input_columns + output_columns
    validation_report = {
        "total_rows": len(df),
        "invalid_rows": [],
        "removed_rows": 0,
        "valid_rows": 0
    }
    
    # Vérifier les colonnes manquantes
    missing_cols = [col for col in all_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes: {missing_cols}")
    
    # Identifier les lignes invalides
    invalid_indices = set()
    
    for idx, row in df.iterrows():
        row_issues = []
        for col in all_columns:
            value = row[col]
            
            # Vérifier NaN
            if pd.isna(value):
                row_issues.append(f"{col}: valeur manquante (NaN)")
                invalid_indices.add(idx)
            # Vérifier si c'est un string (lettre)
            elif isinstance(value, str):
                row_issues.append(f"{col}: valeur non numérique '{value}'")
                invalid_indices.add(idx)
            else:
                # Convertir et vérifier les limites
                try:
                    num_val = float(value)
                    if num_val < -20 or num_val > 150:
                        row_issues.append(f"{col}: valeur hors limites ({num_val})")
                        invalid_indices.add(idx)
                except (ValueError, TypeError):
                    row_issues.append(f"{col}: conversion impossible")
                    invalid_indices.add(idx)
        
        if row_issues:
            validation_report["invalid_rows"].append({
                "index": int(idx),
                "issues": row_issues
            })
    
    # Filtrer les données valides
    valid_mask = ~df.index.isin(invalid_indices)
    cleaned_df = df[valid_mask].copy()
    
    # Convertir en numérique
    for col in all_columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Supprimer les NaN restants
    cleaned_df = cleaned_df.dropna(subset=all_columns)
    
    validation_report["removed_rows"] = len(df) - len(cleaned_df)
    validation_report["valid_rows"] = len(cleaned_df)
    
    return cleaned_df, validation_report


def split_features_target(
    df: pd.DataFrame,
    input_columns: List[str],
    output_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare les features (X) et les targets (y).
    
    Args:
        df: DataFrame nettoyé
        input_columns: Noms des colonnes d'entrée
        output_columns: Noms des colonnes de sortie
    
    Returns:
        Tuple (X, y)
    """
    X = df[input_columns].copy()
    y = df[output_columns].copy()
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divise les données en ensembles d'entraînement et de test.
    
    Args:
        X: Features
        y: Targets
        test_size: Proportion du jeu de test
        random_state: Seed pour reproductibilité
    
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Reset indices
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test