"""
Nodes pour l'entraînement du modèle - Utilise le modèle CNN officiel du cours.
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple


def create_model(input_shape, units=128, activation='relu', l2_value=0.01, dropout_rate=None, learning_rate=1e-3):
    """
    Crée le modèle CNN - Fonction officielle du cours.
    
    Args:
        input_shape: Format (dim, 1) pour Conv1D
        units: Nombre de neurones dans la couche dense
        activation: Fonction d'activation
        l2_value: Valeur de régularisation L2
        dropout_rate: Taux de dropout (None pour désactiver)
        learning_rate: Taux d'apprentissage
    
    Returns:
        Modèle Keras compilé
    """
    inputs = layers.Input(shape=input_shape)

    # Couches de convolution
    x = layers.Conv1D(filters=32, kernel_size=3, activation=activation, padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation=activation, padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    # Aplatir
    x = layers.Flatten()(x)

    # Couches denses
    x = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_value))(x)
    
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    # Sortie: 7 valeurs (une par fréquence)
    outputs = layers.Dense(7, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=['mae']
    )
    
    return model


def prepare_data_for_cnn(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prépare les données pour le modèle CNN (reshape pour Conv1D).
    
    Args:
        X: Features DataFrame
        y: Targets DataFrame
    
    Returns:
        Tuple (X_reshaped, y_array)
    """
    X_array = X.values.astype(np.float32)
    y_array = y.values.astype(np.float32)
    
    # Reshape pour Conv1D: (samples, timesteps, features) -> (samples, 7, 1)
    X_reshaped = X_array.reshape((X_array.shape[0], X_array.shape[1], 1))
    
    return X_reshaped, y_array


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    units: int = 128,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    Entraîne le modèle CNN.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de validation
        units: Neurones dans la couche dense
        epochs: Nombre d'époques
        batch_size: Taille du batch
        learning_rate: Taux d'apprentissage
        dropout_rate: Taux de dropout
    
    Returns:
        Modèle entraîné
    """
    # Préparer les données
    X_train_cnn, y_train_cnn = prepare_data_for_cnn(X_train, y_train)
    X_test_cnn, y_test_cnn = prepare_data_for_cnn(X_test, y_test)
    
    # Créer le modèle
    input_shape = (X_train_cnn.shape[1], 1)  # (7, 1)
    model = create_model(
        input_shape=input_shape,
        units=units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Entraînement
    model.fit(
        X_train_cnn, y_train_cnn,
        validation_data=(X_test_cnn, y_test_cnn),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model


def evaluate_model(
    model: tf.keras.Model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Dict[str, Any]:
    """
    Évalue les performances du modèle.
    """
    X_test_cnn, y_test_array = prepare_data_for_cnn(X_test, y_test)
    
    y_pred = model.predict(X_test_cnn)
    
    mse = mean_squared_error(y_test_array, y_pred)
    mae = mean_absolute_error(y_test_array, y_pred)
    r2 = r2_score(y_test_array, y_pred)
    
    # Métriques par fréquence
    per_frequency_metrics = {}
    columns = y_test.columns.tolist()
    for i, col in enumerate(columns):
        per_frequency_metrics[col] = {
            "mse": float(mean_squared_error(y_test_array[:, i], y_pred[:, i])),
            "mae": float(mean_absolute_error(y_test_array[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_test_array[:, i], y_pred[:, i]))
        }
    
    metrics = {
        "overall": {"mse": float(mse), "mae": float(mae), "r2": float(r2)},
        "per_frequency": per_frequency_metrics
    }
    
    print(f"Model Performance - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    return metrics