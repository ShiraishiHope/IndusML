import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any, Tuple
import mlflow
import mlflow.tensorflow
import platform
import logging

logger = logging.getLogger(__name__)

def configure_device() -> str:
    """Configure TensorFlow pour le GPU du Mac M1."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if platform.system() == "Darwin":
                return "MPS"
        except Exception:
            pass
    return "CPU"

class WithinMarginAccuracy(tf.keras.metrics.Metric):
    def __init__(self, margin=5.0, name='accuracy_5hz', **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = margin
        self.total_within = self.add_weight(name='total_within', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        within = tf.cast(tf.abs(y_true - y_pred) <= self.margin, tf.float32)
        self.total_within.assign_add(tf.reduce_sum(within))
        self.total_count.assign_add(tf.cast(tf.size(within), tf.float32))

    def result(self):
        return self.total_within / self.total_count

    def reset_state(self):
        self.total_within.assign(0.0)
        self.total_count.assign(0.0)

def create_vocal_model(input_shape=(21, 2), learning_rate=1e-3, units=128, dropout_rate=0.2):
    """
    Architecture CNN adaptée pour 2 canaux d'entrée : [Score Vocal, Catégorie].
    """
    model = tf.keras.Sequential([
        # On passe de (21, 1) à (21, 2) ici
        layers.Input(shape=input_shape),
        
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2, padding='same'),
        
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.Flatten(),
        
        layers.Dense(units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu'),
        
        layers.Dense(21, activation='linear') 
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=['mae', WithinMarginAccuracy(margin=5.0)]
    )
    return model

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    units: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout_rate: float
) -> tf.keras.Model:
    
    configure_device()

    # Vérification de la dimension d'entrée
    # X_train doit être (nb_patients, 21, 2)
    in_shape = (X_train.shape[1], X_train.shape[2])

    model = create_vocal_model(
        input_shape=in_shape, 
        learning_rate=learning_rate,
        units=units,
        dropout_rate=dropout_rate
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    # Enregistrement manuel dans MLflow
    if mlflow.active_run():
        mlflow.log_params({
            "units": units,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "input_channels": in_shape[1]
        })
        # Log du modèle avec signature
        mlflow.tensorflow.log_model(model, "modele_vocal")

    return model

def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    error_margin: float = 5.0
) -> Dict[str, Any]:
    
    # X_test possède déjà les 2 canaux grâce au node processing
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    abs_errors = np.abs(y_test - y_pred)
    within_margin = abs_errors <= error_margin
    overall_accuracy = float(np.mean(within_margin))
    
    print(f"Évaluation Vocale Terminée - Accuracy (±{error_margin}): {overall_accuracy:.2%}, MAE: {mae:.2f}")
    
    if mlflow.active_run():
        mlflow.log_metrics({
            "test_accuracy_5hz": overall_accuracy,
            "test_mae": float(mae)
        })

    return {
        "test_accuracy": overall_accuracy,
        "error_margin": error_margin,
        "test_mse": float(mse),
        "test_mae": float(mae),
    }