"""
Nodes for Optuna hyperparameter optimization.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, Any
import optuna
import mlflow
import logging

from audio_prediction.pipelines.training.nodes import create_model, prepare_data_for_cnn

logger = logging.getLogger(__name__)


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    n_trials: int = 30,
    metric: str = "val_loss",
    direction: str = "minimize",
    search_space: Dict[str, Any] = None,
) -> Dict[str, Any]:
    if search_space is None:
        search_space = {}

    X_train_cnn, y_train_cnn = prepare_data_for_cnn(X_train, y_train)
    input_shape = (X_train_cnn.shape[1], 1)

    def objective(trial: optuna.Trial) -> float:
        units = trial.suggest_int(
            "units",
            search_space.get("units_min", 32),
            search_space.get("units_max", 256),
            step=32,
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            search_space.get("learning_rate_min", 1e-4),
            search_space.get("learning_rate_max", 1e-2),
            log=True,
        )
        dropout_rate = trial.suggest_float(
            "dropout_rate",
            search_space.get("dropout_rate_min", 0.1),
            search_space.get("dropout_rate_max", 0.5),
        )
        batch_size = trial.suggest_categorical(
            "batch_size",
            search_space.get("batch_size_choices", [16, 32, 64]),
        )

        # Wrap model creation, training, AND manual logging
        # inside the nested run so autologging targets THIS run
        if mlflow.active_run():
            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                model = create_model(
                    input_shape=input_shape,
                    units=units,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate,
                )

                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )

                history = model.fit(
                    X_train_cnn,
                    y_train_cnn,
                    epochs=50,
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[early_stopping],
                    verbose=0,
                )

                best_val_loss = min(history.history["val_loss"])

                mlflow.log_params({
                    "units": units,
                    "learning_rate": learning_rate,
                    "dropout_rate": dropout_rate,
                    "batch_size": batch_size,
                })
                mlflow.log_metric("best_val_loss", best_val_loss)
        else:
            # Fallback if no active MLflow run
            model = create_model(
                input_shape=input_shape,
                units=units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
            )

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )

            history = model.fit(
                X_train_cnn,
                y_train_cnn,
                epochs=50,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=0,
            )

            best_val_loss = min(history.history["val_loss"])

        return best_val_loss

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["best_val_loss"] = study.best_value

    logger.info("Optuna best parameters: %s", best_params)

    if mlflow.active_run():
        mlflow.log_params({f"optuna_best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("optuna_best_val_loss", study.best_value)

    return best_params