"""
Pipeline d'entraînement du modèle CNN.
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=[
                "X_train_vocal", "y_train_vocal",
                "params:training.units",
                "params:training.epochs",
                "params:training.batch_size",
                "params:training.learning_rate",
                "params:training.dropout_rate"
            ],
            outputs="vocal_model",
            name="vocal_model_node"
        ),
        node(
            func=evaluate_model,
            inputs=["vocal_model", "X_test_vocal", "y_test_vocal"],
            outputs="model_metrics",
            name="evaluate_model_node"
        )
    ])