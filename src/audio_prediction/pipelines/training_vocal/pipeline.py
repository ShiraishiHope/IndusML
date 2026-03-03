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
                "params:training_vocal.units",
                "params:training_vocal.epochs",
                "params:training_vocal.batch_size",
                "params:training_vocal.learning_rate",
                "params:training_vocal.dropout_rate"
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