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
                "X_train", "y_train", "X_test", "y_test",
                "params:training.units",
                "params:training.epochs",
                "params:training.batch_size",
                "params:training.learning_rate",
                "params:training.dropout_rate"
            ],
            outputs="trained_model",
            name="train_model_node"
        ),
        node(
            func=evaluate_model,
            inputs=["trained_model", "X_test", "y_test"],
            outputs="model_metrics",
            name="evaluate_model_node"
        )
    ])