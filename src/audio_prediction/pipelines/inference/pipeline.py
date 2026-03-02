"""
This is a boilerplate pipeline 'inference'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_prediction_input, predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=validate_prediction_input,
            inputs=["inference_input", "params:data_processing.input_columns"],
            outputs=["validated_input", "validation_errors"],
            name="validate_prediction_input_node"
        ),
        node(
            func=predict,
            inputs=["trained_model", "validated_input", "params:data_processing.output_columns"],
            outputs="predictions",
            name="predict_node"
        )
    ])
