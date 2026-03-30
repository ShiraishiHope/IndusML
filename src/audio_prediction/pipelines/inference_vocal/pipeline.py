"""
Pipeline d'inférence vocal
src/audio_prediction/pipelines/inference_vocal/pipeline.py
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_vocal_input, predict_vocal


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_vocal_input,
            inputs="vocal_inference_input",
            outputs="vocal_X",
            name="prepare_vocal_input_node"
        ),
        node(
            func=predict_vocal,
            inputs=["vocal_model", "vocal_X"],
            outputs="vocal_predictions",
            name="predict_vocal_node"
        )
    ])