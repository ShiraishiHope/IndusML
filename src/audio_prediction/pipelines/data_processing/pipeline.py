"""
Pipeline de traitement des données.
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_data, split_features_target, split_train_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=validate_data,
            inputs=["raw_audiograms", "params:data_processing.input_columns", "params:data_processing.output_columns"],
            outputs=["cleaned_audiograms", "validation_report"],
            name="validate_data_node"
        ),
        node(
            func=split_features_target,
            inputs=["cleaned_audiograms", "params:data_processing.input_columns", "params:data_processing.output_columns"],
            outputs=["features", "targets"],
            name="split_features_target_node"
        ),
        node(
            func=split_train_test,
            inputs=["features", "targets", "params:data_processing.test_size", "params:data_processing.random_state"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_train_test_node"
        )
    ])