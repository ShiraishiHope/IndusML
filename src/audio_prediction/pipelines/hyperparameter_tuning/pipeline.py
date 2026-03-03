from kedro.pipeline import Pipeline, node, pipeline
from .nodes import optimize_hyperparameters  # <-- local import, not from training


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=optimize_hyperparameters,
            inputs=[
                "X_train",
                "y_train",
                "params:optuna.n_trials",
                "params:optuna.metric",
                "params:optuna.direction",
                "params:optuna.search_space",
            ],
            outputs="best_hyperparameters",
            name="optuna_optimize_node",
        ),
    ])