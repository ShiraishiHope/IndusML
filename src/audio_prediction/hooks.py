import mlflow
from kedro.framework.hooks import hook_impl


class MLflowHook:
    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        mlflow.set_experiment("audio_prediction")
        mlflow.start_run()
        mlflow.keras.autolog()

    @hook_impl
    def after_pipeline_run(self, run_params, pipeline, catalog):
        if mlflow.active_run():
            mlflow.end_run()

    @hook_impl
    def on_pipeline_error(self, error, run_params, pipeline, catalog):
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")