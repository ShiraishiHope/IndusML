"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

from audio_prediction.pipelines import data_processing, training, inference
from audio_prediction.pipelines import hyperparameter_tuning

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""
    
    data_processing_pipeline = data_processing.create_pipeline()
    training_pipeline = training.create_pipeline()
    inference_pipeline = inference.create_pipeline()
    hp_tuning_pipeline = hyperparameter_tuning.create_pipeline()

    
    return {
        "data_processing": data_processing_pipeline,
        "training": training_pipeline,
        "inference": inference_pipeline,
        "train": data_processing_pipeline + training_pipeline,
        "hp_tuning": data_processing_pipeline + hp_tuning_pipeline,
        "__default__": data_processing_pipeline + training_pipeline,
    }