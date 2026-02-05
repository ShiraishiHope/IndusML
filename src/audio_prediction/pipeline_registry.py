"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

from audio_prediction.pipelines import data_processing, training, inference


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""
    
    data_processing_pipeline = data_processing.create_pipeline()
    training_pipeline = training.create_pipeline()
    
    return {
        "data_processing": data_processing_pipeline,
        "training": training_pipeline,
        "train": data_processing_pipeline + training_pipeline,
        "__default__": data_processing_pipeline + training_pipeline,
    }