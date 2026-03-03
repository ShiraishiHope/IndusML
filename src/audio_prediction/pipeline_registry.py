"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline # type: ignore

from audio_prediction.pipelines import data_processing, training, inference,data_processing_vocal ,training_vocal ,inference_vocal
from audio_prediction.pipelines import hyperparameter_tuning

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""
    
    data_processing_pipeline = data_processing.create_pipeline()
    training_pipeline = training.create_pipeline()
    inference_pipeline = inference.create_pipeline()
    hp_tuning_pipeline = hyperparameter_tuning.create_pipeline()
    data_processing_vocal_pipeline = data_processing_vocal.create_pipeline()
    training_vocal_pipeline = training_vocal.create_pipeline()
    inference_vocal_pipeline = inference_vocal.create_pipeline()

    
    return {
        "data_processing": data_processing_pipeline,
        "training": training_pipeline,
        "inference": inference_pipeline,
        "train": data_processing_pipeline + training_pipeline,
        "hp_tuning": data_processing_pipeline + hp_tuning_pipeline,
        "__default__": data_processing_pipeline + training_pipeline,

        "data_processing_vocal": data_processing_vocal_pipeline,
        "training_vocal": training_vocal_pipeline,
        "train_vocal": data_processing_vocal_pipeline + training_vocal_pipeline,
        "inference_vocal": inference_vocal_pipeline,
    }