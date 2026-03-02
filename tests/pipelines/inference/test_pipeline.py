"""
This is a boilerplate test file for pipeline 'inference'
generated using Kedro 1.2.0.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pytest
import pandas as pd
import numpy as np
from kedro.runner import SequentialRunner
from kedro.io import DataCatalog, MemoryDataset

from audio_prediction.pipelines.inference.pipeline import create_pipeline
from audio_prediction.pipelines.training.nodes import train_model
from audio_prediction.pipelines.data_processing.nodes import split_features_target, split_train_test


INPUT_COLUMNS = [
    "before_exam_125_Hz", "before_exam_250_Hz", "before_exam_500_Hz",
    "before_exam_1000_Hz", "before_exam_2000_Hz", "before_exam_4000_Hz",
    "before_exam_8000_Hz"
]
OUTPUT_COLUMNS = [
    "after_exam_125_Hz", "after_exam_250_Hz", "after_exam_500_Hz",
    "after_exam_1000_Hz", "after_exam_2000_Hz", "after_exam_4000_Hz",
    "after_exam_8000_Hz"
]


@pytest.fixture
def trained_model_fixture():
    np.random.seed(42)
    n = 100
    data = {}
    for col in INPUT_COLUMNS:
        data[col] = np.random.randint(0, 100, n)
    for col in OUTPUT_COLUMNS:
        data[col] = np.random.randint(0, 80, n)
    df = pd.DataFrame(data)
    X, y = split_features_target(df, INPUT_COLUMNS, OUTPUT_COLUMNS)
    X_train, X_test, y_train, y_test = split_train_test(X, y, 0.2, 42)
    return train_model(X_train, y_train, X_test, y_test, units=16, epochs=2, batch_size=32)


def test_inference_pipeline_runs(trained_model_fixture):
    inference_input = pd.DataFrame({
        col: [50, 40, 30] for col in INPUT_COLUMNS
    })
    
    pipeline = create_pipeline()
    catalog = DataCatalog({
        "inference_input": MemoryDataset(inference_input),
        "trained_model": MemoryDataset(trained_model_fixture),
        "params:data_processing.input_columns": MemoryDataset(INPUT_COLUMNS),
        "params:data_processing.output_columns": MemoryDataset(OUTPUT_COLUMNS),
        "validated_input": MemoryDataset(),
        "validation_errors": MemoryDataset(),
        "predictions": MemoryDataset(),
    })
    
    runner = SequentialRunner()
    runner.run(pipeline, catalog)
    
    predictions = catalog.load("predictions")
    assert len(predictions) == 3
    assert list(predictions.columns) == OUTPUT_COLUMNS