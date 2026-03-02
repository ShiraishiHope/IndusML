"""
This is a boilerplate test file for pipeline 'training'
generated using Kedro 1.2.0.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pytest
import pandas as pd
import numpy as np
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner
from kedro.io import DataCatalog, MemoryDataset

from audio_prediction.pipelines.training.pipeline import create_pipeline
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
def sample_data():
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
    return X_train, X_test, y_train, y_test


def test_training_pipeline_runs(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    
    pipeline = create_pipeline()
    catalog = DataCatalog({
        "X_train": MemoryDataset(X_train),
        "X_test": MemoryDataset(X_test),
        "y_train": MemoryDataset(y_train),
        "y_test": MemoryDataset(y_test),
        "params:training.units": MemoryDataset(16),
        "params:training.epochs": MemoryDataset(2),
        "params:training.batch_size": MemoryDataset(32),
        "params:training.learning_rate": MemoryDataset(0.001),
        "params:training.dropout_rate": MemoryDataset(0.2),
        "trained_model": MemoryDataset(),
        "model_metrics": MemoryDataset(),
    })
    
    runner = SequentialRunner()
    runner.run(pipeline, catalog)
    
    model = catalog.load("trained_model")
    metrics = catalog.load("model_metrics")
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert "overall" in metrics
    assert "mse" in metrics["overall"]