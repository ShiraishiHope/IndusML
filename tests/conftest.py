"""
Configuration pytest.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

kedro>=0.18.0
kedro-viz>=6.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
fastapi>=0.100.0
uvicorn>=0.23.0
pytest>=7.4.0
httpx>=0.24.0
pydantic>=2.0.0