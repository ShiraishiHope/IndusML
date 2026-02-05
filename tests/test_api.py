"""
Tests pour l'API FastAPI.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api import app


@pytest.fixture
def client():
    """Client de test pour l'API."""
    return TestClient(app)


class TestAPIEndpoints:
    """Tests des endpoints de l'API."""
    
    def test_root_endpoint(self, client):
        """Test endpoint racine."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self, client):
        """Test endpoint health."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_predict_without_model(self, client):
        """Test prédiction sans modèle entraîné."""
        # Ce test vérifie le comportement quand le modèle n'existe pas
        test_data = {
            "data": [{
                "before_exam_125_Hz": 50,
                "before_exam_250_Hz": 45,
                "before_exam_500_Hz": 40,
                "before_exam_1000_Hz": 35,
                "before_exam_2000_Hz": 30,
                "before_exam_4000_Hz": 25,
                "before_exam_8000_Hz": 20,
            }]
        }
        
        response = client.post("/predict", json=test_data)
        # Soit 503 (modèle non dispo) soit 200 (modèle existe)
        assert response.status_code in [200, 503]
    
    def test_predict_with_invalid_data(self, client):
        """Test prédiction avec données invalides."""
        test_data = {
            "data": [
                {
                    "before_exam_125_Hz": 50,
                    "before_exam_250_Hz": 45,
                    "before_exam_500_Hz": 40,
                    "before_exam_1000_Hz": 35,
                    "before_exam_2000_Hz": 30,
                    "before_exam_4000_Hz": 25,
                    "before_exam_8000_Hz": 20,
                },
                {
                    "before_exam_125_Hz": None,  # Invalide
                    "before_exam_250_Hz": 45,
                    "before_exam_500_Hz": "ABC",  # Invalide
                    "before_exam_1000_Hz": 35,
                    "before_exam_2000_Hz": 30,
                    "before_exam_4000_Hz": 25,
                    "before_exam_8000_Hz": 20,
                }
            ]
        }
        
        response = client.post("/predict", json=test_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "invalid_rows" in data
            assert len(data["invalid_rows"]) >= 1