# audio_prediction

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Présentation
Projet Kedro de Machine Learning pour la prédiction de gains prothétiques audiométriques. Il utilise des modèles CNN pour prédire les valeurs audiométriques post-examen à partir des mesures pré-examen, sur des pipelines d'audiométrie tonale et vocale.

## Installation des dépendances

pip install -r requirements.txt

## Pipelines disponibles

Le projet enregistre les pipelines suivants dans pipeline_registry.py :

## Pipelines d'audiométrie tonale

### Pipeline par défaut (traitement des données + entraînement) :

kedro run

### Traitement des données uniquement

Validation des données, séparation features/cibles, et création des ensembles train/test :

kedro run --pipeline data_processing

### Entraînement uniquement 

entraîne le modèle CNN et l'évalue (nécessite les sorties du traitement des données) :

kedro run --pipeline training

### Entraînement complet (traitement des données + entraînement combinés) :

kedro run --pipeline train

### Inférence

exécute les prédictions sur de nouvelles données à partir d'un modèle entraîné :

kedro run --pipeline inference

### Optimisation des hyperparamètres (traitement des données + optimisation Optuna) :

kedro run --pipeline hp_tuning

## Pipelines d'audiométrie vocale

### Traitement des données vocales uniquement :

kedro run --pipeline data_processing_vocal

### Entraînement vocal uniquement :

kedro run --pipeline training_vocal

### Entraînement vocal complet :

kedro run --pipeline train_vocal

### Inférence vocale :

kedro run --pipeline inference_vocal

## Lancer l'API

Démarrer le serveur FastAPI en local :

uvicorn api:app --host 0.0.0.0 --port 8000

Ou directement via Python :

python api.py

## Points d'accès de l'API :

GET / — Informations sur l'API
GET /health — Vérification de l'état et disponibilité du modèle
POST /train — Lancer le pipeline d'entraînement complet
POST /predict — Prédire à partir de données audiométriques d'entrée

## Docker

Construire l'image Docker :

docker build -t audio_prediction:latest .

Lancer le conteneur :

docker run -p 8000:8000 audio_prediction:latest

Tests

Exécuter l'ensemble des tests avec pytest :

## pytest

Utilisation avec les notebooks Kedro

kedro jupyter notebook

kedro jupyter lab

kedro ipython

## Dépendances du projet

Pour consulter et mettre à jour les dépendances, modifiez requirements.txt. Installez-les avec pip install -r requirements.txt.