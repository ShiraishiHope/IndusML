from flask import Flask, request, jsonify
from pathlib import Path
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
import pandas as pd

app = Flask(__name__)
project_path = Path.cwd()
bootstrap_project(project_path)

# 1. Route par défaut
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "API IndusML opérationnelle"})

# 2. Route de prédiction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    with KedroSession.create(project_path=project_path) as session:
        session.run(pipeline_name="my_pipeline")
        
    output = pd.read_csv('data/07_model_output/predictions.csv') 
    return output.to_json(orient='records')

# 3. Route d'entraînement
@app.route("/train", methods=["POST"])
def train():
    with KedroSession.create(project_path=project_path) as session:
        session.run(pipeline_name="__default__") 
    return jsonify({"message": "Entraînement terminé avec succès"})

if __name__ == '__main__':
    # Le paramètre debug=True permet de recharger l'API à chaque modification du code
    app.run(host='127.0.0.1', port=5000, debug=True)