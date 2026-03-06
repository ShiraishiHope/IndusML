from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import webbrowser
import os
from threading import Timer
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession

app = Flask(__name__)
CORS(app)  # Autorise le frontend du Canvas à communiquer avec l'API

# Initialisation du projet Kedro
project_path = Path.cwd()
bootstrap_project(project_path)

# Chemins des fichiers
HISTORY_FILE = "data/01_raw/history.csv"
INPUT_FILE = "data/01_raw/inference_input.csv"

# 1. Route pour servir l'interface graphique
@app.route("/", methods=["GET"])
@app.route("/interface", methods=["GET"])
def index():
    return send_file("index.html")

# 2. Route de prédiction (POST /predict)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df_input = pd.DataFrame([data])
        
        # Sauvegarde pour l'inférence immédiate (écrase le précédent)
        df_input.to_csv(INPUT_FILE, index=False)
        
        # Sauvegarde dans l'historique global (ajoute à la suite - Consigne TP)
        if not os.path.isfile(HISTORY_FILE):
            df_input.to_csv(HISTORY_FILE, index=False)
        else:
            df_input.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
        
        # Exécution du pipeline d'inférence Kedro
        with KedroSession.create(project_path=project_path) as session:
            session.run(pipeline_name="inference")
            
        output = pd.read_csv('data/07_model_output/predictions.csv')
        return output.to_json(orient='records') 
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 3. Route pour récupérer l'historique (GET /history)
@app.route("/history", methods=["GET"])
def get_history():
    try:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            # On retourne les 10 derniers tests pour ne pas charger trop de données
            return df.tail(10).to_json(orient='records')
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 4. Route d'entraînement (POST /train)
@app.route("/train", methods=["POST"])
def train():
    try:
        with KedroSession.create(project_path=project_path) as session:
            session.run(pipeline_name="__default__")
        return jsonify({"message": "Ré-entraînement du modèle terminé avec succès"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(host='127.0.0.1', port=5000, debug=True) 