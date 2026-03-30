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
CORS(app)

# Initialisation du projet Kedro
project_path = Path.cwd()
bootstrap_project(project_path)

# Chemins des fichiers - TONAL
HISTORY_FILE = "data/01_raw/history.csv"
INPUT_FILE = "data/01_raw/inference_input.csv"

# Chemins des fichiers - VOCAL (Ajoutés)
VOCAL_HISTORY_FILE = "data/01_raw/vocal_history.csv"
VOCAL_INPUT_FILE = "data/01_raw/vocal_inference_input.csv"
VOCAL_OUTPUT_FILE = "data/07_model_output/vocal_predictions.csv" # À vérifier selon ton catalog.yml

@app.route("/", methods=["GET"])
@app.route("/interface", methods=["GET"])
def index():
    return send_file("index.html")

# --- PARTIE TONALE ---

@app.route("/predict", methods=["POST"])
def predict():
    try:
        content = request.get_json()
        # On récupère les données (format {data: [payload]})
        data = content['data'][0] if 'data' in content else content
        
        # 1. On crée le DataFrame complet (Identité + Mesures)
        df_full = pd.DataFrame([data])

        # 2. Sauvegarde dans l'HISTORIQUE (On garde tout : Nom, Prénom, Mesures)
        if not os.path.isfile(HISTORY_FILE):
            # Si le fichier n'existe pas, on le crée avec les en-têtes
            df_full.to_csv(HISTORY_FILE, index=False)
        else:
            # S'il existe, on ajoute à la suite SANS réécrire l'en-tête
            # On s'assure de l'ordre des colonnes pour ne pas décaler le CSV
            df_history = pd.read_csv(HISTORY_FILE)
            df_full = df_full.reindex(columns=df_history.columns) # Aligne les colonnes
            df_full.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

        # 3. Sauvegarde pour l'INFÉRENCE KEDRO (On ne garde QUE les chiffres)
        # On filtre pour ne garder que les colonnes qui commencent par 'before_exam'
        cols_inference = [c for c in df_full.columns if "before_exam" in c]
        df_inference = df_full[cols_inference]
        df_inference.to_csv(INPUT_FILE, index=False)
        
        # 4. Exécution Kedro
        with KedroSession.create(project_path=project_path) as session:
            session.run(pipeline_name="inference")
            
        output = pd.read_csv('data/07_model_output/predictions.csv')
        return output.to_json(orient='records') 

    except Exception as e:
        print(f"Erreur Tonal: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

@app.route("/history", methods=["GET"])
def get_history():
    try:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            return df.tail(10).to_json(orient='records')
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train():
    try:
        with KedroSession.create(project_path=project_path) as session:
            session.run(pipeline_name="train") # Changé de __default__ à train pour être explicite
        return jsonify({"message": "Ré-entraînement du modèle tonal terminé"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- PARTIE VOCALE (NOUVEAU) ---

@app.route("/predict_vocal", methods=["POST"])
def predict_vocal():
    try:
        data = request.get_json()
        # On prépare le DataFrame avec les colonnes attendues par ton pipeline Kedro vocal
        # Assure-toi que les noms correspondent à ton fichier data_processing_vocal
        df_input = pd.DataFrame([{
            "oreille": data.get("oreille"),
            "srt_db": data.get("srt_db"),
            "score_40db": data.get("score_40db"),
            "score_60db": data.get("score_60db"),
            "score_80db": data.get("score_80db")
        }])
        
        # Sauvegarde pour l'inférence Kedro
        df_input.to_csv(VOCAL_INPUT_FILE, index=False)
        
        # Historique vocal
        if not os.path.isfile(VOCAL_HISTORY_FILE):
            df_input.to_csv(VOCAL_HISTORY_FILE, index=False)
        else:
            df_input.to_csv(VOCAL_HISTORY_FILE, mode='a', header=False, index=False)
        
        # Exécution du pipeline d'inférence VOCAL
        with KedroSession.create(project_path=project_path) as session:
            session.run(pipeline_name="inference_vocal")
            
        # Lecture du résultat (le chemin doit correspondre à ton catalogue Kedro)
        if os.path.exists(VOCAL_OUTPUT_FILE):
            output = pd.read_csv(VOCAL_OUTPUT_FILE)
            return output.to_json(orient='records')
        else:
            # Fallback si Kedro n'a pas encore créé le fichier
            return jsonify({"prediction": 0.0, "status": "Fichier de sortie absent"})

    except Exception as e:
        print(f"Erreur Vocal: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/train_vocal", methods=["POST"])
def train_vocal():
    try:
        with KedroSession.create(project_path=project_path) as session:
            session.run(pipeline_name="train_vocal")
        return jsonify({"message": "Ré-entraînement du modèle vocal terminé"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True aide à voir les erreurs