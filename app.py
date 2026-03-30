from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import numpy as np
import webbrowser
import os
from threading import Timer
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession

app = Flask(__name__)
CORS(app)

project_path = Path.cwd()
bootstrap_project(project_path)

# Chemins — TONAL
HISTORY_FILE      = "data/01_raw/history.csv"
INPUT_FILE        = "data/01_raw/inference_input.csv"

# Chemins — VOCAL
VOCAL_HISTORY_FILE = "data/01_raw/vocal_history.csv"
VOCAL_INPUT_FILE   = "data/01_raw/vocal_inference_input.csv"
VOCAL_OUTPUT_FILE  = "data/07_model_output/vocal_predictions.csv"

# Niveaux d'intensité (doit correspondre exactement à l'entraînement)
LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


@app.route("/", methods=["GET"])
@app.route("/interface", methods=["GET"])
def index():
    return send_file("index.html")


# ─── PARTIE TONALE ────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    try:
        content = request.get_json()
        data = content['data'][0] if 'data' in content else content

        df_full = pd.DataFrame([data])

        if not os.path.isfile(HISTORY_FILE):
            df_full.to_csv(HISTORY_FILE, index=False)
        else:
            df_history = pd.read_csv(HISTORY_FILE)
            df_full = df_full.reindex(columns=df_history.columns)
            df_full.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

        cols_inference = [c for c in df_full.columns if "before_exam" in c]
        df_inference = df_full[cols_inference]
        df_inference.to_csv(INPUT_FILE, index=False)

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
            session.run(pipeline_name="train")
        return jsonify({"message": "Ré-entraînement du modèle tonal terminé"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── PARTIE VOCALE ────────────────────────────────────────────────────────────

@app.route("/predict_vocal", methods=["POST"])
def predict_vocal():
    try:
        content = request.get_json()

        # Payload attendu depuis le front :
        input_matrix = content["input"]  

        if len(input_matrix) != 21:
            return jsonify({"error": f"21 niveaux attendus, {len(input_matrix)} reçus"}), 400

        # ── Sauvegarde CSV pour Kedro ──────────────────────────────────
        scores = [row[0] for row in input_matrix]
        srt50  = input_matrix[0][1]  

        row_dict = {f"score_{lvl}": scores[i] for i, lvl in enumerate(LEVELS)}
        row_dict["true_srt50"] = srt50

        df_input = pd.DataFrame([row_dict])
        df_input.to_csv(VOCAL_INPUT_FILE, index=False)

        # ── Historique vocal ──────────────────────────────────────────
        if not os.path.isfile(VOCAL_HISTORY_FILE):
            df_input.to_csv(VOCAL_HISTORY_FILE, index=False)
        else:
            df_input.to_csv(VOCAL_HISTORY_FILE, mode='a', header=False, index=False)

        # ── Pipeline Kedro ────────────────────────────────────────────
        with KedroSession.create(project_path=project_path) as session:
            session.run(pipeline_name="inference_vocal")

        # ── Lecture du résultat ───────────────────────────────────────
        if not os.path.exists(VOCAL_OUTPUT_FILE):
            return jsonify({"error": "Fichier de sortie Kedro absent"}), 500

        output_df = pd.read_csv(VOCAL_OUTPUT_FILE)

        pred_cols = [f"pred_{lvl}" for lvl in LEVELS]

        if all(c in output_df.columns for c in pred_cols):
            predicted_scores = output_df[pred_cols].iloc[0].tolist()
        else:
            predicted_scores = output_df.iloc[0].tolist()

        return jsonify({"predicted_scores": predicted_scores})

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
    app.run(host='0.0.0.0', port=5000, debug=True)