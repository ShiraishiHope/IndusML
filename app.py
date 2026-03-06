from flask import Flask, request, jsonify , send_file
from pathlib import Path
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
import pandas as pd
import webbrowser
from threading import Timer

app = Flask(__name__)
project_path = Path.cwd()
bootstrap_project(project_path)

# 1. Route par défaut
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "API IndusML opérationnelle"})

import pandas as pd

import pandas as pd
from flask import request

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    df_input = pd.DataFrame([data]) 
    df_input.to_csv("data/01_raw/inference_input.csv", index=False)
    print("Données d'entrée reçues et enregistrées pour l'inférence :")
    with KedroSession.create(project_path=project_path) as session:
        session.run(pipeline_name="inference") # 
    print("Pipeline d'inférence exécuté avec succès.")
    output = pd.read_csv('data/07_model_output/predictions.csv') 
    return output.to_json(orient='records')

# 3. Route d'entraînement
@app.route("/train", methods=["POST"])
def train():
    with KedroSession.create(project_path=project_path) as session:
        session.run(pipeline_name="__default__") 
    return jsonify({"message": "Entraînement terminé avec succès"})

# ... vos imports et configuration ...

# Nouvelle route qui appelle DIRECTEMENT votre fichier index.html
@app.route("/interface", methods=["GET"])
def afficher_interface():
    # send_file va chercher le fichier index.html situé dans le même dossier
    return send_file("index.html")

# ... le reste de vos routes ...

def open_browser():

    webbrowser.open_new("http://127.0.0.1:5000/interface") 

if __name__ == '__main__':
    Timer(1, open_browser).start()
    # Le paramètre debug=True permet de recharger l'API à chaque modification du code
    app.run(host='127.0.0.1', port=5000, debug=True)