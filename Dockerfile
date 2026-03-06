# Utiliser une image Python légère
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires (si besoin pour pandas/numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask flask-cors pandas kedro

# Copier tout le contenu du projet dans le conteneur
COPY . .

# Exposer le port sur lequel Flask tourne
EXPOSE 5000

# Commande pour lancer l'API au démarrage du conteneur
# Note : On force l'host à 0.0.0.0 pour qu'il soit accessible hors du conteneur
CMD ["python", "app.py"]