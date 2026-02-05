FROM python:3.13-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le projet
COPY . .

# Installer le projet en mode editable
RUN pip install -e src/

# Exposer le port
EXPOSE 8000

# Commande par défaut
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

`.dockerignore` :
```
__pycache__
*.pyc
*.pyo
.git
.gitignore
*.egg-info
.pytest_cache
.coverage
htmlcov
.mypy_cache
logs/
notebooks/
docs/