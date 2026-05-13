# Image de base : Python 3.13 en version allégée
FROM python:3.13-slim

# Installer les dépendances nécessaires à Kaleido
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail dans le conteneur
WORKDIR /app

# Copier les dépendances et les installer (optimise le cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Port utilisé par Streamlit
EXPOSE 8501

# Commande de lancement
# --server.port=8501       -> port d'écoute de Streamlit
# --server.address=0.0.0.0 -> accepte les connexions de n'importe quelle adresse IP
# --server.headless=true   -> desactive l'ouverture automatique du navigateur
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]  

