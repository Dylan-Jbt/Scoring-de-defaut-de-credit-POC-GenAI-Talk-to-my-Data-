"""
Configuration de l'agent — chargement des variables d'environnement.

Ce module est importé en premier par agent.py au démarrage de l'application.
Il remonte l'arborescence pour trouver le fichier .env, ce qui permet de lancer
l'app depuis n'importe quel répertoire (app/agents/, app/, racine, Docker…).

Ordre de recherche du .env : app/agents/.env → app/.env → .env (racine du projet)
Si aucun fichier n'est trouvé, load_dotenv ne lève pas d'erreur — les variables
restent None et agent.py lèvera un EnvironmentError explicite si OPENAI_API_KEY manque.

Variables exportées :
    OPENAI_MODEL        — modèle OpenAI à utiliser (modifiable ici, ex. gpt-4o)
    OPENAI_API_KEY      — clé secrète OpenAI (obligatoire)
    LANGFUSE_SECRET_KEY — clé secrète Langfuse pour l'observabilité (optionnel)
    LANGFUSE_PUBLIC_KEY — clé publique Langfuse (optionnel)
    LANGFUSE_HOST       — URL du serveur Langfuse (cloud public par défaut)
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Recherche et chargement du fichier .env
#
# Remonte l'arborescence depuis app/agents/ pour trouver le .env, ce qui
# permet de lancer l'app depuis n'importe quel répertoire de travail.
# ──────────────────────────────────────────────────────────────────────────────
_here = Path(__file__).parent
_env = next(
    (p for p in [_here / ".env", _here.parent / ".env", _here.parent.parent / ".env"] if p.exists()),
    _here / ".env",
)
load_dotenv(dotenv_path=_env)

# ──────────────────────────────────────────────────────────────────────────────
# Variables exportées — lues depuis l'environnement après chargement du .env
#
# OPENAI_MODEL est la seule valeur codée en dur ; toutes les autres sont lues
# depuis les variables d'environnement (retournent None si absentes).
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
