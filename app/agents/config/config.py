import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# =============================================================================
# Ce module charge automatiquement le fichier .env situé à la racine du projet.
# Consultez OBTENTION_CLES_API.md pour créer et remplir votre fichier .env.
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# =============================================================================
# MODÈLES PAR DÉFAUT
# =============================================================================
LITELLM_MODEL = "gpt-4o-mini"
OPENAI_MODEL = "gpt-4o-mini"

# =============================================================================
# OPENAI
# =============================================================================
# Clé API OpenAI — https://platform.openai.com/api-keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# LANGFUSE 
# =============================================================================
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
