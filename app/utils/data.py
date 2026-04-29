"""Utilitaires de chargement des données et du modèle pour l'app Streamlit."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import streamlit as st

# ── Chemins — fonctionne en local (racine du projet) et en Docker ────────────
_APP  = Path(__file__).resolve().parents[1]   # app/
_ROOT = _APP.parent                            # racine du projet


def _find(relative: str) -> Optional[Path]:
    """Cherche 'relative' d'abord dans app/, puis à la racine du projet."""
    for base in (_APP, _ROOT):
        candidate = base / relative
        if candidate.exists():
            return candidate
    return None


_DATA_PATH  = _find("data/credit_card_default.csv")
_MODEL_PATH = _find("models/best_model_pipeline.joblib")
_META_PATH  = _find("models/best_model_metadata.json")

FEATURE_COLS = [
    "limit_balance", "sex", "education_level", "marital_status", "age",
    "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
    "bill_amt_1", "bill_amt_2", "bill_amt_3", "bill_amt_4", "bill_amt_5", "bill_amt_6",
    "pay_amt_1", "pay_amt_2", "pay_amt_3", "pay_amt_4", "pay_amt_5", "pay_amt_6",
]
TARGET_COL = "default_payment_next_month"

LABEL_MAPS = {
    "sex":             {1: "Homme", 2: "Femme"},
    "education_level": {1: "Études sup.", 2: "Université", 3: "Lycée", 4: "Autre"},
    "marital_status":  {1: "Marié(e)", 2: "Célibataire", 3: "Autre"},
}

# ── Injection du module feature_engineering (faite une seule fois) ───────────
_FE_INJECTED = False

def _inject_feature_engineering() -> None:
    """Résout et injecte feature_engineering dans __main__ (une seule fois)."""
    global _FE_INJECTED
    if _FE_INJECTED:
        return

    fe_path = _find("utils/feature_engineering.py")
    if fe_path is None:
        raise FileNotFoundError(
            "feature_engineering.py introuvable dans app/utils/ ni utils/"
        )

    spec = importlib.util.spec_from_file_location("feature_engineering", fe_path)
    fe_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe_module)
    sys.modules["__main__"].feature_engineering = fe_module.feature_engineering  # type: ignore[attr-defined]
    _FE_INJECTED = True


# ── Loaders ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Charge et renvoie le dataset principal (mis en cache)."""
    if _DATA_PATH is None:
        st.error("❌ Fichier de données introuvable (`data/credit_card_default.csv`).")
        st.stop()

    try:
        df = pd.read_csv(_DATA_PATH)
    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier de données : {e}")
        st.stop()

    # Supprimer la colonne leakage si présente
    df = df.drop(columns=["predicted_default_payment_next_month"], errors="ignore")

    # Validation légère : colonnes attendues
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colonnes manquantes dans le dataset : {missing}")

    return df


@st.cache_resource(show_spinner=False)
def load_model():
    """Charge et renvoie le pipeline scikit-learn sérialisé (mis en cache).

    Le pipeline contient un FunctionTransformer(feature_engineering) défini
    à l'origine dans les notebooks (scope __main__). joblib cherche la fonction
    dans __main__ au moment de la désérialisation : on l'y injecte avant l'appel.
    """
    if _MODEL_PATH is None:
        st.error("❌ Modèle introuvable (`models/best_model_pipeline.joblib`).")
        st.stop()

    try:
        _inject_feature_engineering()
        return joblib.load(_MODEL_PATH)
    except FileNotFoundError as e:
        st.error(f"❌ Fichier manquant : {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Échec du chargement du modèle : {e}")
        st.stop()


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """Charge et renvoie les métadonnées JSON du meilleur modèle."""
    if _META_PATH is None:
        st.warning("⚠️ Métadonnées du modèle introuvables (`models/best_model_metadata.json`).")
        return {}

    try:
        with open(_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        st.warning(f"⚠️ Métadonnées JSON invalides : {e}")
        return {}
    except Exception as e:
        st.warning(f"⚠️ Impossible de lire les métadonnées : {e}")
        return {}


def load_all_with_status() -> tuple[pd.DataFrame, object, dict]:
    """Charge données, modèle et métadonnées avec un retour visuel unifié.

    À appeler depuis le point d'entrée de l'app (ex. app.py) pour afficher
    une seule barre de progression au démarrage, sans répéter les spinners.

    Returns:
        (df, model, metadata)
    """
    with st.status("Chargement de l'application…", expanded=False) as status:
        status.update(label="📂 Chargement des données…")
        df = load_data()

        status.update(label="🤖 Chargement du modèle…")
        model = load_model()

        status.update(label="📋 Chargement des métadonnées…")
        metadata = load_metadata()

        status.update(label="✅ Prêt !", state="complete", expanded=False)

    return df, model, metadata