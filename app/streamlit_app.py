"""
Point d'entrée principal de l'application Streamlit.

Scoring de Défaut de Crédit + POC GenAI « Talk to my Data »
Direction Recouvrement & Risque 
"""

import sys
from pathlib import Path

import streamlit as st

# ── Résolution des imports internes ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.data import load_data, load_metadata

# ── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Scoring Défaut de Crédit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar — navigation & branding ─────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding-bottom: 0.5rem;'>
            <span style='font-size:2rem;'>💳</span><br/>
            <strong style='font-size:1.1rem;'>Scoring Défaut de Crédit</strong><br/>
            <span style='color:#5B9BD5; font-size:0.85rem;'>POC GenAI · Talk to my Data</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Direction Recouvrement & Risque")

# ── Chargement des ressources (mise en cache) ────────────────────────────────
@st.cache_data(show_spinner=False)
def _kpis():
    df   = load_data()
    meta = load_metadata()
    return df, meta

with st.spinner("Chargement du modèle et des données…"):
    df, meta = _kpis()

# ── Hero section ─────────────────────────────────────────────────────────────
st.title("💳 Scoring de Défaut de Crédit")
st.markdown(
    "**POC GenAI « Talk to my Data »** — Analyse tabulaire en langage naturel "
    "avec LangChain v1 + Streamlit  \n"
    "Banque de détail · Direction Recouvrement & Risque"
)

st.divider()

# ── KPIs du dataset ───────────────────────────────────────────────────────────
st.subheader("📋 Dataset — Vue d'ensemble")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clients", f"{len(df):,}")
k2.metric("Variables", f"{len(df.columns)}")
k3.metric("Taux de défaut", f"{df['default_payment_next_month'].mean():.1%}")
k4.metric("PR-AUC (test)", f"{meta['metrics_xtest']['PR-AUC']:.4f}")
k5.metric("ROC-AUC (test)", f"{meta['metrics_xtest']['ROC-AUC']:.4f}")

st.divider()

# ── Présentation des pages ───────────────────────────────────────────────────
st.subheader("🗂️ Navigation")

col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.markdown(
        """
        **🔎 Exploration**

        Analyse exploratoire interactive du dataset :
        distributions, taux de défaut par segment,
        corrélations et statistiques descriptives.
        """
    )

with col_b:
    st.markdown(
        """
        **💳 Scoring**

        Prédiction individuelle du défaut de crédit
        via le modèle Random Forest optimisé.
        Scoring en masse avec analyse Top-K%.
        """
    )

with col_c:
    st.markdown(
        """
        **📊 Rapport**

        Performances du modèle sur le jeu de test :
        métriques, lift, courbe de capture,
        importance des features et figures d'analyse.
        """
    )

with col_d:
    st.markdown(
        """
        **🤖 Agent IA** *(Talk to my Data)*

        Posez vos questions en français sur le dataset.
        L'agent génère et exécute du code Python (pandas)
        et affiche la réponse + le code + le résultat.
        """
    )

st.divider()

# ── Rappel modèle ────────────────────────────────────────────────────────────
st.subheader("🧠 Modèle en production")
m = meta["metrics_xtest"]
p = meta["best_params"]

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown(
        f"""
        | Paramètre | Valeur |
        |---|---|
        | **Modèle** | {meta['model_name']} |
        | **Date d'entraînement** | {meta['training_date']} |
        | **Taille train** | {meta['train_shape'][0]:,} observations |
        | **Taille test** | {meta['test_shape'][0]:,} observations |
        | **Taux défaut (train)** | {meta['target_rate_train']:.1%} |
        """
    )

with c2:
    st.markdown(
        f"""
        | Métrique | Valeur |
        |---|---|
        | **PR-AUC** ⭐ | {m['PR-AUC']:.4f} |
        | **ROC-AUC** | {m['ROC-AUC']:.4f} |
        | **F1-Score** | {m['F1']:.4f} |
        | **Recall** | {m['Recall']:.4f} |
        | **Precision** | {m['Precision']:.4f} |
        """
    )

st.divider()

# ── Contraintes du POC ───────────────────────────────────────────────────────
with st.expander("⚙️ Contraintes techniques du POC GenAI", expanded=False):
    st.markdown(
        """
        - **LangChain v1** — orchestration de l'agent, pas de version ultérieure
        - **Python/pandas uniquement** — pas de SQL, pas de requêtes réseau
        - **Pas d'écriture disque** depuis l'agent
        - **Format de sortie imposé** : ① Réponse en français ② Code Python exécuté ③ Résultat (table / graphique)
        - **Gestion des refus** : si la question sort du périmètre du dataset, l'agent répond
          *« Impossible avec les données disponibles. »*
        """
    )

st.caption(
    "Projet Scoring de Défaut de Crédit — POC GenAI « Talk to my Data » réalisé avec LangChain v1 et Streamlit | "
)
