"""
Page d'exploration des données (EDA).

Permet d'explorer le dataset de défaut de crédit de manière interactive :
  - Filtres sidebar : sexe, niveau d'éducation, tranche d'âge
  - KPIs en temps réel sur la population filtrée
  - Onglets : variable cible, socio-démographie, finances,
              historique de paiement, données brutes

Toutes les visualisations utilisent Plotly Express (graphiques interactifs).
Les libellés des modalités (sex, education_level…) sont traduits via LABEL_MAPS.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.charts import bar_default_rate, hist_numeric, pie_target
from utils.data import LABEL_MAPS, TARGET_COL, load_data

# ──────────────────────────────────────────────────────────────────────────────
# Configuration de la page — titre, présentation et chargement du dataset
#
# set_page_config doit être le premier appel Streamlit du script.
# load_data() lit le CSV une seule fois (mis en cache par @st.cache_data dans data.py).
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Exploration", page_icon=None, layout="wide")
st.title("Exploration des données")
st.markdown(
    "Dataset : **2 965 lignes · 26 colonnes** — Défaut de carte de crédit "
    "(*BigQuery ML datasets*). Variable cible : `default_payment_next_month` (0/1)."
)

df = load_data()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — filtres interactifs sur la population
#
# Trois filtres indépendants (sexe, éducation, tranche d'âge) combinés
# par boolean indexing. df_f est le sous-ensemble filtré utilisé par
# tous les graphiques et métriques ci-dessous.
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filtres")
    sex_opts = st.multiselect("Sexe", options=[1, 2],
                              default=[1, 2], format_func=lambda x: LABEL_MAPS["sex"][x])
    edu_opts = st.multiselect("Niveau d'éducation", options=[1, 2, 3, 4],
                              default=[1, 2, 3, 4],
                              format_func=lambda x: LABEL_MAPS["education_level"][x])
    age_range = st.slider("Tranche d'âge", int(df["age"].min()), int(df["age"].max()),
                          (int(df["age"].min()), int(df["age"].max())))

df_f = df[
    df["sex"].isin(sex_opts) &
    df["education_level"].isin(edu_opts) &
    df["age"].between(*age_range)
]

st.caption(f"Données filtrées : **{len(df_f):,}** clients")

# ──────────────────────────────────────────────────────────────────────────────
# KPIs dynamiques — résumé de la population filtrée
#
# 4 métriques recalculées en temps réel à chaque changement de filtre :
# volume total, nombre de défauts, taux de défaut et âge médian.
# ──────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Clients", f"{len(df_f):,}")
k2.metric("Défauts", f"{df_f[TARGET_COL].sum():,}")
k3.metric("Taux de défaut", f"{df_f[TARGET_COL].mean():.1%}")
k4.metric("Âge médian", f"{df_f['age'].median():.0f} ans")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Onglets de visualisation — 5 vues complémentaires du dataset filtré
#
# Chaque onglet analyse un axe différent : variable cible, socio-démographie,
# finances, historique de paiement et données brutes. Tous opèrent sur df_f.
# ──────────────────────────────────────────────────────────────────────────────
tab_cible, tab_socio, tab_fin, tab_paiement, tab_raw = st.tabs([
    "Variable cible",
    "Socio-démographie",
    "Finances",
    "Historique paiement",
    "Données brutes",
])

# ──────────────────────────────────────────────────────────────────────────────
# Onglet 1 — Variable cible : répartition des défauts et non-défauts
#
# Deux vues : camembert (proportions globales) et bar chart (counts 0/1).
# ──────────────────────────────────────────────────────────────────────────────
with tab_cible:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Répartition globale**")
        st.plotly_chart(pie_target(df_f), use_container_width=True)
    with c2:
        st.markdown("**Distribution des probabilités de défaut (dataset filtré)**")
        counts = df_f[TARGET_COL].value_counts().rename({0: "Non-défaut (0)", 1: "Défaut (1)"})
        st.bar_chart(counts)

# ──────────────────────────────────────────────────────────────────────────────
# Onglet 2 — Socio-démographie : taux de défaut par segment démographique
#
# 4 graphiques : sexe, niveau d'éducation, statut marital, distribution de l'âge.
# LABEL_MAPS traduit les codes numériques en libellés lisibles (ex. 1 → "Homme").
# ──────────────────────────────────────────────────────────────────────────────
with tab_socio:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Taux de défaut par sexe**")
        st.plotly_chart(bar_default_rate(df_f, "sex", LABEL_MAPS["sex"]),
                        use_container_width=True)
    with c2:
        st.markdown("**Taux de défaut par niveau d'éducation**")
        st.plotly_chart(bar_default_rate(df_f, "education_level", LABEL_MAPS["education_level"]),
                        use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Taux de défaut par statut marital**")
        st.plotly_chart(bar_default_rate(df_f, "marital_status", LABEL_MAPS["marital_status"]),
                        use_container_width=True)
    with c4:
        st.markdown("**Distribution de l'âge par statut de défaut**")
        st.plotly_chart(hist_numeric(df_f, "age"), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Onglet 3 — Finances : distribution des montants et statistiques groupées
#
# Distribution de la limite de crédit + tableau mean/median sur limit_balance,
# bill_amt_1 et pay_amt_1, groupé par statut de défaut (0/1).
# ──────────────────────────────────────────────────────────────────────────────
with tab_fin:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Distribution de la limite de crédit**")
        st.plotly_chart(hist_numeric(df_f, "limit_balance"), use_container_width=True)
    with c2:
        st.markdown("**Statistiques par groupe**")
        stats = (
            df_f.groupby(TARGET_COL)[["limit_balance", "bill_amt_1", "pay_amt_1"]]
            .agg(["mean", "median"])
            .round(0)
        )
        stats.index = stats.index.map({0: "Non-défaut", 1: "Défaut"})
        st.dataframe(stats, use_container_width=True)

    st.markdown("**Distribution de bill_amt_1 par statut de défaut**")
    st.plotly_chart(hist_numeric(df_f, "bill_amt_1"), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Onglet 4 — Historique de paiement : retards par mois et par statut de défaut
#
# Taux de défaut selon pay_0 (mois le plus récent) + tableau des retards
# moyens sur 6 mois (pay_0 à pay_6), groupé par défaut/non-défaut.
# ──────────────────────────────────────────────────────────────────────────────
with tab_paiement:
    st.markdown("**Taux de défaut selon le statut de paiement du mois précédent (pay_0)**")
    st.plotly_chart(bar_default_rate(df_f, "pay_0"), use_container_width=True)

    pay_cols = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
    st.markdown("**Retard moyen par groupe (défaut vs non-défaut)**")
    pay_stats = (
        df_f.groupby(TARGET_COL)[pay_cols]
        .mean()
        .T
        .rename(columns={0: "Non-défaut", 1: "Défaut"})
        .round(3)
    )
    st.dataframe(pay_stats, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Onglet 5 — Données brutes : aperçu du DataFrame filtré
#
# Affiche les 200 premières lignes de df_f + .describe() sur toutes les colonnes.
# ──────────────────────────────────────────────────────────────────────────────
with tab_raw:
    st.markdown(f"**Aperçu — {len(df_f):,} lignes × {len(df_f.columns)} colonnes**")
    st.dataframe(df_f.head(200), use_container_width=True, height=400)
    st.markdown("**Statistiques descriptives**")
    st.dataframe(df_f.describe().round(2), use_container_width=True)
