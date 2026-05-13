"""Page d'exploration des données (EDA)."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.charts import bar_default_rate, hist_numeric, pie_target
from utils.data import LABEL_MAPS, TARGET_COL, load_data

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Exploration", page_icon="🔎", layout="wide")
st.title("🔎 Exploration des données")
st.markdown(
    "Dataset : **2 965 lignes · 26 colonnes** — Défaut de carte de crédit "
    "(*BigQuery ML datasets*). Variable cible : `default_payment_next_month` (0/1)."
)

df = load_data()

# ── Sidebar — filtres ────────────────────────────────────────────────────────
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

# ── KPIs ─────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Clients", f"{len(df_f):,}")
k2.metric("Défauts", f"{df_f[TARGET_COL].sum():,}")
k3.metric("Taux de défaut", f"{df_f[TARGET_COL].mean():.1%}")
k4.metric("Âge médian", f"{df_f['age'].median():.0f} ans")

st.divider()

# ── Onglets ───────────────────────────────────────────────────────────────────
tab_cible, tab_socio, tab_fin, tab_paiement, tab_raw = st.tabs([
    "🎯 Variable cible",
    "👤 Socio-démographie",
    "💰 Finances",
    "📅 Historique paiement",
    "📋 Données brutes",
])

# ─ Onglet 1 : cible ──────────────────────────────────────────────────────────
with tab_cible:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Répartition globale**")
        st.plotly_chart(pie_target(df_f), use_container_width=True)
    with c2:
        st.markdown("**Distribution des probabilités de défaut (dataset filtré)**")
        counts = df_f[TARGET_COL].value_counts().rename({0: "Non-défaut (0)", 1: "Défaut (1)"})
        st.bar_chart(counts)

# ─ Onglet 2 : socio-démographie ──────────────────────────────────────────────
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

# ─ Onglet 3 : finances ───────────────────────────────────────────────────────
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

# ─ Onglet 4 : historique paiement ───────────────────────────────────────────
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

# ─ Onglet 5 : données brutes ─────────────────────────────────────────────────
with tab_raw:
    st.markdown(f"**Aperçu — {len(df_f):,} lignes × {len(df_f.columns)} colonnes**")
    st.dataframe(df_f.head(200), use_container_width=True, height=400)
    st.markdown("**Statistiques descriptives**")
    st.dataframe(df_f.describe().round(2), use_container_width=True)
