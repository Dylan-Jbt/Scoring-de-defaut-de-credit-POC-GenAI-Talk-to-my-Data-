"""Page de scoring de défaut de crédit — prédiction individuelle."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.charts import gauge_proba
from utils.data import FEATURE_COLS, LABEL_MAPS, TARGET_COL, load_data, load_model

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Scoring Défaut", page_icon="💳", layout="wide")
st.title("💳 Scoring de défaut de crédit")
st.markdown(
    "Renseignez le profil d'un client pour obtenir sa **probabilité estimée de défaut** "
    "au prochain mois, calculée par le modèle Random Forest optimisé."
)

model = load_model()

# ── Formulaire de saisie ─────────────────────────────────────────────────────
with st.form("scoring_form"):
    st.subheader("Profil client")
    c1, c2, c3 = st.columns(3)

    with c1:
        limit_balance = st.number_input("Limite de crédit (NT$)", min_value=0, max_value=1_000_000,
                                        value=50_000, step=5_000)
        sex = st.selectbox("Sexe", options=[1, 2], format_func=lambda x: LABEL_MAPS["sex"][x])
        education_level = st.selectbox("Niveau d'éducation", options=[1, 2, 3, 4],
                                       format_func=lambda x: LABEL_MAPS["education_level"][x])
        marital_status = st.selectbox("Statut marital", options=[1, 2, 3],
                                      format_func=lambda x: LABEL_MAPS["marital_status"][x])
        age = st.slider("Âge", min_value=18, max_value=80, value=35)

    with c2:
        st.markdown("**Historique de paiement** *(−1=à temps, 1=1 mois de retard, …)*")
        pay_0 = st.slider("pay_0 (mois −1)", -2, 8, 0)
        pay_2 = st.slider("pay_2 (mois −2)", -2, 8, 0)
        pay_3 = st.slider("pay_3 (mois −3)", -2, 8, 0)
        pay_4 = st.slider("pay_4 (mois −4)", -2, 8, 0)
        pay_5 = st.slider("pay_5 (mois −5)", -2, 8, 0)
        pay_6 = st.slider("pay_6 (mois −6)", -2, 8, 0)

    with c3:
        st.markdown("**Montants facturés (NT$)**")
        bill_amt_1 = st.number_input("bill_amt_1", min_value=0, value=10_000, step=1_000)
        bill_amt_2 = st.number_input("bill_amt_2", min_value=0, value=9_500,  step=1_000)
        bill_amt_3 = st.number_input("bill_amt_3", min_value=0, value=9_000,  step=1_000)
        bill_amt_4 = st.number_input("bill_amt_4", min_value=0, value=8_500,  step=1_000)
        bill_amt_5 = st.number_input("bill_amt_5", min_value=0, value=8_000,  step=1_000)
        bill_amt_6 = st.number_input("bill_amt_6", min_value=0, value=7_500,  step=1_000)

    st.markdown("**Montants remboursés (NT$)**")
    pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
    pay_amt_1 = pc1.number_input("pay_amt_1", min_value=0, value=2_000, step=500)
    pay_amt_2 = pc2.number_input("pay_amt_2", min_value=0, value=2_000, step=500)
    pay_amt_3 = pc3.number_input("pay_amt_3", min_value=0, value=2_000, step=500)
    pay_amt_4 = pc4.number_input("pay_amt_4", min_value=0, value=2_000, step=500)
    pay_amt_5 = pc5.number_input("pay_amt_5", min_value=0, value=2_000, step=500)
    pay_amt_6 = pc6.number_input("pay_amt_6", min_value=0, value=2_000, step=500)

    submitted = st.form_submit_button("🔍 Calculer le score", use_container_width=True)

# ── Résultat ─────────────────────────────────────────────────────────────────
if submitted:
    input_data = pd.DataFrame([{
        "limit_balance": limit_balance, "sex": sex,
        "education_level": education_level, "marital_status": marital_status, "age": age,
        "pay_0": pay_0, "pay_2": pay_2, "pay_3": pay_3,
        "pay_4": pay_4, "pay_5": pay_5, "pay_6": pay_6,
        "bill_amt_1": bill_amt_1, "bill_amt_2": bill_amt_2, "bill_amt_3": bill_amt_3,
        "bill_amt_4": bill_amt_4, "bill_amt_5": bill_amt_5, "bill_amt_6": bill_amt_6,
        "pay_amt_1": pay_amt_1, "pay_amt_2": pay_amt_2, "pay_amt_3": pay_amt_3,
        "pay_amt_4": pay_amt_4, "pay_amt_5": pay_amt_5, "pay_amt_6": pay_amt_6,
    }])

    proba = float(model.predict_proba(input_data[FEATURE_COLS])[:, 1][0])
    decision = proba >= 0.5

    st.divider()
    col_gauge, col_verdict = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(gauge_proba(proba), use_container_width=True)

    with col_verdict:
        st.markdown("### Verdict")
        if decision:
            st.error(f"⚠️ **Risque élevé de défaut** — probabilité : {proba:.1%}")
            st.markdown(
                "Ce client présente un profil à **relancer en priorité**. "
                "Il est recommandé de déclencher une action de recouvrement préventif."
            )
        else:
            st.success(f"✅ **Risque faible** — probabilité : {proba:.1%}")
            st.markdown(
                "Ce client ne présente pas de signal de défaut imminent "
                "sur la base de son profil actuel."
            )

        st.markdown("---")
        st.caption(
            "Seuil de décision : 0,50 | Modèle : Random Forest optimisé | "
            "PR-AUC test : 0,6208 | ROC-AUC test : 0,8137"
        )

# ── Scoring en masse sur le dataset ─────────────────────────────────────────
st.divider()
st.subheader("Scoring sur l'ensemble du dataset")

with st.expander("Voir le tableau de scoring complet", expanded=False):
    df = load_data()
    X = df[FEATURE_COLS]
    probas = model.predict_proba(X)[:, 1]
    labels = (probas >= 0.5).astype(int)

    df_score = df[["id"] + FEATURE_COLS + [TARGET_COL]].copy()
    df_score["proba_defaut"] = np.round(probas, 4)
    df_score["label_pred"]   = labels
    df_score["correct"]      = df_score["label_pred"] == df_score[TARGET_COL]

    top_k = st.slider("Afficher le Top-K% à risque", 5, 50, 20, step=5)
    cutoff = np.percentile(probas, 100 - top_k)
    df_top = df_score[df_score["proba_defaut"] >= cutoff].sort_values("proba_defaut", ascending=False)

    st.metric("Clients dans le Top-K%", len(df_top))
    recall_topk = df_top[TARGET_COL].sum() / df[TARGET_COL].sum()
    st.metric("Recall dans le Top-K%", f"{recall_topk:.1%}")
    st.dataframe(
        df_top[["id", "proba_defaut", "label_pred", TARGET_COL]].reset_index(drop=True),
        use_container_width=True,
        height=350,
    )
