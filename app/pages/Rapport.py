"""
Page de rapport du modèle — métriques, figures et analyse par décile.

Affiche les résultats du Random Forest optimisé sur le jeu de test :
  - Métriques globales : PR-AUC (métrique principale sur dataset déséquilibré),
    ROC-AUC, F1, Recall, Precision
  - Courbes de lift et de capture par décile (ciblage des actions de recouvrement)
  - Figures générées lors de l'entraînement (reports/figures/)
  - Comparaison de plusieurs modèles (depuis best_model_metadata.json)

Les données du rapport proviennent de models/best_model_metadata.json (métadonnées
du modèle) et du recalcul à la volée des scores sur le dataset complet.
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.charts import bar_lift, line_capture
from utils.data import FEATURE_COLS, TARGET_COL, load_data, load_metadata, load_model

_APP  = Path(__file__).resolve().parents[1]   # app/
_ROOT = _APP.parent                            # racine du projet

# ──────────────────────────────────────────────────────────────────────────────
# _find_figures() — localise le dossier des figures du rapport
#
# Cherche reports/figures/ dans app/ puis à la racine du projet.
# Retourne le premier chemin existant, ou app/reports/figures/ par défaut
# (même si vide) — les sections concernant les figures gèreront l'absence.
# ──────────────────────────────────────────────────────────────────────────────
def _find_figures() -> Path:
    for base in (_APP, _ROOT):
        p = base / "reports" / "figures"
        if p.exists():
            return p
    return _APP / "reports" / "figures"

_FIGURES = _find_figures()

# ──────────────────────────────────────────────────────────────────────────────
# Configuration de la page — titre, présentation et chargement des ressources
#
# Charge métadonnées (JSON), pipeline joblib et dataset CSV en une seule passe.
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rapport modèle", page_icon=None, layout="wide")
st.title("Rapport du modèle")
st.markdown(
    "Synthèse des performances du **Random Forest optimisé** entraîné pour la prédiction "
    "du défaut de crédit (jeu de test retenu, jamais vu pendant l'entraînement)."
)

meta  = load_metadata()
model = load_model()
df    = load_data()

# ──────────────────────────────────────────────────────────────────────────────
# KPIs métriques — 5 scores du Random Forest sur le jeu de test
#
# Données issues de models/best_model_metadata.json (clé "metrics_xtest").
# PR-AUC est la métrique principale sur ce dataset déséquilibré (~22 % de défauts).
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Métriques sur le jeu de test")
m = meta["metrics_xtest"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("PR-AUC", f"{m['PR-AUC']:.4f}")
k2.metric("ROC-AUC",  f"{m['ROC-AUC']:.4f}")
k3.metric("F1-Score",  f"{m['F1']:.4f}")
k4.metric("Recall",    f"{m['Recall']:.4f}")
k5.metric("Precision", f"{m['Precision']:.4f}")

st.caption(
    f"Modèle : **{meta['model_name']}** | "
    f"Train : {meta['train_shape'][0]:,} obs · {meta['train_shape'][1]} features | "
    f"Test : {meta['test_shape'][0]:,} obs · Taux de défaut test : {meta['target_rate_test']:.1%}"
)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Onglets — 4 vues du rapport : hyperparamètres, lift, figures, comparaison
#
# Les calculs lourds (predict_proba sur l'ensemble du dataset) sont réalisés
# uniquement dans l'onglet Lift, à la demande de l'utilisateur.
# ──────────────────────────────────────────────────────────────────────────────
tab_params, tab_lift, tab_figs, tab_compare = st.tabs([
    "Hyperparamètres",
    "Lift & Capture",
    "Figures d'analyse",
    "Comparaison modèles",
])

# ──────────────────────────────────────────────────────────────────────────────
# Onglet 1 — Hyperparamètres : meilleurs paramètres et importance des features
#
# Paramètres extraits du JSON (best_params). Importance des features calculée
# depuis model.named_steps["model"].feature_importances_ (Random Forest).
# ──────────────────────────────────────────────────────────────────────────────
with tab_params:
    st.markdown("**Meilleurs hyperparamètres (GridSearchCV / RandomSearch)**")
    params_df = (
        {"Paramètre": k.replace("model__", ""), "Valeur": v}
        for k, v in meta["best_params"].items()
    )
    st.table(list(params_df))

    st.markdown("**Importance des features**")
    try:
        importances = model.named_steps["model"].feature_importances_
        import pandas as pd
        imp_df = (
            pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
            .sort_values("Importance", ascending=False)
        )
        st.bar_chart(imp_df.set_index("Feature")["Importance"])
    except Exception:
        st.info("Importance des features non disponible pour ce pipeline.")

# ──────────────────────────────────────────────────────────────────────────────
# Onglet 2 — Lift & Capture : évaluation du ciblage par décile
#
# predict_proba() est appelé sur le dataset complet pour classer les clients
# par score décroissant. Le calcul est délégué à utils/infer.py (compute_gains)
# avec un fallback inline si l'import échoue.
# ──────────────────────────────────────────────────────────────────────────────
with tab_lift:
    st.markdown(
        "Le **lift** mesure combien de fois le modèle fait mieux que le ciblage aléatoire. "
        "La **courbe de capture** indique le % de défauts détectés selon le seuil de relance."
    )

    # Calcul à la volée sur le dataset complet
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].values
    scores = model.predict_proba(X)[:, 1]

    # Import de compute_gains depuis utils/infer.py du projet racine
    _ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_ROOT / "utils"))
    try:
        from infer import compute_gains
        df_gains = compute_gains(scores, y, n_deciles=10)
    except Exception:
        global_rate = y.mean()
        total = y.sum()
        _df = pd.DataFrame({"score": scores, "label": y})
        _df["decile"] = pd.qcut(_df["score"], q=10, labels=False, duplicates="drop")
        _df["decile"] = 9 - _df["decile"]
        rows, cumul = [], 0
        for d in sorted(_df["decile"].unique()):
            sub = _df[_df["decile"] == d]
            n_t = sub["label"].sum()
            cumul += n_t
            rows.append({
                "Decile": int(d) + 1,
                "Clients": len(sub),
                "Cibles": int(n_t),
                "Taux": n_t / len(sub) if len(sub) else 0,
                "Lift": (n_t / len(sub) / global_rate) if len(sub) else 0,
                "Capture_Cumul": cumul / total if total else 0,
            })
        df_gains = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Lift par décile**")
        st.plotly_chart(bar_lift(df_gains), use_container_width=True)
    with c2:
        st.markdown("**Courbe de capture cumulée**")
        st.plotly_chart(line_capture(df_gains), use_container_width=True)

    st.markdown("**Tableau des déciles**")
    st.dataframe(
        df_gains.style.format({
            "Taux": "{:.1%}", "Lift": "{:.2f}", "Capture_Cumul": "{:.1%}"
        }),
        use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Onglet 3 — Figures d'analyse : visualisations générées à l'entraînement
#
# Selectbox parmi 15 figures PNG stockées dans reports/figures/.
# _FIGURES contient le chemin résolu par _find_figures() au chargement du module.
# ──────────────────────────────────────────────────────────────────────────────
with tab_figs:
    fig_map = {
        "Distribution de la cible":           "01_target_distribution.png",
        "Variables socio-démographiques":      "02_socio_demo.png",
        "Âge et défaut":                       "03_age_default.png",
        "Statut de paiement et défaut":        "04_pay_status_default.png",
        "Tendance de paiement":                "05_pay_trend.png",
        "Limite de crédit":                    "06_limit_balance.png",
        "Tendance factures / remboursements":  "07_bill_pay_trend.png",
        "Taux d'utilisation":                  "08_utilization.png",
        "Corrélations":                        "09_correlations.png",
        "Matrice de corrélation":              "10_corr_matrix.png",
        "Analyse RF (test)":                   "21_rf_tuned_analysis.png",
        "ROC / PR (test)":                     "25_xtest_roc_pr.png",
        "Matrices de confusion (test)":        "26_xtest_confusion_matrices.png",
        "Décile / Lift / Capture (test)":      "27_xtest_decile_lift_capture.png",
        "Comparaison modèles tunés":           "23_comparison_tuned_roc_pr.png",
    }

    selected = st.selectbox("Choisir une figure", list(fig_map.keys()))
    img_path = _FIGURES / fig_map[selected]
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.warning(f"Figure non trouvée : `{img_path.name}`")

# ──────────────────────────────────────────────────────────────────────────────
# Onglet 4 — Comparaison modèles : figures de baseline et de tuning
#
# 6 figures comparant LR, RF et XGB avant et après optimisation.
# ──────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown(
        "Comparaison des performances des modèles évalués avant et après optimisation. "
        "Les figures ci-dessous sont issues des notebooks d'entraînement."
    )
    for label, fname in [
        ("Baseline — comparaison ROC/PR", "14_models_comparison.png"),
        ("Matrices de confusion (baseline)", "15_confusion_matrices.png"),
        ("Tuning — LR", "20_lr_tuned_analysis.png"),
        ("Tuning — RF", "21_rf_tuned_analysis.png"),
        ("Tuning — XGB", "22_xgb_tuned_analysis.png"),
        ("Comparaison ROC/PR (modèles tunés)", "23_comparison_tuned_roc_pr.png"),
    ]:
        p = _FIGURES / fname
        if p.exists():
            with st.expander(label):
                st.image(str(p), use_container_width=True)

st.divider()
st.caption(
    "Random Forest optimisé · random_state=1204 · class_weight=balanced_subsample · "
    
)
