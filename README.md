# Scoring de Défaut de Crédit + POC GenAI « Talk to my Data »

> Conception d'un modèle de scoring du risque de crédit et d'un POC GenAI pour l'analyse de données en langage naturel, avec mise en place d'une pipeline ML end-to-end et d'une interface interactive.

---

## Contexte

Vous endossez le rôle de Data Scientist au sein d'une banque de détail. La direction **Recouvrement & Risque** souhaite renforcer sa politique de relance et de recouvrement au prochain trimestre.

La mission se déroule en 3 étapes :
1. **EDA + Repo Git** — prise en main du dataset et mise en place d'un dépôt reproductible
2. **Modèle ML production-ready** — scoring de défaut en tenant compte du déséquilibre de classes
3. **POC GenAI « Talk to my Data »** — assistant d'analyse conversationnel piloté en langage naturel

---

## Dataset

| Caractéristique | Valeur |
|---|---|
| Source | BigQuery ML datasets — export CSV |
| Taille | 2 965 lignes × 26 colonnes |
| Variable cible | `default_payment_next_month` (0 = pas de défaut, 1 = défaut) |
| Taux de défaut | ~21 % |
| Leakage | La colonne `predicted_default_payment_next_month` **ne doit pas** être utilisée comme feature |

**Variables :** socio-démographiques (`sex`, `education_level`, `marital_status`, `age`), financières (`limit_balance`, `bill_amt_1..6`, `pay_amt_1..6`), historique de paiement (`pay_0`, `pay_2..6`).

---

## Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── Dockerfile
├── data/
│   └── credit_card_default.csv          # Dataset brut
├── notebooks/
│   ├── 01_data_exploration.ipynb        # EDA, qualité, protocole d'évaluation
│   ├── 02_modelisation_baseline.ipynb   # LR, RF, XGBoost — scores baseline
│   └── 03_models_fintuned.ipynb         # RandomizedSearchCV, évaluation finale
├── utils/
│   ├── data_prep.py                     # Chargement et feature engineering
│   ├── train.py                         # Entraînement du pipeline
│   ├── infer.py                         # Inférence : predict_scores(), rank_by_score()
│   ├── metrics.py                       # evaluate_model(), calculate_lift()
│   └── feature_engineering.py          # Transformations (utilization_proxy, pay_trend…)
├── models/
│   ├── best_model_pipeline.joblib       # Pipeline complet (FE + Scaler + RF optimisé)
│   └── best_model_metadata.json        # Métriques, hyperparamètres, shapes
├── reports/
│   ├── model_report.md                  # Rapport synthétique du modèle
│   └── figures/                         # Visualisations EDA et performances
└── app/
    ├── streamlit_app.py                 # Point d'entrée de l'application
    ├── pages/
    │   ├── Defaut_credit.py             # Scoring individuel
    │   ├── Exploration.py               # Analyse exploratoire interactive
    │   ├── Rapport.py                   # Rapport métriques et figures
    │   ├── Talk_to_my_Data.py           # Interface chat LLM
    │   └── A_propos.py
    ├── agents/
    │   ├── agent.py                     # Agent LangChain v1 (ReAct + InMemorySaver)
    │   ├── config.py                    # Clés API, variables d'environnement
    │   ├── promts.py                    # Prompts système (avec contexte modèle RF)
    │   ├── tools.py                     # Outils d'analyse pandas (4 outils)
    │   └── pyproject.toml
    └── utils/
        ├── data.py
        └── charts.py
```

---

## Résultats du modèle

**Modèle retenu : Random Forest optimisé** (RandomizedSearchCV, `random_state=1204`)

| Métrique | Jeu de test (hold-out) |
|---|---|
| **PR-AUC** | **0.6208** |
| ROC-AUC | 0.8137 |
| F1-Score | 0.5839 |
| Recall | 0.6299 |
| Precision | 0.5442 |

> Le PR-AUC est la métrique prioritaire sur un dataset déséquilibré (~21 % de défauts).  
> Le modèle capture **63 % des défauts réels** du jeu de test.

**Stratégie de ciblage :** top 20-30 % de clients par score décroissant → capture ~55-70 % des défauts avec un effort de relance maîtrisé.

---

## Installation

### Prérequis
- Python 3.10+
- Compte OpenAI avec une clé API (pour le POC GenAI)

### 1. Cloner le dépôt et créer l'environnement

```bash
git clone <url-du-repo>
cd Scoring-de-defaut-de-credit-POC-GenAI-Talk-to-my-Data-
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurer les clés API

Créez un fichier `.env` à la racine du projet :

```env
OPENAI_API_KEY=sk-...votre-clé-openai...

# Optionnel — observabilité Langfuse
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

> Sans `OPENAI_API_KEY`, les pages de scoring et d'exploration fonctionnent mais la page **Talk to my Data** ne démarrera pas.

---

## Lancer l'application

```bash
streamlit run app/streamlit_app.py
```

L'application s'ouvre sur `http://localhost:8501` avec 5 pages :

| Page | Description |
|---|---|
| Accueil | KPIs globaux du dataset et du modèle |
| Scoring Défaut | Prédiction individuelle via formulaire |
| Exploration | Analyse exploratoire interactive |
| Rapport modèle | Métriques, lift, figures d'analyse |
| Talk to my Data | Chat en langage naturel avec l'agent LLM |

---

## Notebooks

| Notebook | Contenu |
|---|---|
| [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) | EDA complète : dictionnaire des données, détection du leakage, distribution de la cible, analyse socio-démo, historique de paiement, corrélations |
| [02_modelisation_baseline.ipynb](notebooks/02_modelisation_baseline.ipynb) | Pipeline de prétraitement, 3 modèles baseline (LR, RF, XGBoost), comparaison en CV 5-fold, lift par décile |
| [03_models_fintuned.ipynb](notebooks/03_models_fintuned.ipynb) | Fine-tuning RandomizedSearchCV, évaluation finale sur X_test (hold-out), export du pipeline |

---

## POC GenAI — Architecture de l'agent

L'agent suit le pattern **ReAct** (Reasoning + Acting) de LangChain v1 avec mémoire conversationnelle (`InMemorySaver`) et contexte du modèle RF injecté dans le prompt système :

```
Question utilisateur
      │
      ▼
  Agent ReAct (gpt-4o-mini, température 0)
  Prompt système : contexte métier + métriques RF (PR-AUC, ROC-AUC…)
      │
      ├─► get_dataset_info         → structure du dataset, dictionnaire des colonnes
      ├─► get_column_statistics    → statistiques descriptives d'une colonne
      ├─► get_default_rate_by_group → taux de défaut par segment (groupby)
      └─► execute_pandas_query     → code pandas personnalisé (analyses complexes)
      │
      ▼
  Réponse structurée en 3 sections :
    **Réponse**       — explication en français
    **Code exécuté**  — affiché avec coloration syntaxique (st.code)
    **Résultat**      — chiffres, tableaux ou agrégats
      +
  Expander « Etapes de l'agent » — outils appelés, arguments et résultats
```

**Mémoire :** historique de conversation isolé par session via `InMemorySaver` + `thread_id` UUID.  
**Observabilité :** traces Langfuse optionnelles (activées si `LANGFUSE_*` présents dans `.env`).  
**Contraintes :** Python/pandas uniquement, pas de SQL, pas d'accès réseau, pas d'écriture disque.

---

## Stack technique

| Couche | Technologies |
|---|---|
| Données & ML | `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| GenAI | `langchain` v1, `langchain-openai`, `langgraph` |
| Observabilité | `langfuse` (optionnel) |
| Interface | `streamlit` |
| Environnement | `python-dotenv`, `.env` |

---

## Rapport synthétique

Voir [reports/model_report.md](reports/model_report.md) pour :
- Détail du protocole d'évaluation (split, CV, hold-out)
- Feature engineering appliqué
- Comparaison baseline vs modèles tuned
- Stratégie de seuil / top K%
- Limites et recommandations

---


