"""Page d'information — contexte métier, dataset, architecture et technologies du POC."""
import streamlit as st

st.title("A propos")

st.markdown(
    "Cette application a été développée dans le cadre du projet Scoring de Défaut de Crédit. "
    "Elle constitue le POC GenAI \u00ab\u00a0Talk to my Data\u00a0\u00bb réalisé avec LangChain v1 et Streamlit."
)

st.markdown("""
### Contexte métier

Vous intégrez la direction **Recouvrement & Risque** d'une banque de détail.
L'objectif est de renforcer la politique de relance et de recouvrement en s'appuyant sur
un modèle de scoring de défaut de crédit et un assistant d'analyse en langage naturel.

### Jeu de données

Dataset public de défaut de carte de crédit (**BigQuery ML datasets**) : **2\u202f965 lignes** et **26 colonnes** couvrant :

- **Socio-démographiques** : `sex`, `education_level`, `marital_status`, `age`
- **Financières** : `limit_balance`, `bill_amt_1`\u2013`bill_amt_6`, `pay_amt_1`\u2013`pay_amt_6`
- **Historique de paiement** : `pay_0`, `pay_2`\u2013`pay_6`
- **Cible** : `default_payment_next_month` (0\u00a0= pas de défaut, 1\u00a0= défaut)

### Architecture du POC GenAI

L'assistant analyse le dataset en générant et exécutant du code Python (pandas)
directement sur le DataFrame chargé en mémoire. Chaque réponse affiche :

1. La **réponse** en français
2. Le **code Python** exécuté
3. Le **résultat** (tableau, agrégat ou graphique)

### Technologies

- **Streamlit** \u2014 interface web interactive
- **LangChain v1** \u2014 orchestration de l'agent GenAI
- **Pandas** \u2014 manipulation et analyse des données
- **Plotly Express** \u2014 graphiques interactifs
- **scikit-learn** \u2014 pipeline ML et modèle de scoring
- **joblib** \u2014 sérialisation du modèle entraîné

### Contraintes du POC

- Pas de SQL \u2014 Python/pandas uniquement
- Pas d'accès réseau au moment de l'analyse
- Pas d'écriture disque depuis l'agent
- Si une question sort du périmètre du dataset, l'assistant répond : **\u00ab\u00a0Impossible avec les données disponibles.\u00a0\u00bb**
""")
