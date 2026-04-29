import streamlit as st

st.title(" À propos")

st.write(
    "Cette application a été développée dans le cadre du projet Scoring de Défaut de Crédit " \
    "Elle constitue le POC GenAI « Talk to my Data » réalisé avec LangChain v1 et Streamlit."
)

st.subheader("Contexte métier")
st.write(
    "Vous intégrez la direction **Recouvrement & Risque** d'une banque de détail. "
    "L'objectif est de renforcer la politique de relance et de recouvrement en s'appuyant sur "
    "un modèle de scoring de défaut de crédit et un assistant d'analyse en langage naturel."
)

st.subheader("Jeu de données")
st.write(
    "Le dataset utilisé est un extrait tabulaire issu du dataset public de défaut de carte de crédit "
    "disponible sur **BigQuery ML datasets**. Il contient **2 965 lignes** et **26 colonnes** couvrant :"
)
st.write("- **Socio-démographiques** : `sex`, `education_level`, `marital_status`, `age`")
st.write("- **Financières** : `limit_balance`, `bill_amt_1`–`bill_amt_6`, `pay_amt_1`–`pay_amt_6`")
st.write("- **Historique de paiement** : `pay_0`, `pay_2`–`pay_6`")
st.write("- **Cible** : `default_payment_next_month` (0 = pas de défaut, 1 = défaut)")

st.subheader("Architecture du POC GenAI")
st.write(
    "L'assistant analyse le dataset en générant et exécutant du code Python (pandas) "
    "directement sur le DataFrame chargé en mémoire. Chaque réponse affiche :"
)
st.write("1. La **réponse** en français")
st.write("2. Le **code Python** exécuté")
st.write("3. Le **résultat** (tableau, agrégat ou graphique)")

st.subheader("Technologies")
st.write("- **Streamlit** — interface web interactive")
st.write("- **LangChain v1** — orchestration de l'agent GenAI")
st.write("- **Pandas** — manipulation et analyse des données")
st.write("- **Plotly Express** — graphiques interactifs")
st.write("- **scikit-learn** — pipeline ML et modèle de scoring")
st.write("- **joblib** — sérialisation du modèle entraîné")

st.subheader("Contraintes du POC")
st.write("- Pas de SQL — Python/pandas uniquement")
st.write("- Pas d'accès réseau au moment de l'analyse")
st.write("- Pas d'écriture disque depuis l'agent")
st.write(
    "- Si une question sort du périmètre du dataset, l'assistant répond : "
    "**« Impossible avec les données disponibles. »**"
)
