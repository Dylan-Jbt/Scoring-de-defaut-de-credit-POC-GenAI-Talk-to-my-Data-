"""Page de chat « Talk to my Data » — interface conversationnelle avec l'agent LLM."""

import sys
import uuid
from pathlib import Path

import streamlit as st

# ── Résolution des imports internes ─────────────────────────────────────────
_APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_APP))
sys.path.insert(0, str(_APP / "agents" / "config"))

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Talk to my Data", page_icon="🤖", layout="wide")
st.title("🤖 Talk to my Data")
st.markdown(
    "Posez vos questions en **langage naturel** sur le dataset de défaut de crédit. "
    "L'agent utilise ses outils d'analyse pour vous répondre avec des chiffres réels."
)

# ── Chargement de l'agent (avec gestion d'erreur claire) ────────────────────
@st.cache_resource(show_spinner="Chargement de l'agent LLM…")
def _load_agent():
    from agent import agent  # noqa: PLC0415
    return agent

try:
    agent = _load_agent()
except EnvironmentError as e:
    st.error(
        f"**Clé API manquante** : {e}\n\n"
        "Créez un fichier `.env` à la racine du projet avec :\n"
        "```\nOPENAI_API_KEY=sk-...\n```"
    )
    st.stop()
except Exception as e:
    st.error(f"**Erreur au chargement de l'agent** : {e}")
    st.stop()

# ── Session : thread_id unique par utilisateur ───────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Affichage de l'historique ────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Exemples de questions ────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("**💡 Exemples de questions :**")
    examples = [
        "Quel est le taux de défaut global dans le dataset ?",
        "Quel groupe d'âge présente le plus fort risque de défaut ?",
        "Quelle est la distribution du niveau d'éducation des clients ?",
        "Y a-t-il une corrélation entre la limite de crédit et le défaut ?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()

# ── Zone de saisie ───────────────────────────────────────────────────────────
if prompt := st.chat_input("Posez votre question sur les données…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("L'agent analyse vos données…"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config=config,
                )
                answer = result["messages"][-1].content
            except Exception as e:
                answer = f"❌ Erreur lors de l'appel à l'agent : {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Bouton reset ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    if st.button("🗑️ Réinitialiser la conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
