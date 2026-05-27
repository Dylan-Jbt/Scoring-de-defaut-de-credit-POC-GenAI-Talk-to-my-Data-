"""
Page de chat « Talk to my Data » — interface conversationnelle avec l'agent LLM.

Flux d'une requête :
  1. L'utilisateur saisit une question (ou clique sur un exemple)
  2. La question est envoyée à l'agent LangGraph via agent.invoke()
  3. L'agent appelle ses outils (tools.py) pour interroger le dataset
  4. La réponse (dernier AIMessage) est affichée et sauvegardée dans l'historique

Points clés :
  - thread_id UUID par session : isole la mémoire InMemorySaver de chaque utilisateur
  - pending_question : contourne la limitation Streamlit (bouton → chat_input)
  - Langfuse : observabilité optionnelle, activée si les clés sont présentes dans .env
"""

import re
import sys
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

# ──────────────────────────────────────────────────────────────────────────────
# Résolution des imports internes — app/ et app/agents/ dans sys.path
#
# Ajoute deux répertoires pour permettre les imports depuis utils/ (app/)
# et depuis agent.py / tools.py (app/agents/). Guard pour éviter les doublons.
# ──────────────────────────────────────────────────────────────────────────────
APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
if str(APP / "agents") not in sys.path:
    sys.path.insert(0, str(APP / "agents"))

# ──────────────────────────────────────────────────────────────────────────────
# _render_structured_response() — affichage sectionné d'une réponse agent
#
# Parse les trois sections imposées par le SYSTEM_PROMPT :
#   **Réponse**       → st.markdown  (explication en français)
#   **Code exécuté** → st.code(..., language="python")  (bloc coloré avec copie)
#   **Résultat**      → st.markdown  (tableau / chiffres)
# Si aucune section n'est détectée (réponse libre ou refus), fallback markdown.
# ──────────────────────────────────────────────────────────────────────────────
_SECTION_RE = re.compile(
    r'\*\*(Réponse|Code exécuté|Résultat)\s*\*\*\s*:?\s*',
    re.IGNORECASE,
)
_CODE_BLOCK_RE = re.compile(r'```(?:python)?\n?(.*?)```', re.DOTALL)


def _render_structured_response(text: str) -> None:
    """Parse et affiche une réponse agent avec sections visuelles distinctes."""
    parts = _SECTION_RE.split(text)
    if len(parts) <= 1:
        # Pas de sections structurées — réponse libre ou message de refus
        st.markdown(text)
        return

    # parts = [préambule, titre1, contenu1, titre2, contenu2, ...]
    if parts[0].strip():
        st.markdown(parts[0])

    i = 1
    while i + 1 < len(parts):
        title = parts[i].strip()
        content = parts[i + 1].strip()
        i += 2

        if not content:
            continue

        if "code" in title.lower():
            st.markdown(f"**{title}**")
            # Extrait le code hors des balises ``` si l'agent les a incluses
            code_match = _CODE_BLOCK_RE.search(content)
            st.code(
                code_match.group(1).strip() if code_match else content,
                language="python",
            )
        else:
            st.markdown(f"**{title}**")
            st.markdown(content)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration de la page — titre et introduction de la page chat
#
# set_page_config doit être appelé avant tout autre élément Streamlit.
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Talk to my Data", page_icon=None, layout="wide")
st.title("Talk to my Data")
st.markdown(
    "Posez vos questions en **langage naturel** sur le dataset de défaut de crédit. "
    "L'agent utilise ses outils d'analyse pour vous répondre avec des chiffres réels."
)

# ──────────────────────────────────────────────────────────────────────────────
# _load_agent() — import et mise en cache de l'agent LangGraph
#
# @st.cache_resource garantit que l'agent est instancié une seule fois pour
# tous les utilisateurs du même processus Streamlit (singleton partagé).
# L'import est différé (dans le corps de la fonction) pour que les erreurs
# de clé API soient capturées proprement dans le bloc try/except ci-dessous.
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement de l'agent LLM…")
def _load_agent():
    from agent import agent, build_langfuse_handler  # noqa: PLC0415
    return agent, build_langfuse_handler

try:
    agent, build_langfuse_handler = _load_agent()
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

# ──────────────────────────────────────────────────────────────────────────────
# Session state — isolation de la mémoire et de l'historique par utilisateur
#
# thread_id : UUID unique par onglet navigateur. LangGraph InMemorySaver isole
# les checkpoints par thread_id — chaque utilisateur a sa propre mémoire.
# pending_question : contourne la limitation Streamlit où un st.button() ne
# peut pas déclencher un st.chat_input() dans le même cycle de rendu. Le
# bouton stocke la question ici ; st.rerun() relance le cycle et la question
# est consommée ci-dessous comme si l'utilisateur l'avait saisie manuellement.
# ──────────────────────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
st.session_state.setdefault("messages", [])   # historique affiché dans le chat
st.session_state.setdefault("pending_question", None)

# ──────────────────────────────────────────────────────────────────────────────
# Affichage de l'historique — rendu des messages de la session courante
#
# Boucle sur st.session_state.messages (liste de dicts {role, content}).
# Chaque message est affiché avec le bon avatar (user / assistant).
# ──────────────────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_structured_response(msg["content"])
        else:
            st.markdown(msg["content"])

# ──────────────────────────────────────────────────────────────────────────────
# Exemples de questions — boutons de démarrage rapide (visibles si historique vide)
#
# Affichés uniquement au début d'une nouvelle conversation pour guider l'utilisateur.
# Un clic stocke la question dans pending_question et relance le cycle Streamlit.
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("**Exemples de questions :**")
    examples = [
        "Quel est le taux de défaut global dans le dataset ?",
        "Quel groupe d'âge présente le plus fort risque de défaut ?",
        "Quelle est la distribution du niveau d'éducation des clients ?",
        "Y a-t-il une corrélation entre la limite de crédit et le défaut ?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Zone de saisie — résolution de la question active et appel à l'agent
#
# Priorité : saisie clavier (chat_input) > bouton exemple (pending_question).
# Si une question est active, elle est envoyée à l'agent LangGraph via invoke().
# ──────────────────────────────────────────────────────────────────────────────
_chat_input = st.chat_input("Posez votre question sur les données…")

# Priorité : saisie directe > question en attente (bouton exemple)
if _chat_input:
    active_prompt = _chat_input
elif st.session_state.pending_question:
    active_prompt = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    active_prompt = None

if active_prompt:
    prompt = active_prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("L'agent analyse vos données…"):
            lf_handler = build_langfuse_handler()
            # "configurable.thread_id" est la clé attendue par LangGraph pour
            # sélectionner le bon checkpoint InMemorySaver (mémoire de l'utilisateur).
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            if lf_handler is not None:
                config["callbacks"] = [lf_handler]
                config["metadata"] = {
                    "langfuse_session_id": st.session_state.thread_id,
                    "langfuse_tags": ["talk-to-my-data", "credit-scoring"],
                }
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config=config,
                )
                # LangGraph retourne tous les messages (HumanMessage, ToolMessage, AIMessage).
                # On garde uniquement le dernier AIMessage — c'est la réponse finale.
                all_msgs = result.get("messages", [])
                ai_msgs = [m for m in all_msgs if isinstance(m, AIMessage)]
                answer = str(ai_msgs[-1].content) if ai_msgs else "L'agent n'a produit aucune réponse."
            except Exception as e:
                all_msgs = []
                answer = f"Erreur lors de l'appel à l'agent : {e}"

        # ── Étapes intermédiaires — outils appelés pendant l'exécution ────────
        # Reconstruit les paires (appel, résultat) depuis les messages LangGraph :
        # AIMessage.tool_calls contient le nom, les arguments et l'id de chaque appel ;
        # ToolMessage.tool_call_id permet de retrouver l'appel correspondant même si
        # plusieurs outils sont appelés en parallèle dans le même cycle ReAct.
        steps: dict[str, dict] = {}   # tool_call_id → {name, args, output}
        steps_order: list[str] = []   # conserve l'ordre d'appel

        for msg in all_msgs:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    call_id = tc["id"]
                    steps[call_id] = {"name": tc["name"], "args": tc.get("args", {})}
                    steps_order.append(call_id)
            elif isinstance(msg, ToolMessage):
                call_id = msg.tool_call_id
                if call_id in steps:
                    steps[call_id]["output"] = msg.content

        if steps_order:
            with st.expander(f"Etapes de l'agent — {len(steps_order)} outil(s) appelé(s)", expanded=False):
                for i, call_id in enumerate(steps_order, 1):
                    step = steps[call_id]
                    st.markdown(f"**{i}. `{step['name']}`**")
                    if step.get("args"):
                        st.json(step["args"])
                    if step.get("output"):
                        out = step["output"]
                        st.text(out if len(out) <= 800 else out[:800] + "…")

        _render_structured_response(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ──────────────────────────────────────────────────────────────────────────────
# Bouton reset — réinitialisation de la conversation dans la sidebar
#
# Efface l'historique et génère un nouveau thread_id pour repartir d'une
# mémoire LangGraph vierge (l'ancien thread reste en cache mais n'est plus utilisé).
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    if st.button("Réinitialiser la conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
