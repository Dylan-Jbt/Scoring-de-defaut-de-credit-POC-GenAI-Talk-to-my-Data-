"""
Initialisation de l'agent "Talk to my Data" — LangChain v1.0 + LangGraph.

Architecture :
    LLM          : ChatOpenAI (gpt-4o-mini, température 0 — réponses reproductibles
                   et déterministes sur des données chiffrées)
    Agent        : create_agent (boucle ReAct via LangGraph) — l'agent appelle ses
                   outils en séquence jusqu'à pouvoir formuler une réponse complète
    Mémoire      : InMemorySaver — conserve l'historique de conversation par thread_id ;
                   chaque utilisateur est isolé via un UUID généré côté Streamlit
    Observabilité : Langfuse (optionnel) — trace chaque appel LLM, tool-call et latence ;
                   désactivé automatiquement si les clés LANGFUSE_* sont absentes du .env

Pattern singleton :
    `agent` est instancié une seule fois au chargement du module (import-time).
    Streamlit le met ensuite en cache via @st.cache_resource pour éviter de recréer
    le graphe LangGraph à chaque rechargement de page.
    `build_langfuse_handler()` est appelé à chaque requête pour créer un
    CallbackHandler frais — nécessaire car Langfuse lie chaque handler à une seule trace.

Exports :
    agent                    — CompiledStateGraph (LangGraph) prêt à invoquer
    build_langfuse_handler() — CallbackHandler Langfuse par requête, ou None
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, OPENAI_API_KEY, OPENAI_MODEL
from promts import SYSTEM_PROMPT
from tools import execute_pandas_query, get_column_statistics, get_dataset_info, get_default_rate_by_group

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY absente. Ajoutez-la dans le fichier .env : OPENAI_API_KEY=sk-..."
    )

# ──────────────────────────────────────────────────────────────────────────────
# LLM — ChatOpenAI avec température 0 (réponses déterministes sur les données)
#
# gpt-4o-mini : bon équilibre performance/coût pour des analyses tabulaires.
# température=0 garantit que la même question produit toujours le même résultat.
# ──────────────────────────────────────────────────────────────────────────────
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# Observabilité Langfuse — initialisation optionnelle au démarrage du module
#
# Tente de se connecter à Langfuse au chargement. En cas d'échec (clés absentes,
# import manquant, réseau indisponible…), _langfuse_enabled reste False et aucune
# trace n'est envoyée — l'agent continue de fonctionner sans observabilité.
# ──────────────────────────────────────────────────────────────────────────────
_langfuse_enabled = False
if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
    try:
        from langfuse import Langfuse  # noqa: PLC0415
        Langfuse(secret_key=LANGFUSE_SECRET_KEY, public_key=LANGFUSE_PUBLIC_KEY, host=LANGFUSE_HOST)
        _langfuse_enabled = True
    except Exception:  # noqa: BLE001
        pass


# ──────────────────────────────────────────────────────────────────────────────
# build_langfuse_handler() — CallbackHandler Langfuse par requête
#
# Retourne None si Langfuse est désactivé (clés absentes ou import impossible).
# Un handler frais est requis à chaque invocation de l'agent : Langfuse lie
# chaque CallbackHandler à une seule trace — réutiliser le même mélangerait
# les requêtes successives dans la même trace Langfuse.
# ──────────────────────────────────────────────────────────────────────────────
def build_langfuse_handler():
    """Retourne un CallbackHandler Langfuse frais par invocation, ou None."""
    if not _langfuse_enabled:
        return None
    try:
        from langfuse.langchain import CallbackHandler  # noqa: PLC0415
        return CallbackHandler()
    except ImportError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Agent LangGraph — singleton chargé une seule fois au démarrage du module
#
# create_agent() compile le graphe ReAct : instancie le modèle, lie les outils
# et configure InMemorySaver pour la mémoire par thread_id.
# Streamlit met ensuite ce module en cache via @st.cache_resource.
# ──────────────────────────────────────────────────────────────────────────────
agent = create_agent(
    model=_llm,
    tools=[get_dataset_info, get_column_statistics, get_default_rate_by_group, execute_pandas_query],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)
