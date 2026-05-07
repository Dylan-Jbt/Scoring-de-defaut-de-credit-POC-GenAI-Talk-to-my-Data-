"""
Création de l'agent "Talk to my Data".

Pattern : create_agent (LangChain v1.0) + InMemorySaver (mémoire court-terme)

"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from config import (
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from promts import SYSTEM_PROMPT
from tools import (
    execute_pandas_query,
    get_column_statistics,
    get_dataset_info,
    get_default_rate_by_group,
)

# =============================================================================
# VÉRIFICATION DES CLÉS OBLIGATOIRES
# =============================================================================

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "La clé OPENAI_API_KEY est absente ou vide. "
        "Ajoutez-la dans le fichier .env situé à la racine du projet. "
        "Exemple : OPENAI_API_KEY=sk-..."
    )

if not OPENAI_MODEL:
    raise EnvironmentError(
        "La variable OPENAI_MODEL est absente ou vide dans config.py."
    )

# =============================================================================
# MODÈLE LLM
# =============================================================================
# Température 0 → réponses déterministes, essentielles pour l'analyse de données
# (évite les hallucinations de chiffres — cf. notebook 07, section 2.3)

_llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# =============================================================================
# MÉMOIRE CONVERSATIONNELLE (court-terme)
# =============================================================================
# InMemorySaver conserve l'historique des échanges tant que l'app tourne.
# Chaque session utilisateur est isolée via un thread_id unique.
# Cf. notebook 07, section 4 — Mémoire avec Checkpointer.

_checkpointer = InMemorySaver()

# =============================================================================
# OUTILS DISPONIBLES
# =============================================================================

AGENT_TOOLS = [
    get_dataset_info,
    get_column_statistics,
    get_default_rate_by_group,
    execute_pandas_query,
]

# =============================================================================
# CALLBACKS (observabilité Langfuse — optionnel)
# =============================================================================
# Le CallbackHandler Langfuse trace automatiquement chaque appel LLM et outil.
# Il s'active uniquement si les trois clés sont renseignées dans le .env.
# Cf. notebook 06_Observabilite_et_Monitoring_Langfuse.

def _build_callbacks() -> list:
    if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY and LANGFUSE_HOST:
        try:
            from langfuse.langchain import CallbackHandler  # noqa: PLC0415

            return [
                CallbackHandler(
                    secret_key=LANGFUSE_SECRET_KEY,
                    public_key=LANGFUSE_PUBLIC_KEY,
                    host=LANGFUSE_HOST,
                )
            ]
        except ImportError:
            pass
    return []


# =============================================================================
# CRÉATION DE L'AGENT
# =============================================================================

def create_data_agent(system_prompt: str = SYSTEM_PROMPT):
    """Instancie et retourne l'agent d'analyse des données de crédit.

    L'agent utilise le pattern ReAct (Reasoning + Acting) de LangChain v1.0 :
    il raisonne sur la requête, choisit le bon outil, observe le résultat, et
    répète jusqu'à produire une réponse finale.

    Args:
        system_prompt: Prompt système à injecter (SYSTEM_PROMPT par défaut).

    Returns:
        Agent LangChain compilé, prêt à être invoqué via .invoke().

    Exemple d'utilisation :
        agent = create_data_agent()
        config = {"configurable": {"thread_id": "session-001"}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Quel est le taux de défaut global ?"}]},
            config=config,
        )
        print(result["messages"][-1].content)
    """
    callbacks = _build_callbacks()

    llm = _llm
    if callbacks:
        # On attache les callbacks au modèle pour tracer tous les appels LLM
        llm = _llm.bind(callbacks=callbacks)

    return create_agent(
        model=llm,
        tools=AGENT_TOOLS,
        system_prompt=system_prompt,
        checkpointer=_checkpointer,
    )


# =============================================================================
# INSTANCE SINGLETON
# =============================================================================
# Instance partagée par l'application Streamlit.
# Elle est chargée une seule fois au démarrage (import du module).

agent = create_data_agent()
