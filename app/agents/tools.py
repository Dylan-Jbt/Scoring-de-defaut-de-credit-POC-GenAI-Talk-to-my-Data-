"""
Outils (tools) pour l'agent "Talk to my Data".

Chaque outil expose une capacité d'analyse sur le dataset de défaut de crédit
et suit le pattern @tool de LangChain v1.0.

Outils disponibles :
    get_dataset_info()           — structure du dataset et descriptions des colonnes
    get_column_statistics()      — statistiques descriptives d'une colonne
    get_default_rate_by_group()  — taux de défaut segmenté par groupe
    execute_pandas_query()       — exécution sandboxed de code pandas personnalisé

Tous les outils accèdent au même DataFrame via _get_df() (chargement paresseux).
Le sandboxing d'execute_pandas_query() est géré par _FORBIDDEN_PATTERNS + _SAFE_NAMES.
"""

import builtins as _builtins
import io
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain.tools import tool

# ──────────────────────────────────────────────────────────────────────────────
# Chargement paresseux du dataset
#
# Sentinel initialisé à None. Lors du premier appel à _get_df(), le CSV est lu
# et stocké ici pour toute la durée de vie du processus Streamlit.
# Les appels suivants court-circuitent la lecture disque et retournent ce cache.
# ──────────────────────────────────────────────────────────────────────────────
_DF: Optional[pd.DataFrame] = None


# ──────────────────────────────────────────────────────────────────────────────
# _get_df() — accès centralisé au DataFrame
#
# Lit credit_card_default.csv en remontant l'arborescence (racine → app/) pour
# fonctionner quel que soit le répertoire de lancement. Exclut la colonne de
# prédictions pré-calculées pour n'exposer à l'agent que les variables d'entrée.
# ──────────────────────────────────────────────────────────────────────────────
def _get_df() -> pd.DataFrame:
    """Retourne le DataFrame principal (chargé une seule fois en mémoire)."""
    global _DF
    if _DF is not None:
        return _DF
    # Remonte deux niveaux depuis tools.py pour localiser data/credit_card_default.csv.
    # parents[3] = racine du projet | parents[2] = app/
    # Cette double recherche rend le chargement robuste quel que soit le
    # répertoire de lancement (app/agents/, app/, racine, Docker…).
    for base in [Path(__file__).parents[3], Path(__file__).parents[2]]:
        candidate = base / "data" / "credit_card_default.csv"
        if candidate.exists():
            # Supprime la colonne de prédictions pré-calculées si elle existe.
            # Elle n'est pas une variable d'entrée du modèle et ne doit pas
            # être accessible à l'agent (fuite d'information vers la cible).
            _DF = pd.read_csv(candidate).drop(
                columns=["predicted_default_payment_next_month"], errors="ignore"
            )
            return _DF
    raise FileNotFoundError("Fichier de données introuvable : data/credit_card_default.csv")


# Descriptions des colonnes (contexte métier injecté dans les prompts)
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "id": "Identifiant unique du client",
    "limit_balance": "Limite de crédit en NT$ (Nouveau dollar taïwanais)",
    "sex": "Sexe du client (1=Homme, 2=Femme)",
    "education_level": (
        "Niveau d'éducation (1=Études supérieures, 2=Université, 3=Lycée, 4=Autre)"
    ),
    "marital_status": "Statut marital (1=Marié(e), 2=Célibataire, 3=Autre)",
    "age": "Âge du client en années",
    "pay_0": (
        "Statut de paiement du mois précédent "
        "(-2=pas de consommation, -1=à temps, 0=revolving, 1=1 mois de retard, ...)"
    ),
    "pay_2": "Statut de paiement il y a 2 mois (même échelle que pay_0)",
    "pay_3": "Statut de paiement il y a 3 mois",
    "pay_4": "Statut de paiement il y a 4 mois",
    "pay_5": "Statut de paiement il y a 5 mois",
    "pay_6": "Statut de paiement il y a 6 mois",
    "bill_amt_1": "Montant facturé le mois précédent (NT$)",
    "bill_amt_2": "Montant facturé il y a 2 mois (NT$)",
    "bill_amt_3": "Montant facturé il y a 3 mois (NT$)",
    "bill_amt_4": "Montant facturé il y a 4 mois (NT$)",
    "bill_amt_5": "Montant facturé il y a 5 mois (NT$)",
    "bill_amt_6": "Montant facturé il y a 6 mois (NT$)",
    "pay_amt_1": "Montant remboursé le mois précédent (NT$)",
    "pay_amt_2": "Montant remboursé il y a 2 mois (NT$)",
    "pay_amt_3": "Montant remboursé il y a 3 mois (NT$)",
    "pay_amt_4": "Montant remboursé il y a 4 mois (NT$)",
    "pay_amt_5": "Montant remboursé il y a 5 mois (NT$)",
    "pay_amt_6": "Montant remboursé il y a 6 mois (NT$)",
    "default_payment_next_month": (
        "Variable cible : défaut de paiement le mois suivant (0=non, 1=oui)"
    ),
}

# Opérations interdites dans le sandbox d'exécution
_FORBIDDEN_PATTERNS = [
    "open(", "write(", "to_csv(", "to_excel(", "to_json(", "to_parquet(", "to_pickle(",
    "import os", "import sys", "import subprocess",
    "socket", "urllib", "requests", "http",
    "__import__", "eval(", "exec(", "compile(",
]

# Builtins autorisés dans le sandbox d'exécution — ajouter ici pour étendre
_SAFE_NAMES = {
    "len", "range", "enumerate", "zip", "list", "dict", "set", "tuple",
    "str", "int", "float", "bool", "round", "abs", "min", "max", "sum",
    "sorted", "reversed",
}


# ──────────────────────────────────────────────────────────────────────────────
# Outil 1 : informations générales sur le dataset
#
# Calcule le taux de défaut global (.mean() sur colonne binaire) et construit
# un bloc texte avec les KPIs principaux (lignes, colonnes, taux) suivi de la
# description métier de chaque colonne (COLUMN_DESCRIPTIONS). Appelé en priorité
# par l'agent pour comprendre la structure du dataset avant toute autre analyse.
# ──────────────────────────────────────────────────────────────────────────────
@tool
def get_dataset_info() -> str:
    """Retourne les informations générales sur le dataset de défaut de crédit.

    Fournit : nombre de lignes, colonnes, taux de défaut global,
    et une description métier de chaque variable.

    Returns:
        Description complète du dataset sous forme de texte structuré.
    """
    df = _get_df()
    # .mean() sur une colonne binaire (0/1) donne directement le taux de défaut
    # (proportion de 1, c'est-à-dire de clients ayant fait défaut dans le dataset).
    target_rate = df["default_payment_next_month"].mean()

    lines = [
        "DATASET : Défaut de carte de crédit — Banque de détail",
        f"Lignes       : {len(df):,}",
        f"Colonnes     : {len(df.columns)}",
        f"Taux défaut  : {target_rate:.1%}",
        f"Clients en défaut : {int(df['default_payment_next_month'].sum()):,}",
        "",
        "COLONNES ET DESCRIPTIONS :",
    ]
    for col in df.columns:
        desc = COLUMN_DESCRIPTIONS.get(col, "Colonne du dataset")
        lines.append(f"  - {col} : {desc}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Outil 2 : statistiques descriptives d'une colonne
#
# Bascule automatiquement entre deux modes selon le type de variable :
#   - Numérique continue  (nunique > 15) : 8 statistiques via .describe()
#   - Catégorielle / discrète (≤15 valeurs) : distribution des modalités
# Retourne un message d'erreur explicite si la colonne demandée n'existe pas.
# ──────────────────────────────────────────────────────────────────────────────
@tool
def get_column_statistics(column_name: str) -> str:
    """Calcule et retourne les statistiques descriptives d'une colonne du dataset.

    Pour les variables numériques continues : min, max, moyenne, médiane, écart-type.
    Pour les variables catégorielles / discrètes : distribution des modalités.

    Args:
        column_name: Nom exact de la colonne à analyser (sensible à la casse).

    Returns:
        Statistiques de la colonne sous forme de texte.
    """
    df = _get_df()

    if column_name not in df.columns:
        available = ", ".join(df.columns.tolist())
        return (
            f"Erreur : la colonne '{column_name}' n'existe pas.\n"
            f"Colonnes disponibles : {available}"
        )

    series = df[column_name]
    lines = [f"STATISTIQUES — colonne '{column_name}' :"]

    # Seuil de 15 valeurs uniques : en dessous, la variable est considérée
    # comme catégorielle même si son dtype est int64
    # (ex : sex=2 modalités, education_level=4, marital_status=3).
    # Au-dessus, c'est une variable continue (ex : age, limit_balance, bill_amt_*).
    if series.dtype in ["int64", "float64"] and series.nunique() > 15:
        # .describe() calcule en une seule passe : count, mean, std,
        # min, 25%, 50% (médiane), 75%, max. Les clés sont accessibles par nom.
        stats = series.describe()
        lines += [
            "  Type        : numérique continu",
            f"  Comptage    : {int(stats['count']):,}",
            f"  Moyenne     : {stats['mean']:.4f}",
            f"  Écart-type  : {stats['std']:.4f}",
            f"  Minimum     : {stats['min']:.4f}",
            f"  Q25         : {stats['25%']:.4f}",
            f"  Médiane     : {stats['50%']:.4f}",
            f"  Q75         : {stats['75%']:.4f}",
            f"  Maximum     : {stats['max']:.4f}",
            f"  Valeurs nulles : {int(series.isna().sum())}",
        ]
    else:
        # sort_index() trie les modalités par valeur (1, 2, 3, 4…) plutôt que
        # par effectif décroissant — affichage ordonné et reproductible pour l'agent.
        vc = series.value_counts().sort_index()
        lines += [
            "  Type       : catégoriel / discret",
            f"  Modalités  : {series.nunique()}",
            "  Distribution :",
        ]
        for val, count in vc.items():
            lines.append(f"    {val} → {count:,} clients ({count / len(series):.1%})")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Outil 3 : taux de défaut par groupe
#
# Segmente le dataset par une variable et calcule pour chaque modalité :
# taux de défaut (mean), effectif (count) et nombre de défauts (sum) en une passe.
# Résultat trié par taux décroissant pour mettre en tête les segments à risque,
# facilitant la priorisation des actions de relance et de recouvrement.
# ──────────────────────────────────────────────────────────────────────────────
@tool
def get_default_rate_by_group(column_name: str) -> str:
    """Calcule le taux de défaut pour chaque modalité d'une variable du dataset.

    Permet d'identifier quels segments de clients présentent le plus fort risque
    de défaut (utile pour le ciblage des actions de recouvrement).

    Args:
        column_name: Nom de la colonne de regroupement
                     (ex : 'sex', 'education_level', 'marital_status', 'pay_0').

    Returns:
        Tableau texte : groupe | taux_défaut | nb_clients | nb_défauts,
        trié par taux décroissant.
    """
    df = _get_df()
    target = "default_payment_next_month"

    if column_name not in df.columns:
        return f"Erreur : la colonne '{column_name}' n'existe pas dans le dataset."

    if target not in df.columns:
        return f"Erreur : la variable cible '{target}' est absente du dataset."

    # Calcule en une seule passe : taux de défaut (mean), effectif (count),
    # nombre de défauts (sum). Trié par taux décroissant pour mettre en tête
    # les segments les plus risqués — utile pour prioriser les actions de relance.
    grouped = (
        df.groupby(column_name)[target]
        .agg(["mean", "count", "sum"])
        .rename(
            columns={
                "mean": "taux_defaut",
                "count": "nb_clients",
                "sum": "nb_defauts",
            }
        )
        .sort_values("taux_defaut", ascending=False)
        .reset_index()
    )

    lines = [
        f"TAUX DE DÉFAUT PAR '{column_name}' (trié par risque décroissant) :",
        f"{'Groupe':<15} {'Taux défaut':>12} {'Nb clients':>12} {'Nb défauts':>12}",
        "-" * 55,
    ]
    for _, row in grouped.iterrows():
        lines.append(
            f"{str(row[column_name]):<15} "
            f"{row['taux_defaut']:>11.1%} "
            f"{int(row['nb_clients']):>12,} "
            f"{int(row['nb_defauts']):>12,}"
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Outil 4 : exécution de code pandas personnalisé (sandbox)
#
# Permet à l'agent d'écrire du code Python/pandas ad hoc pour des analyses
# non couvertes par les 3 outils précédents. Flux d'exécution :
#   1. Vérifie l'absence de patterns interdits (_FORBIDDEN_PATTERNS)
#   2. Construit un namespace isolé : df (copie), pd, np + builtins restreints
#   3. Exécute via exec(compile(...)) — compile() vérifie la syntaxe avant
#   4. Collecte la sortie : print() capturé OU variable `result` assignée
# ──────────────────────────────────────────────────────────────────────────────
@tool
def execute_pandas_query(code: str) -> str:
    """Exécute du code Python/pandas sur le dataset de défaut de crédit.

    Le DataFrame est accessible via la variable `df`.
    Les bibliothèques `pd` (pandas) et `np` (numpy) sont pré-importées.

    Pour afficher un résultat, utilisez `print()` ou assignez-le à `result`.
    Exemple : `result = df.groupby('sex')['age'].mean()`

    Contraintes de sécurité :
    - Pas d'accès réseau, pas d'écriture disque
    - Opérations de lecture uniquement sur le dataset
    - Les opérations `open()`, `import os`, `import sys`, etc. sont bloquées

    Args:
        code: Code Python utilisant pandas sur le DataFrame `df`.

    Returns:
        Résultat de l'exécution (stdout capturé + valeur de `result` si définie).
    """
    # Sécurité : détecte la première opération interdite
    if bad := next((p for p in _FORBIDDEN_PATTERNS if p in code), None):
        return (
            f"Erreur de sécurité : opération interdite détectée ('{bad}'). "
            "Seules les opérations pandas/numpy de lecture sont autorisées."
        )

    import numpy as np  # noqa: PLC0415  (import local intentionnel)

    stdout_capture = io.StringIO()
    # Namespace injecté dans exec() — ce que le code soumis peut voir et utiliser :
    # - df         : copie du dataset (les modifications ne touchent pas l'original)
    # - pd / np    : pandas et numpy disponibles sans import dans le code
    # - __builtins__ : réduit aux fonctions sûres de _SAFE_NAMES ;
    #                  print() est redéfini pour écrire dans stdout_capture
    #                  plutôt que sur la console, afin de capturer la sortie.
    namespace: dict = {
        "df": _get_df().copy(),
        "pd": pd,
        "np": np,
        "__builtins__": {
            **{name: getattr(_builtins, name) for name in _SAFE_NAMES},
            "print": lambda *args, **kwargs: print(*args, **kwargs, file=stdout_capture),
        },
    }

    try:
        # compile() vérifie la syntaxe avant d'exécuter et assigne le nom "<analyst>"
        # au fichier source — ce nom apparaît dans les tracebacks pour identifier
        # l'origine du code (distincts des erreurs internes de l'application).
        exec(compile(code, "<analyst>", "exec"), namespace)  # noqa: S102
    except Exception:
        return f"Erreur lors de l'exécution :\n{traceback.format_exc(limit=3)}"

    # Deux façons de retourner un résultat depuis le code soumis :
    # 1. print()  → capturé dans stdout_capture, retourné tel quel
    # 2. result   → variable assignée dans le code → convertie en texte structuré
    #              (to_string() pour DataFrame/Series, str() sinon)
    parts: list[str] = []
    if out := stdout_capture.getvalue().strip():
        parts.append(out)
    if (res := namespace.get("result")) is not None:
        parts.append(res.to_string() if isinstance(res, (pd.DataFrame, pd.Series)) else str(res))

    return "\n\n".join(parts) if parts else "Code exécuté avec succès (aucune sortie produite)."
