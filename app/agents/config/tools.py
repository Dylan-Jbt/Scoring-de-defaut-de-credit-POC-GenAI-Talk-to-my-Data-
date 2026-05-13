"""
Outils (tools) pour l'agent "Talk to my Data".

Chaque outil expose une capacité d'analyse sur le dataset de défaut de crédit.
Ils suivent le pattern @tool de LangChain v1.0
"""

import io
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain.tools import tool

# =============================================================================
# CHARGEMENT PARESSEUX DU DATASET
# =============================================================================
# Stratégie identique à app/utils/data.py : on cherche d'abord à la racine
# du projet, puis dans app/.  Le chargement n'est effectué qu'une seule fois.

_CSV_PATH: Optional[Path] = None
_DF: Optional[pd.DataFrame] = None


def _get_csv_path() -> Path:
    global _CSV_PATH
    if _CSV_PATH is not None:
        return _CSV_PATH
    for base in [Path(__file__).parents[3], Path(__file__).parents[2]]:
        candidate = base / "data" / "credit_card_default.csv"
        if candidate.exists():
            _CSV_PATH = candidate
            return _CSV_PATH
    raise FileNotFoundError(
        "Fichier de données introuvable : data/credit_card_default.csv"
    )


def _get_df() -> pd.DataFrame:
    """Retourne le DataFrame principal (chargé une seule fois en mémoire)."""
    global _DF
    if _DF is not None:
        return _DF
    _DF = pd.read_csv(_get_csv_path())
    _DF = _DF.drop(columns=["predicted_default_payment_next_month"], errors="ignore")
    return _DF


# =============================================================================
# DESCRIPTIONS DES COLONNES (contexte métier injecté dans les prompts)
# =============================================================================

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

# =============================================================================
# OPÉRATIONS INTERDITES (sécurité des exécutions de code)
# =============================================================================

_FORBIDDEN_PATTERNS = [
    "open(",
    "write(",
    "to_csv(",
    "to_excel(",
    "to_json(",
    "to_parquet(",
    "to_pickle(",
    "import os",
    "import sys",
    "import subprocess",
    "socket",
    "urllib",
    "requests",
    "http",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
]


# =============================================================================
# OUTIL 1 : informations générales sur le dataset
# =============================================================================

@tool
def get_dataset_info() -> str:
    """Retourne les informations générales sur le dataset de défaut de crédit.

    Fournit : nombre de lignes, colonnes, taux de défaut global,
    et une description métier de chaque variable.

    Returns:
        Description complète du dataset sous forme de texte structuré.
    """
    df = _get_df()
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


# =============================================================================
# OUTIL 2 : statistiques descriptives d'une colonne
# =============================================================================

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

    if series.dtype in ["int64", "float64"] and series.nunique() > 15:
        stats = series.describe()
        lines += [
            f"  Type        : numérique continu",
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
        vc = series.value_counts().sort_index()
        lines += [
            f"  Type       : catégoriel / discret",
            f"  Modalités  : {series.nunique()}",
            f"  Distribution :",
        ]
        for val, count in vc.items():
            lines.append(f"    {val} → {count:,} clients ({count / len(series):.1%})")

    return "\n".join(lines)


# =============================================================================
# OUTIL 3 : taux de défaut par groupe
# =============================================================================

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


# =============================================================================
# OUTIL 4 : exécution de code pandas personnalisé
# =============================================================================

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
    # ── Vérification de sécurité ────────────────────────────────────────────
    for pattern in _FORBIDDEN_PATTERNS:
        if pattern in code:
            return (
                f"Erreur de sécurité : opération interdite détectée ('{pattern}'). "
                "Seules les opérations pandas/numpy de lecture sont autorisées."
            )

    # ── Préparation du namespace d'exécution ────────────────────────────────
    import numpy as np  # noqa: PLC0415  (import local intentionnel)

    df = _get_df()
    stdout_capture = io.StringIO()

    safe_builtins = {
        "print": lambda *args, **kwargs: print(*args, **kwargs, file=stdout_capture),
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "round": round,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "reversed": reversed,
        "True": True,
        "False": False,
        "None": None,
    }

    namespace: dict = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "__builtins__": safe_builtins,
    }

    # ── Exécution du code ────────────────────────────────────────────────────
    try:
        exec(compile(code, "<analyst>", "exec"), namespace)  # noqa: S102
    except Exception:
        return f"Erreur lors de l'exécution :\n{traceback.format_exc(limit=3)}"

    # ── Collecte des sorties ─────────────────────────────────────────────────
    output_parts: list[str] = []

    stdout_val = stdout_capture.getvalue()
    if stdout_val.strip():
        output_parts.append(stdout_val.strip())

    if "result" in namespace and namespace["result"] is not None:
        result = namespace["result"]
        if isinstance(result, pd.DataFrame):
            output_parts.append(result.to_string())
        elif isinstance(result, pd.Series):
            output_parts.append(result.to_string())
        else:
            output_parts.append(str(result))

    if not output_parts:
        return "Code exécuté avec succès (aucune sortie produite)."

    return "\n\n".join(output_parts)
