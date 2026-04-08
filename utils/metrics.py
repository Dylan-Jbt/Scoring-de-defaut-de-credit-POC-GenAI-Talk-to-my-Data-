"""
Module de calcul de métriques et d'évaluation.

Ce module contient les fonctions nécessaires pour :
- Évaluer la performance des modèles
- Calculer des métriques de classification
- Analyser la qualité des prédictions
- Évaluer la calibration des modèles
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Any
import scipy.stats as ss

try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve


def process(data: List[float], factor: float) -> List[float]:
    """
    Applique une transformation linéaire à une liste de données.
    
    Multiplie chaque élément par un facteur.
    
    Args:
        data: Liste de nombres
        factor: Facteur multiplicatif
    
    Returns:
        Liste transformée
    
    Exemple:
        >>> result = process([1, 2, 3], factor=2)
        >>> print(result)  # [2, 4, 6]
    """
    return [x * factor for x in data]


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    metric: str = 'auc'
) -> Tuple[float, float]:
    """
    Évalue la performance d'un modèle sur les jeux train et test.
    
    Args:
        model: Modèle entraîné
        X_train: Données d'entraînement
        y_train: Labels d'entraînement
        X_test: Données de test
        y_test: Labels de test
        metric: Métrique à calculer ('auc', 'accuracy', etc.)
    
    Returns:
        Tuple (score_train, score_test)
    
    Exemple:
        >>> from sklearn.metrics import roc_auc_score
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> # (après entraînement du modèle)
        >>> train_score, test_score = evaluate_model(model, X_train, y_train, X_test, y_test)
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    if metric == 'auc':
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_test_pred = model.predict_proba(X_test)[:, 1]
        train_score = roc_auc_score(y_train, y_train_pred)
        test_score = roc_auc_score(y_test, y_test_pred)
    elif metric == 'accuracy':
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_score = accuracy_score(y_train, y_train_pred)
        test_score = accuracy_score(y_test, y_test_pred)
    else:
        raise ValueError(f"Métrique '{metric}' non supportée")
    
    return train_score, test_score


def confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Calcule les métriques issues de la matrice de confusion.
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
    
    Returns:
        Dictionnaire avec TP, FP, TN, FN, Sensitivity, Specificity, Precision, F1
    
    Exemple:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> metrics = confusion_matrix_metrics(y_true, y_pred)
        >>> print(f"Précision: {metrics['Precision']:.2f}")
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1': f1
    }


def calculate_roc_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> float:
    """
    Calcule l'AUC-ROC (Area Under the Receiver Operating Characteristic curve).
    
    Métrique entre 0 et 1 mesurant la capacité du modèle à distinguer les classes :
    - 0.5 = performance aléatoire
    - 1.0 = prédictions parfaites
    
    Args:
        y_true: Labels réels (0 ou 1)
        y_pred_proba: Probabilités prédites (entre 0 et 1)
    
    Returns:
        AUC score
    
    Exemple:
        >>> from sklearn.metrics import roc_auc_score
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
        >>> auc = calculate_roc_auc(y_true, y_pred)
        >>> print(f"AUC: {auc:.3f}")
    """
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred_proba)


def calculate_lift(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    percentile: float = 0.1
) -> float:
    """
    Calcule le lift pour un percentile donné.
    
    Le lift mesure combien de fois on fait mieux qu'un ciblage aléatoire.
    
    Args:
        y_true: Labels réels
        y_pred_proba: Probabilités prédites
        percentile: Percentile d'intérêt (défaut: 0.1 = top 10%)
    
    Returns:
        Valeur du lift
    
    Exemple:
        >>> y_true = np.array([0, 0, 1, 1, 1, 0, 1])
        >>> y_pred = np.array([0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        >>> lift = calculate_lift(y_true, y_pred, percentile=0.1)
        >>> print(f"Lift (top 10%): {lift:.2f}x")
    """
    # Seuil du percentile
    threshold = np.percentile(y_pred_proba, (1 - percentile) * 100)
    
    # Sélection des clients au-dessus du seuil
    selected = y_pred_proba >= threshold
    
    # Taux de réponse dans la sélection
    response_rate_selected = np.mean(y_true[selected]) if np.sum(selected) > 0 else 0
    
    # Taux de réponse global
    response_rate_global = np.mean(y_true)
    
    # Calcul du lift
    lift = response_rate_selected / response_rate_global if response_rate_global > 0 else 0
    
    return lift




# === Fonctions des Modules 5, 6, 7 === #

def score_spiegelhalter(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray
) -> float:
    """
    Calcule le score de Spiegelhalter pour évaluer la calibration d'un modèle.

    Paramètres
    ----------
    y_true : np.ndarray | pd.Series
        Labels réels (0 ou 1).
    y_pred : np.ndarray
        Probabilités prédites par le modèle.

    Retourne
    --------
    float
        Score de Spiegelhalter. Une valeur proche de 0 indique une bonne calibration.
    """
    numerateur = np.sum(np.multiply(y_true - y_pred, 1 - 2 * y_pred))
    denominateur = np.sqrt(
        np.sum(
            np.multiply(
                np.multiply(np.power(1 - 2 * y_pred, 2), y_pred), 1 - y_pred
            )
        )
    )

    return numerateur / denominateur

__all__ = [
    'process',
    'evaluate_model',
    'confusion_matrix_metrics',
    'calculate_roc_auc',
    'calculate_lift',
    'score_spiegelhalter'
]


# ====================================================================================================
# Fonctions de métriques des Modules 7
# ====================================================================================================

def score_spiegelhalter(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray
) -> float:
    """
    Calcule le score de Spiegelhalter pour évaluer la calibration d'un modèle.

    Paramètres
    ----------
    y_true : np.ndarray | pd.Series
        Labels réels (0 ou 1).
    y_pred : np.ndarray
        Probabilités prédites par le modèle.

    Retourne
    --------
    float
        Score de Spiegelhalter. Une valeur proche de 0 indique une bonne calibration.
    """
    numerateur = np.sum(np.multiply(y_true - y_pred, 1 - 2 * y_pred))
    denominateur = np.sqrt(
        np.sum(
            np.multiply(
                np.multiply(np.power(1 - 2 * y_pred, 2), y_pred), 1 - y_pred
            )
        )
    )

    return numerateur / denominateur


# ====================================================================================================
# Fonctions de calibration du Module 7
# ====================================================================================================

def sklearn_calibration(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    n_bins: int = 20
) -> pd.DataFrame:
    """
    Calcule les données de calibration en utilisant scikit-learn.

    Paramètres
    ----------
    y_true : np.ndarray | pd.Series
        Labels réels (0 ou 1).
    y_pred : np.ndarray
        Probabilités prédites par le modèle.
    n_bins : int, optional
        Nombre d'intervalles pour le découpage (défaut: 20).

    Retourne
    --------
    pd.DataFrame
        DataFrame avec les colonnes 'prob_pred' et 'prob_true'.
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred,
        n_bins=n_bins,
        strategy="quantile"
    )

    return pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})
