"""
Module de prédiction et inférence.

Ce module contient les fonctions nécessaires pour :
- Faire des prédictions avec les modèles entraînés
- Scorer et classer les données
- Analyser les déciles et gains marketing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


def compute_gains(
    scores_array: np.ndarray,
    labels_array: np.ndarray,
    n_deciles: int = 10
) -> pd.DataFrame:
    """
    Calcule le lift et le gain par décile pour une analyse de scoring/marketing.
    
    Cette fonction est essentielle pour évaluer la performance d'un modèle 
    de scoring en contexte marketing. Elle segmente les données en déciles
    (groupes de 10% ordonnés par score décroissant) et calcule pour chaque décile:
    - Le nombre de clients
    - Le nombre de cibles (conversions/souscriptions)
    - Le taux de réponse (cibles/clients)
    - Le lift (écart à la moyenne)
    - La capture cumulée
    
    Args:
        scores_array: Array contenant les scores de prédiction (entre 0 et 1)
        labels_array: Array contenant les labels réels (0 ou 1)
        n_deciles: Nombre de déciles (défaut: 10)
    
    Returns:
        DataFrame avec colonnes:
        - Decile: Numéro du décile (1 = meilleurs scores)
        - Clients: Nombre de clients dans le décile
        - Cibles: Nombre de véritables positifs
        - Taux: Taux de réponse (cibles/clients)
        - Lift: Multiplicateur par rapport au modèle aléatoire
        - Capture_Cumul: % de la cible totale capturée jusqu'à ce décile
    
    Exemple:
        >>> import numpy as np
        >>> scores = np.array([0.95, 0.87, 0.65, 0.52, 0.38, 0.25, 0.15, 0.08])
        >>> labels = np.array([1, 1, 0, 1, 0, 0, 0, 0])
        >>> gains = compute_gains(scores, labels, n_deciles=4)
        >>> print(gains)
        
    Notes:
        Lift > 1.0 = le modèle surperforme par rapport au hasard
        Lift = 1.0 = performance égale au hasard
        Lift < 1.0 = performance inférieure au hasard
        
        Exemple d'interprétation:
        - Décile 1: Lift = 3.5 → En ciblant le top 10%, on touche 3.5x plus de cibles
        - Décile 2: Capture_Cumul = 62% → Top 20% capturent 62% de la cible totale
    """
    df_gains = pd.DataFrame({
        'score': scores_array,
        'label': labels_array
    })
    
    # Attribution des déciles (1 = meilleurs scores)
    df_gains['decile'] = pd.qcut(
        df_gains['score'], 
        q=n_deciles, 
        labels=False, 
        duplicates='drop'
    )
    df_gains['decile'] = n_deciles - df_gains['decile']  # Inverser
    
    # Taux de réponse global
    global_rate = df_gains['label'].mean()
    
    gains = []
    cumul = 0
    total_targets = df_gains['label'].sum()
    
    # Calculer les métriques par décile
    for dec in sorted(df_gains['decile'].unique()):
        dec_data = df_gains[df_gains['decile'] == dec]
        n_targets = dec_data['label'].sum()
        cumul += n_targets
        
        gains.append({
            'Decile': int(dec),
            'Clients': len(dec_data),
            'Cibles': int(n_targets),
            'Taux': n_targets / len(dec_data) if len(dec_data) > 0 else 0,
            'Lift': (n_targets / len(dec_data) / global_rate) if len(dec_data) > 0 else 0,
            'Capture_Cumul': cumul / total_targets if total_targets > 0 else 0
        })
    
    return pd.DataFrame(gains)


def predict_scores(
    model: Any,
    X: pd.DataFrame | np.ndarray
) -> np.ndarray:
    """
    Prédit les probabilités de scoring pour un ensemble de données.
    
    Args:
        model: Modèle entraîné ayant une méthode predict_proba()
        X: Données d'entrée
    
    Returns:
        Array de probabilités (classe positive)
    """
    return model.predict_proba(X)[:, 1]


def rank_by_score(
    scores_array: np.ndarray,
    ids: np.ndarray | None = None,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Classe les éléments par score et retourne un DataFrame trié.
    
    Args:
        scores_array: Array de scores
        ids: Identifiants (optionnel)
        ascending: Tri croissant si True, décroissant si False (défaut)
    
    Returns:
        DataFrame trié avec 'id' et 'score'
    
    Exemple:
        >>> scores = np.array([0.8, 0.5, 0.9])
        >>> ids = np.array([1, 2, 3])
        >>> ranked = rank_by_score(scores, ids)
        >>> print(ranked)
           id  score
        0   3    0.9
        1   1    0.8
        2   2    0.5
    """
    if ids is None:
        ids = np.arange(len(scores_array))
    
    df = pd.DataFrame({'id': ids, 'score': scores_array})
    return df.sort_values('score', ascending=ascending).reset_index(drop=True)




# === Fonctions des Modules 5, 6, 7 === #

def predict_reg_lineaire(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Applique une régression linéaire sur le jeu de données.

    Paramètres
    ----------
    X : np.ndarray de forme (n_observations, n_features)
        Matrice des variables explicatives
    beta : np.ndarray de forme (n_features + 1,)
        Vecteur de paramètres estimés sur le jeu d'entraînement (incluant l'intercept)

    Retourne
    --------
    y_pred : np.ndarray de forme (n_observations,)
        Vecteur des prédictions

    Notes
    -----
    - L'intercept (β₀) est automatiquement géré
    - La fonction ajoute une colonne de 1 pour l'intercept avant la prédiction

    Exemples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> beta = np.array([0.5, 1.0, 0.8])
    >>> y_pred = predict_reg_lineaire(X, beta)
    """

    # Etape 1
    # Ici on crée un vecteur NumPy contenant uniquement des 1
    # Ce vecteur contient une seule colonne (contenant que des 1) et autant de lignes qu'il y a
    # d'observations dans le jeu de données
    # Ce vecteur est créé pour pouvoir attribuer beta_0 à la constante du modèle linéaire

    constante = np.ones((X.shape[0], 1))

    # Etape 2
    # Ajoute la constante à la matrice des variables explicatives
    X = np.concatenate((constante, X), axis=1)

    # Réalisez la multiplication matricielle permettant d'appliquer les coefficients à la matrice X
    y_pred = np.matmul(beta, X.T)

    # Renvoie le vecteur de prédictions
    return y_pred

def predict_reg_ridge(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Applique une régression Ridge sur le jeu de données.

    Paramètres
    ----------
    X : np.ndarray de forme (n_observations, n_features)
        Matrice des variables explicatives
    beta : np.ndarray de forme (n_features + 1,)
        Vecteur de paramètres estimés sur le jeu d'entraînement (incluant l'intercept)

    Retourne
    --------
    y_pred : np.ndarray de forme (n_observations,)
        Vecteur des prédictions

    Notes
    -----
    - L'intercept (β₀) est automatiquement géré
    - La fonction ajoute une colonne de 1 pour l'intercept avant la prédiction
    - Cette fonction est identique à predict_reg_lineaire (pas de λ en prédiction)

    Exemples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> beta = np.array([0.5, 1.0, 0.8])
    >>> y_pred = predict_reg_ridge(X, beta)
    """

    # Étape 1 : Création du vecteur de constantes
    # Ici on crée un vecteur NumPy contenant uniquement des 1
    # Ce vecteur contient une seule colonne (contenant que des 1) et autant de lignes qu'il y a
    # d'observations dans le jeu de données
    # Ce vecteur est créé pour pouvoir attribuer beta_0 à la constante du modèle linéaire
    constante = np.ones((X.shape[0], 1))

    # Étape 2 : Ajout de la constante à la matrice des variables explicatives
    X = np.concatenate((constante, X), axis=1)

    # Réalisez la multiplication matricielle permettant d'appliquer les coefficients à la matrice X
    y_pred = np.matmul(beta, X.T)

    # Renvoie le vecteur de prédictions
    return y_pred

def predict_reg_lasso(X: np.ndarray, beta_0: float, beta_j: np.ndarray) -> np.ndarray:
    """
    Prédit la target pour une matrice X donnée en utilisant les paramètres estimés.

    Applique la formule : y_pred = X * beta_j + beta_0

    Paramètres
    ----------
    X : np.ndarray
        Matrice de features de dimensions (N, P).
    beta_0 : float
        Valeur de beta_0 aussi appelé intercept ou biais.
    beta_j : np.ndarray
        Vecteur de dimension (P,) contenant les coefficients associés
        aux variables explicatives.

    Retourne
    --------
    np.ndarray
        Prédictions de dimension (N,).
    """
    # Réalisation des prédictions
    y_pred = X.dot(beta_j) + beta_0
    return y_pred

__all__ = [
    'compute_gains',
    'predict_scores',
    'rank_by_score',
    'predict_reg_lineaire',
    'predict_reg_ridge',
    'predict_reg_lasso'
]


# ====================================================================================================
# Fonctions d'inférence des Modules 5-7
# ====================================================================================================

def predict_reg_lineaire(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Applique une régression linéaire sur le jeu de données.

    Paramètres
    ----------
    X : np.ndarray de forme (n_observations, n_features)
        Matrice des variables explicatives
    beta : np.ndarray de forme (n_features + 1,)
        Vecteur de paramètres estimés sur le jeu d'entraînement (incluant l'intercept)

    Retourne
    --------
    y_pred : np.ndarray de forme (n_observations,)
        Vecteur des prédictions

    Notes
    -----
    - L'intercept (β₀) est automatiquement géré
    - La fonction ajoute une colonne de 1 pour l'intercept avant la prédiction

    Exemples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> beta = np.array([0.5, 1.0, 0.8])
    >>> y_pred = predict_reg_lineaire(X, beta)
    """

    # Etape 1
    # Ici on crée un vecteur NumPy contenant uniquement des 1
    # Ce vecteur contient une seule colonne (contenant que des 1) et autant de lignes qu'il y a
    # d'observations dans le jeu de données
    # Ce vecteur est créé pour pouvoir attribuer beta_0 à la constante du modèle linéaire

    constante = np.ones((X.shape[0], 1))

    # Etape 2
    # Ajoute la constante à la matrice des variables explicatives
    X = np.concatenate((constante, X), axis=1)

    # Réalisez la multiplication matricielle permettant d'appliquer les coefficients à la matrice X
    y_pred = np.matmul(beta, X.T)

    # Renvoie le vecteur de prédictions
    return y_pred

def predict_reg_ridge(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Applique une régression Ridge sur le jeu de données.

    Paramètres
    ----------
    X : np.ndarray de forme (n_observations, n_features)
        Matrice des variables explicatives
    beta : np.ndarray de forme (n_features + 1,)
        Vecteur de paramètres estimés sur le jeu d'entraînement (incluant l'intercept)

    Retourne
    --------
    y_pred : np.ndarray de forme (n_observations,)
        Vecteur des prédictions

    Notes
    -----
    - L'intercept (β₀) est automatiquement géré
    - La fonction ajoute une colonne de 1 pour l'intercept avant la prédiction
    - Cette fonction est identique à predict_reg_lineaire (pas de λ en prédiction)

    Exemples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> beta = np.array([0.5, 1.0, 0.8])
    >>> y_pred = predict_reg_ridge(X, beta)
    """

    # Étape 1 : Création du vecteur de constantes
    # Ici on crée un vecteur NumPy contenant uniquement des 1
    # Ce vecteur contient une seule colonne (contenant que des 1) et autant de lignes qu'il y a
    # d'observations dans le jeu de données
    # Ce vecteur est créé pour pouvoir attribuer beta_0 à la constante du modèle linéaire
    constante = np.ones((X.shape[0], 1))

    # Étape 2 : Ajout de la constante à la matrice des variables explicatives
    X = np.concatenate((constante, X), axis=1)

    # Réalisez la multiplication matricielle permettant d'appliquer les coefficients à la matrice X
    y_pred = np.matmul(beta, X.T)

    # Renvoie le vecteur de prédictions
    return y_pred

def predict_reg_lasso(X: np.ndarray, beta_0: float, beta_j: np.ndarray) -> np.ndarray:
    """
    Prédit la target pour une matrice X donnée en utilisant les paramètres estimés.

    Applique la formule : y_pred = X * beta_j + beta_0

    Paramètres
    ----------
    X : np.ndarray
        Matrice de features de dimensions (N, P).
    beta_0 : float
        Valeur de beta_0 aussi appelé intercept ou biais.
    beta_j : np.ndarray
        Vecteur de dimension (P,) contenant les coefficients associés
        aux variables explicatives.

    Retourne
    --------
    np.ndarray
        Prédictions de dimension (N,).
    """
    # Réalisation des prédictions
    y_pred = X.dot(beta_j) + beta_0
    return y_pred
