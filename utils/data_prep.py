"""
Module de préparation et nettoyage des données.

Ce module contient les fonctions nécessaires pour :
- Charger et importer les données
- Valider et transformer les données
- Détecter et gérer les outliers
- Formatter et nettoyer les données
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import scipy.stats as ss

try:
    from numpy.typing import ArrayLike
except ImportError:
    from typing import Any as ArrayLike

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

try:
    from sklearn.pipeline import Pipeline
except ImportError:
    Pipeline = None


def process_user_bad(name: str, age: int, email: str) -> Dict[str, Any] | None:
    """
    Valide et transforme les informations utilisateur.
    
    Args:
        name: Nom de l'utilisateur
        age: Âge de l'utilisateur
        email: Email de l'utilisateur
    
    Returns:
        Dictionnaire avec les infos utilisateur ou None si validation échoue
    
    Exemple:
        >>> user = process_user_bad("john doe", 30, "john@example.com")
        >>> print(user)
        {'name': 'John Doe', 'age': 30, 'email': 'john@example.com'}
    """
    # Validation
    if not name or len(name) < 2:
        return None
    if age < 0 or age > 150:
        return None
    if "@" not in email:
        return None

    # Transformation
    name = name.strip().title()
    email = email.lower()

    # Création
    user = {"name": name, "age": age, "email": email}

    # Sauvegarde (simulée)
    print(f"User saved: {user}")

    return user


def detect_possible_outliers(
    df: pd.DataFrame, 
    column: str
) -> Tuple[List[int], float, float]:
    """
    Détecte les outliers dans une colonne numérique selon la méthode de l'IQR.
    
    Utilise la méthode des quartiles : outliers = valeurs < Q1-1.5*IQR ou > Q3+1.5*IQR
    
    Args:
        df: DataFrame pandas
        column: Nom de la colonne numérique
    
    Returns:
        Tuple contenant:
        - Liste des indices des outliers
        - Limite basse (outer_lower)
        - Limite haute (outer_upper)
    
    Exemple:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 100]})
        >>> outliers, lower, upper = detect_possible_outliers(df, 'values')
        >>> print(f"Outliers indices: {outliers}")
    """
    # 1er Quartile
    Q1 = np.nanpercentile(df[column], 25)

    # 3ème Quartile
    Q3 = np.nanpercentile(df[column], 75)

    # Inter-Quartile Range (IQR)
    IQR = Q3 - Q1

    # limites, basse & haute
    outer_fence = IQR * 1.5
    outer_lower = Q1 - outer_fence
    outer_upper = Q3 + outer_fence

    # Détection des outliers potentiels
    mask = (df[column] < outer_lower) | (df[column] > outer_upper)

    # Stockage de leurs indices
    possible_outlier_index = df[mask].index

    # Passage sous forme de list
    outliers_index = possible_outlier_index.tolist()

    return sorted(outliers_index), outer_lower, outer_upper


def replace_outlier(value: float, threshold: float = 20, replacement: float = 15) -> float:
    """
    Remplace les valeurs aberrantes par une valeur de remplacement.
    
    Args:
        value: Valeur à traiter
        threshold: Seuil au-delà duquel on remplace
        replacement: Valeur de remplacement
    
    Returns:
        Valeur remplacée ou originale
    
    Exemple:
        >>> val = replace_outlier(25, threshold=20, replacement=15)
        >>> print(val)  # 15
    """
    if value > threshold:
        value = replacement
    return value


def currency(x: float, pos: int) -> str:
    """
    Formate une valeur en devise euros (K€ ou M€).
    
    Args:
        x: Valeur en euros
        pos: Position du tick (unused)
    
    Returns:
        Chaîne formatée (ex: "12.5M€")
    
    Exemple:
        >>> val = currency(1500000, 0)
        >>> print(val)  # "1.5M€"
    """
    if x >= 1e6:
        pos = '{:1.1f}M€'.format(x * 1e-6)
    else:
        pos = '{:1.0f}K€'.format(x * 1e-3)
    return pos


def distance_km(x: float, pos: int) -> str:
    """
    Formate une distance en kilomètres (k km ou M km).
    
    Args:
        x: Distance en kilomètres
        pos: Position du tick (unused)
    
    Returns:
        Chaîne formatée (ex: "2.5M km")
    
    Exemple:
        >>> val = distance_km(2500000, 0)
        >>> print(val)  # "2.5M km"
    """
    if x >= 1e6:
        pos = '{:1.1f}M km'.format(x * 1e-6)
    else:
        pos = '{:1.0f}k km'.format(x * 1e-3)
    return pos


def standardize(array: np.ndarray) -> np.ndarray:
    """
    Standardise un array (normalisation z-score).
    
    Formule: (X - μ) / σ
    
    Args:
        array: Array NumPy 1D ou 2D
    
    Returns:
        Array standardisé (moyenne=0, std=1)
    
    Exemple:
        >>> import numpy as np
        >>> data = np.array([[1, 2], [3, 4]])
        >>> norm = standardize(data)
    """
    means = np.mean(array, axis=0)
    stds = np.std(array, axis=0)
    return (array - means) / (stds + 1e-10)  # 1e-10 pour éviter division par zéro




# === Fonctions des Modules 5, 6, 7 === #

def creer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée de nouvelles variables à partir des variables existantes.

    Cette fonction génère deux nouvelles caractéristiques :
    - family_size : taille totale de la famille présente à bord (SibSp + Parch + 1)
    - is_alone : indicateur binaire si le passager voyage seul (1) ou non (0)

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant au minimum les colonnes 'SibSp' et 'Parch'

    Retourne
    --------
    pd.DataFrame
        Nouveau DataFrame avec les colonnes 'family_size' et 'is_alone' ajoutées

    Exemples
    --------
    >>> df_enrichi = creer_features(df_train)
    >>> df_enrichi[['family_size', 'is_alone']].head()
    """
    df_nouveau = df.copy()
    df_nouveau['family_size'] = df_nouveau['SibSp'] + df_nouveau['Parch'] + 1
    df_nouveau["is_alone"] = (df_nouveau['family_size'] == 1).astype(int)

    return df_nouveau

def get_original_feature_names(
    dummies_features: List[str], separator: str = "___"
) -> List[str]:
    """
    Récupère les noms des variables originales à partir des variables dummies.

    Pour les variables encodées avec un séparateur (par défaut '___'), cette fonction
    extrait le préfixe (nom de la variable originale). Les variables numériques
    non encodées sont conservées telles quelles.

    Paramètres
    ----------
    dummies_features : List[str]
        Liste des noms de variables dummies issues de get_dummies().
    separator : str, optional
        Séparateur utilisé lors de l'encodage (défaut: '___').

    Retourne
    --------
    List[str]
        Liste unique des noms de variables originales, triée par ordre alphabétique.

    Exemple
    -------
    >>> get_original_feature_names(["age", "job___admin.", "job___student"])
    ['age', 'job']
    """
    import re

    original_features = []
    for col in dummies_features:
        # Recherche du préfixe avant le séparateur
        search_result = re.search(rf"(.+){separator}", col)
        if search_result:
            original_features.append(search_result.group(1))
        else:
            original_features.append(col)
    return np.unique(original_features).tolist()

def compute_gradient(beta_j: np.ndarray, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, lambda_: float) -> tuple[float, np.ndarray]:
    """
    Calcule la dérivée de la fonction de perte en beta pour la régression Lasso.

    Le gradient tient compte de la pénalité L1 qui dépend du signe de beta_j.
    L'intercept beta_0 n'est pas pénalisé.

    Paramètres
    ----------
    beta_j : np.ndarray
        Vecteur de dimensions (P,) contenant les coefficients associés
        aux variables explicatives à l'étape t.
    X : np.ndarray
        Matrice de features de dimensions (N, P).
    y : np.ndarray
        Target de dimension (N,).
    y_pred : np.ndarray
        Prédictions obtenues avec les paramètres beta de l'étape t, dimension (N,).
    lambda_ : float
        Paramètre de pénalisation lambda de la régression Lasso (doit être > 0).

    Retourne
    --------
    tuple[float, np.ndarray]
        grad_beta_0 : float
            La dérivée en beta_0 (intercept).
        grad_beta_j : np.ndarray
            La dérivée en beta_j avec j != 0, de dimension (P,).
    """
    # Initialisation du gradient pour les coefficients beta_1 jusqu'à beta_P
    # Note : beta_0 (intercept) est exclu car il n'est pas pénalisé
    grad_beta_j = np.zeros(X.shape[1])

    # Calcul du gradient pour chaque coefficient beta_j
    for j in range(len(beta_j)):
        # Le signe de beta_j détermine le signe de la pénalité L1
        if beta_j[j] > 0:
            grad_beta_j[j] = (-(2 * (X[:, j]).dot(y - y_pred)) + lambda_) / X.shape[0]
        else:
            grad_beta_j[j] = (-(2 * (X[:, j]).dot(y - y_pred)) - lambda_) / X.shape[0]

    # Calcul du gradient pour beta_0 (pas de pénalisation)
    grad_beta_0 = -2 * np.sum(y - y_pred) / X.shape[0]

    return grad_beta_0, grad_beta_j

def update_weights(beta_0: float, beta_j: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float, learning_rate: float) -> tuple[float, np.ndarray]:
    """
    Met à jour les paramètres beta en utilisant le gradient et la valeur précédente de beta.

    Applique la formule de la descente de gradient : beta_nouveau = beta_ancien - learning_rate * gradient

    Paramètres
    ----------
    beta_0 : float
        Valeur de beta_0 aussi appelée intercept ou biais.
    beta_j : np.ndarray
        Vecteur de dimension (P,) contenant les coefficients associés
        aux variables explicatives à l'étape t.
    X : np.ndarray
        Matrice de features de dimension (N, P).
    y : np.ndarray
        Target de dimension (N,).
    lambda_ : float
        Paramètre de pénalisation lambda de la régression Lasso.
    learning_rate : float
        Le learning rate (taux d'apprentissage) qui contrôle la taille du pas.

    Retourne
    --------
    tuple[float, np.ndarray]
        beta_0 : float
            La valeur mise à jour de beta_0.
        beta_j : np.ndarray
            La valeur mise à jour de beta_j.
    """
    # Étape 1 : Prédiction en utilisant les coefficients beta de l'étape t
    y_pred = X.dot(beta_j) + beta_0

    # Étape 2 : Calcul du gradient
    grad_beta_0, grad_beta_j = compute_gradient(beta_j, X, y, y_pred, lambda_)

    # Étape 3 : Mise à jour des paramètres
    beta_j = beta_j - learning_rate * grad_beta_j
    beta_0 = beta_0 - learning_rate * grad_beta_0

    return beta_0, beta_j

def cercle_correlations(pca: PCA, dim_x: int, dim_y: int, features: List[str]) -> None:
    """
    Affiche le cercle des corrélations pour une ACP en utilisant Plotly.

    Le cercle des corrélations permet de visualiser les relations entre les variables
    initiales et les composantes principales. Plus une flèche est longue et proche
    du cercle, mieux la variable est représentée sur les axes choisis.

    Paramètres
    ----------
    pca : PCA
        L'objet PCA de scikit-learn déjà entraîné sur les données.
    dim_x : int
        L'indice de la dimension à afficher sur l'axe des abscisses (commence à 0).
    dim_y : int
        L'indice de la dimension à afficher sur l'axe des ordonnées (commence à 0).
    features : List[str]
        La liste des noms des variables (features) à représenter.

    Retourne
    --------
    None
        Affiche directement le graphique interactif Plotly.

    Exemple
    -------
    >>> from sklearn.decomposition import PCA
    >>> pca = PCA()
    >>> pca.fit(X_normalise)
    >>> cercle_correlations(pca, 0, 1, ['var1', 'var2', 'var3'])
    """

    # Création des flèches et des annotations
    arrows = []
    annotations = []

    for i, feature in enumerate(features):
        # Coordonnées des flèches (corrélations avec les composantes)
        x = pca.components_[dim_x, i]
        y = pca.components_[dim_y, i]

        # Ajout des flèches
        arrows.append(go.Scatter(
            x=[0, x],
            y=[0, y],
            mode="lines+markers",
            line=dict(color='#3b4859', width=2),
            marker=dict(size=8, color='#ff7373'),
            name=feature,
            showlegend=True
        ))

        # Ajout des annotations pour les noms des variables
        annotations.append(dict(
            x=x * 1.1,  # Légèrement décalé pour la lisibilité
            y=y * 1.1,
            xanchor="center",
            yanchor="middle",
            text=f"<b>{feature}</b>",
            showarrow=False,
            font=dict(size=11, color="#3b4859")
        ))

    # Ajout du cercle unitaire
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    arrows.append(go.Scatter(
        x=circle_x,
        y=circle_y,
        mode="lines",
        line=dict(color='gray', dash='dash'),
        name="Cercle unitaire",
        showlegend=False
    ))

    # Configuration des axes et de la mise en page
    variance_x = round(pca.explained_variance_ratio_[dim_x] * 100, 2)
    variance_y = round(pca.explained_variance_ratio_[dim_y] * 100, 2)

    layout = go.Layout(
        title=f"Cercle des corrélations (PC{dim_x+1} et PC{dim_y+1})",
        xaxis=dict(
            title=f"PC{dim_x+1} ({variance_x}% de variance)",
            range=[-1.2, 1.2],
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ),
        yaxis=dict(
            title=f"PC{dim_y+1} ({variance_y}% de variance)",
            range=[-1.2, 1.2],
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ),
        annotations=annotations,
        width=800,
        height=800,
        hovermode='closest'
    )

    # Création et affichage de la figure
    fig = go.Figure(data=arrows, layout=layout)
    fig.show()

def tracer_courbes_apprentissage(
    modele,
    X: ArrayLike,
    y: ArrayLike,
    cv: int = 5,
    scoring: str = "f1_weighted",
    n_points: int = 10,
    couleur_train: str = "#3b4859",
    couleur_val: str = "#ff7373",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Calcule et trace les courbes d'apprentissage d'un modèle.

    Les courbes d'apprentissage montrent l'évolution de la performance sur
    l'entraînement et en validation croisée en fonction de la taille de
    l'échantillon d'entraînement. Elles permettent de diagnostiquer le
    sur-apprentissage (écart important) ou le sous-apprentissage (stagnation).

    Paramètres
    ----------
    modele : estimator
        Modèle scikit-learn (classifieur ou régresseur) à évaluer.
    X : ArrayLike
        Caractéristiques d'entrée pour l'entraînement.
    y : ArrayLike
        Variable cible (étiquettes ou valeurs continues).
    cv : int, par défaut 5
        Nombre de folds pour la validation croisée.
    scoring : str, par défaut "f1_weighted"
        Métrique d'évaluation (ex: "f1_weighted", "accuracy", "r2", "neg_mse").
    n_points : int, par défaut 10
        Nombre de points pour lesquels la performance est évaluée.
    couleur_train : str, par défaut "#3b4859"
        Couleur de la courbe d'entraînement.
    couleur_val : str, par défaut "#ff7373"
        Couleur de la courbe de validation.
    ax : matplotlib.axes.Axes, optionnel
        Axe matplotlib cible. Si ``None``, un nouvel axe est créé.

    Retours
    -------
    matplotlib.axes.Axes
        Axe contenant les courbes d'apprentissage.
    dict
        Dictionnaire contenant les statistiques finales :
        - "train_mean" : score moyen d'entraînement à 100% des données
        - "train_std" : écart-type d'entraînement
        - "val_mean" : score moyen de validation à 100% des données
        - "val_std" : écart-type de validation
        - "ecart" : différence train_mean - val_mean (indicateur d'overfit)

    Exemples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = make_pipeline(SimpleImputer(), DecisionTreeClassifier())
    >>> ax, stats = tracer_courbes_apprentissage(clf, X, y)
    >>> print(f"Écart train/val : {stats['ecart']:.3f}")
    """
    from sklearn.model_selection import learning_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Calcul des courbes d'apprentissage
    fractions = np.linspace(0.1, 1.0, n_points)
    tailles_train, scores_train, scores_val = learning_curve(
        estimator=modele,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        train_sizes=fractions,
        n_jobs=-1,
        random_state=42,
    )

    # Calcul des statistiques
    train_mean = scores_train.mean(axis=1)
    train_std = scores_train.std(axis=1)
    val_mean = scores_val.mean(axis=1)
    val_std = scores_val.std(axis=1)

    # Tracé des courbes
    ax.plot(
        tailles_train,
        train_mean,
        label="Score d'entraînement",
        marker="o",
        color=couleur_train,
        linewidth=2,
    )
    ax.fill_between(
        tailles_train,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color=couleur_train,
    )

    ax.plot(
        tailles_train,
        val_mean,
        label="Score de validation (CV)",
        marker="s",
        color=couleur_val,
        linewidth=2,
    )
    ax.fill_between(
        tailles_train,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color=couleur_val,
    )

    ax.set_xlabel("Taille de l'échantillon d'entraînement", fontsize=11)
    ax.set_ylabel(f"Score ({scoring})", fontsize=11)
    ax.set_title("Courbes d'apprentissage", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    # Retourne aussi les statistiques finales
    stats = {
        "train_mean": train_mean[-1],
        "train_std": train_std[-1],
        "val_mean": val_mean[-1],
        "val_std": val_std[-1],
        "ecart": train_mean[-1] - val_mean[-1],
    }

    return ax, stats

def creation_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des variables dérivées pour enrichir le dataset Titanic.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les données du Titanic

    Retours
    -------
    pd.DataFrame
        DataFrame enrichi avec les nouvelles variables
    """
    df_nouveau = df.copy()
    df_nouveau['taille_famille'] = df_nouveau['SibSp'] + df_nouveau['Parch'] + 1
    df_nouveau["est_seul"] = (df_nouveau['taille_famille'] == 1).astype(int)
    return df_nouveau

def cramer_v_coeff(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """
    Calcule le V de Cramer entre deux variables catégorielles.

    Le V de Cramer mesure l'intensité de l'association entre deux variables
    catégorielles. Il est basé sur la statistique du Khi-deux.

    Parameters
    ----------
    x : pd.Series
        Première variable catégorielle.
    y : pd.Series
        Deuxième variable catégorielle.

    Returns
    -------
    tuple[float, float]
        - V de Cramer (entre 0 et 1)
        - p-value du test du Khi-deux
    """
    # Suppression des valeurs manquantes
    complete_cases = x.isna() | y.isna()
    x = x[~complete_cases]
    y = y[~complete_cases]

    # Calcul du Khi-deux max (dénominateur du V de Cramer)
    n = len(x)
    khi2_max = n * (min(x.nunique(), y.nunique()) - 1)

    # Construction du tableau de contingence et calcul du Khi-deux
    contingency_table = pd.crosstab(x, y)
    khi2_result = ss.chi2_contingency(observed=contingency_table, correction=True)

    # Calcul du V de Cramer
    cramer = round(np.sqrt(khi2_result.statistic / khi2_max), 4)
    p_value = khi2_result.pvalue

    return cramer, p_value

def compute_cramer_v(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la matrice des V de Cramer pour toutes les paires de variables.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame contenant uniquement des variables catégorielles.

    Returns
    -------
    pd.DataFrame
        Matrice carrée contenant les V de Cramer entre chaque paire de variables.
        La diagonale contient des 1 (association parfaite d'une variable avec elle-même).
    """
    ncols = data.shape[1]
    cols = data.columns
    cramer_matrix = np.eye(ncols)

    for j in range(ncols - 1):
        for i in range(j + 1, ncols):
            v_cramer, _ = cramer_v_coeff(x=data.iloc[:, j], y=data.iloc[:, i])
            cramer_matrix[i, j] = v_cramer
            cramer_matrix[j, i] = v_cramer

    return pd.DataFrame(cramer_matrix, columns=cols, index=cols)

def cramer_v_coeff(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """
    Calcule le V de Cramer entre deux variables catégorielles.

    Le V de Cramer mesure l'intensité de l'association entre deux variables
    catégorielles. Il est basé sur la statistique du Khi-deux.

    Paramètres
    ----------
    x : pd.Series
        Première variable catégorielle.
    y : pd.Series
        Deuxième variable catégorielle.

    Retourne
    --------
    Tuple[float, float]
        - V de Cramer (entre 0 et 1)
        - p-value du test du Khi-deux
    """
    # Suppression des valeurs manquantes
    complete_cases = x.isna() | y.isna()
    x = x[~complete_cases]
    y = y[~complete_cases]

    # Calcul du Khi-deux max (dénominateur du V de Cramer)
    n = len(x)
    khi2_max = n * (min(x.nunique(), y.nunique()) - 1)

    # Construction du tableau de contingence et calcul du Khi-deux
    contingency_table = pd.crosstab(x, y)
    khi2_result = ss.chi2_contingency(observed=contingency_table, correction=True)

    # Calcul du V de Cramer
    cramer = round(np.sqrt(khi2_result.statistic / khi2_max), 4)
    p_value = khi2_result.pvalue

    return cramer, p_value

def categorical_to_discrete(data: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """
    Encode les variables catégorielles en variables discrètes.

    Paramètres
    ----------
    data : pd.DataFrame
        DataFrame contenant les données à encoder.
    cat_cols : List[str]
        Liste des colonnes catégorielles à encoder.

    Retourne
    --------
    pd.DataFrame
        Copie du DataFrame avec les variables catégorielles encodées.
    """
    data_encoded = data.copy()
    for col in cat_cols:
        data_encoded[col] = data_encoded[col].map(code_dic)
    return data_encoded

def load_scoring_data(
    url: str = "https://raw.githubusercontent.com/datagong/data/main/bank-additional-full.csv",
) -> pd.DataFrame:
    """
    Charge le jeu de données de scoring bancaire.

    Paramètres
    ----------
    url : str, optional
        URL du fichier CSV à charger.

    Retourne
    --------
    pd.DataFrame
        DataFrame contenant les données brutes.
    """
    return pd.read_csv(filepath_or_buffer=url, sep=";")

def get_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Identifie les colonnes catégorielles et numériques d'un DataFrame.

    Paramètres
    ----------
    X : pd.DataFrame
        DataFrame contenant les variables explicatives.

    Retourne
    --------
    Tuple[List[str], List[str], List[str]]
        - Liste des colonnes catégorielles
        - Liste des colonnes numériques
        - Liste de toutes les colonnes
    """
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    all_cols = X.columns.tolist()
    return cat_cols, num_cols, all_cols

def model_evaluation(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[float, float]:
    """
    Évalue les performances d'un modèle sur les jeux train et test.

    Paramètres
    ----------
    model : Pipeline
        Modèle entraîné (pipeline scikit-learn).
    X_train : pd.DataFrame
        Variables explicatives du jeu d'entraînement.
    y_train : pd.Series
        Variable cible du jeu d'entraînement.
    X_test : pd.DataFrame
        Variables explicatives du jeu de test.
    y_test : pd.Series
        Variable cible du jeu de test.

    Retourne
    --------
    Tuple[float, float]
        AUC sur le jeu d'entraînement et AUC sur le jeu de test.
    """
    # Prédiction des probabilités
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    # Calcul des AUC
    auc_train = roc_auc_score(y_train, y_train_pred)
    auc_test = roc_auc_score(y_test, y_test_pred)

    return auc_train, auc_test

def lift_curve_data(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Calcule les données nécessaires pour tracer la courbe de lift.

    Paramètres
    ----------
    y_true : np.ndarray | pd.Series
        Labels réels (0 ou 1).
    y_pred : np.ndarray
        Probabilités prédites par le modèle.

    Retourne
    --------
    pd.DataFrame
        DataFrame contenant les colonnes nécessaires pour tracer la courbe de lift,
        notamment la colonne 'concentration' représentant le pourcentage cumulé
        de la cible capturée.
    """
    # Trier par ordre décroissant des probabilités
    sorted_score = np.argsort(y_pred)[::-1]

    # Convertir en arrays si nécessaire
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.to_numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.to_numpy()

    y_true = y_true[sorted_score]
    y_pred = y_pred[sorted_score]

    # Créer un dataframe avec les vecteurs triés
    lift_df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})

    # Découper en percentiles
    lift_df["perc"] = (lift_df.index / lift_df.shape[0]) * 100
    lift_df["perc_bins"] = pd.cut(lift_df["perc"], np.arange(0, 101, 1))

    # Grouper par intervalle et calculer l'effectif des positifs
    lift_df = (
        lift_df.groupby(["perc_bins"], as_index=False, observed=True)
        .agg(y_true_sum=("y_true", "sum"))
    )

    # Calculer l'effectif cumulé des positifs
    lift_df["y_true_cumsum"] = lift_df["y_true_sum"].cumsum()

    # Calculer la concentration (% cumulé de positifs capturés)
    lift_df["concentration"] = (
        lift_df["y_true_cumsum"] / lift_df["y_true_sum"].sum()
    ) * 100

    # Ajouter le point (0, 0) pour la représentation graphique
    zeros_df = pd.DataFrame(np.zeros((1, 4)), columns=lift_df.columns)
    lift_df = pd.concat([zeros_df, lift_df]).reset_index(drop=True)

    return lift_df

def compute_lift_by_decile(
    scores: np.ndarray,
    labels: np.ndarray,
    n_deciles: int = 10
) -> pd.DataFrame:
    """
    Calcule le lift par décile pour un modèle de scoring.

    Paramètres
    ----------
    scores : np.ndarray
        Scores de probabilité prédits par le modèle.
    labels : np.ndarray
        Labels réels (0 ou 1).
    n_deciles : int, optional
        Nombre de déciles à calculer (défaut: 10).

    Retourne
    --------
    pd.DataFrame
        DataFrame contenant pour chaque décile :
        - decile : numéro du décile (1 = meilleurs scores)
        - n_obs : nombre d'observations
        - n_targets : nombre de cibles
        - response_rate : taux de réponse dans le décile
        - lift : lift par rapport au taux global
        - cumul_targets : cibles cumulées
        - cumul_capture : taux de capture cumulé
    """
    # Création d'un DataFrame de travail
    df = pd.DataFrame({"score": scores, "label": labels})

    # Attribution des déciles (1 = meilleurs scores)
    df["decile"] = pd.qcut(df["score"], q=n_deciles, labels=False)
    df["decile"] = n_deciles - df["decile"]  # Inverser pour que 1 = meilleurs

    # Taux de réponse global
    global_rate = df["label"].mean()

    # Agrégation par décile
    lift_table = (
        df.groupby("decile")
        .agg(
            n_obs=("label", "count"),
            n_targets=("label", "sum"),
        )
        .reset_index()
    )

    # Calcul des métriques
    lift_table["response_rate"] = lift_table["n_targets"] / lift_table["n_obs"]
    lift_table["lift"] = lift_table["response_rate"] / global_rate
    lift_table["cumul_targets"] = lift_table["n_targets"].cumsum()
    lift_table["cumul_capture"] = lift_table["cumul_targets"] / df["label"].sum()

    return lift_table

__all__ = [
    'process_user_bad',
    'detect_possible_outliers', 
    'replace_outlier',
    'currency',
    'distance_km',
    'standardize'
]


# ====================================================================================================
# Fonctions utilitaires - Modules 5-7
# ====================================================================================================

def compute_gradient(beta_j: np.ndarray, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, lambda_: float) -> tuple[float, np.ndarray]:
    """
    Calcule la dérivée de la fonction de perte en beta pour la régression Lasso.

    Le gradient tient compte de la pénalité L1 qui dépend du signe de beta_j.
    L'intercept beta_0 n'est pas pénalisé.

    Paramètres
    ----------
    beta_j : np.ndarray
        Vecteur de dimensions (P,) contenant les coefficients associés
        aux variables explicatives à l'étape t.
    X : np.ndarray
        Matrice de features de dimensions (N, P).
    y : np.ndarray
        Target de dimension (N,).
    y_pred : np.ndarray
        Prédictions obtenues avec les paramètres beta de l'étape t, dimension (N,).
    lambda_ : float
        Paramètre de pénalisation lambda de la régression Lasso (doit être > 0).

    Retourne
    --------
    tuple[float, np.ndarray]
        grad_beta_0 : float
            La dérivée en beta_0 (intercept).
        grad_beta_j : np.ndarray
            La dérivée en beta_j avec j != 0, de dimension (P,).
    """
    # Initialisation du gradient pour les coefficients beta_1 jusqu'à beta_P
    # Note : beta_0 (intercept) est exclu car il n'est pas pénalisé
    grad_beta_j = np.zeros(X.shape[1])

    # Calcul du gradient pour chaque coefficient beta_j
    for j in range(len(beta_j)):
        # Le signe de beta_j détermine le signe de la pénalité L1
        if beta_j[j] > 0:
            grad_beta_j[j] = (-(2 * (X[:, j]).dot(y - y_pred)) + lambda_) / X.shape[0]
        else:
            grad_beta_j[j] = (-(2 * (X[:, j]).dot(y - y_pred)) - lambda_) / X.shape[0]

    # Calcul du gradient pour beta_0 (pas de pénalisation)
    grad_beta_0 = -2 * np.sum(y - y_pred) / X.shape[0]

    return grad_beta_0, grad_beta_j

def update_weights(beta_0: float, beta_j: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float, learning_rate: float) -> tuple[float, np.ndarray]:
    """
    Met à jour les paramètres beta en utilisant le gradient et la valeur précédente de beta.

    Applique la formule de la descente de gradient : beta_nouveau = beta_ancien - learning_rate * gradient

    Paramètres
    ----------
    beta_0 : float
        Valeur de beta_0 aussi appelée intercept ou biais.
    beta_j : np.ndarray
        Vecteur de dimension (P,) contenant les coefficients associés
        aux variables explicatives à l'étape t.
    X : np.ndarray
        Matrice de features de dimension (N, P).
    y : np.ndarray
        Target de dimension (N,).
    lambda_ : float
        Paramètre de pénalisation lambda de la régression Lasso.
    learning_rate : float
        Le learning rate (taux d'apprentissage) qui contrôle la taille du pas.

    Retourne
    --------
    tuple[float, np.ndarray]
        beta_0 : float
            La valeur mise à jour de beta_0.
        beta_j : np.ndarray
            La valeur mise à jour de beta_j.
    """
    # Étape 1 : Prédiction en utilisant les coefficients beta de l'étape t
    y_pred = X.dot(beta_j) + beta_0

    # Étape 2 : Calcul du gradient
    grad_beta_0, grad_beta_j = compute_gradient(beta_j, X, y, y_pred, lambda_)

    # Étape 3 : Mise à jour des paramètres
    beta_j = beta_j - learning_rate * grad_beta_j
    beta_0 = beta_0 - learning_rate * grad_beta_0

    return beta_0, beta_j

def cercle_correlations(pca: PCA, dim_x: int, dim_y: int, features: List[str]) -> None:
    """
    Affiche le cercle des corrélations pour une ACP en utilisant Plotly.

    Le cercle des corrélations permet de visualiser les relations entre les variables
    initiales et les composantes principales. Plus une flèche est longue et proche
    du cercle, mieux la variable est représentée sur les axes choisis.

    Paramètres
    ----------
    pca : PCA
        L'objet PCA de scikit-learn déjà entraîné sur les données.
    dim_x : int
        L'indice de la dimension à afficher sur l'axe des abscisses (commence à 0).
    dim_y : int
        L'indice de la dimension à afficher sur l'axe des ordonnées (commence à 0).
    features : List[str]
        La liste des noms des variables (features) à représenter.

    Retourne
    --------
    None
        Affiche directement le graphique interactif Plotly.

    Exemple
    -------
    >>> from sklearn.decomposition import PCA
    >>> pca = PCA()
    >>> pca.fit(X_normalise)
    >>> cercle_correlations(pca, 0, 1, ['var1', 'var2', 'var3'])
    """

    # Création des flèches et des annotations
    arrows = []
    annotations = []

    for i, feature in enumerate(features):
        # Coordonnées des flèches (corrélations avec les composantes)
        x = pca.components_[dim_x, i]
        y = pca.components_[dim_y, i]

        # Ajout des flèches
        arrows.append(go.Scatter(
            x=[0, x],
            y=[0, y],
            mode="lines+markers",
            line=dict(color='#3b4859', width=2),
            marker=dict(size=8, color='#ff7373'),
            name=feature,
            showlegend=True
        ))

        # Ajout des annotations pour les noms des variables
        annotations.append(dict(
            x=x * 1.1,  # Légèrement décalé pour la lisibilité
            y=y * 1.1,
            xanchor="center",
            yanchor="middle",
            text=f"<b>{feature}</b>",
            showarrow=False,
            font=dict(size=11, color="#3b4859")
        ))

    # Ajout du cercle unitaire
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    arrows.append(go.Scatter(
        x=circle_x,
        y=circle_y,
        mode="lines",
        line=dict(color='gray', dash='dash'),
        name="Cercle unitaire",
        showlegend=False
    ))

    # Configuration des axes et de la mise en page
    variance_x = round(pca.explained_variance_ratio_[dim_x] * 100, 2)
    variance_y = round(pca.explained_variance_ratio_[dim_y] * 100, 2)

    layout = go.Layout(
        title=f"Cercle des corrélations (PC{dim_x+1} et PC{dim_y+1})",
        xaxis=dict(
            title=f"PC{dim_x+1} ({variance_x}% de variance)",
            range=[-1.2, 1.2],
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ),
        yaxis=dict(
            title=f"PC{dim_y+1} ({variance_y}% de variance)",
            range=[-1.2, 1.2],
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ),
        annotations=annotations,
        width=800,
        height=800,
        hovermode='closest'
    )

    # Création et affichage de la figure
    fig = go.Figure(data=arrows, layout=layout)
    fig.show()

def tracer_courbes_apprentissage(
    modele,
    X: ArrayLike,
    y: ArrayLike,
    cv: int = 5,
    scoring: str = "f1_weighted",
    n_points: int = 10,
    couleur_train: str = "#3b4859",
    couleur_val: str = "#ff7373",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Calcule et trace les courbes d'apprentissage d'un modèle.

    Les courbes d'apprentissage montrent l'évolution de la performance sur
    l'entraînement et en validation croisée en fonction de la taille de
    l'échantillon d'entraînement. Elles permettent de diagnostiquer le
    sur-apprentissage (écart important) ou le sous-apprentissage (stagnation).

    Paramètres
    ----------
    modele : estimator
        Modèle scikit-learn (classifieur ou régresseur) à évaluer.
    X : ArrayLike
        Caractéristiques d'entrée pour l'entraînement.
    y : ArrayLike
        Variable cible (étiquettes ou valeurs continues).
    cv : int, par défaut 5
        Nombre de folds pour la validation croisée.
    scoring : str, par défaut "f1_weighted"
        Métrique d'évaluation (ex: "f1_weighted", "accuracy", "r2", "neg_mse").
    n_points : int, par défaut 10
        Nombre de points pour lesquels la performance est évaluée.
    couleur_train : str, par défaut "#3b4859"
        Couleur de la courbe d'entraînement.
    couleur_val : str, par défaut "#ff7373"
        Couleur de la courbe de validation.
    ax : matplotlib.axes.Axes, optionnel
        Axe matplotlib cible. Si ``None``, un nouvel axe est créé.

    Retours
    -------
    matplotlib.axes.Axes
        Axe contenant les courbes d'apprentissage.
    dict
        Dictionnaire contenant les statistiques finales :
        - "train_mean" : score moyen d'entraînement à 100% des données
        - "train_std" : écart-type d'entraînement
        - "val_mean" : score moyen de validation à 100% des données
        - "val_std" : écart-type de validation
        - "ecart" : différence train_mean - val_mean (indicateur d'overfit)

    Exemples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = make_pipeline(SimpleImputer(), DecisionTreeClassifier())
    >>> ax, stats = tracer_courbes_apprentissage(clf, X, y)
    >>> print(f"Écart train/val : {stats['ecart']:.3f}")
    """
    from sklearn.model_selection import learning_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Calcul des courbes d'apprentissage
    fractions = np.linspace(0.1, 1.0, n_points)
    tailles_train, scores_train, scores_val = learning_curve(
        estimator=modele,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        train_sizes=fractions,
        n_jobs=-1,
        random_state=42,
    )

    # Calcul des statistiques
    train_mean = scores_train.mean(axis=1)
    train_std = scores_train.std(axis=1)
    val_mean = scores_val.mean(axis=1)
    val_std = scores_val.std(axis=1)

    # Tracé des courbes
    ax.plot(
        tailles_train,
        train_mean,
        label="Score d'entraînement",
        marker="o",
        color=couleur_train,
        linewidth=2,
    )
    ax.fill_between(
        tailles_train,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color=couleur_train,
    )

    ax.plot(
        tailles_train,
        val_mean,
        label="Score de validation (CV)",
        marker="s",
        color=couleur_val,
        linewidth=2,
    )
    ax.fill_between(
        tailles_train,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color=couleur_val,
    )

    ax.set_xlabel("Taille de l'échantillon d'entraînement", fontsize=11)
    ax.set_ylabel(f"Score ({scoring})", fontsize=11)
    ax.set_title("Courbes d'apprentissage", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    # Retourne aussi les statistiques finales
    stats = {
        "train_mean": train_mean[-1],
        "train_std": train_std[-1],
        "val_mean": val_mean[-1],
        "val_std": val_std[-1],
        "ecart": train_mean[-1] - val_mean[-1],
    }

    return ax, stats

def creation_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des variables dérivées pour enrichir le dataset Titanic.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les données du Titanic

    Retours
    -------
    pd.DataFrame
        DataFrame enrichi avec les nouvelles variables
    """
    df_nouveau = df.copy()
    df_nouveau['taille_famille'] = df_nouveau['SibSp'] + df_nouveau['Parch'] + 1
    df_nouveau["est_seul"] = (df_nouveau['taille_famille'] == 1).astype(int)
    return df_nouveau

def cramer_v_coeff(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """
    Calcule le V de Cramer entre deux variables catégorielles.

    Le V de Cramer mesure l'intensité de l'association entre deux variables
    catégorielles. Il est basé sur la statistique du Khi-deux.

    Parameters
    ----------
    x : pd.Series
        Première variable catégorielle.
    y : pd.Series
        Deuxième variable catégorielle.

    Returns
    -------
    tuple[float, float]
        - V de Cramer (entre 0 et 1)
        - p-value du test du Khi-deux
    """
    # Suppression des valeurs manquantes
    complete_cases = x.isna() | y.isna()
    x = x[~complete_cases]
    y = y[~complete_cases]

    # Calcul du Khi-deux max (dénominateur du V de Cramer)
    n = len(x)
    khi2_max = n * (min(x.nunique(), y.nunique()) - 1)

    # Construction du tableau de contingence et calcul du Khi-deux
    contingency_table = pd.crosstab(x, y)
    khi2_result = ss.chi2_contingency(observed=contingency_table, correction=True)

    # Calcul du V de Cramer
    cramer = round(np.sqrt(khi2_result.statistic / khi2_max), 4)
    p_value = khi2_result.pvalue

    return cramer, p_value

def compute_cramer_v(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la matrice des V de Cramer pour toutes les paires de variables.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame contenant uniquement des variables catégorielles.

    Returns
    -------
    pd.DataFrame
        Matrice carrée contenant les V de Cramer entre chaque paire de variables.
        La diagonale contient des 1 (association parfaite d'une variable avec elle-même).
    """
    ncols = data.shape[1]
    cols = data.columns
    cramer_matrix = np.eye(ncols)

    for j in range(ncols - 1):
        for i in range(j + 1, ncols):
            v_cramer, _ = cramer_v_coeff(x=data.iloc[:, j], y=data.iloc[:, i])
            cramer_matrix[i, j] = v_cramer
            cramer_matrix[j, i] = v_cramer

    return pd.DataFrame(cramer_matrix, columns=cols, index=cols)

def cramer_v_coeff(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """
    Calcule le V de Cramer entre deux variables catégorielles.

    Le V de Cramer mesure l'intensité de l'association entre deux variables
    catégorielles. Il est basé sur la statistique du Khi-deux.

    Paramètres
    ----------
    x : pd.Series
        Première variable catégorielle.
    y : pd.Series
        Deuxième variable catégorielle.

    Retourne
    --------
    Tuple[float, float]
        - V de Cramer (entre 0 et 1)
        - p-value du test du Khi-deux
    """
    # Suppression des valeurs manquantes
    complete_cases = x.isna() | y.isna()
    x = x[~complete_cases]
    y = y[~complete_cases]

    # Calcul du Khi-deux max (dénominateur du V de Cramer)
    n = len(x)
    khi2_max = n * (min(x.nunique(), y.nunique()) - 1)

    # Construction du tableau de contingence et calcul du Khi-deux
    contingency_table = pd.crosstab(x, y)
    khi2_result = ss.chi2_contingency(observed=contingency_table, correction=True)

    # Calcul du V de Cramer
    cramer = round(np.sqrt(khi2_result.statistic / khi2_max), 4)
    p_value = khi2_result.pvalue

    return cramer, p_value

def categorical_to_discrete(data: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """
    Encode les variables catégorielles en variables discrètes.

    Paramètres
    ----------
    data : pd.DataFrame
        DataFrame contenant les données à encoder.
    cat_cols : List[str]
        Liste des colonnes catégorielles à encoder.

    Retourne
    --------
    pd.DataFrame
        Copie du DataFrame avec les variables catégorielles encodées.
    """
    data_encoded = data.copy()
    for col in cat_cols:
        data_encoded[col] = data_encoded[col].map(code_dic)
    return data_encoded

def load_scoring_data(
    url: str = "https://raw.githubusercontent.com/datagong/data/main/bank-additional-full.csv",
) -> pd.DataFrame:
    """
    Charge le jeu de données de scoring bancaire.

    Paramètres
    ----------
    url : str, optional
        URL du fichier CSV à charger.

    Retourne
    --------
    pd.DataFrame
        DataFrame contenant les données brutes.
    """
    return pd.read_csv(filepath_or_buffer=url, sep=";")

def get_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Identifie les colonnes catégorielles et numériques d'un DataFrame.

    Paramètres
    ----------
    X : pd.DataFrame
        DataFrame contenant les variables explicatives.

    Retourne
    --------
    Tuple[List[str], List[str], List[str]]
        - Liste des colonnes catégorielles
        - Liste des colonnes numériques
        - Liste de toutes les colonnes
    """
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    all_cols = X.columns.tolist()
    return cat_cols, num_cols, all_cols

def model_evaluation(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[float, float]:
    """
    Évalue les performances d'un modèle sur les jeux train et test.

    Paramètres
    ----------
    model : Pipeline
        Modèle entraîné (pipeline scikit-learn).
    X_train : pd.DataFrame
        Variables explicatives du jeu d'entraînement.
    y_train : pd.Series
        Variable cible du jeu d'entraînement.
    X_test : pd.DataFrame
        Variables explicatives du jeu de test.
    y_test : pd.Series
        Variable cible du jeu de test.

    Retourne
    --------
    Tuple[float, float]
        AUC sur le jeu d'entraînement et AUC sur le jeu de test.
    """
    # Prédiction des probabilités
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    # Calcul des AUC
    auc_train = roc_auc_score(y_train, y_train_pred)
    auc_test = roc_auc_score(y_test, y_test_pred)

    return auc_train, auc_test

def lift_curve_data(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Calcule les données nécessaires pour tracer la courbe de lift.

    Paramètres
    ----------
    y_true : np.ndarray | pd.Series
        Labels réels (0 ou 1).
    y_pred : np.ndarray
        Probabilités prédites par le modèle.

    Retourne
    --------
    pd.DataFrame
        DataFrame contenant les colonnes nécessaires pour tracer la courbe de lift,
        notamment la colonne 'concentration' représentant le pourcentage cumulé
        de la cible capturée.
    """
    # Trier par ordre décroissant des probabilités
    sorted_score = np.argsort(y_pred)[::-1]

    # Convertir en arrays si nécessaire
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.to_numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.to_numpy()

    y_true = y_true[sorted_score]
    y_pred = y_pred[sorted_score]

    # Créer un dataframe avec les vecteurs triés
    lift_df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})

    # Découper en percentiles
    lift_df["perc"] = (lift_df.index / lift_df.shape[0]) * 100
    lift_df["perc_bins"] = pd.cut(lift_df["perc"], np.arange(0, 101, 1))

    # Grouper par intervalle et calculer l'effectif des positifs
    lift_df = (
        lift_df.groupby(["perc_bins"], as_index=False, observed=True)
        .agg(y_true_sum=("y_true", "sum"))
    )

    # Calculer l'effectif cumulé des positifs
    lift_df["y_true_cumsum"] = lift_df["y_true_sum"].cumsum()

    # Calculer la concentration (% cumulé de positifs capturés)
    lift_df["concentration"] = (
        lift_df["y_true_cumsum"] / lift_df["y_true_sum"].sum()
    ) * 100

    # Ajouter le point (0, 0) pour la représentation graphique
    zeros_df = pd.DataFrame(np.zeros((1, 4)), columns=lift_df.columns)
    lift_df = pd.concat([zeros_df, lift_df]).reset_index(drop=True)

    return lift_df

def compute_lift_by_decile(
    scores: np.ndarray,
    labels: np.ndarray,
    n_deciles: int = 10
) -> pd.DataFrame:
    """
    Calcule le lift par décile pour un modèle de scoring.

    Paramètres
    ----------
    scores : np.ndarray
        Scores de probabilité prédits par le modèle.
    labels : np.ndarray
        Labels réels (0 ou 1).
    n_deciles : int, optional
        Nombre de déciles à calculer (défaut: 10).

    Retourne
    --------
    pd.DataFrame
        DataFrame contenant pour chaque décile :
        - decile : numéro du décile (1 = meilleurs scores)
        - n_obs : nombre d'observations
        - n_targets : nombre de cibles
        - response_rate : taux de réponse dans le décile
        - lift : lift par rapport au taux global
        - cumul_targets : cibles cumulées
        - cumul_capture : taux de capture cumulé
    """
    # Création d'un DataFrame de travail
    df = pd.DataFrame({"score": scores, "label": labels})

    # Attribution des déciles (1 = meilleurs scores)
    df["decile"] = pd.qcut(df["score"], q=n_deciles, labels=False)
    df["decile"] = n_deciles - df["decile"]  # Inverser pour que 1 = meilleurs

    # Taux de réponse global
    global_rate = df["label"].mean()

    # Agrégation par décile
    lift_table = (
        df.groupby("decile")
        .agg(
            n_obs=("label", "count"),
            n_targets=("label", "sum"),
        )
        .reset_index()
    )

    # Calcul des métriques
    lift_table["response_rate"] = lift_table["n_targets"] / lift_table["n_obs"]
    lift_table["lift"] = lift_table["response_rate"] / global_rate
    lift_table["cumul_targets"] = lift_table["n_targets"].cumsum()
    lift_table["cumul_capture"] = lift_table["cumul_targets"] / df["label"].sum()

    return lift_table


# ====================================================================================================
# Sélection de features - Modules 6-7
# ====================================================================================================

def creer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée de nouvelles variables à partir des variables existantes.

    Cette fonction génère deux nouvelles caractéristiques :
    - family_size : taille totale de la famille présente à bord (SibSp + Parch + 1)
    - is_alone : indicateur binaire si le passager voyage seul (1) ou non (0)

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant au minimum les colonnes 'SibSp' et 'Parch'

    Retourne
    --------
    pd.DataFrame
        Nouveau DataFrame avec les colonnes 'family_size' et 'is_alone' ajoutées

    Exemples
    --------
    >>> df_enrichi = creer_features(df_train)
    >>> df_enrichi[['family_size', 'is_alone']].head()
    """
    df_nouveau = df.copy()
    df_nouveau['family_size'] = df_nouveau['SibSp'] + df_nouveau['Parch'] + 1
    df_nouveau["is_alone"] = (df_nouveau['family_size'] == 1).astype(int)

    return df_nouveau

def get_original_feature_names(
    dummies_features: List[str], separator: str = "___"
) -> List[str]:
    """
    Récupère les noms des variables originales à partir des variables dummies.

    Pour les variables encodées avec un séparateur (par défaut '___'), cette fonction
    extrait le préfixe (nom de la variable originale). Les variables numériques
    non encodées sont conservées telles quelles.

    Paramètres
    ----------
    dummies_features : List[str]
        Liste des noms de variables dummies issues de get_dummies().
    separator : str, optional
        Séparateur utilisé lors de l'encodage (défaut: '___').

    Retourne
    --------
    List[str]
        Liste unique des noms de variables originales, triée par ordre alphabétique.

    Exemple
    -------
    >>> get_original_feature_names(["age", "job___admin.", "job___student"])
    ['age', 'job']
    """
    import re

    original_features = []
    for col in dummies_features:
        # Recherche du préfixe avant le séparateur
        search_result = re.search(rf"(.+){separator}", col)
        if search_result:
            original_features.append(search_result.group(1))
        else:
            original_features.append(col)
    return np.unique(original_features).tolist()
