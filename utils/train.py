"""
Module d'entraînement et d'optimisation de modèles.

Ce module contient les fonctions nécessaires pour :
- Entraîner les modèles de machine learning
- Optimiser les hyperparamètres
- Effectuer l'ingénierie des features
- Opérations matricielles et transformations
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Sequence, Optional

try:
    from numpy.typing import ArrayLike
except ImportError:
    from typing import Any as ArrayLike
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline


def five_year_interval(x: datetime) -> str:
    """
    Retourne l'intervalle de 5 ans pour une date donnée.
    
    Utilisé pour segmenter les données temporelles en périodes de 5 ans.
    
    Args:
        x: Date (objet datetime)
    
    Returns:
        Chaîne représentant l'intervalle (ex: "2010-2015")
    
    Exemple:
        >>> from datetime import datetime
        >>> date = datetime.strptime('2012-06-15', '%Y-%m-%d')
        >>> interval = five_year_interval(date)
        >>> print(interval)  # "2010-2015"
    """
    if x < datetime.strptime('2005-01-01', '%Y-%m-%d'):
        return "2000-2004"
    elif x < datetime.strptime('2010-01-01', '%Y-%m-%d'):
        return "2005-2009"
    elif x < datetime.strptime('2015-01-01', '%Y-%m-%d'):
        return "2010-2015"
    else:
        return "2015-2020"


def broadcast_addition(matrix: List[List[float]], vector: List[float]) -> List[List[float]]:
    """
    Additionne un vecteur à chaque ligne d'une matrice en utilisant le broadcasting.
    
    Cette fonction implémente la notion de broadcasting NumPy :
    m x n matrix + n vector = m x n result
    
    Args:
        matrix: Liste de listes (matrice m x n)
        vector: Liste (vecteur de taille n)
    
    Returns:
        Matrice résultante de même taille
    
    Exemple:
        >>> matrix = [[1, 2], [3, 4]]
        >>> vector = [10, 20]
        >>> result = broadcast_addition(matrix, vector)
        >>> print(result)  # [[11, 22], [13, 24]]
    """
    new_matrix = []
    for row in matrix:
        new_row = []
        for a, b in zip(row, vector):
            new_row.append(a + b)
        new_matrix.append(new_row)
    return new_matrix


def estimate_pi(n: int) -> float:
    """
    Estime la valeur de π en utilisant la méthode de Monte Carlo.
    
    Génère n points aléatoires dans un carré unitaire et compte
    combien tombent dans le cercle inscrit. 
    Formule: π ≈ 4 * (points_in_circle / total_points)
    
    Args:
        n: Nombre de points aléatoires à générer
    
    Returns:
        Approximation de π
    
    Exemple:
        >>> import numpy as np
        >>> pi_approx = estimate_pi(1000000)
        >>> print(pi_approx)  # ≈ 3.14159...
    """
    random_points = np.random.rand(n, 2)  # n points aléatoires (x,y) in [0,1]²
    is_in_circle = np.sum(random_points ** 2, axis=1) <= 1  # booleans (true si point in cercle)
    n_in_circle = np.sum(is_in_circle)  # nombre de points dans le cercle
    return (n_in_circle / n) * 4


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """
    Calcule la fonction sigmoïde (fonction d'activation).
    
    Formule: σ(x) = 1 / (1 + e^(-x))
    
    La fonction sigmoïde mappe toute valeur réelle à l'intervalle (0, 1).
    Utilisée en machine learning comme fonction d'activation.
    
    Args:
        x: Scalaire ou array NumPy
    
    Returns:
        Sigmoid(x) - même type que l'input
    
    Exemple:
        >>> import numpy as np
        >>> x = np.array([0, 1, -1])
        >>> y = sigmoid(x)
        >>> print(y)  # [0.5, 0.73105858, 0.26894142]
    """
    return 1 / (1 + np.exp(-x))




# === Fonctions des Modules 5, 6, 7 === #

def fit_reg_lineaire(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Estime les paramètres beta de la régression linéaire via la méthode analytique (OLS).

    Formule : β = (X'X)⁻¹X'y

    Paramètres
    ----------
    X : np.ndarray de forme (n_observations, n_features)
        Matrice des variables explicatives
    y : np.ndarray de forme (n_observations,)
        Vecteur de la variable cible

    Retourne
    --------
    beta : np.ndarray de forme (n_features + 1,)
        Vecteur des coefficients (incluant l'intercept β₀)

    Notes
    -----
    - L'intercept (β₀) est automatiquement ajouté comme première composante
    - Cette méthode suppose que X'X est inversible (absence de multicolinéarité)
    - Utilise la décomposition matricielle optimisée de NumPy

    Exemples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 3])
    >>> beta = fit_reg_lineaire(X, y)
    """

    # Etape 1
    # Ici on crée un vecteur NumPy contenant uniquement des 1
    # Ce vecteur contient une seule colonne (contenant que des 1) et le même nombre d'observations du jeu de données
    # Ce vecteur est créé pour estimer la constante beta_0 du modèle linéaire

    constante = np.ones((X.shape[0], 1))

    # Etape 2
    # Ajout de la constante dans les vecteurs de variables explicatives
    X = np.concatenate((constante, X), axis=1)

    # Le vecteur de variables explicatives est maintenant de taille 3 + 1 = 4
    # 3 désigne le nombre de variables explicatives
    # 1 désigne le vecteur contenant des 1 qui permet d'estimer le beta_0

    # Etape 3
    # Calculez la première composante c1 = X' * X
    c1 = np.matmul(X.T, X)

    # Etape 4
    # Calculez l'inverse de c1
    c1_inv = np.linalg.inv(c1)

    # Etape 5
    # Calculez c2 = c1_inv * X'
    c2 = np.matmul(c1_inv, X.T)

    # Etape 6
    # Calculez beta = c2 * y
    beta = np.matmul(c2, y)

    return beta  # Retourne le vecteur beta

def fit_reg_ridge(X: np.ndarray, y: np.ndarray, lambda_: float = 1) -> np.ndarray:
    """
    Estime les paramètres beta de la régression Ridge via la méthode analytique.

    Formule : β = (X'X + λI)⁻¹X'y

    Paramètres
    ----------
    X : np.ndarray de forme (n_observations, n_features)
        Matrice des variables explicatives
    y : np.ndarray de forme (n_observations,)
        Vecteur de la variable cible
    lambda_ : float, optionnel (défaut=1)
        Hyperparamètre de régularisation Ridge (λ)
        Contrôle l'intensité de la pénalité L2 sur les coefficients

    Retourne
    --------
    beta : np.ndarray de forme (n_features + 1,)
        Vecteur des coefficients (incluant l'intercept β₀)

    Notes
    -----
    - L'intercept (β₀) est automatiquement ajouté comme première composante
    - L'intercept n'est pas pénalisé par la régularisation (I[0,0] = 0)
    - Plus λ est élevé, plus la pénalisation est forte
    - Cette méthode est stable même en cas de multicolinéarité

    Exemples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 3])
    >>> beta = fit_reg_ridge(X, y, lambda_=10)
    """

    # Étape 1 : Création du vecteur de constantes
    # Ici on crée un vecteur NumPy contenant uniquement des 1
    # Ce vecteur contient une seule colonne (contenant que des 1) et le même nombre
    # d'observations que le jeu de données
    # Ce vecteur est créé pour estimer la constante beta_0 du modèle linéaire
    constante = np.ones((X.shape[0], 1))

    # Étape 2 : Ajout de la constante dans la matrice des variables explicatives
    X = np.concatenate((constante, X), axis=1)

    # Le vecteur de variables explicatives est désormais de taille 13 + 1 = 14
    # 13 désigne le nombre de variables explicatives du dataset Boston Housing
    # 1 désigne le vecteur contenant des 1 qui permet d'estimer beta_0

    # Étape 3 : Création de la matrice identité
    # Créer la matrice identité contenant que des 1 sur la diagonale
    I = np.identity(X.shape[1])

    # La première composante de la diagonale correspond au paramètre beta_0
    # que l'on ne souhaite pas pénaliser ici
    I[0, 0] = 0

    # Étape 4 : Calculez la première composante c1 = X' * X
    c1 = np.matmul(X.T, X)

    # Étape 5 : Ajoutez l'hyperparamètre lambda : c1 = c1 + lambda_ * I
    c1 = c1 + lambda_ * I

    # Étape 6 : Calculez l'inverse de c1
    c1_inv = np.linalg.inv(c1)

    # Étape 7 : Calculez c2 = c1_inv * X'
    c2 = np.matmul(c1_inv, X.T)

        # Étape 8 : Calculez beta = c2 * y
    beta = np.matmul(c2, y)

    return beta  # retourne le vecteur beta

def fit_lasso(X: np.ndarray, y: np.ndarray, lambda_: float, learning_rate: float, n_iteration: int) -> tuple[float, np.ndarray]:
    """
    Estime les paramètres beta par la méthode de la descente de gradient pour Lasso.

    L'algorithme initialise les paramètres à zéro puis les met à jour itérativement
    en suivant le gradient de la fonction de perte avec pénalité L1.

    Paramètres
    ----------
    X : np.ndarray
        Matrice de features de dimension (N, P).
    y : np.ndarray
        Target de dimension (N,).
    lambda_ : float
        Paramètre de pénalisation lambda de la régression Lasso (doit être > 0).
    learning_rate : float
        Le learning rate (taux d'apprentissage) qui contrôle la taille du pas.
    n_iteration : int
        Le nombre d'itérations à effectuer dans la descente de gradient.

    Retourne
    --------
    tuple[float, np.ndarray]
        beta_0 : float
            La valeur optimisée de beta_0 (intercept).
        beta_j : np.ndarray
            Les valeurs optimisées de beta_j (coefficients).
    """
    # Initialisation des paramètres à zéro
    beta_0 = 0
    beta_j = np.zeros(X.shape[1])

    # Itérations de la descente de gradient
    for i in range(n_iteration):
        beta_0, beta_j = update_weights(beta_0, beta_j, X, y, lambda_, learning_rate)

        # Affichage de la progression tous les 50 itérations
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Itération {i+1:3d} : beta_0 = {beta_0:7.3f}  beta_j = {beta_j.round(3)}")

    return beta_0, beta_j

def plot_decision_boundaries(
    clf: ClassifierMixin,
    X: pd.DataFrame,
    y: ArrayLike,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Affiche les frontières de décision d'un classifieur sur des données 2D.

    Cette visualisation montre comment le modèle sépare l'espace des
    caractéristiques en régions de classes distinctes et superpose les points
    d'entraînement.

    Paramètres
    ----------
    clf : ClassifierMixin
        Classifieur scikit-learn entraîné (doit implémenter ``predict`` et
        idéalement ``decision_function``/``predict_proba`` pour certaines
        visualisations).
    X : pandas.DataFrame, shape (n_samples, 2)
        Caractéristiques d'entrée (exactement 2 colonnes pour la projection 2D).
    y : ArrayLike, shape (n_samples,)
        Étiquettes de classes correspondantes.
    ax : matplotlib.axes.Axes, optional
        Axe matplotlib cible. Si ``None``, un nouvel axe est créé.

    Retours
    -------
    matplotlib.axes.Axes
        Axe contenant le tracé des frontières et des points.

    Notes
    -----
    Utilise ``sklearn.inspection.DecisionBoundaryDisplay.from_estimator`` pour
    un tracé standard conforme à scikit-learn.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Frontières de décision via API publique sklearn (>= 1.1)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel=X.columns[0],
        ylabel=X.columns[1],
    )

    # Points de données
    ax.scatter(
        X.values[:, 0],
        X.values[:, 1],
        c=y,
        cmap=plt.cm.coolwarm,
        s=20,
        edgecolors="k",
    )
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title("Frontières de décision")
    return ax

def fit_and_plot_classification(
    modele: ClassifierMixin,
    donnees: pd.DataFrame,
    noms_caracteristiques: Sequence[str],
    nom_cible: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Ajuste un classifieur puis trace sa frontière de décision et les points.

    Paramètres
    ----------
    modele : ClassifierMixin
        Modèle de classification scikit-learn (doit implémenter ``fit`` et
        ``predict`` ou ``decision_function``/``predict_proba``).
    donnees : pandas.DataFrame
        Données tabulaires contenant les colonnes des caractéristiques et la cible.
    noms_caracteristiques : Sequence[str]
        Noms des 2 caractéristiques utilisées pour l'entraînement et la visualisation.
    nom_cible : str
        Nom de la colonne cible (catégorielle) dans ``donnees``.
    ax : matplotlib.axes.Axes, optionnel
        Axe matplotlib sur lequel tracer. Si ``None``, un axe est créé.

    Retours
    -------
    matplotlib.axes.Axes
        Axe sur lequel la figure est tracée.

    Notes
    -----
    La frontière de décision est tracée via
    ``sklearn.inspection.DecisionBoundaryDisplay.from_estimator``. Les points
    d'entraînement sont superposés avec une palette à deux couleurs.
    """
    if ax is None:
        _, ax = plt.subplots()

    X_vis = donnees[list(noms_caracteristiques)]
    y_vis = donnees[nom_cible]

    palette = ["#d38f00", "#011c5d"]
    cmap_perso = mpl.colors.ListedColormap(palette)

    # Frontières de décision
    DecisionBoundaryDisplay.from_estimator(
        modele,
        X_vis,
        response_method="predict",
        cmap=cmap_perso,
        alpha=0.5,
        ax=ax,
    )

    # Nuage de points
    sns.scatterplot(
        data=donnees,
        x=noms_caracteristiques[0],
        y=noms_caracteristiques[1],
        hue=nom_cible,
        palette=palette,
        ax=ax,
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_xlabel(noms_caracteristiques[0])
    ax.set_ylabel(noms_caracteristiques[1])
    ax.set_title("Classification — frontières de décision")
    return ax

def fit_and_plot_regression(
    modele: RegressorMixin,
    donnees: pd.DataFrame,
    noms_caracteristiques: Sequence[str],
    nom_cible: str,
    nb_points: int = 200,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Ajuste un régleur (régression) puis trace la courbe prédite 1D.

    Paramètres
    ----------
    modele : RegressorMixin
        Modèle de régression scikit-learn (implémente ``fit`` et ``predict``).
    donnees : pandas.DataFrame
        Données tabulaires contenant la feature et la cible continues.
    noms_caracteristiques : Sequence[str]
        Noms des caractéristiques. Cette fonction attend une seule feature
        (la première est utilisée pour l'axe des abscisses).
    nom_cible : str
        Nom de la colonne cible continue dans ``donnees``.
    nb_points : int, par défaut 200
        Nombre de points pour l'abscisse régulière utilisée lors du tracé.
    ax : matplotlib.axes.Axes, optionnel
        Axe matplotlib sur lequel tracer. Si ``None``, un axe est créé.

    Retours
    -------
    matplotlib.axes.Axes
        Axe sur lequel la figure est tracée.

    Notes
    -----
    La courbe prédite est superposée au nuage de points. La fonction ne gère
    qu'une unique feature explicative pour la visualisation 1D.
    """
    if ax is None:
        _, ax = plt.subplots()

    feature = noms_caracteristiques[0]
    X_train = donnees[[feature]]
    y_train = donnees[nom_cible]

    # Ajustement du modèle
    modele.fit(X_train, y_train)

    # Abscisse régulière pour le tracé
    x_min, x_max = np.nanmin(X_train.values), np.nanmax(X_train.values)
    x_plot = np.linspace(x_min, x_max, nb_points)
    X_plot = pd.DataFrame({feature: x_plot})
    y_pred = modele.predict(X_plot)

    # Nuage + courbe
    sns.scatterplot(x=X_train[feature], y=y_train, color="black", alpha=0.5, ax=ax)
    ax.plot(x_plot, y_pred, color="#ff7373", lw=3)

    ax.set_xlabel(feature)
    ax.set_ylabel(nom_cible)
    ax.set_title("Régression — fonction prédite (constante par morceaux)")
    return ax

def fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: Any,
    **kwargs
) -> Pipeline:
    """
    Entraîne un modèle de classification avec preprocessing automatique.

    Cette fonction crée une pipeline complète incluant :
    - Imputation des valeurs manquantes
    - Normalisation des variables numériques (RobustScaler)
    - Encodage des variables catégorielles (OneHotEncoder)
    - Entraînement du classifieur

    Paramètres
    ----------
    X : pd.DataFrame
        Variables explicatives.
    y : pd.Series
        Variable cible.
    estimator : Any
        Classifieur scikit-learn à entraîner.
    **kwargs : dict
        Arguments supplémentaires passés à la méthode fit.

    Retourne
    --------
    Pipeline
        Pipeline entraînée contenant le preprocessing et le modèle.
    """
    # Séparer les colonnes en numériques et catégorielles
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    # Traitement des variables numériques
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler()
    )

    # Traitement des variables catégorielles
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore")
    )

    # Combinaison des deux pipelines
    preprocessor = make_column_transformer(
        (num_pipeline, num_cols),
        (cat_pipeline, cat_cols)
    )

    # Construction de la pipeline finale
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", estimator)
        ]
    )

    return model.fit(X, y, **kwargs)

__all__ = [
    'five_year_interval',
    'broadcast_addition',
    'estimate_pi',
    'sigmoid',
    'fit_reg_lineaire',
    'fit_reg_ridge',
    'fit_lasso',
    'plot_decision_boundaries',
    'fit_and_plot_classification',
    'fit_and_plot_regression',
    'fit_model'
]


# ====================================================================================================
# Fonctions d'entraînement des Modules 5-7
# ====================================================================================================

def fit_reg_lineaire(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Estime les paramètres beta de la régression linéaire via la méthode analytique (OLS).

    Formule : β = (X'X)⁻¹X'y

    Paramètres
    ----------
    X : np.ndarray de forme (n_observations, n_features)
        Matrice des variables explicatives
    y : np.ndarray de forme (n_observations,)
        Vecteur de la variable cible

    Retourne
    --------
    beta : np.ndarray de forme (n_features + 1,)
        Vecteur des coefficients (incluant l'intercept β₀)

    Notes
    -----
    - L'intercept (β₀) est automatiquement ajouté comme première composante
    - Cette méthode suppose que X'X est inversible (absence de multicolinéarité)
    - Utilise la décomposition matricielle optimisée de NumPy

    Exemples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 3])
    >>> beta = fit_reg_lineaire(X, y)
    """

    # Etape 1
    # Ici on crée un vecteur NumPy contenant uniquement des 1
    # Ce vecteur contient une seule colonne (contenant que des 1) et le même nombre d'observations du jeu de données
    # Ce vecteur est créé pour estimer la constante beta_0 du modèle linéaire

    constante = np.ones((X.shape[0], 1))

    # Etape 2
    # Ajout de la constante dans les vecteurs de variables explicatives
    X = np.concatenate((constante, X), axis=1)

    # Le vecteur de variables explicatives est maintenant de taille 3 + 1 = 4
    # 3 désigne le nombre de variables explicatives
    # 1 désigne le vecteur contenant des 1 qui permet d'estimer le beta_0

    # Etape 3
    # Calculez la première composante c1 = X' * X
    c1 = np.matmul(X.T, X)

    # Etape 4
    # Calculez l'inverse de c1
    c1_inv = np.linalg.inv(c1)

    # Etape 5
    # Calculez c2 = c1_inv * X'
    c2 = np.matmul(c1_inv, X.T)

    # Etape 6
    # Calculez beta = c2 * y
    beta = np.matmul(c2, y)

    return beta  # Retourne le vecteur beta

def fit_reg_ridge(X: np.ndarray, y: np.ndarray, lambda_: float = 1) -> np.ndarray:
    """
    Estime les paramètres beta de la régression Ridge via la méthode analytique.

    Formule : β = (X'X + λI)⁻¹X'y

    Paramètres
    ----------
    X : np.ndarray de forme (n_observations, n_features)
        Matrice des variables explicatives
    y : np.ndarray de forme (n_observations,)
        Vecteur de la variable cible
    lambda_ : float, optionnel (défaut=1)
        Hyperparamètre de régularisation Ridge (λ)
        Contrôle l'intensité de la pénalité L2 sur les coefficients

    Retourne
    --------
    beta : np.ndarray de forme (n_features + 1,)
        Vecteur des coefficients (incluant l'intercept β₀)

    Notes
    -----
    - L'intercept (β₀) est automatiquement ajouté comme première composante
    - L'intercept n'est pas pénalisé par la régularisation (I[0,0] = 0)
    - Plus λ est élevé, plus la pénalisation est forte
    - Cette méthode est stable même en cas de multicolinéarité

    Exemples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 3])
    >>> beta = fit_reg_ridge(X, y, lambda_=10)
    """

    # Étape 1 : Création du vecteur de constantes
    # Ici on crée un vecteur NumPy contenant uniquement des 1
    # Ce vecteur contient une seule colonne (contenant que des 1) et le même nombre
    # d'observations que le jeu de données
    # Ce vecteur est créé pour estimer la constante beta_0 du modèle linéaire
    constante = np.ones((X.shape[0], 1))

    # Étape 2 : Ajout de la constante dans la matrice des variables explicatives
    X = np.concatenate((constante, X), axis=1)

    # Le vecteur de variables explicatives est désormais de taille 13 + 1 = 14
    # 13 désigne le nombre de variables explicatives du dataset Boston Housing
    # 1 désigne le vecteur contenant des 1 qui permet d'estimer beta_0

    # Étape 3 : Création de la matrice identité
    # Créer la matrice identité contenant que des 1 sur la diagonale
    I = np.identity(X.shape[1])

    # La première composante de la diagonale correspond au paramètre beta_0
    # que l'on ne souhaite pas pénaliser ici
    I[0, 0] = 0

    # Étape 4 : Calculez la première composante c1 = X' * X
    c1 = np.matmul(X.T, X)

    # Étape 5 : Ajoutez l'hyperparamètre lambda : c1 = c1 + lambda_ * I
    c1 = c1 + lambda_ * I

    # Étape 6 : Calculez l'inverse de c1
    c1_inv = np.linalg.inv(c1)

    # Étape 7 : Calculez c2 = c1_inv * X'
    c2 = np.matmul(c1_inv, X.T)

        # Étape 8 : Calculez beta = c2 * y
    beta = np.matmul(c2, y)

    return beta  # retourne le vecteur beta

def fit_lasso(X: np.ndarray, y: np.ndarray, lambda_: float, learning_rate: float, n_iteration: int) -> tuple[float, np.ndarray]:
    """
    Estime les paramètres beta par la méthode de la descente de gradient pour Lasso.

    L'algorithme initialise les paramètres à zéro puis les met à jour itérativement
    en suivant le gradient de la fonction de perte avec pénalité L1.

    Paramètres
    ----------
    X : np.ndarray
        Matrice de features de dimension (N, P).
    y : np.ndarray
        Target de dimension (N,).
    lambda_ : float
        Paramètre de pénalisation lambda de la régression Lasso (doit être > 0).
    learning_rate : float
        Le learning rate (taux d'apprentissage) qui contrôle la taille du pas.
    n_iteration : int
        Le nombre d'itérations à effectuer dans la descente de gradient.

    Retourne
    --------
    tuple[float, np.ndarray]
        beta_0 : float
            La valeur optimisée de beta_0 (intercept).
        beta_j : np.ndarray
            Les valeurs optimisées de beta_j (coefficients).
    """
    # Initialisation des paramètres à zéro
    beta_0 = 0
    beta_j = np.zeros(X.shape[1])

    # Itérations de la descente de gradient
    for i in range(n_iteration):
        beta_0, beta_j = update_weights(beta_0, beta_j, X, y, lambda_, learning_rate)

        # Affichage de la progression tous les 50 itérations
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Itération {i+1:3d} : beta_0 = {beta_0:7.3f}  beta_j = {beta_j.round(3)}")

    return beta_0, beta_j

def plot_decision_boundaries(
    clf: ClassifierMixin,
    X: pd.DataFrame,
    y: ArrayLike,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Affiche les frontières de décision d'un classifieur sur des données 2D.

    Cette visualisation montre comment le modèle sépare l'espace des
    caractéristiques en régions de classes distinctes et superpose les points
    d'entraînement.

    Paramètres
    ----------
    clf : ClassifierMixin
        Classifieur scikit-learn entraîné (doit implémenter ``predict`` et
        idéalement ``decision_function``/``predict_proba`` pour certaines
        visualisations).
    X : pandas.DataFrame, shape (n_samples, 2)
        Caractéristiques d'entrée (exactement 2 colonnes pour la projection 2D).
    y : ArrayLike, shape (n_samples,)
        Étiquettes de classes correspondantes.
    ax : matplotlib.axes.Axes, optional
        Axe matplotlib cible. Si ``None``, un nouvel axe est créé.

    Retours
    -------
    matplotlib.axes.Axes
        Axe contenant le tracé des frontières et des points.

    Notes
    -----
    Utilise ``sklearn.inspection.DecisionBoundaryDisplay.from_estimator`` pour
    un tracé standard conforme à scikit-learn.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Frontières de décision via API publique sklearn (>= 1.1)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel=X.columns[0],
        ylabel=X.columns[1],
    )

    # Points de données
    ax.scatter(
        X.values[:, 0],
        X.values[:, 1],
        c=y,
        cmap=plt.cm.coolwarm,
        s=20,
        edgecolors="k",
    )
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title("Frontières de décision")
    return ax

def fit_and_plot_classification(
    modele: ClassifierMixin,
    donnees: pd.DataFrame,
    noms_caracteristiques: Sequence[str],
    nom_cible: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Ajuste un classifieur puis trace sa frontière de décision et les points.

    Paramètres
    ----------
    modele : ClassifierMixin
        Modèle de classification scikit-learn (doit implémenter ``fit`` et
        ``predict`` ou ``decision_function``/``predict_proba``).
    donnees : pandas.DataFrame
        Données tabulaires contenant les colonnes des caractéristiques et la cible.
    noms_caracteristiques : Sequence[str]
        Noms des 2 caractéristiques utilisées pour l'entraînement et la visualisation.
    nom_cible : str
        Nom de la colonne cible (catégorielle) dans ``donnees``.
    ax : matplotlib.axes.Axes, optionnel
        Axe matplotlib sur lequel tracer. Si ``None``, un axe est créé.

    Retours
    -------
    matplotlib.axes.Axes
        Axe sur lequel la figure est tracée.

    Notes
    -----
    La frontière de décision est tracée via
    ``sklearn.inspection.DecisionBoundaryDisplay.from_estimator``. Les points
    d'entraînement sont superposés avec une palette à deux couleurs.
    """
    if ax is None:
        _, ax = plt.subplots()

    X_vis = donnees[list(noms_caracteristiques)]
    y_vis = donnees[nom_cible]

    palette = ["#d38f00", "#011c5d"]
    cmap_perso = mpl.colors.ListedColormap(palette)

    # Frontières de décision
    DecisionBoundaryDisplay.from_estimator(
        modele,
        X_vis,
        response_method="predict",
        cmap=cmap_perso,
        alpha=0.5,
        ax=ax,
    )

    # Nuage de points
    sns.scatterplot(
        data=donnees,
        x=noms_caracteristiques[0],
        y=noms_caracteristiques[1],
        hue=nom_cible,
        palette=palette,
        ax=ax,
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_xlabel(noms_caracteristiques[0])
    ax.set_ylabel(noms_caracteristiques[1])
    ax.set_title("Classification — frontières de décision")
    return ax

def fit_and_plot_regression(
    modele: RegressorMixin,
    donnees: pd.DataFrame,
    noms_caracteristiques: Sequence[str],
    nom_cible: str,
    nb_points: int = 200,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Ajuste un régleur (régression) puis trace la courbe prédite 1D.

    Paramètres
    ----------
    modele : RegressorMixin
        Modèle de régression scikit-learn (implémente ``fit`` et ``predict``).
    donnees : pandas.DataFrame
        Données tabulaires contenant la feature et la cible continues.
    noms_caracteristiques : Sequence[str]
        Noms des caractéristiques. Cette fonction attend une seule feature
        (la première est utilisée pour l'axe des abscisses).
    nom_cible : str
        Nom de la colonne cible continue dans ``donnees``.
    nb_points : int, par défaut 200
        Nombre de points pour l'abscisse régulière utilisée lors du tracé.
    ax : matplotlib.axes.Axes, optionnel
        Axe matplotlib sur lequel tracer. Si ``None``, un axe est créé.

    Retours
    -------
    matplotlib.axes.Axes
        Axe sur lequel la figure est tracée.

    Notes
    -----
    La courbe prédite est superposée au nuage de points. La fonction ne gère
    qu'une unique feature explicative pour la visualisation 1D.
    """
    if ax is None:
        _, ax = plt.subplots()

    feature = noms_caracteristiques[0]
    X_train = donnees[[feature]]
    y_train = donnees[nom_cible]

    # Ajustement du modèle
    modele.fit(X_train, y_train)

    # Abscisse régulière pour le tracé
    x_min, x_max = np.nanmin(X_train.values), np.nanmax(X_train.values)
    x_plot = np.linspace(x_min, x_max, nb_points)
    X_plot = pd.DataFrame({feature: x_plot})
    y_pred = modele.predict(X_plot)

    # Nuage + courbe
    sns.scatterplot(x=X_train[feature], y=y_train, color="black", alpha=0.5, ax=ax)
    ax.plot(x_plot, y_pred, color="#ff7373", lw=3)

    ax.set_xlabel(feature)
    ax.set_ylabel(nom_cible)
    ax.set_title("Régression — fonction prédite (constante par morceaux)")
    return ax

def fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: Any,
    **kwargs
) -> Pipeline:
    """
    Entraîne un modèle de classification avec preprocessing automatique.

    Cette fonction crée une pipeline complète incluant :
    - Imputation des valeurs manquantes
    - Normalisation des variables numériques (RobustScaler)
    - Encodage des variables catégorielles (OneHotEncoder)
    - Entraînement du classifieur

    Paramètres
    ----------
    X : pd.DataFrame
        Variables explicatives.
    y : pd.Series
        Variable cible.
    estimator : Any
        Classifieur scikit-learn à entraîner.
    **kwargs : dict
        Arguments supplémentaires passés à la méthode fit.

    Retourne
    --------
    Pipeline
        Pipeline entraînée contenant le preprocessing et le modèle.
    """
    # Séparer les colonnes en numériques et catégorielles
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    # Traitement des variables numériques
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler()
    )

    # Traitement des variables catégorielles
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore")
    )

    # Combinaison des deux pipelines
    preprocessor = make_column_transformer(
        (num_pipeline, num_cols),
        (cat_pipeline, cat_cols)
    )

    # Construction de la pipeline finale
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", estimator)
        ]
    )

    return model.fit(X, y, **kwargs)
