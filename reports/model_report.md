# Rapport du modèle — Scoring de Défaut de Crédit

**Projet :** Scoring de Défaut de Crédit + POC GenAI « Talk to my Data »  
**Contexte :** Direction Recouvrement & Risque — Banque de détail  
**Date d'entraînement :** 28 avril 2026  
**Random state :** 1204

---

## 1. Contexte métier

La direction Recouvrement & Risque souhaite renforcer sa politique de relance et de recouvrement au prochain trimestre. Pour ce faire, un modèle de scoring probabiliste est nécessaire afin d'identifier, parmi les clients porteurs d'une carte de crédit, ceux susceptibles de faire **défaut le mois suivant**.

L'objectif est de produire un score de probabilité de défaut par client permettant de **prioriser les campagnes de relance** : on ciblera les K% de clients avec les scores les plus élevés.

---

## 2. Dataset

| Caractéristique | Valeur |
|---|---|
| Source | `credit_card_default.csv` — export BigQuery ML datasets |
| Population | Clients porteurs de carte de crédit (Taiwan) |
| Taille totale | 2 965 lignes × 26 colonnes |
| Variable cible | `default_payment_next_month` (binaire : 1 = défaut, 0 = pas de défaut) |
| Taux de défaut global | ~21 % |
| Colonne de leakage | `predicted_default_payment_next_month` — **exclue** du modèle |

### Variables disponibles

- **Socio-démographiques :** `sex`, `education_level`, `marital_status`, `age`
- **Financières :** `limit_balance`, `bill_amt_1..6`, `pay_amt_1..6`
- **Historique de paiement :** `pay_0`, `pay_2..6` (−1 = à temps, 1 = 1 mois de retard…)

### Leakage détecté

La colonne `predicted_default_payment_next_month` présente une corrélation supérieure à 0.50 avec la cible (ROC-AUC ~0.796 à elle seule). Elle a été définitivement exclue de toutes les analyses et du pipeline de modélisation.

---

## 3. Protocole d'évaluation

| Paramètre | Valeur |
|---|---|
| Split | Stratifié 80 / 20 (`random_state=1204`) |
| Train | 2 372 observations |
| Test (hold-out) | 593 observations — utilisé **une seule fois** en évaluation finale |
| Taux de défaut train | 21,4 % |
| Taux de défaut test | 21,4 % |
| Validation interne | Cross-validation 5-fold sur `X_train` uniquement |
| Métrique prioritaire | **PR-AUC** (plus discriminante que ROC-AUC sur données déséquilibrées) |

---

## 4. Feature Engineering

Les transformations suivantes ont été appliquées **dans le pipeline scikit-learn** (aucun leakage possible) :

- **Codes aberrants** : les valeurs `pay_0..6 ∈ {-2, -1}` recodées en 0 (paiement à temps ou anticipé)
- **`utilization_proxy`** : ratio `sum(bill_amt) / (limit_balance × 6)` — mesure du taux d'utilisation moyen du crédit
- **`pay_trend`** : différence entre `pay_0` et `pay_6` — capte la dégradation récente du comportement de paiement
- **`avg_pay_status`** : moyenne des `pay_0..6` — résumé de l'historique de paiement
- **`total_pay_amt`** : somme des `pay_amt_1..6` — effort de remboursement global
- **Scaling** : `RobustScaler` (résistant aux outliers, cohérent avec les distributions observées en EDA)

---

## 5. Modèles entraînés

### 5.1 Baseline (Notebook 02)

Trois modèles comparés en cross-validation 5-fold sur `X_train` :

| Modèle | PR-AUC CV | ROC-AUC CV |
|---|---|---|
| Régression Logistique | ~0.575 | ~0.775 |
| Random Forest | ~0.610 | ~0.795 |
| XGBoost | ~0.598 | ~0.790 |

> Le Random Forest ressort comme meilleur baseline — robustesse aux outliers et gestion native de la non-linéarité.

### 5.2 Fine-tuning (Notebook 03)

Optimisation via **`RandomizedSearchCV`** (scoring = `average_precision`, 5-fold CV sur `X_train`) :

| Modèle | PR-AUC CV (tuné) | Gain vs baseline |
|---|---|---|
| Régression Logistique tuned | ~0.582 | +0.007 |
| **Random Forest tuned** | **~0.621** | **+0.011** |
| XGBoost tuned | ~0.618 | +0.020 |

### Traitement du déséquilibre des classes

Le Random Forest optimisé utilise `class_weight="balanced_subsample"` :  
chaque arbre ré-équilibre automatiquement les classes lors de son échantillonnage — solution plus robuste que le sur-échantillonnage global (SMOTE) sur ce dataset.

---

## 6. Modèle retenu — Random Forest optimisé

### Hyperparamètres

| Paramètre | Valeur |
|---|---|
| `n_estimators` | 392 |
| `max_depth` | 6 |
| `max_features` | 0.5 |
| `min_samples_leaf` | 33 |
| `min_samples_split` | 69 |
| `class_weight` | `balanced_subsample` |

### Performances sur le jeu de test (hold-out)

| Métrique | Valeur |
|---|---|
| **PR-AUC ⭐** | **0.6208** |
| ROC-AUC | 0.8137 |
| F1-Score | 0.5839 |
| Recall | 0.6299 |
| Precision | 0.5442 |

Le modèle détecte **63 % des défauts réels** (recall) avec une précision de 54 %, ce qui représente un ratio signal/bruit acceptable pour cibler les campagnes de relance.

---

## 7. Stratégie de seuil et ciblage

L'approche retenue est celle du **top K% à relancer** :

- Le modèle produit un score de probabilité `proba_default ∈ [0, 1]` pour chaque client
- On trie les clients par score décroissant et on cible les K% les plus risqués
- **Lecture du lift par décile :** le décile supérieur (D1) concentre ~3× plus de défauts que le hasard

| Décile ciblé | % défauts capturés (estimé) |
|---|---|
| Top 10% (D1) | ~35-40 % |
| Top 20% (D2) | ~55-60 % |
| Top 30% (D3) | ~68-72 % |

**Recommandation opérationnelle :** cibler les **top 20-30%** permet de capturer plus de 55-70 % des défauts tout en maintenant un effort de relance raisonnable (ratio coût/bénéfice positif avec un coût de relance < coût de défaut).

---

## 8. Figures d'analyse

| Fichier | Description |
|---|---|
| `01_target_distribution.png` | Distribution de la variable cible |
| `02_socio_demo.png` | Taux de défaut par variables socio-démographiques |
| `03_age_default.png` | Taux de défaut par tranche d'âge |
| `04_pay_status_default.png` | Impact du statut de paiement sur le défaut |
| `09_correlations.png` | Corrélations features / cible |
| `14_models_comparison.png` | Comparaison courbes ROC/PR des 3 modèles baseline |
| `24_decile_tuned_lift_capture.png` | Lift & capture cumulée (modèles tuned, X_train) |
| `25_xtest_roc_pr.png` | Courbes ROC et PR sur le jeu de test |
| `26_xtest_confusion_matrices.png` | Matrices de confusion sur le jeu de test |
| `27_xtest_decile_lift_capture.png` | Lift & capture cumulée sur le jeu de test |

---

## 9. Limites et recommandations

### Limites identifiées

- **Dataset de taille modeste** : 2 965 observations — les performances pourraient être instables sur de nouveaux segments clients
- **Absence de calibration** : le modèle n'est pas calibré (Platt scaling ou isotonic regression non appliqués) ; les probabilités ne sont pas directement interprétables comme des probabilités fréquentistes
- **Données statiques** : le modèle ne gère pas la dérive temporelle (concept drift) — il devra être réentraîné régulièrement
- **Origine des données** : dataset taiwanais (2005) — la transférabilité à une banque française contemporaine n'est pas garantie

### Recommandations

1. **Calibration** : appliquer un `CalibratedClassifierCV` en post-traitement si les probabilités brutes sont utilisées dans les outils décisionnels
2. **Monitoring** : mettre en place un suivi mensuel du PSI (Population Stability Index) et du ROC-AUC en production
3. **Enrichissement** : intégrer des données comportementales supplémentaires (ancienneté client, nombre de produits détenus) pour améliorer la discrimination
4. **Seuil dynamique** : réviser le seuil/topK à chaque trimestre en fonction des coûts réels de relance et du taux de défaut observé
5. **Explicabilité** : intégrer SHAP values pour les analyses individuelles en production

---

## 10. Artefacts produits

| Fichier | Description |
|---|---|
| `models/best_model_pipeline.joblib` | Pipeline complet (FE + Scaler + RF optimisé), prêt à l'inférence |
| `models/best_model_metadata.json` | Métriques, hyperparamètres, shapes train/test |
| `utils/infer.py` | Script d'inférence : `predict_scores()`, `rank_by_score()`, `compute_gains()` |
