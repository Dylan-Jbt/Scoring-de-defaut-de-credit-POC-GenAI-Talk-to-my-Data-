"""
Feature engineering — définition canonique.

Ce module expose `feature_engineering` de façon importable afin que
joblib.load() puisse résoudre la référence au FunctionTransformer
sérialisé dans le pipeline (défini à l'origine dans les notebooks).
"""

import numpy as np
import pandas as pd


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le DataFrame brut en ajoutant des features dérivées.

    Opérations appliquées :
    - Regroupe les modalités rares de education_level (0, 5, 6 → 4)
    - Regroupe les modalités rares de marital_status (0 → 3)
    - utilization_proxy : ratio montant moyen facturé / limite de crédit
    - pay_trend         : écart entre le statut de paiement récent (pay_0) et ancien (pay_6)
    - avg_pay_status    : moyenne des statuts de paiement sur 6 mois
    - total_pay_amt     : total des remboursements sur 6 mois
    """
    X = X.copy()

    # Nettoyage des modalités
    X["education_level"] = X["education_level"].replace({0: 4, 5: 4, 6: 4})
    X["marital_status"]  = X["marital_status"].replace({0: 3})

    bill_cols    = [f"bill_amt_{i}" for i in range(1, 7)]
    pay_amt_cols = [f"pay_amt_{i}" for i in range(1, 7)]
    pay_cols     = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]

    avg_bill = X[bill_cols].mean(axis=1)
    X["utilization_proxy"] = np.where(
        X["limit_balance"] > 0,
        avg_bill / X["limit_balance"].clip(lower=1),
        0.0,
    )
    X["pay_trend"]      = X["pay_0"] - X["pay_6"]
    X["avg_pay_status"] = X[pay_cols].mean(axis=1)
    X["total_pay_amt"]  = X[pay_amt_cols].sum(axis=1)

    return X
