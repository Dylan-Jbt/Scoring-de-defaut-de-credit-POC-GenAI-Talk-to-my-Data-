"""Fonctions graphiques Plotly réutilisables pour l'app Streamlit."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_PRIMARY   = "#1E3A5F"
_DANGER    = "#C0392B"
_SECONDARY = "#5B9BD5"
_PALETTE   = [_PRIMARY, _DANGER]


# ──────────────────────────────────────────────────────────────────────────────
# pie_target() — camembert répartition défaut / non-défaut
#
# Compte les effectifs de la variable cible et les affiche en camembert
# Plotly avec un anneau central (hole=0.45).
# ──────────────────────────────────────────────────────────────────────────────
def pie_target(df: pd.DataFrame, target_col: str = "default_payment_next_month") -> go.Figure:
    """Camembert répartition défaut / non-défaut."""
    counts = df[target_col].value_counts().rename({0: "Non-défaut", 1: "Défaut"})
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        color_discrete_sequence=_PALETTE,
        hole=0.45,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(showlegend=True, margin=dict(t=20, b=10))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# bar_default_rate() — taux de défaut par catégorie d’une variable
#
# Groupe le DataFrame par la colonne cible, calcule le taux de défaut
# moyen par modalité et applique optionnellement un dictionnaire de labels.
# ──────────────────────────────────────────────────────────────────────────────
def bar_default_rate(df: pd.DataFrame, col: str, label_map: dict | None = None) -> go.Figure:
    """Taux de défaut par catégorie d'une variable."""
    grp = df.groupby(col)["default_payment_next_month"].mean().reset_index()
    grp.columns = [col, "taux_defaut"]
    if label_map:
        grp[col] = grp[col].map(label_map).fillna(grp[col].astype(str))
    fig = px.bar(
        grp,
        x=col,
        y="taux_defaut",
        color_discrete_sequence=[_PRIMARY],
        text_auto=".1%",
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Taux de défaut",
        xaxis_title=col,
        margin=dict(t=20, b=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# hist_numeric() — histogramme d’une variable numérique coloré par cible
#
# Superpose deux histogrammes (Défaut / Non-défaut) en mode overlay
# pour visualiser la distribution d’une feature continue selon la cible.
# ──────────────────────────────────────────────────────────────────────────────
def hist_numeric(df: pd.DataFrame, col: str, target_col: str = "default_payment_next_month") -> go.Figure:
    """Histogramme d'une variable numérique, coloré par cible."""
    _df = df.copy()
    _df[target_col] = _df[target_col].map({0: "Non-défaut", 1: "Défaut"})
    fig = px.histogram(
        _df, x=col, color=target_col,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={"Non-défaut": _SECONDARY, "Défaut": _DANGER},
        nbins=40,
    )
    fig.update_layout(legend_title_text="", margin=dict(t=20, b=10))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# gauge_proba() — jauge Plotly de la probabilité de défaut
#
# Affiche un indicateur Indicator avec zones colorées (vert / orange / rouge).
# La couleur de la barre passe au rouge dès que proba ≥ 0.5.
# ──────────────────────────────────────────────────────────────────────────────
def gauge_proba(proba: float) -> go.Figure:
    """Jauge de probabilité de défaut."""
    color = _DANGER if proba >= 0.5 else _PRIMARY
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix": " %", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  40], "color": "#D5E8D4"},
                {"range": [40, 60], "color": "#FFF2CC"},
                {"range": [60, 100], "color": "#F8CECC"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# bar_lift() — graphique en barres du lift par décile
#
# Représente chaque décile avec son lift et trace une ligne horizontale
# pointillée à y=1 correspondant à la performance aléatoire.
# ──────────────────────────────────────────────────────────────────────────────
def bar_lift(df_gains: pd.DataFrame) -> go.Figure:
    """Graphique en barres du lift par décile."""
    fig = px.bar(
        df_gains, x="Decile", y="Lift",
        color_discrete_sequence=[_PRIMARY],
        text_auto=".2f",
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color=_DANGER, annotation_text="Aléatoire")
    fig.update_layout(xaxis_title="Décile", yaxis_title="Lift", margin=dict(t=20, b=10))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# line_capture() — courbe de capture cumulée (% de la cible vs. décile)
#
# Superpose la courbe modèle (Capture_Cumul) et une droite aléatoire
# en pointillés rouges pour visualiser le gain de ciblage.
# ──────────────────────────────────────────────────────────────────────────────
def line_capture(df_gains: pd.DataFrame) -> go.Figure:
    """Courbe de capture cumulée."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_gains["Decile"],
        y=df_gains["Capture_Cumul"] * 100,
        mode="lines+markers",
        line=dict(color=_PRIMARY, width=2),
        name="Modèle",
    ))
    # Ligne aléatoire
    n = len(df_gains)
    fig.add_trace(go.Scatter(
        x=df_gains["Decile"],
        y=[i / n * 100 for i in range(1, n + 1)],
        mode="lines",
        line=dict(color=_DANGER, dash="dash"),
        name="Aléatoire",
    ))
    fig.update_layout(
        xaxis_title="Décile",
        yaxis_title="% cible capturée",
        yaxis_ticksuffix="%",
        legend=dict(orientation="h"),
        margin=dict(t=20, b=10),
    )
    return fig
