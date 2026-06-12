"""
features.py — feature engineering pipeline for the World Cup prediction model

For each match we build a feature vector from the HOME team's perspective.
Away team features are the mirror image.

Features produced:
  elo_home, elo_away, elo_diff          — ELO ratings at match time
  form_home, form_away                   — points per game in last N matches
  goals_scored_home/away                 — recent attacking output
  goals_conceded_home/away               — recent defensive output
  h2h_win_rate_home                      — historical head-to-head win rate
  days_since_last_match_home/away        — fatigue / rest proxy
  is_neutral                             — venue type
  tournament_weight                      — match importance (maps to K-factor scale)
"""

import pandas as pd
import numpy as np
from src.elo import EloSystem, get_k
from src.utils import get_logger

logger = get_logger(__name__)


# ── Form helpers ──────────────────────────────────────────────────────────────

def _team_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Return all matches involving a team, with a unified perspective:
    columns: date, opponent, scored, conceded, points, tournament
    """
    home = df[df.home_team == team].copy()
    home = home.rename(columns={
        "home_score": "scored",
        "away_score": "conceded",
        "away_team":  "opponent",
    })[["date", "opponent", "scored", "conceded", "tournament", "neutral"]]
    home["points"] = home.apply(
        lambda r: 3 if r.scored > r.conceded else (1 if r.scored == r.conceded else 0), axis=1
    )

    away = df[df.away_team == team].copy()
    away = away.rename(columns={
        "away_score": "scored",
        "home_score": "conceded",
        "home_team":  "opponent",
    })[["date", "opponent", "scored", "conceded", "tournament", "neutral"]]
    away["points"] = away.apply(
        lambda r: 3 if r.scored > r.conceded else (1 if r.scored == r.conceded else 0), axis=1
    )

    combined = pd.concat([home, away]).sort_values("date").reset_index(drop=True)
    return combined


def compute_form(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    window: int = 10,
) -> dict:
    """
    Compute form metrics for a team in the N matches before a given date.
    Returns a dict of feature values.
    """
    matches = _team_matches(df, team)
    past = matches[matches.date < before_date].tail(window)

    if len(past) == 0:
        return {
            "form_ppg":        0.0,
            "goals_scored":    0.0,
            "goals_conceded":  0.0,
            "win_rate":        0.0,
            "days_since_last": 365,
            "n_matches":       0,
        }

    return {
        "form_ppg":        past.points.mean(),
        "goals_scored":    past.scored.mean(),
        "goals_conceded":  past.conceded.mean(),
        "win_rate":        (past.points == 3).mean(),
        "days_since_last": (before_date - past.date.max()).days,
        "n_matches":       len(past),
    }


# ── Head-to-head ──────────────────────────────────────────────────────────────

def compute_h2h(
    df: pd.DataFrame,
    team_a: str,
    team_b: str,
    before_date: pd.Timestamp,
    window: int = 10,
) -> dict:
    """
    Head-to-head stats between two teams before a given date.
    Perspective: team_a
    """
    mask = (
        ((df.home_team == team_a) & (df.away_team == team_b)) |
        ((df.home_team == team_b) & (df.away_team == team_a))
    )
    h2h = df[mask & (df.date < before_date)].tail(window)

    if len(h2h) == 0:
        return {"h2h_win_rate": 0.5, "h2h_goal_diff": 0.0, "h2h_n": 0}

    a_wins, b_wins = 0, 0
    goal_diff = 0
    for _, r in h2h.iterrows():
        if r.home_team == team_a:
            gd = r.home_score - r.away_score
        else:
            gd = r.away_score - r.home_score
        goal_diff += gd
        if gd > 0:
            a_wins += 1
        elif gd < 0:
            b_wins += 1

    total = len(h2h)
    return {
        "h2h_win_rate":   a_wins / total,
        "h2h_goal_diff":  goal_diff / total,
        "h2h_n":          total,
    }


# ── Main feature builder ──────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    elo_system: EloSystem,
    form_window: int = 10,
    h2h_window: int = 10,
) -> pd.DataFrame:
    """
    Build the full feature matrix from the match results DataFrame.

    Each row = one match (from home team perspective).
    No data leakage: all features are computed from data BEFORE match date.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned match results (output of 01_eda notebook).
    elo_system : EloSystem
        Fitted EloSystem instance.
    form_window : int
        Number of recent matches for form calculation.
    h2h_window : int
        Number of recent H2H matches to consider.

    Returns
    -------
    pd.DataFrame with one row per match and feature + target columns.
    """
    logger.info(f"Building features for {len(df):,} matches...")
    elo_history = elo_system.history_df()

    rows = []

    for _, match in df.iterrows():
        date      = match["date"]
        home_team = match["home_team"]
        away_team = match["away_team"]

        # ── ELO features (pre-match, from history) ──
        hist_row = elo_history[elo_history.date == date]
        hist_match = hist_row[
            (hist_row.home_team == home_team) &
            (hist_row.away_team == away_team)
        ]

        if hist_match.empty:
            elo_home = elo_system.rating(home_team)
            elo_away = elo_system.rating(away_team)
        else:
            elo_home = hist_match.iloc[0]["elo_home_pre"]
            elo_away = hist_match.iloc[0]["elo_away_pre"]

        # ── Form features ──
        home_form = compute_form(df, home_team, date, window=form_window)
        away_form = compute_form(df, away_team, date, window=form_window)

        # ── Head-to-head ──
        h2h = compute_h2h(df, home_team, away_team, date, window=h2h_window)

        # ── Tournament weight ──
        t_weight = get_k(match["tournament"]) / 60.0  # normalise to [0, 1]

        # ── Outcome targets ──
        hs, as_ = match["home_score"], match["away_score"]
        if hs > as_:
            outcome = 2   # home win
        elif hs == as_:
            outcome = 1   # draw
        else:
            outcome = 0   # away win

        rows.append({
            # identifiers (not used as features)
            "date":           date,
            "home_team":      home_team,
            "away_team":      away_team,
            "tournament":     match["tournament"],

            # ELO
            "elo_home":       elo_home,
            "elo_away":       elo_away,
            "elo_diff":       elo_home - elo_away,

            # Form — home
            "form_ppg_home":          home_form["form_ppg"],
            "goals_scored_home":      home_form["goals_scored"],
            "goals_conceded_home":    home_form["goals_conceded"],
            "win_rate_home":          home_form["win_rate"],
            "days_since_last_home":   home_form["days_since_last"],

            # Form — away
            "form_ppg_away":          away_form["form_ppg"],
            "goals_scored_away":      away_form["goals_scored"],
            "goals_conceded_away":    away_form["goals_conceded"],
            "win_rate_away":          away_form["win_rate"],
            "days_since_last_away":   away_form["days_since_last"],

            # Form differentials (often more predictive than absolutes)
            "elo_diff":               elo_home - elo_away,
            "form_diff":              home_form["form_ppg"] - away_form["form_ppg"],
            "goals_scored_diff":      home_form["goals_scored"] - away_form["goals_scored"],
            "goals_conceded_diff":    home_form["goals_conceded"] - away_form["goals_conceded"],

            # Head-to-head
            "h2h_win_rate":           h2h["h2h_win_rate"],
            "h2h_goal_diff":          h2h["h2h_goal_diff"],
            "h2h_n":                  h2h["h2h_n"],

            # Match context
            "is_neutral":             int(match["neutral"]),
            "tournament_weight":      t_weight,

            # Targets
            "home_score":             int(hs),
            "away_score":             int(as_),
            "outcome":                outcome,    # 2=home win, 1=draw, 0=away win
        })

    features_df = pd.DataFrame(rows)
    logger.info(f"Feature matrix shape: {features_df.shape}")
    return features_df


# ── Feature column lists (used by model.py) ──────────────────────────────────

FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff",
    "form_ppg_home", "goals_scored_home", "goals_conceded_home",
    "win_rate_home", "days_since_last_home",
    "form_ppg_away", "goals_scored_away", "goals_conceded_away",
    "win_rate_away", "days_since_last_away",
    "form_diff", "goals_scored_diff", "goals_conceded_diff",
    "h2h_win_rate", "h2h_goal_diff",
    "is_neutral", "tournament_weight",
]

TARGET_OUTCOME  = "outcome"      # 3-class: 0=away, 1=draw, 2=home
TARGET_HOME_GLS = "home_score"   # Poisson target
TARGET_AWAY_GLS = "away_score"   # Poisson target
