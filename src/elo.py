"""
elo.py — ELO rating system for international football

How it works:
  - Every team starts at 1500 ELO
  - After each match, ratings are updated based on expected vs actual outcome
  - K-factor varies by match importance (World Cup > friendly)
  - Margin of victory multiplier rewards dominant wins
  - Ratings decay slightly toward the mean between tournaments

References:
  - World Football ELO: https://www.eloratings.net/about
  - FiveThirtyEight methodology (adapted)
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.utils import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_ELO = 1500.0
ELO_MEAN    = 1500.0  # global mean for decay

# K-factors by tournament importance
K_FACTORS: dict[str, float] = {
    "FIFA World Cup":               60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro":                    50,
    "Copa America":                 50,
    "Africa Cup of Nations":        45,
    "UEFA Nations League":          40,
    "Confederations Cup":           45,
    "Olympic Games":                35,
    "Friendly":                     20,
}
K_DEFAULT = 30  # for any tournament not listed above


def get_k(tournament: str) -> float:
    for key, k in K_FACTORS.items():
        if key.lower() in tournament.lower():
            return k
    return K_DEFAULT


# ── Core ELO functions ────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that team A wins (0–1)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def margin_multiplier(goal_diff: int) -> float:
    """
    Scale K by winning margin — bigger wins = bigger rating change.
    Formula from World Football ELO.
    """
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    elif gd == 3:
        return 1.75
    else:
        return 1.75 + (gd - 3) * 0.1


def actual_score(home_goals: int, away_goals: int) -> tuple[float, float]:
    """
    Returns (home_result, away_result) where:
      win=1.0, draw=0.5, loss=0.0
    """
    if home_goals > away_goals:
        return 1.0, 0.0
    elif home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def elo_update(
    rating_home: float,
    rating_away: float,
    home_goals: int,
    away_goals: int,
    tournament: str,
    is_neutral: bool = False,
    home_advantage: float = 100.0,
) -> tuple[float, float]:
    """
    Compute new ELO ratings after a single match.

    Returns (new_rating_home, new_rating_away).
    """
    # Apply home advantage if not a neutral venue
    adjusted_home = rating_home + (0 if is_neutral else home_advantage)

    exp_home = expected_score(adjusted_home, rating_away)
    exp_away = 1.0 - exp_home

    act_home, act_away = actual_score(home_goals, away_goals)
    goal_diff = home_goals - away_goals
    K = get_k(tournament)
    mult = margin_multiplier(goal_diff)

    delta_home = K * mult * (act_home - exp_home)
    delta_away = K * mult * (act_away - exp_away)

    return rating_home + delta_home, rating_away + delta_away


# ── Full history computation ──────────────────────────────────────────────────

class EloSystem:
    """
    Compute and store ELO ratings for all teams across the full match history.

    Usage:
        elo = EloSystem()
        elo.fit(results_df)
        ratings = elo.ratings          # current ratings dict
        history = elo.history_df()     # full match-by-match history DataFrame
    """

    def __init__(
        self,
        default_elo: float = DEFAULT_ELO,
        home_advantage: float = 100.0,
        decay_factor: float = 0.0,   # 0 = no decay; try 0.05 for light decay
    ):
        self.default_elo    = default_elo
        self.home_advantage = home_advantage
        self.decay_factor   = decay_factor
        self.ratings: dict[str, float] = {}
        self._history: list[dict] = []

    def _get(self, team: str) -> float:
        return self.ratings.get(team, self.default_elo)

    def _decay(self) -> None:
        """Pull all ratings slightly toward the mean (call between tournaments)."""
        if self.decay_factor == 0:
            return
        for team in self.ratings:
            self.ratings[team] += self.decay_factor * (ELO_MEAN - self.ratings[team])

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "EloSystem":
        """
        Process all matches in chronological order and build rating history.

        df must have: date, home_team, away_team, home_score, away_score,
                      tournament, neutral
        """
        self.ratings = {}
        self._history = []

        prev_year = None

        for _, row in df.iterrows():
            year = row["date"].year
            if prev_year is not None and year != prev_year:
                self._decay()
            prev_year = year

            home, away = row["home_team"], row["away_team"]
            r_home = self._get(home)
            r_away = self._get(away)

            # Store pre-match ratings in history
            self._history.append({
                "date":           row["date"],
                "home_team":      home,
                "away_team":      away,
                "home_score":     row["home_score"],
                "away_score":     row["away_score"],
                "tournament":     row["tournament"],
                "neutral":        row["neutral"],
                "elo_home_pre":   r_home,
                "elo_away_pre":   r_away,
                "elo_diff":       r_home - r_away,
            })

            new_home, new_away = elo_update(
                r_home, r_away,
                int(row["home_score"]), int(row["away_score"]),
                row["tournament"],
                is_neutral=bool(row["neutral"]),
                home_advantage=self.home_advantage,
            )

            self.ratings[home] = new_home
            self.ratings[away] = new_away

        if verbose:
            logger.info(f"ELO computed for {len(self.ratings)} teams over {len(self._history):,} matches.")

        return self

    def history_df(self) -> pd.DataFrame:
        """Return the full match-by-match ELO history as a DataFrame."""
        return pd.DataFrame(self._history)

    def top_n(self, n: int = 20) -> pd.Series:
        """Return top N teams by current ELO."""
        return (
            pd.Series(self.ratings)
            .sort_values(ascending=False)
            .head(n)
        )

    def rating(self, team: str) -> float:
        """Get current ELO for a team."""
        return self._get(team)

    def snapshot(self, date: str | pd.Timestamp) -> dict[str, float]:
        """
        Return ELO ratings as they stood just before a given date.
        Useful for building training features without data leakage.
        """
        date = pd.Timestamp(date)
        snap: dict[str, float] = {}
        for row in self._history:
            if row["date"] >= date:
                break
            snap[row["home_team"]] = row["elo_home_pre"]
            snap[row["away_team"]] = row["elo_away_pre"]
        return snap
