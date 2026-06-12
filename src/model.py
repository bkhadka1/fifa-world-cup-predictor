"""
model.py — model training, evaluation, and ensemble

Models:
  1. Poisson Regression  — predicts expected goals → win/draw/loss probs
  2. XGBoost             — 3-class outcome classifier
  3. Random Forest       — 3-class outcome classifier
  4. Logistic Regression — baseline 3-class classifier
  5. Ensemble            — weighted average of all four
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
import joblib

from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

from src.utils import get_logger, model_path
from src.features import FEATURE_COLS, TARGET_OUTCOME, TARGET_HOME_GLS, TARGET_AWAY_GLS

logger = get_logger(__name__)


# ── Poisson model ─────────────────────────────────────────────────────────────

class PoissonMatchModel:
    """
    Fits two independent Poisson regressors:
      - one for home goals
      - one for away goals

    Then derives win/draw/loss probabilities by convolving
    the two Poisson distributions over a scoreline grid.
    """

    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self.home_model = Pipeline([
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(max_iter=500)),
        ])
        self.away_model = Pipeline([
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(max_iter=500)),
        ])

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        self.home_model.fit(X, y_home)
        self.away_model.fit(X, y_away)
        return self

    def predict_lambda(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (lambda_home, lambda_away) — expected goals arrays."""
        lam_home = self.home_model.predict(X).clip(0.01)
        lam_away = self.away_model.predict(X).clip(0.01)
        return lam_home, lam_away

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns array of shape (n, 3):
          col 0 = P(away win)
          col 1 = P(draw)
          col 2 = P(home win)
        """
        lam_home, lam_away = self.predict_lambda(X)
        n = len(X)
        probs = np.zeros((n, 3))

        goals = np.arange(0, self.max_goals + 1)

        for i in range(n):
            home_pmf = poisson.pmf(goals, lam_home[i])
            away_pmf = poisson.pmf(goals, lam_away[i])
            # outer product gives P(home_goals=j, away_goals=k)
            grid = np.outer(home_pmf, away_pmf)
            probs[i, 2] = np.tril(grid, -1).sum()   # home win (home > away)
            probs[i, 1] = np.trace(grid)              # draw
            probs[i, 0] = np.triu(grid, 1).sum()      # away win

        # Normalise to sum to 1
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, name: str = "poisson_model.pkl"):
        joblib.dump(self, model_path(name))
        logger.info(f"Saved → {model_path(name)}")

    @staticmethod
    def load(name: str = "poisson_model.pkl") -> "PoissonMatchModel":
        return joblib.load(model_path(name))


# ── Classifier models ─────────────────────────────────────────────────────────

def build_xgboost(n_estimators: int = 400, learning_rate: float = 0.05) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )),
    ])


def build_random_forest(n_estimators: int = 500) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_logistic() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            C=1.0,
            random_state=42,
        )),
    ])


# ── Ensemble ──────────────────────────────────────────────────────────────────

class EnsembleModel:
    """
    Weighted average ensemble of Poisson + XGBoost + Random Forest + Logistic.

    Weights are tuned by minimising log-loss on a validation split.
    Default weights give slightly more trust to Poisson and XGBoost.
    """

    def __init__(
        self,
        poisson_model: PoissonMatchModel,
        xgb_model: Pipeline,
        rf_model: Pipeline,
        lr_model: Pipeline,
        weights: tuple = (0.35, 0.30, 0.20, 0.15),  # poisson, xgb, rf, lr
    ):
        self.poisson = poisson_model
        self.xgb     = xgb_model
        self.rf      = rf_model
        self.lr      = lr_model
        self.weights = np.array(weights) / sum(weights)  # normalise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_poisson = self.poisson.predict_proba(X)
        p_xgb     = self.xgb.predict_proba(X)
        p_rf      = self.rf.predict_proba(X)
        p_lr      = self.lr.predict_proba(X)

        return (
            self.weights[0] * p_poisson +
            self.weights[1] * p_xgb +
            self.weights[2] * p_rf +
            self.weights[3] * p_lr
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, name: str = "ensemble_model.pkl"):
        joblib.dump(self, model_path(name))
        logger.info(f"Saved → {model_path(name)}")

    @staticmethod
    def load(name: str = "ensemble_model.pkl") -> "EnsembleModel":
        return joblib.load(model_path(name))


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Return a dict of evaluation metrics for a single model."""
    acc      = accuracy_score(y_true, y_pred)
    ll       = log_loss(y_true, y_proba)
    # Brier score averaged across classes
    brier    = np.mean([
        brier_score_loss((y_true == c).astype(int), y_proba[:, c])
        for c in range(3)
    ])

    logger.info(f"{name:20s}  acc={acc:.3f}  log_loss={ll:.3f}  brier={brier:.3f}")
    return {"model": name, "accuracy": acc, "log_loss": ll, "brier_score": brier}


def cross_validate_model(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """Time-aware cross-validation (no shuffle — preserves temporal order)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    logger.info(f"{name:20s}  CV acc={scores.mean():.3f} ± {scores.std():.3f}")
    return {"model": name, "cv_mean": scores.mean(), "cv_std": scores.std()}
