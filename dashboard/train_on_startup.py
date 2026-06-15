"""
train_on_startup.py — retrain and save models if pkl files are missing or unloadable.
Called automatically by app.py on Streamlit Cloud startup.
This ensures models are always trained with the correct Python/library versions.
"""

import os
import sys
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def models_loadable() -> bool:
    """Check if existing models can actually be loaded."""
    ensemble_path = os.path.join(MODELS_DIR, "ensemble_model.pkl")
    if not os.path.exists(ensemble_path):
        return False
    try:
        joblib.load(ensemble_path)
        return True
    except Exception:
        return False


def train_and_save():
    """Train all models from scratch and save them."""
    from src.elo import EloSystem
    from src.features import build_features, FEATURE_COLS, TARGET_OUTCOME, TARGET_HOME_GLS, TARGET_AWAY_GLS
    from src.model import (
        PoissonMatchModel, build_xgboost, build_random_forest,
        build_logistic, EnsembleModel
    )
    from src.utils import normalize_teams, add_outcome_column

    print("Loading match data...")
    matches_path = os.path.join(DATA_DIR, "matches_modern.csv")
    matches = pd.read_csv(matches_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    matches = matches.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)

    print("Building ELO ratings...")
    elo = EloSystem(default_elo=1500, home_advantage=100)
    elo.fit(matches, verbose=False)

    print("Building feature matrix (this takes a few minutes)...")
    features_path = os.path.join(DATA_DIR, "features.csv")
    if os.path.exists(features_path):
        features = pd.read_csv(features_path, parse_dates=["date"])
        print(f"  Loaded cached features: {features.shape}")
    else:
        features = build_features(matches, elo)
        features.to_csv(features_path, index=False)
        print(f"  Built and saved features: {features.shape}")

    # Train/test split
    SPLIT_DATE = "2018-01-01"
    train = features[features.date < SPLIT_DATE]
    X_train        = train[FEATURE_COLS]
    y_train        = train[TARGET_OUTCOME]
    y_train_home_g = train[TARGET_HOME_GLS]
    y_train_away_g = train[TARGET_AWAY_GLS]

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Training Poisson model...")
    poisson_model = PoissonMatchModel(max_goals=10)
    poisson_model.fit(X_train, y_train_home_g, y_train_away_g)
    joblib.dump(poisson_model, os.path.join(MODELS_DIR, "poisson_model.pkl"))

    print("Training XGBoost...")
    xgb_model = build_xgboost(n_estimators=400, learning_rate=0.05)
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.pkl"))

    print("Training Random Forest...")
    rf_model = build_random_forest(n_estimators=300)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "rf_model.pkl"))

    print("Training Logistic Regression...")
    lr_model = build_logistic()
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "lr_model.pkl"))

    print("Building ensemble...")
    ensemble = EnsembleModel(
        poisson_model=poisson_model,
        xgb_model=xgb_model,
        rf_model=rf_model,
        lr_model=lr_model,
        weights=(0.35, 0.30, 0.20, 0.15),
    )
    joblib.dump(ensemble, os.path.join(MODELS_DIR, "ensemble_model.pkl"))

    print("All models trained and saved ✓")
    return ensemble


def get_or_train_model():
    """Load model if possible, retrain if not."""
    if models_loadable():
        return joblib.load(os.path.join(MODELS_DIR, "ensemble_model.pkl"))
    print("Models not loadable — retraining on this Python version...")
    return train_and_save()
