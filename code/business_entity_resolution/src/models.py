"""
models.py - Matching model training and rule-based baseline.

Provides train functions for:
  1. RuleBasedMatcher (no training)
  2. Logistic Regression
  3. Random Forest
  4. XGBoost
  5. LightGBM
"""

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

log = logging.getLogger(__name__)

RANDOM_STATE = 42


class RuleBasedMatcher:
    """
    No-training baseline: scores a pair by its weighted_combined feature (index 26).
    predict_proba mirrors sklearn interface.
    """
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.clip(X[:, 26], 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])


class ScaledLogisticRegression:
    """Logistic regression with internal StandardScaler."""

    def __init__(self, C: float = 1.0):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=C, max_iter=1000, random_state=RANDOM_STATE,
            class_weight='balanced', solver='lbfgs'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ScaledLogisticRegression':
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))


def train_rule_based() -> RuleBasedMatcher:
    """Return rule-based baseline (no training needed)."""
    return RuleBasedMatcher()


def train_logistic(X: np.ndarray, y: np.ndarray) -> ScaledLogisticRegression:
    """Fit a scaled logistic regression."""
    log.info(f"  LogReg: fitting on {X.shape[0]:,} pairs...")
    model = ScaledLogisticRegression(C=1.0)
    model.fit(X, y)
    return model


def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Fit a balanced random forest."""
    log.info(f"  RandomForest: fitting on {X.shape[0]:,} pairs...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=3,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X, y)
    return rf


def train_xgboost(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    """Fit an XGBoost classifier with scale_pos_weight for imbalance."""
    log.info(f"  XGBoost: fitting on {X.shape[0]:,} pairs...")
    scale_pos = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric='logloss',
        random_state=RANDOM_STATE, n_jobs=-1,
        verbosity=0, use_label_encoder=False
    )
    clf.fit(X, y)
    return clf


def train_lightgbm(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    """Fit a LightGBM classifier."""
    log.info(f"  LightGBM: fitting on {X.shape[0]:,} pairs...")
    scale_pos = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE, n_jobs=-1,
        verbosity=-1
    )
    clf.fit(X, y)
    return clf


MODEL_TRAINERS = {
    'Rule-based':   lambda X, y: train_rule_based(),
    'LogReg':       train_logistic,
    'RandomForest': train_random_forest,
    'XGBoost':      train_xgboost,
    'LightGBM':     train_lightgbm,
}
