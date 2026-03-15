"""
app/ml/predictor.py
Predictor class responsible for:
 - Loading all trained model artifacts at startup
 - Encoding raw form inputs using saved LabelEncoders + StandardScaler
 - Running inference with any registered model
 - Computing SHAP values (TreeExplainer) with fallback demo mode
 - Returning structured PredictionResult-compatible dicts
"""

import logging
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.ml.features import (
    FEATURE_ORDER,
    FEATURE_DISPLAY,
    SEVERITY_LABELS,
    ARTIFACTS_DIR,
    MODEL_REGISTRY,
)

logger = logging.getLogger(__name__)

# ── Globals populated by load_artifacts() on startup ──────────────────────────
_models: dict[str, Any] = {}
_scaler = None
_encoders: dict[str, Any] = {}
_shap_explainer = None
_metrics_report: dict = {}
_demo_mode = False


# ──────────────────────────────────────────────────────────────────────────────
def load_artifacts() -> None:
    """
    Load all .pkl artifacts from app/ml/artifacts/ into module-level globals.
    If any artifact is missing, the system falls back to demo mode which returns
    random (but structurally valid) predictions so the UI always renders.
    """
    global _scaler, _encoders, _shap_explainer, _metrics_report, _demo_mode

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []

    # ── Scaler ─────────────────────────────────────────────────────────────────
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    if scaler_path.exists():
        _scaler = joblib.load(scaler_path)
        logger.info("Loaded scaler.pkl")
    else:
        missing.append("scaler.pkl")

    # ── Label encoders ─────────────────────────────────────────────────────────
    encoders_path = ARTIFACTS_DIR / "encoders.pkl"
    if encoders_path.exists():
        _encoders = joblib.load(encoders_path)
        logger.info("Loaded encoders.pkl")
    else:
        missing.append("encoders.pkl")

    # ── Models ─────────────────────────────────────────────────────────────────
    for key, info in MODEL_REGISTRY.items():
        model_path: Path = info["file"]
        if model_path.exists():
            _models[key] = joblib.load(model_path)
            logger.info("Loaded model: %s", key)
        else:
            missing.append(model_path.name)

    # ── SHAP explainer ─────────────────────────────────────────────────────────
    shap_path = ARTIFACTS_DIR / "shap_explainer.pkl"
    if shap_path.exists():
        _shap_explainer = joblib.load(shap_path)
        logger.info("Loaded shap_explainer.pkl")
    else:
        missing.append("shap_explainer.pkl")

    # ── Metrics report ─────────────────────────────────────────────────────────
    import json
    metrics_path = ARTIFACTS_DIR / "metrics_report.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            _metrics_report = json.load(f)
        logger.info("Loaded metrics_report.json")

    if missing:
        logger.warning(
            "Missing artifacts: %s — running in DEMO MODE (random predictions).",
            ", ".join(missing),
        )
        _demo_mode = True
    else:
        _demo_mode = False
        logger.info("All artifacts loaded. Predictor ready.")


# ──────────────────────────────────────────────────────────────────────────────
def _encode_inputs(raw_inputs: dict[str, str]) -> np.ndarray:
    """
    Transform raw string feature dict → scaled numpy array ready for inference.
    Applies LabelEncoders then StandardScaler in FEATURE_ORDER.
    """
    row: list[float] = []
    for feature in FEATURE_ORDER:
        value = raw_inputs.get(feature, "Unknown")
        encoder = _encoders.get(feature)
        if encoder is not None:
            try:
                encoded = encoder.transform([value])[0]
            except ValueError:
                # Unseen label — use 0 (mode fallback)
                encoded = 0
        else:
            # Numeric feature (hour_of_day)
            try:
                encoded = float(value)
            except (ValueError, TypeError):
                encoded = 0.0
        row.append(encoded)

    X = np.array(row, dtype=float).reshape(1, -1)
    if _scaler is not None:
        X = _scaler.transform(X)
    return X


# ──────────────────────────────────────────────────────────────────────────────
def _compute_shap(X: np.ndarray, predicted_class: int) -> dict[str, float]:
    """
    Compute SHAP values for the predicted class.
    Returns a dict of {display_name: shap_value} for the top 10 features by magnitude.
    Falls back to demo random values if explainer is unavailable.
    """
    if _shap_explainer is None:
        # Demo mode: random ±0.25 values
        values = {
            FEATURE_DISPLAY[f]: round(random.uniform(-0.25, 0.25), 4)
            for f in FEATURE_ORDER
        }
        top10 = dict(
            sorted(values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        )
        return top10

    try:
        shap_vals = _shap_explainer.shap_values(X)
        # shap_vals shape: (n_classes, n_samples, n_features)  for TreeExplainer
        if isinstance(shap_vals, list):
            class_shap = shap_vals[predicted_class][0]
        else:
            class_shap = shap_vals[0]

        values = {
            FEATURE_DISPLAY[FEATURE_ORDER[i]]: round(float(class_shap[i]), 4)
            for i in range(len(FEATURE_ORDER))
        }
        top10 = dict(
            sorted(values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        )
        return top10
    except Exception as exc:
        logger.exception("SHAP computation failed: %s", exc)
        return {}


# ──────────────────────────────────────────────────────────────────────────────
def predict(raw_inputs: dict[str, str], model_key: str = "xgb") -> dict:
    """
    Run a full prediction pipeline and return a result dict.

    Args:
        raw_inputs: Dict of {feature_name: string_value} for all 31 features.
        model_key:  Key from MODEL_REGISTRY (e.g. "xgb", "rf", "lgbm").

    Returns:
        Dict with keys: severity_label, severity_code, confidence,
                        probabilities, shap_values, model_key
    """
    if _demo_mode or model_key not in _models:
        return _demo_predict(model_key)

    model = _models[model_key]
    X = _encode_inputs(raw_inputs)

    # ── Probabilities ──────────────────────────────────────────────────────────
    try:
        proba = model.predict_proba(X)[0]  # shape: (3,)
        predicted_class = int(np.argmax(proba))
        confidence = float(np.max(proba))
    except AttributeError:
        # Regression models (ridge, lasso) don't have predict_proba
        raw_pred = float(model.predict(X)[0])
        # Threshold float → class
        if raw_pred < 0.5:
            predicted_class = 0
        elif raw_pred < 1.5:
            predicted_class = 1
        else:
            predicted_class = 2
        proba = np.array([0.0, 0.0, 0.0])
        proba[predicted_class] = 1.0
        confidence = 1.0

    severity_label = SEVERITY_LABELS[predicted_class]
    probabilities = {
        SEVERITY_LABELS[i]: round(float(proba[i]), 4) for i in range(3)
    }

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_values = _compute_shap(X, predicted_class)

    return {
        "severity_label": severity_label,
        "severity_code": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "shap_values": shap_values,
        "model_key": model_key,
    }


# ──────────────────────────────────────────────────────────────────────────────
def _demo_predict(model_key: str = "xgb") -> dict:
    """Return a structurally valid random prediction for demo / dev mode."""
    predicted_class = random.choices([0, 1, 2], weights=[75, 20, 5])[0]
    severity_label = SEVERITY_LABELS[predicted_class]

    raw_proba = [random.uniform(0.01, 0.99) for _ in range(3)]
    total = sum(raw_proba)
    proba = [round(p / total, 4) for p in raw_proba]

    confidence = max(proba)
    probabilities = {SEVERITY_LABELS[i]: proba[i] for i in range(3)}

    shap_values = {
        FEATURE_DISPLAY[f]: round(random.uniform(-0.25, 0.25), 4)
        for f in FEATURE_ORDER
    }
    top10 = dict(
        sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    )

    return {
        "severity_label": severity_label,
        "severity_code": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "shap_values": top10,
        "model_key": model_key,
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_metrics_report() -> dict:
    """Return the loaded metrics_report.json (empty dict if not yet trained)."""
    return _metrics_report


def is_demo_mode() -> bool:
    """Return True if running without real artifacts."""
    return _demo_mode


def get_loaded_models() -> list[str]:
    """Return list of model keys that were successfully loaded."""
    return list(_models.keys())
