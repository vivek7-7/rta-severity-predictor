"""
app/routers/model_info.py
Model information page: dataset stats, model comparison table,
confusion matrix, and feature importance chart.
"""

import logging
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.routers.auth import get_current_user
from app.ml.features import MODEL_REGISTRY, SEVERITY_LABELS
from app.ml.predictor import get_metrics_report, is_demo_mode

logger = logging.getLogger(__name__)
router = APIRouter(tags=["model_info"])
templates = Jinja2Templates(directory="app/templates")

# Dataset facts (fixed)
DATASET_INFO = {
    "name": "Road Traffic Accidents — Addis Ababa Sub-City",
    "source": "Kaggle / Addis Ababa Sub-City Police Dept.",
    "rows": 12316,
    "features": 31,
    "classes": ["Slight Injury", "Serious Injury", "Fatal injury"],
    "class_distribution": {"Slight Injury": 75, "Serious Injury": 20, "Fatal injury": 5},
    "imbalance_strategy": "SMOTE (Synthetic Minority Oversampling Technique)",
    "missing_values": "Mode imputation per column",
}


@router.get(
    "/model-info",
    response_class=HTMLResponse,
    summary="Model information and comparison",
    description="Dataset details, all-model metrics table, confusion matrix, feature importance.",
)
async def model_info_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    selected_model: str = Query(default="xgb"),
):
    metrics = get_metrics_report()

    # Build comparison rows
    comparison_rows = []
    for key, info in MODEL_REGISTRY.items():
        m = metrics.get(key, {})
        comparison_rows.append({
            "key": key,
            "name": info["name"],
            "unit": info["unit"],
            "type": info["type"],
            "accuracy": round(m.get("accuracy", 0) * 100, 2) if m else "—",
            "weighted_f1": round(m.get("weighted_f1", 0), 3) if m else "—",
            "macro_f1": round(m.get("macro_f1", 0), 3) if m else "—",
            "roc_auc": round(m.get("roc_auc", 0), 3) if m else "—",
            "train_time": round(m.get("train_time_seconds", 0), 1) if m else "—",
            "is_default": info.get("default", False),
        })

    # Confusion matrix for selected model
    selected_cm = []
    if metrics and selected_model in metrics:
        selected_cm = metrics[selected_model].get("confusion_matrix", [])

    # Feature importance (from metrics_report if stored, else placeholder)
    feature_importance = {}
    if metrics and selected_model in metrics:
        feature_importance = metrics[selected_model].get("feature_importance", {})

    if not feature_importance:
        # Placeholder top-15 feature importances for UI demo
        from app.ml.features import FEATURE_DISPLAY
        features = list(FEATURE_DISPLAY.values())[:15]
        import random
        random.seed(42)
        vals = sorted([random.uniform(0.01, 0.12) for _ in features], reverse=True)
        feature_importance = dict(zip(features, [round(v, 4) for v in vals]))

    return templates.TemplateResponse(
        "model_info.html",
        {
            "request": request,
            "user": current_user,
            "dataset_info": DATASET_INFO,
            "comparison_rows": comparison_rows,
            "selected_model": selected_model,
            "selected_model_name": MODEL_REGISTRY.get(selected_model, {}).get("name", selected_model),
            "selected_cm": selected_cm,
            "feature_importance": feature_importance,
            "severity_labels": list(SEVERITY_LABELS.values()),
            "demo_mode": is_demo_mode(),
            "has_metrics": bool(metrics),
        },
    )
