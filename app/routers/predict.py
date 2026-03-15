"""
app/routers/predict.py
Handles the 31-field prediction form (GET) and processes submissions (POST).
Stores results to the database and redirects to /result/{id}.
"""

import logging
from fastapi import APIRouter, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.prediction import Prediction
from app.models.user import User
from app.routers.auth import get_current_user
from app.ml import predictor
from app.ml.features import FEATURE_OPTIONS, FEATURE_DISPLAY, MODEL_REGISTRY

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predict"])
templates = Jinja2Templates(directory="app/templates")


@router.get(
    "/predict",
    response_class=HTMLResponse,
    summary="Prediction form",
    description="Renders the 31-field accident feature form with model selector.",
)
async def predict_form(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "user": current_user,
            "feature_options": FEATURE_OPTIONS,
            "feature_display": FEATURE_DISPLAY,
            "model_registry": MODEL_REGISTRY,
            "demo_mode": predictor.is_demo_mode(),
        },
    )


@router.post(
    "/predict",
    summary="Process prediction form",
    description="Runs ML inference on submitted features, stores result, redirects to /result/{id}.",
)
async def predict_submit(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    # ── All 31 features as form fields ──────────────────────────────────────
    day_of_week: str = Form(...),
    age_band_of_driver: str = Form(...),
    sex_of_driver: str = Form(...),
    educational_level: str = Form(...),
    vehicle_driver_relation: str = Form(...),
    driving_experience: str = Form(...),
    type_of_vehicle: str = Form(...),
    owner_of_vehicle: str = Form(...),
    service_year_of_vehicle: str = Form(...),
    defect_of_vehicle: str = Form(...),
    area_accident_occured: str = Form(...),
    lanes_or_medians: str = Form(...),
    road_allignment: str = Form(...),
    types_of_junction: str = Form(...),
    road_surface_type: str = Form(...),
    road_surface_conditions: str = Form(...),
    light_conditions: str = Form(...),
    weather_conditions: str = Form(...),
    type_of_collision: str = Form(...),
    number_of_vehicles_involved: str = Form(...),
    number_of_casualties: str = Form(...),
    vehicle_movement: str = Form(...),
    casualty_class: str = Form(...),
    sex_of_casualty: str = Form(...),
    age_band_of_casualty: str = Form(...),
    casualty_severity: str = Form(...),
    work_of_casuality: str = Form(...),
    fitness_of_casuality: str = Form(...),
    pedestrian_movement: str = Form(...),
    cause_of_accident: str = Form(...),
    hour_of_day: str = Form(...),
    model_key: str = Form(default="xgb"),
):
    raw_inputs = {
        "day_of_week": day_of_week,
        "age_band_of_driver": age_band_of_driver,
        "sex_of_driver": sex_of_driver,
        "educational_level": educational_level,
        "vehicle_driver_relation": vehicle_driver_relation,
        "driving_experience": driving_experience,
        "type_of_vehicle": type_of_vehicle,
        "owner_of_vehicle": owner_of_vehicle,
        "service_year_of_vehicle": service_year_of_vehicle,
        "defect_of_vehicle": defect_of_vehicle,
        "area_accident_occured": area_accident_occured,
        "lanes_or_medians": lanes_or_medians,
        "road_allignment": road_allignment,
        "types_of_junction": types_of_junction,
        "road_surface_type": road_surface_type,
        "road_surface_conditions": road_surface_conditions,
        "light_conditions": light_conditions,
        "weather_conditions": weather_conditions,
        "type_of_collision": type_of_collision,
        "number_of_vehicles_involved": number_of_vehicles_involved,
        "number_of_casualties": number_of_casualties,
        "vehicle_movement": vehicle_movement,
        "casualty_class": casualty_class,
        "sex_of_casualty": sex_of_casualty,
        "age_band_of_casualty": age_band_of_casualty,
        "casualty_severity": casualty_severity,
        "work_of_casuality": work_of_casuality,
        "fitness_of_casuality": fitness_of_casuality,
        "pedestrian_movement": pedestrian_movement,
        "cause_of_accident": cause_of_accident,
        "hour_of_day": hour_of_day,
    }

    # Validate model_key
    if model_key not in MODEL_REGISTRY:
        model_key = "xgb"

    # Run prediction
    result = predictor.predict(raw_inputs, model_key=model_key)

    # Persist to database
    pred = Prediction(
        user_id=current_user.id,
        severity_label=result["severity_label"],
        severity_code=result["severity_code"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        shap_values=result["shap_values"],
        inputs=raw_inputs,
        model_key=model_key,
        cause_of_accident=cause_of_accident,
        weather_conditions=weather_conditions,
    )
    db.add(pred)
    await db.commit()
    await db.refresh(pred)

    logger.info(
        "Prediction #%d: %s (%.1f%%) by user %s using model %s",
        pred.id,
        result["severity_label"],
        result["confidence"] * 100,
        current_user.email,
        model_key,
    )

    return RedirectResponse(
        url=f"/result/{pred.id}", status_code=status.HTTP_303_SEE_OTHER
    )
