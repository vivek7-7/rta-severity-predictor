"""
app/ml/features.py
Single source of truth for:
 - FEATURE_ORDER: exact column order expected by the trained models (31 features)
 - FEATURE_OPTIONS: valid dropdown values for every categorical feature
 - FEATURE_DISPLAY: human-readable labels for each feature (used in result/SHAP UI)
 - MODEL_REGISTRY: all supported model keys with display names and file paths
"""

from pathlib import Path

# ── Feature order must match the training script encoding order ────────────────
FEATURE_ORDER = [
    "day_of_week",
    "age_band_of_driver",
    "sex_of_driver",
    "educational_level",
    "vehicle_driver_relation",
    "driving_experience",
    "type_of_vehicle",
    "owner_of_vehicle",
    "service_year_of_vehicle",
    "defect_of_vehicle",
    "area_accident_occured",
    "lanes_or_medians",
    "road_allignment",
    "types_of_junction",
    "road_surface_type",
    "road_surface_conditions",
    "light_conditions",
    "weather_conditions",
    "type_of_collision",
    "number_of_vehicles_involved",
    "number_of_casualties",
    "vehicle_movement",
    "casualty_class",
    "sex_of_casualty",
    "age_band_of_casualty",
    "casualty_severity",
    "work_of_casuality",
    "fitness_of_casuality",
    "pedestrian_movement",
    "cause_of_accident",
    "hour_of_day",
]

# ── Human-readable display labels ─────────────────────────────────────────────
FEATURE_DISPLAY = {
    "day_of_week": "Day of Week",
    "age_band_of_driver": "Driver Age Band",
    "sex_of_driver": "Driver Sex",
    "educational_level": "Driver Education Level",
    "vehicle_driver_relation": "Driver-Vehicle Relation",
    "driving_experience": "Driving Experience",
    "type_of_vehicle": "Vehicle Type",
    "owner_of_vehicle": "Vehicle Owner",
    "service_year_of_vehicle": "Vehicle Service Year",
    "defect_of_vehicle": "Vehicle Defect",
    "area_accident_occured": "Accident Area",
    "lanes_or_medians": "Lanes / Medians",
    "road_allignment": "Road Alignment",
    "types_of_junction": "Junction Type",
    "road_surface_type": "Road Surface Type",
    "road_surface_conditions": "Road Surface Conditions",
    "light_conditions": "Light Conditions",
    "weather_conditions": "Weather Conditions",
    "type_of_collision": "Collision Type",
    "number_of_vehicles_involved": "Vehicles Involved",
    "number_of_casualties": "Number of Casualties",
    "vehicle_movement": "Vehicle Movement",
    "casualty_class": "Casualty Class",
    "sex_of_casualty": "Casualty Sex",
    "age_band_of_casualty": "Casualty Age Band",
    "casualty_severity": "Casualty Severity",
    "work_of_casuality": "Casualty Occupation",
    "fitness_of_casuality": "Casualty Fitness",
    "pedestrian_movement": "Pedestrian Movement",
    "cause_of_accident": "Cause of Accident",
    "hour_of_day": "Hour of Day",
}

# ── Valid option lists for every dropdown ─────────────────────────────────────
FEATURE_OPTIONS: dict[str, list[str]] = {
    "day_of_week": [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ],
    "age_band_of_driver": [
        "Under 18", "18-30", "31-50", "Over 51", "Unknown"
    ],
    "sex_of_driver": [
        "Male", "Female", "Unknown"
    ],
    "educational_level": [
        "Above high school", "High school", "Junior high school",
        "Elementary school", "Illiterate", "Writing & reading", "Unknown"
    ],
    "vehicle_driver_relation": [
        "Employee", "Owner", "Other"
    ],
    "driving_experience": [
        "No Licence", "Below 1yr", "1-2yr", "2-5yr", "5-10yr", "Above 10yr", "Unknown"
    ],
    "type_of_vehicle": [
        "Automobile", "Lorry (11-40Q)", "Public (12 seats)", "Public (13-45 seats)",
        "Public (> 45 seats)", "Long lorry", "Taxi", "Pick up upto 10Q",
        "Stationwagen", "Ridden horse", "Motorcycle", "Special vehicle",
        "Bajaj", "Turbo", "Bicycle", "Other"
    ],
    "owner_of_vehicle": [
        "Owner", "Governmental", "Organization", "Other"
    ],
    "service_year_of_vehicle": [
        "Below 1yr", "1-2yr", "2-5yr", "5-10yr", "Above 10yr", "Unknown"
    ],
    "defect_of_vehicle": [
        "No defect", "Tyre burst", "No brakes", "Brake issue", "Other"
    ],
    "area_accident_occured": [
        "Residential areas", "Office areas", "Church areas", "Industrial areas",
        "School areas", "Recreational areas", "Outside Addis Abeba",
        "Hospital areas", "Market areas", "Rural areas", "Unknown"
    ],
    "lanes_or_medians": [
        "Undivided Two way", "Double carriageway (median)", "One way",
        "Two-way (divided with solid lines road marking)",
        "Two-way (divided with broken lines road marking)", "Unknown", "Other"
    ],
    "road_allignment": [
        "Tangent road with flat terrain", "Tangent road with mild grade and flat terrain",
        "Escarpment", "Tangent road with rolling terrain", "Gentle horizontal curve",
        "Sharp reverse curve", "Steep grade downward with mountainous terrain",
        "Unknown"
    ],
    "types_of_junction": [
        "No junction", "Y Shape", "Crossing", "O Shape", "Other", "Unknown",
        "T Shape", "X Shape"
    ],
    "road_surface_type": [
        "Asphalt roads", "Earth roads", "Asphalt roads with some patches",
        "Gravel roads", "Other"
    ],
    "road_surface_conditions": [
        "Dry", "Wet or damp", "Snow", "Flood over 3cm. deep"
    ],
    "light_conditions": [
        "Daylight", "Darkness - lights lit", "Darkness - no lighting",
        "Darkness - lights unlit"
    ],
    "weather_conditions": [
        "Normal", "Raining", "Raining and Windy", "Cloudy", "Windy",
        "Snow", "Fog or mist", "Other"
    ],
    "type_of_collision": [
        "Vehicle with vehicle collision", "Collision with roadside-parked vehicles",
        "Collision with roadside objects", "Collision with animals",
        "Rollover", "Fall from vehicles", "Collision with pedestrians",
        "With Train", "Other"
    ],
    "number_of_vehicles_involved": [
        "1", "2", "3", "4", "5", "6", "7"
    ],
    "number_of_casualties": [
        "1", "2", "3", "4", "5", "6", "7", "8"
    ],
    "vehicle_movement": [
        "Going straight", "U-Turn", "Moving Backward", "Turnover",
        "Going round", "Reversing", "Parked", "Stopping", "Overtaking",
        "Waiting to go", "Getting off", "Unknown", "Other"
    ],
    "casualty_class": [
        "Pedestrian", "Driver or rider", "Passenger"
    ],
    "sex_of_casualty": [
        "Male", "Female", "Unknown"
    ],
    "age_band_of_casualty": [
        "Under 18", "18-30", "31-50", "Over 51", "Unknown"
    ],
    "casualty_severity": [
        "3", "2", "1"
    ],
    "work_of_casuality": [
        "Driver", "Self-employed", "Unemployed", "Employee", "Student",
        "Government emp.", "Private emp.", "Retired", "Other", "Unknown"
    ],
    "fitness_of_casuality": [
        "Normal", "Deaf", "Blind", "Under the influence of drugs", "Other"
    ],
    "pedestrian_movement": [
        "Not a Pedestrian", "Crossing from driver's nearside",
        "Crossing from nearside - masked by parked or stationary vehicle",
        "In carriageway, stationary - not crossing",
        "Walking along in carriageway, back to traffic",
        "Walking along in carriageway, facing traffic",
        "In carriageway, stationary - masked by parked or stationary vehicle",
        "Unknown or other"
    ],
    "cause_of_accident": [
        "Moving Backward", "Overtaking", "Changing lane to the left",
        "Changing lane to the right", "Overloading", "Other",
        "No priority to vehicle", "No priority to pedestrian",
        "No distancing", "Getting off the vehicle",
        "Improper parking", "Overspeed", "Driving carelessly",
        "Drunk driving", "Changing lane to left",
        "Turnover", "Brakes failure", "Unknown"
    ],
    "hour_of_day": [str(h) for h in range(0, 24)],
}

# ── Severity label mapping ─────────────────────────────────────────────────────
SEVERITY_LABELS = {
    0: "Slight Injury",
    1: "Serious Injury",
    2: "Fatal injury",
}

SEVERITY_COLORS = {
    "Slight Injury": "green",
    "Serious Injury": "amber",
    "Fatal injury": "red",
}

# ── Model registry ─────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

MODEL_REGISTRY = {
    "xgb": {
        "name": "XGBoost",
        "file": ARTIFACTS_DIR / "model_xgb.pkl",
        "unit": "III",
        "type": "Ensemble",
        "default": True,
    },
    "rf": {
        "name": "Random Forest",
        "file": ARTIFACTS_DIR / "model_rf.pkl",
        "unit": "III",
        "type": "Ensemble",
        "default": False,
    },
    "lgbm": {
        "name": "LightGBM",
        "file": ARTIFACTS_DIR / "model_lgbm.pkl",
        "unit": "III",
        "type": "Ensemble",
        "default": False,
    },
    "gb": {
        "name": "Gradient Boosting",
        "file": ARTIFACTS_DIR / "model_gb.pkl",
        "unit": "III",
        "type": "Ensemble",
        "default": False,
    },
    "svm": {
        "name": "Support Vector Machine",
        "file": ARTIFACTS_DIR / "model_svm.pkl",
        "unit": "III",
        "type": "Kernel",
        "default": False,
    },
    "lr": {
        "name": "Logistic Regression",
        "file": ARTIFACTS_DIR / "model_lr.pkl",
        "unit": "III",
        "type": "Linear",
        "default": False,
    },
    "dt": {
        "name": "Decision Tree",
        "file": ARTIFACTS_DIR / "model_dt.pkl",
        "unit": "III",
        "type": "Tree",
        "default": False,
    },
    "knn": {
        "name": "k-Nearest Neighbors",
        "file": ARTIFACTS_DIR / "model_knn.pkl",
        "unit": "III",
        "type": "Instance-based",
        "default": False,
    },
    "nb": {
        "name": "Naïve Bayes",
        "file": ARTIFACTS_DIR / "model_nb.pkl",
        "unit": "III",
        "type": "Probabilistic",
        "default": False,
    },
    "mlp": {
        "name": "MLP Neural Network",
        "file": ARTIFACTS_DIR / "model_mlp.pkl",
        "unit": "IV",
        "type": "Neural Network",
        "default": False,
    },
    "ridge": {
        "name": "Ridge Regression",
        "file": ARTIFACTS_DIR / "model_ridge.pkl",
        "unit": "II",
        "type": "Regression",
        "default": False,
    },
    "lasso": {
        "name": "Lasso Regression",
        "file": ARTIFACTS_DIR / "model_lasso.pkl",
        "unit": "II",
        "type": "Regression",
        "default": False,
    },
}
