import sys
from pathlib import Path
import yaml

from rest_framework.decorators import api_view
from rest_framework.response import Response

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from app.loader import load_models
from app.predict import two_stage_predict
from app.utils import build_applicant

with open("config.yaml") as f:
    config = yaml.safe_load(f)

clf, reg = load_models(config)

REQUIRED_FIELDS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

ALLOWED_EDUCATION = {"Graduate", "Not Graduate"}
ALLOWED_SELF_EMPLOYED = {"Yes", "No"}

def validate_applicant(data):
    for field in REQUIRED_FIELDS:
        if field not in data:
            return f"Missing field: {field}"

    if data["education"] not in ALLOWED_EDUCATION:
        return "Invalid education value"

    if data["self_employed"] not in ALLOWED_SELF_EMPLOYED:
        return "Invalid self_employed value"

    if not (300 <= data["cibil_score"] <= 900):
        return "CIBIL score out of range (300 - 900)"

    return None


@api_view(["POST"])
def predict(request):
    error = validate_applicant(request.data)
    if error:
        return Response({"error": error}, status=400)

    applicant = request.data
    expected_cols = list(clf.feature_names_in_)
    df = build_applicant(applicant, expected_cols)
    result = two_stage_predict(clf, reg, df)[0]
    return Response(result)
