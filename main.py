from backend.app.loader import load_models
from backend.app.utils import build_applicant
from backend.app.predict import two_stage_predict
import yaml

#loading models

config = yaml.safe_load(open("backend/config.yaml"))

clf, reg = load_models(config)

def run_cli():
    data = {
    'no_of_dependents': 0,
    'education': 'Graduate',
    'self_employed': 'No',
    'income_annum': 50000,
    'loan_term': 360,
    'cibil_score': 550,
    'residential_assets_value': 100000,
    'commercial_assets_value': 50000,
    'luxury_assets_value': 25000,
    'bank_asset_value': 15000,
    'loan_amount': 200000
    }
    df = build_applicant(data, list(clf.feature_names_in_))
    print(two_stage_predict(clf, reg, df))

if __name__=="__main__":
    run_cli()