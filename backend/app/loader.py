import os
import joblib

def load_models(config):
    
    clf_path = config['models']['classifier']
    reg_path = config['models']['regressor']
    
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"classifier not found at path - {clf_path}")
    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"regressor not found at path - {clf_path}")
    
    cls = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    
    return cls, reg
