import pandas as pd

def build_applicant(d, expected_cols):
    ''' 
    d: Input user data dictionary
    expected_cols: user data columns expected by the model
    returns: 
    
    '''
    df = pd.DataFrame([d])
    
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].str.strip()
    
    missing = [c for c in expected_cols if c not in df.columns]
    
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    
    return df[expected_cols]