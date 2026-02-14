import pandas as pd

def two_stage_predict(clf, reg, applicant_df):
    """
    clf: classifier model
    reg: regressor model
    applicant_df: input dataframe for prediction
    returns: dict with keys 'approved', 'approval_prob', 'reg_pred'
    """
    preds = clf.predict(applicant_df) 
    prob = clf.predict_proba(applicant_df) 
    
    results = []
    i = 0
    approve_idx = 1
    
    approve  = int(preds[i])
    approval_prob = float(prob[i,approve_idx])
    reg_pred = None
    
    if approve == 1:
        # stage 2 prediction only if loan is approved
        applicant_df_reg = applicant_df.copy()
        reg_pred = float(reg.predict(applicant_df_reg)[0])
    
    results.append({
        "approve": approve,
        "approval_prob":approval_prob,
        "reg_pred":reg_pred
    })
    return results