import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from utils.constants import RANDOM_STATE

def calc_accuracy(model, X_eval, y_eval):
    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    performance = roc_auc_score(y_eval, y_pred_proba)
    return performance

def align_eval_columns(X_eval: pd.DataFrame, trained_columns: list) -> pd.DataFrame:
    """Adds missing columns as 0 and drops extra columns so X_eval matches the trained feature set."""
    X_eval_aligned = X_eval.copy()
    
    # Add missing columns
    for col in trained_columns:
        if col not in X_eval_aligned.columns:
            X_eval_aligned[col] = 0
    
    # Drop columns not present during training
    extra_cols = [col for col in X_eval_aligned.columns if col not in trained_columns]
    X_eval_aligned.drop(columns=extra_cols, inplace=True)
    
    # Match the exact ordering
    X_eval_aligned = X_eval_aligned[trained_columns]
    return X_eval_aligned