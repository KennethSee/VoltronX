import pandas as pd
from sklearn.metrics import roc_auc_score

from DB import DB
from FedModel import FedModel
from GlobModel import GlobModel
from utils.constants import FULL_TRANSACTIONS_FILE, FULL_ACCOUNTS_FILE, TRAIN_TARGET_FILE

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


db = DB(FULL_TRANSACTIONS_FILE, FULL_ACCOUNTS_FILE, TRAIN_TARGET_FILE)
categories = db.retrieve_categories()
encoding_sql = db.encode_data_in_sql(categories)
column_headers = db.retrieve_encoded_transactions(0, 0, encoding_sql).columns

# initialize local models
fed_model_1 = FedModel('large_digital_1', column_headers)
# fed_model_2 = FedModel('large_1', column_headers)
fed_model_3 = FedModel('small_digital_1', column_headers)
# fed_model_4 = FedModel('local_1', column_headers)

# initialize global model
global_model = GlobModel(column_headers)
df_eval = db.retrieve_encoded_transactions(1, 6, encoding_sql)
global_model.set_evaluation_set(df_eval)

# Send trained models to the global model
# global_model.receive_local_model(fed_model_1)
# global_model.receive_local_model(fed_model_2)
# global_model.receive_local_model(fed_model_3)
# global_model.receive_local_model(fed_model_4)

# seed original 3 months data
# new_data_1 = db.retrieve_encoded_bank_transactions(fed_model_1.bank, 1, 6, encoding_sql)
# fed_model_1.add_data(new_data_1)
# new_data_2 = db.retrieve_encoded_bank_transactions(fed_model_2.bank, 1, 6, encoding_sql)
# fed_model_2.add_data(new_data_2)
# new_data_3 = db.retrieve_encoded_bank_transactions(fed_model_3.bank, 1, 6, encoding_sql)
# fed_model_3.add_data(new_data_3)
# new_data_4 = db.retrieve_encoded_bank_transactions(fed_model_4.bank, 1, 6, encoding_sql)
# fed_model_4.add_data(new_data_4)


# add new data
new_data_1 = db.retrieve_encoded_bank_transactions(fed_model_1.bank, fed_model_1.month, fed_model_1.month + 2, encoding_sql)
fed_model_1.add_data(new_data_1)
# new_data_2 = db.retrieve_encoded_bank_transactions(fed_model_2.bank, fed_model_2.month, fed_model_2.month + 2, encoding_sql)
# fed_model_2.add_data(new_data_2)
new_data_3 = db.retrieve_encoded_bank_transactions(fed_model_3.bank, fed_model_3.month, fed_model_3.month + 2, encoding_sql)
fed_model_3.add_data(new_data_3)
# new_data_4 = db.retrieve_encoded_bank_transactions(fed_model_4.bank, fed_model_4.month, fed_model_4.month + 2, encoding_sql)
# fed_model_4.add_data(new_data_4)

# train local models
model_1 = fed_model_1.train_model("IsFraud", [col for col in fed_model_1.get_input_columns()])
# model_2 = fed_model_2.train_model("IsFraud", [col for col in fed_model_2.get_input_columns()])
model_3 = fed_model_3.train_model("IsFraud", [col for col in fed_model_3.get_input_columns()])
# model_4 = fed_model_4.train_model("IsFraud", [col for col in fed_model_4.get_input_columns()])

# Send trained models to the global model
global_model.receive_local_model(fed_model_1)
# global_model.receive_local_model(fed_model_2)
global_model.receive_local_model(fed_model_3)
# global_model.receive_local_model(fed_model_4)

# global model aggregation
global_model.aggregate_models()
final_model = global_model.get_global_model()

# Evaluate models
global_model.calculate_shapley_values()
print(global_model.get_shapley_values())

X_eval, y_eval = global_model.evaluation_set
# Align columns for each fed_model before calc_accuracy
X_eval_1 = align_eval_columns(X_eval, fed_model_1.trained_columns)
# X_eval_2 = align_eval_columns(X_eval, fed_model_2.trained_columns)
X_eval_3 = align_eval_columns(X_eval, fed_model_3.trained_columns)
# X_eval_4 = align_eval_columns(X_eval, fed_model_4.trained_columns)

print(f'Model 1 accuracy: {calc_accuracy(model_1, X_eval, y_eval)}')
# print(f'Model 2 accuracy: {calc_accuracy(model_2, X_eval, y_eval)}')
print(f'Model 3 accuracy: {calc_accuracy(model_3, X_eval, y_eval)}')
# print(f'Model 4 accuracy: {calc_accuracy(model_4, X_eval, y_eval)}')