import pandas as pd

from DB import DB
from FedModel import FedModel
from GlobModel import GlobModel
from utils.constants import FULL_TRANSACTIONS_FILE, FULL_ACCOUNTS_FILE, TRAIN_TARGET_FILE

db = DB(FULL_TRANSACTIONS_FILE, FULL_ACCOUNTS_FILE, TRAIN_TARGET_FILE)
categories = db.retrieve_categories()
encoding_sql = db.encode_data_in_sql(categories)
column_headers = db.retrieve_encoded_transactions(0, 0, encoding_sql).columns

# initialize local models
fed_model_1 = FedModel('large_digital_1', column_headers)
fed_model_2 = FedModel('large_1', column_headers)

# add new data
new_data_1 = db.retrieve_encoded_bank_transactions(fed_model_1.bank, fed_model_1.month, fed_model_1.month + 2, encoding_sql)
fed_model_1.add_data(new_data_1)
new_data_2 = db.retrieve_encoded_bank_transactions(fed_model_2.bank, fed_model_2.month, fed_model_2.month + 2, encoding_sql)
fed_model_2.add_data(new_data_2)

print(fed_model_1.df['IsFraud'].value_counts())
print(fed_model_2.df['IsFraud'].value_counts())