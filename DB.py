import duckdb
import pandas as pd

class DB:

    def __init__(self, full_transactions_file, full_accounts_file, train_target_file):
        self.full_transactions_file = full_transactions_file
        self.full_accounts_file = full_accounts_file
        self.train_target_file = train_target_file

    def retrieve_categories(self, categorical_variables = ['channel', 'payment_system', 'category_0', 'category_1', 'category_2']) -> dict:
        """extract all the various categories within each categorical feature"""
        category_dict = {}
        for variable in categorical_variables:
            cat_items = duckdb.sql(f"""
                            SELECT DISTINCT {variable}
                            FROM  '{self.full_transactions_file}' txn;
                    """).df()[variable].tolist()
            category_dict[variable] = cat_items
        
        return category_dict
    
    def retrieve_banks(self) -> list:
        banks = duckdb.sql(f"""
            SELECT DISTINCT assigned_bank
            FROM '{self.full_transactions_file}' txn
                JOIN '{self.full_accounts_file}' acc ON txn.account_id = acc.account_id AND txn.transaction_direction = 'inbound'
        """).df()['assigned_bank'].tolist()
        return banks
    
    def retrieve_encoded_bank_transactions(self, bank: str, start_month: int, end_month: int, encoding_sql: str) -> pd.DataFrame:
        df = duckdb.sql(f"""
            SELECT txn.*, 
                CASE WHEN txn.transaction_id IN (
                    SELECT transaction_id FROM '{self.train_target_file}'
                ) THEN 1 ELSE 0 END AS IsFraud
            FROM  '{self.full_transactions_file}' txn
                JOIN '{self.full_accounts_file}' acc 
                ON txn.account_id = acc.account_id 
                AND txn.transaction_direction = 'inbound'
            WHERE acc.assigned_bank = '{bank}'
                AND txn.month >= {start_month}
                AND txn.month <= {end_month}
            """).df()
        return df
    
    def retrieve_encoded_transactions(self, start_month: int, end_month: int, encoding_sql: str) -> pd.DataFrame:
        df = duckdb.sql(f"""
            SELECT txn.*, 
                CASE WHEN txn.transaction_id IN (
                    SELECT transaction_id FROM '{self.train_target_file}'
                ) THEN 1 ELSE 0 END AS IsFraud
            FROM  '{self.full_transactions_file}' txn
                JOIN '{self.full_accounts_file}' acc 
                ON txn.account_id = acc.account_id 
                AND txn.transaction_direction = 'inbound'
            WHERE txn.month >= {start_month}
                AND txn.month <= {end_month}
            """).df()
        return df
    
    @staticmethod
    def encode_data_in_sql(categories: dict) -> str:
        """
        Generates SQL for one-hot encoding categorical columns directly in the query.
        """
        encoding_sql = []
        
        for col, cats in categories.items():
            for cat in cats:
                # Ensure the alias is wrapped in double quotes
                encoded_col = f"CASE WHEN {col} = '{cat}' THEN 1 ELSE 0 END AS \"{col}_{cat}\""
                encoding_sql.append(encoded_col)
        
        return ", ".join(encoding_sql)