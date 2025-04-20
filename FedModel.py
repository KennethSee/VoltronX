import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from imblearn.over_sampling import SMOTE
from utils.constants import RANDOM_STATE

class FedModel:
    def __init__(self, bank: str, column_headers: list):
        self.df = pd.DataFrame(columns=column_headers)
        self.bank = bank
        self.month = 7
        self.local_model = None
        self.model_accuracy = None  # Store model AUC
        self.model_log_loss = None  # Store log loss
        self.last_loaded_period = None

    def next_period(self):
        self.month += 1

    def add_data(self, new_data: pd.DataFrame):
        """Add new transaction data."""
        self.df = pd.concat([self.df, new_data], ignore_index=True)

    def get_input_columns(self) -> list:
        """Get feature columns excluding non-informative columns."""
        columns_to_drop = [
            "transaction_id", "account_id", "month", "day", "weekday", "hour", "min", "sec",
            "transaction_direction", "counterpart_id", '__index_level_0__', 'IsFraud'
        ]
        return [col for col in self.df.columns if col not in columns_to_drop]

    def align_features(self, reference_columns: list):
        """Ensures the feature space is consistent across all banks."""
        # Remove unexpected columns
        self.df = self.df[[col for col in self.df.columns if col in reference_columns]]
        # Add missing columns
        for col in reference_columns:
            if col not in self.df.columns:
                self.df[col] = 0
        # Ensure the column order is correct
        self.df = self.df[reference_columns]

    def train_model(self, target_column: str, input_columns: list):
        """Train a local LightGBM fraud detection model using reduced hyperparameter tuning."""
        # Align features
        self.align_features(input_columns + [target_column])
        print(f'Training model for {self.bank} with input columns:')
        print(input_columns)

        X = self.df[input_columns]
        y = self.df[target_column].astype(int)

        # One-hot encode categorical features
        X = pd.get_dummies(X, drop_first=True)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Store the exact columns used for training
        self.trained_columns = X.columns.tolist()

        # Train-test split (stratify for class balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=int(RANDOM_STATE)
        )

        # Handle class imbalance (SMOTE)
        smote = SMOTE(sampling_strategy=0.1, random_state=int(RANDOM_STATE))
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Reduced hyperparameter grid
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'n_estimators': [50, 100, 200],
            'num_leaves': [4, 8, 16],
            'max_depth': [3, 5, 7],
            'min_data_in_leaf': [50, 100, 200],
            'min_gain_to_split': [0.0, 0.1],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 1, 10],
            'bagging_fraction': [0.6, 0.8, 1.0],
            'feature_fraction': [0.6, 0.8, 1.0],
            'bagging_freq': [0, 1, 5],
        }

        # StratifiedKFold for validation
        cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(RANDOM_STATE))

        # Base LightGBM model
        model = lgb.LGBMClassifier(is_unbalance=True, random_state=int(RANDOM_STATE))

        # Reduced RandomizedSearchCV: fewer iterations for quick testing
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring='roc_auc',
            n_iter=3,  # Fewer iterations
            cv=cv_folds,
            verbose=1,
            random_state=int(RANDOM_STATE),
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        # Retrieve the best estimator
        best_model = grid_search.best_estimator_
        self.local_model = best_model

        # Evaluate performance
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        self.model_accuracy = roc_auc_score(y_test, y_pred_proba)
        self.model_log_loss = log_loss(y_test, y_pred_proba)

        print(f"Bank: {self.bank}, Best Model AUC: {self.model_accuracy:.4f}, Log Loss: {self.model_log_loss:.4f}")
        return best_model

    def get_model_weights(self):
        """Extract and return model weights."""
        return self.local_model.booster_.dump_model()

    def get_model_performance(self):
        """Return computed model accuracy & log loss."""
        return self.model_accuracy, self.model_log_loss
