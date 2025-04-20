import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from itertools import combinations

from utils.constants import RANDOM_STATE
from utils.evaluation import calc_accuracy

class GlobModel:
    def __init__(self, column_headers: list):
        self.global_model = None
        self.client_models = {}  # Dictionary: bank -> local model
        self.client_accuracies = {}  # Dictionary: bank -> local model accuracy
        self.evaluation_set = None  # (X_eval, y_eval)
        self.shapley_values = {}  # Dictionary: bank -> Shapley value
        self.column_headers = column_headers  # Feature space for evaluation
        self.fedmodel_param_keys = [
            "learning_rate", "n_estimators", "num_leaves", "max_depth",
            "min_data_in_leaf", "min_gain_to_split", "reg_alpha", "reg_lambda", 
            "bagging_fraction", "feature_fraction", "bagging_freq"
        ]
        # We now create fixed hold-out splits as soon as evaluation_set is set.
        self.global_model_holdout_train = None
        self.global_model_holdout_test = None

    def set_evaluation_set(self, eval_df):
        """Process the evaluation data and create fixed hold-out splits."""
        target_column = "IsFraud"
        input_columns = [col for col in self.column_headers if col not in [
            "transaction_id", "account_id", "month", "day", "weekday", "hour", "min", "sec",
            "transaction_direction", "counterpart_id", '__index_level_0__', 'IsFraud'
        ]]
        eval_df = eval_df.fillna(0)
        X_eval, y_eval = eval_df[input_columns], eval_df[target_column].astype(int)
        X_eval = pd.get_dummies(X_eval, drop_first=True)
        X_eval = X_eval.apply(pd.to_numeric, errors='coerce').fillna(0)
        self.evaluation_set = (X_eval, y_eval)
        print("Evaluation set stored successfully.")
        
        # Create fixed hold-out splits (50/50 split)
        X_train_fixed, X_hold_fixed, y_train_fixed, y_hold_fixed = train_test_split(
            X_eval, y_eval, test_size=0.5, random_state=int(RANDOM_STATE)
        )
        self.global_model_holdout_train = (X_train_fixed, y_train_fixed)
        self.global_model_holdout_test = (X_hold_fixed, y_hold_fixed)
        print("Fixed hold-out splits created.")

    def receive_local_model(self, fed_model):
        """Receive a trained local model and update its accuracy."""
        if fed_model.local_model is not None:
            self.client_models[fed_model.bank] = fed_model.local_model
            accuracy, _ = fed_model.get_model_performance()
            self.client_accuracies[fed_model.bank] = accuracy

    def aggregate_models(self):
        """Aggregate local models by averaging hyperparameters and fit the global model on fixed hold-out train split."""
        if not self.client_models:
            print("No models received.")
            return None

        models = list(self.client_models.values())
        accuracies = np.array(list(self.client_accuracies.values()), dtype=np.float64)

        if np.isnan(accuracies).any() or accuracies.sum() == 0:
            print("Warning: Invalid model accuracies. Using equal weighting instead.")
            accuracies = np.ones(len(models)) / len(models)
        else:
            accuracies /= accuracies.sum()

        model_params = [model.get_params() for model in models]
        numeric_keys = [key for key in model_params[0].keys() if key in self.fedmodel_param_keys]
        integer_params = ["num_leaves", "n_estimators", "max_depth", "min_data_in_leaf", "bagging_freq"]
        aggregated_params = {}
        for key in numeric_keys:
            value = np.average([params[key] for params in model_params],
                               weights=accuracies.astype(np.float64))
            if key in integer_params:
                aggregated_params[key] = int(round(value))
            else:
                aggregated_params[key] = value
        aggregated_params["random_state"] = int(RANDOM_STATE)

        print("Aggregated hyperparameters:", aggregated_params)

        self.global_model = lgb.LGBMClassifier(is_unbalance=True)
        self.global_model.set_params(**aggregated_params)
        
        # Fit the global model on the fixed training portion of the evaluation set
        X_train_fixed, y_train_fixed = self.global_model_holdout_train
        self.global_model.fit(X_train_fixed, y_train_fixed)
        
        print("Global model aggregated and fitted successfully.")

    def _evaluate_model(self, subset):
        """
        Evaluate the performance of a subset of client models by:
         - Aggregating hyperparameters (weighted average) for the subset.
         - Creating a temporary aggregated model.
         - Evaluating the model using 5-fold cross-validation on the entire evaluation set.
           (Cross-validation internally fits the model multiple times.)
        """
        if not subset:
            return 0

        client_keys = list(self.client_models.keys())
        subset_models = [self.client_models[client_keys[i]] for i in subset]
        subset_accuracies = [self.client_accuracies[client_keys[i]] for i in subset]
        accuracy_weights = np.array(subset_accuracies, dtype=np.float64)
        if np.isnan(accuracy_weights).any() or accuracy_weights.sum() == 0:
            accuracy_weights = np.ones(len(subset_models)) / len(subset_models)
        else:
            accuracy_weights /= accuracy_weights.sum()

        model_params = [model.get_params() for model in subset_models]
        numeric_keys = [key for key in model_params[0].keys() if key in self.fedmodel_param_keys]
        integer_params = ["num_leaves", "n_estimators", "max_depth", "min_data_in_leaf", "bagging_freq"]

        aggregated_params = {}
        for key in numeric_keys:
            value = np.average([params[key] for params in model_params],
                               weights=accuracy_weights.astype(np.float64))
            if key in integer_params:
                aggregated_params[key] = int(round(value))
            else:
                aggregated_params[key] = value

        temp_model = lgb.LGBMClassifier(is_unbalance=True)
        temp_model.set_params(**aggregated_params)

        X_train_fixed, y_train_fixed = self.global_model_holdout_train
        X_hold_fixed, y_hold_fixed = self.global_model_holdout_test
        temp_model.fit(X_train_fixed, y_train_fixed)
        performance = calc_accuracy(temp_model, X_hold_fixed, y_hold_fixed)
        return performance

    def calculate_shapley_values(self):
        """Calculate Shapley values for each client model and store them in a dictionary keyed by bank name."""
        if self.evaluation_set is None:
            print("Evaluation set not provided.")
            return None

        client_keys = list(self.client_models.keys())
        n = len(client_keys)
        self.shapley_values = {bank: 0 for bank in client_keys}
        base_performance = self._evaluate_model([])

        for i in range(n):
            for subset in combinations(range(n), i + 1):
                subset = list(subset)
                marginal_contribution = self._evaluate_model(subset) - base_performance
                for idx in subset:
                    bank = client_keys[idx]
                    self.shapley_values[bank] += marginal_contribution / (n * self._binom(n - 1, len(subset) - 1))
        print("Shapley values calculated successfully.")

    def _binom(self, n, k):
        from math import factorial
        return factorial(n) / (factorial(k) * factorial(n - k))

    def get_shapley_values(self):
        """Return the dictionary of calculated Shapley values."""
        return self.shapley_values

    def get_global_model(self):
        """Return the fitted global model."""
        return self.global_model
