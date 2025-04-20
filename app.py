import streamlit as st
from FedModel import FedModel
from GlobModel import GlobModel
from DB import DB
from utils.constants import FULL_ACCOUNTS_FILE, FULL_TRANSACTIONS_FILE, TRAIN_TARGET_FILE
from utils.page import create_model_row
from utils.evaluation import align_eval_columns, calc_accuracy

# Initialize Database
db = DB(FULL_TRANSACTIONS_FILE, FULL_ACCOUNTS_FILE, TRAIN_TARGET_FILE)
categories = db.retrieve_categories()
encoding_sql = db.encode_data_in_sql(categories)
column_headers = db.retrieve_encoded_transactions(0, 0, encoding_sql).columns

st.title("VoltronX: A Federated Learning Solution for Privacy-Preserving Cooperation in Fraud Detection")

# Define bank names
banks = ['local_1', 'local_2', 'local_3', 'large_1', 'large_2', 'small_digital_1', 'small_digital_2', 'large_digital_1']

# Initialize session state for local models if not already done
if 'fed_models' not in st.session_state:
    st.session_state.fed_models = {bank: FedModel(bank, column_headers) for bank in banks}

# Initialize global model once at the start of the page
if 'global_model' not in st.session_state:
    global_model = GlobModel(column_headers)
    df_eval = db.retrieve_encoded_transactions(1, 6, encoding_sql)
    global_model.set_evaluation_set(df_eval)
    st.session_state.global_model = global_model

st.subheader("Local Models")
# Display the models in two rows
create_model_row(banks[:4], db, encoding_sql)
create_model_row(banks[4:], db, encoding_sql)

st.subheader("Global Model Evaluation")
if st.button("Evaluate Local Model Contributions"):
    if not st.session_state.global_model.client_models:
        st.write("No local models have been received by the global model yet.")
    else:
        with st.spinner('Calculating Shapley Values...'):
            st.session_state.global_model.calculate_shapley_values()
            shapley_dict = st.session_state.global_model.get_shapley_values()
            for bank, contribution in shapley_dict.items():
                st.write(f"Local model from **{bank}** contributed **{contribution:.2f}** to the global model performance.")
            st.write('A higher value indicates a greater positive impact on overall accuracy.')

if st.button("Calculate Global Model Accuracy"):
    global_model = st.session_state.global_model
    with st.status("Calculating Global Model Accuracy...", expanded=True):
        st.write("Aggregating local models...")
        global_model.aggregate_models()
        st.write("Retrieving aggregated model...")
        final_model = global_model.get_global_model()
        st.write("Running aggregated model against evaluation set...")
        X_eval, y_eval = global_model.global_model_holdout_test
        glob_auc = calc_accuracy(final_model, X_eval, y_eval)
    st.write(f"Evaluation of the Global Model returns an AUC of {glob_auc}.")