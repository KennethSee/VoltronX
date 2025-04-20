import streamlit as st

from .evaluation import align_eval_columns, calc_accuracy

def create_model_row(bank_list, db, encoding_sql):
    cols = st.columns(len(bank_list))
    for bank in bank_list:
        with cols[bank_list.index(bank)]:
            st.write(f"**{bank}**")
            fed_model = st.session_state.fed_models[bank]
            
            # Create a placeholder for current period display
            period_placeholder = st.empty()
            period_placeholder.markdown(
                f"<p style='text-align: center; font-size: 14px;'>Current Period: {fed_model.month}</p>", 
                unsafe_allow_html=True
            )
            
            # Seed the model with data for the current period (only if not already loaded)
            if not hasattr(fed_model, "last_loaded_period") or fed_model.last_loaded_period != fed_model.month:
                new_data = db.retrieve_encoded_bank_transactions(fed_model.bank, fed_model.month, fed_model.month, encoding_sql)
                fed_model.add_data(new_data)
                fed_model.last_loaded_period = fed_model.month
            
            # Train button: when a local model is trained, send it to the global model
            if st.button(f"Train {bank}", key=f"train_{bank}"):
                with st.spinner(f"Training model for {bank}..."):
                    model = fed_model.train_model("IsFraud", fed_model.get_input_columns())
                    st.session_state.global_model.receive_local_model(fed_model)
                st.success(f"Model for {bank} trained successfully!")

            # Evaluate model button
            if st.button(f"Evaluate {bank} model"):
                global_model = st.session_state.global_model
                with st.spinner("Calculating model accuracy..."):
                    X_eval, y_eval = global_model.global_model_holdout_test
                    X_eval = align_eval_columns(X_eval, fed_model.trained_columns)
                    loc_auc = calc_accuracy(fed_model.local_model, X_eval, y_eval)
                    st.write(f"Evaluation of {bank} local model returns an AUC of {loc_auc}.")
            
            # Next Period button: advance only this bank to the next period
            if st.button(f"Next Period for {bank}", key=f"next_{bank}"):
                with st.spinner(f"Advancing {bank} to the next period..."):
                    fed_model.next_period()
                    new_data = db.retrieve_encoded_bank_transactions(fed_model.bank, fed_model.month, fed_model.month, encoding_sql)
                    fed_model.add_data(new_data)
                    fed_model.last_loaded_period = fed_model.month
                    period_placeholder.markdown(
                        f"<p style='text-align: center; font-size: 14px;'>Current Period: {fed_model.month}</p>", 
                        unsafe_allow_html=True
                    )
                st.success(f"{bank} advanced to month {fed_model.month} and new data added.")
