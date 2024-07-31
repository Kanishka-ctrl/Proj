import streamlit as st
import pandas as pd
import pickle

# Load the saved model and scaler
model_filename = "best_catboost_model.pkl"
scaler_filename = "scaler (1).pkl"


with open(model_filename, 'rb') as model_file:
    best_catboost = pickle.load(model_file)

with open(scaler_filename, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to preprocess future data
def preprocess_future_data(future_data, scaler, selected_features):
    future_data = future_data[selected_features]
    future_data = scaler.transform(future_data)
    return future_data

# Streamlit app
def main():
    st.title("Bond Price Prediction")

    # User inputs
    bond_name = st.text_input("Enter Bond Name:")
    market_demand = st.number_input("Enter Market Demand (e.g., 0.5):", min_value=0.0, max_value=1.0, value=0.5)
    government_support = st.number_input("Enter Government Support (e.g., 0.3):", min_value=0.0, max_value=1.0, value=0.3)

    # Prepare the future data for prediction
    future_data = pd.DataFrame({
        'Market_Demand': [market_demand],
        'Government_Support': [government_support]
    })

    selected_features = ['Market_Demand', 'Government_Support']
    future_data_preprocessed = preprocess_future_data(future_data, scaler, selected_features)

    # Predict future bond price
    if st.button('Predict Bond Price'):
        future_predictions = best_catboost.predict(future_data_preprocessed)
        if bond_name:
            st.write(f"Predicted future bond price for '{bond_name}': ${future_predictions[0]:.2f}")
        else:
            st.write(f"Predicted future bond price: ${future_predictions[0]:.2f}")

if __name__ == "__main__":
    main()
