import pandas as pd
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load('models/RandomForest_model2.pkl')
state_encoder = joblib.load('models/state_encoder.pkl')
district_encoder = joblib.load('models/district_encoder.pkl')
season_encoder = joblib.load('models/season_encoder.pkl')

# Function to safely transform categorical inputs
def safe_transform(encoder, label):
    try:
        return encoder.transform([label])[0]
    except ValueError:
        print(f"Unseen label encountered: {label}")
        return -1  # Handle unseen labels

def preprocess_input(user_input):
    try:
        state_encoded = safe_transform(state_encoder, user_input['State'])
        district_encoded = safe_transform(district_encoder, user_input['District'])
        season_encoded = safe_transform(season_encoder, user_input['Season'])

        if state_encoded == -1 or district_encoded == -1 or season_encoded == -1:
            print(f"Encoding failed for input: {user_input}")
            return None

        input_data = pd.DataFrame([[
            state_encoded,
            district_encoded,
            user_input['Year'],
            season_encoded,
            user_input['Area'],
            user_input['Production']
        ]], columns=["State", "District", "Year", "Season", "Area (Hectare)", "Production (Tonnes)"])

        return input_data
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


# Predict yield using state-based input
def predict_yield(user_input):
    input_data = preprocess_input(user_input)
    if input_data is None:
        return None, None  # Return None if preprocessing fails

    predicted_yield = model.predict(input_data)[0]
    return predicted_yield, (predicted_yield - 0.03, predicted_yield + 0.03)

# Predict yield using fertilizer input
def predict_rice_yield(N, P, K):
    NR_N, NR_P, NR_K = 2.5, 1.0, 2.0
    CS_N, CS_P, CS_K = 0.3, 0.2, 0.25
    CF_N, CF_P, CF_K = 0.6, 0.4, 0.5
    STV_N, STV_P, STV_K = 80, 40, 60

    yield_N = (N * CF_N + CS_N * STV_N) / NR_N
    yield_P = (P * CF_P + CS_P * STV_P) / NR_P
    yield_K = (K * CF_K + CS_K * STV_K) / NR_K

    return min(yield_N, yield_P, yield_K)
