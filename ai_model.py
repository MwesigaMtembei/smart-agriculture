import joblib
import numpy as np
import pandas as pd

# Load model, encoders, and the original dataset
model = joblib.load('crop_predictor_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
df = pd.read_csv('planting_harvesting_times.csv')  # Your CSV

def predict_crop_insights(input_data):
    features = []

    for feature_name in label_encoders.keys():
        value = input_data.get(feature_name)
        if value is None:
            raise ValueError(f"Missing value for {feature_name}")
        encoder = label_encoders[feature_name]
        try:
            encoded_value = encoder.transform([value])[0]
        except ValueError:
            raise ValueError(f"Invalid value '{value}' for {feature_name}. Allowed values: {list(encoder.classes_)}")
        features.append(encoded_value)

    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]

    crop_encoder = label_encoders['crop']
    predicted_crop = crop_encoder.inverse_transform([prediction])[0]

    # Find rows matching the predicted crop
    crop_rows = df[df['crop'] == predicted_crop]

    forecast_details = []
    for _, row in crop_rows.iterrows():
        forecast_details.append({
            'region': row['region'],  # Assuming you have region
            'wakati_unaofaa_kupanda': row['wakati unaofaa kupanda'],
            'wakati_unaofaa_kuvuna': row['wakati unaofaa kuvuna'],
        })

    return {
        'crop': predicted_crop,
        'details': forecast_details
    }
