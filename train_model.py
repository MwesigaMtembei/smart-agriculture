import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
df = pd.read_csv("planting_harvesting_times.csv")

# Encode non-numeric columns
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Save encoders for use later in predictions
joblib.dump(label_encoders, 'label_encoders.pkl')

# Define features and target
X = df.drop("crop", axis=1)
y = df["crop"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'crop_predictor_model.pkl')
print("Model and label encoders saved successfully.")
